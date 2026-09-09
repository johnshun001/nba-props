"""Contract tests for the leakage-free production forecasting path."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis.replay import grouped_metrics, metric_summary, settle_predictions
from execution.betting import pair_prop_prices, select_opportunities
from execution.close_spec import ensure_schema as ensure_close_schema, settle_clv
from execution.drift_monitor import compute_ece, compute_rolling_clv, load_settled_predictions
from execution.shadow_tracker import ensure_schema as ensure_shadow_schema
from features.pregame_features import FEATURE_COLUMNS, apply_snapshot_context, build_pregame_features
from models.ensemble import PooledPropEnsemble, chronological_split, train_pooled_ensemble
from models.train import attach_historical_sportsbook_lines
from models.train import attach_historical_sportsbook_lines
from storage.materialize import parse_odds_row
from storage.pipeline_schema import ensure_pipeline_schemas


def game_rows(players: int = 4, dates: int = 100) -> pd.DataFrame:
    rows = []
    for day in range(dates):
        timestamp = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)
        for player in range(players):
            wave = np.sin(day / 5 + player)
            rows.append({
                "player_id": player + 1,
                "game_id": f"g-{day}",
                "game_date": timestamp,
                "prediction_time": timestamp + pd.Timedelta(hours=18),
                "season": "2023-24",
                "minutes": 26 + player + 3 * wave,
                "pts": 14 + player * 2 + 5 * wave + (day % 3),
                "reb": 4 + player + 2 * np.cos(day / 6),
                "ast": 3 + player / 2 + np.sin(day / 7),
                "matchup": "AAA @ BBB" if day % 2 else "BBB vs. AAA",
            })
    return pd.DataFrame(rows)


def test_current_and_future_results_cannot_change_pregame_features():
    source = game_rows(players=1, dates=30)
    baseline = build_pregame_features(source)
    changed = source.copy()
    changed.loc[10:, ["pts", "reb", "ast", "minutes"]] += 10_000
    rebuilt = build_pregame_features(changed)
    np.testing.assert_allclose(
        baseline.loc[10, FEATURE_COLUMNS].to_numpy(dtype=float),
        rebuilt.loc[10, FEATURE_COLUMNS].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert baseline.loc[10, "pts_avg_5"] == pytest.approx(source.loc[5:9, "pts"].mean())


def test_chronological_periods_are_disjoint_and_ordered():
    features = build_pregame_features(game_rows(players=2, dates=30))
    split = chronological_split(features)
    periods = [split.train, split.validation, split.calibration, split.test]
    assert not any(set(left).intersection(right) for i, left in enumerate(periods) for right in periods[i + 1:])
    times = pd.to_datetime(features["prediction_time"], utc=True)
    assert times.iloc[split.train].max() < times.iloc[split.validation].min()
    assert times.iloc[split.validation].max() < times.iloc[split.calibration].min()
    assert times.iloc[split.calibration].max() < times.iloc[split.test].min()


def test_odds_pairing_never_crosses_events_or_snapshots():
    rows = pd.DataFrame([
        {"event_id": event, "player_name": "A Player", "market": "player_points", "line": 20.5,
         "bookmaker": "book", "asof_time": snapshot, "side": side, "price": price}
        for event, snapshot, side, price in [
            ("event-1", "2024-01-01T10:00Z", "Over", -105),
            ("event-1", "2024-01-01T10:00Z", "Under", -115),
            ("event-2", "2024-01-02T10:00Z", "Over", 110),
        ]
    ])
    paired = pair_prop_prices(rows)
    assert paired["event_id"].tolist() == ["event-1"]
    assert paired.iloc[0]["over_price"] == -105
    assert paired.iloc[0]["under_price"] == -115


def test_materializer_creates_one_deterministic_price_pair():
    payload = {
        "id": "event-1", "home_team": "H", "away_team": "A", "commence_time": "2024-01-02T00:00Z",
        "bookmakers": [{"key": "book", "markets": [{"key": "player_points", "outcomes": [
            {"description": "A Player", "name": "Over", "point": 20.5, "price": -105},
            {"description": "A Player", "name": "Under", "point": 20.5, "price": -115},
        ]}]}],
    }
    raw_time = pd.Timestamp("2024-01-01T09:55Z")
    parsed = parse_odds_row("raw", raw_time, json.dumps(payload), pd.Timestamp("2024-01-01T10:00Z"))
    assert len(parsed) == 1
    assert parsed[0]["event_id"] == "event-1"
    assert (parsed[0]["over_price"], parsed[0]["under_price"]) == (-105, -115)
    assert parsed[0]["asof_time"] == raw_time


def test_training_line_attachment_rejects_post_tip_snapshot():
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE event_game_map (event_id VARCHAR, game_id VARCHAR)")
    con.execute("CREATE TABLE player_lookup (player_id INTEGER, full_name VARCHAR)")
    con.execute("""CREATE TABLE prop_lines (
        event_id VARCHAR, player_name VARCHAR, market VARCHAR, line FLOAT,
        asof_time TIMESTAMP, bookmaker VARCHAR, line_id VARCHAR,
        over_price INTEGER, under_price INTEGER
    )""")
    con.execute("INSERT INTO event_game_map VALUES ('event', 'game')")
    con.execute("INSERT INTO player_lookup VALUES (1, 'Player One')")
    con.execute("""
        INSERT INTO prop_lines VALUES
        ('event', 'Player One', 'player_points', 20.5, '2024-01-01 18:00:00', 'a', 'before', -110, -110),
        ('event', 'Player One', 'player_points', 99.5, '2024-01-01 21:00:00', 'a', 'after', -110, -110)
    """)
    features = pd.DataFrame([{
        "player_id": 1, "game_id": "game", "prediction_time": "2024-01-01T20:00:00Z",
    }])
    attached = attach_historical_sportsbook_lines(con, features)
    assert attached.loc[0, "sportsbook_line_pts"] == pytest.approx(20.5)
    con.close()


def test_snapshot_context_cannot_see_later_lineup_update():
    snapshots = pd.DataFrame([
        {"player_id": 1, "game_id": "game", "prediction_time": "2024-01-01T17:00:00Z"},
        {"player_id": 1, "game_id": "game", "prediction_time": "2024-01-01T19:00:00Z"},
    ])
    lineups = pd.DataFrame([{
        "player_id": 1, "game_id": "game", "asof_time": "2024-01-01T18:00:00Z",
        "starter_probability": 0.9, "bench_probability": 0.1,
    }])
    contextualized = apply_snapshot_context(snapshots, lineup_predictions=lineups)
    assert contextualized.loc[0, "starter_probability"] == pytest.approx(0.5)
    assert contextualized.loc[1, "starter_probability"] == pytest.approx(0.9)


def test_training_line_is_the_latest_snapshot_known_by_prediction_time():
    import duckdb

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE player_lookup (player_id INTEGER, full_name VARCHAR)")
    con.execute("CREATE TABLE event_game_map (event_id VARCHAR, game_id VARCHAR)")
    con.execute("""
        CREATE TABLE prop_lines (
            event_id VARCHAR, player_name VARCHAR, market VARCHAR, line FLOAT,
            over_price INTEGER, under_price INTEGER, asof_time TIMESTAMP,
            bookmaker VARCHAR, line_id VARCHAR
        )
    """)
    con.execute("INSERT INTO player_lookup VALUES (7, 'Exact Player')")
    con.execute("INSERT INTO event_game_map VALUES ('event-1', 'game-1')")
    con.execute("""
        INSERT INTO prop_lines VALUES
            ('event-1', 'Exact Player', 'player_points', 20.5, -110, -110,
             '2024-01-01 17:00:00', 'book', 'past'),
            ('event-1', 'Exact Player', 'player_points', 99.5, -110, -110,
             '2024-01-01 19:00:00', 'book', 'future')
    """)
    features = pd.DataFrame([{
        "player_id": 7, "game_id": "game-1",
        "prediction_time": pd.Timestamp("2024-01-01 18:00:00", tz="UTC"),
    }])
    attached = attach_historical_sportsbook_lines(con, features)
    assert attached.loc[0, "sportsbook_line_pts"] == pytest.approx(20.5)
    con.close()


def test_line_shopping_uses_best_offer_after_deterministic_sort():
    predictions = pd.DataFrame([
        {"event_id": "event", "player_id": 1, "stat": "pts", "bookmaker": "low",
         "sportsbook_line": 19.5, "over_price": 110, "under_price": 110,
         "p_over": 0.5, "p_under": 0.5},
        {"event_id": "event", "player_id": 1, "stat": "pts", "bookmaker": "high",
         "sportsbook_line": 21.5, "over_price": 110, "under_price": 110,
         "p_over": 0.5, "p_under": 0.5},
    ])
    selected = select_opportunities(predictions, minimum_edge=-1.0).set_index("side")
    assert selected.loc["OVER", "bookmaker"] == "low"
    assert selected.loc["UNDER", "bookmaker"] == "high"


@pytest.fixture(scope="module")
def trained_model():
    features = build_pregame_features(game_rows())
    features["sportsbook_line"] = features["pts_avg_10"].fillna(16.5) + 0.5
    return train_pooled_ensemble(features, "pts"), features


def test_artifact_versions_and_prediction_contract_match(trained_model, tmp_path):
    model, features = trained_model
    artifact = model.artifact
    assert artifact.feature_version and artifact.data_version and artifact.model_version
    assert artifact.feature_columns == FEATURE_COLUMNS
    assert "minutes" not in artifact.feature_columns
    path = model.save(tmp_path / "pts.pkl")
    loaded = PooledPropEnsemble.load(path)
    prediction = loaded.predict_distribution(features.tail(3), features.tail(3)["sportsbook_line"])
    assert list(loaded.artifact.feature_columns) == list(model.artifact.feature_columns)
    assert {"q10", "q25", "median", "q75", "q90", "p_over", "p_under"}.issubset(prediction.columns)
    assert np.allclose(prediction["p_over"] + prediction["p_under"], 1.0)
    assert artifact.split_boundaries["validation_end"] < artifact.split_boundaries["calibration_end"]


def test_prediction_settlement_and_full_metric_report(trained_model):
    model, features = trained_model
    sample = features.tail(20).copy().reset_index(drop=True)
    forecast = model.predict_distribution(sample, sample["sportsbook_line"]).reset_index(drop=True)
    replay = pd.concat([
        pd.DataFrame({
            "event_id": sample["game_id"], "player_id": sample["player_id"], "stat": "pts",
            "bookmaker": "book", "sportsbook_line": sample["sportsbook_line"],
            "over_price": -110, "under_price": -110, "actual": sample["pts"],
        }), forecast,
    ], axis=1)
    opportunities = select_opportunities(replay, minimum_edge=-1.0)
    assert {"OVER", "UNDER"}.issubset(set(opportunities["side"]))
    settled = settle_predictions(opportunities.assign(clv=0.2))
    report = metric_summary(settled)
    assert {
        "mae", "rmse", "direction_accuracy", "brier_score", "log_loss",
        "calibration_error", "win_rate", "roi_after_vig", "clv", "maximum_drawdown",
    }.issubset(report)
    breakdowns = grouped_metrics(settled)
    assert {"player", "statistic", "edge_threshold", "sportsbook"}.issubset(breakdowns)


def test_clv_and_result_settlement_match_exact_event():
    import duckdb
    con = duckdb.connect(":memory:")
    ensure_pipeline_schemas(con)
    ensure_shadow_schema(con)
    ensure_close_schema(con)
    con.execute("CREATE TABLE player_lookup (player_id INTEGER, full_name VARCHAR)")
    con.execute("INSERT INTO player_lookup VALUES (7, 'Exact Player')")
    con.execute("INSERT INTO event_game_map VALUES ('right-event', 'game-1', now(), 'test')")
    con.execute("INSERT INTO player_results VALUES (7, 'game-1', now(), 24, 8, 4, 33, 'raw', 'v1')")
    for event, close_line in [("wrong-event", 99.5), ("right-event", 21.5)]:
        con.execute("""
            INSERT INTO close_snapshots VALUES (?, ?, 'game-1', 'Exact Player',
                'player_points', 'book', ?, -110, -110, now(), now(), now())
        """, [f"snap-{event}", event, close_line])
    con.execute("""
        INSERT INTO shadow_predictions (
            pred_id, created_at, event_id, player_id, player_name, stat,
            market, book, side, our_line, settled
        ) VALUES
            ('pred-over', now(), 'right-event', 7, 'Exact Player', 'pts',
             'player_points', 'book', 'OVER', 20.5, FALSE),
            ('pred-under', now(), 'right-event', 7, 'Exact Player', 'pts',
             'player_points', 'book', 'UNDER', 20.5, FALSE)
    """)
    settle_clv(con)
    rows = con.execute("""
        SELECT side, close_line, clv, actual_result, outcome, settled
        FROM shadow_predictions ORDER BY side
    """).fetchall()
    assert rows == [
        ("OVER", 21.5, 1.0, 24.0, 1.0, True),
        ("UNDER", 21.5, -1.0, 24.0, 1.0, True),
    ]
    con.close()


def test_missing_drift_or_calibration_evidence_is_not_ready():
    empty = pd.DataFrame(columns=["calibrated_p_over", "outcome", "clv"])
    assert compute_rolling_clv(empty)["status"] == "NOT_READY"
    assert compute_ece(empty)["status"] == "NOT_READY"


def test_under_calibration_uses_under_probability():
    import duckdb
    con = duckdb.connect(":memory:")
    ensure_shadow_schema(con)
    con.execute("""
        INSERT INTO shadow_predictions (
            pred_id, created_at, player_id, stat, book, side, signal, our_line,
            calibrated_p_over, calibration_status, clv, outcome, settled
        ) VALUES ('under', now(), 1, 'pts', 'book', 'UNDER', 'UNDER', 20.5,
                  0.2, 'READY', 0.5, 1.0, TRUE)
    """)
    loaded = load_settled_predictions(con)
    assert loaded.loc[0, "model_probability"] == pytest.approx(0.8)
    con.close()
