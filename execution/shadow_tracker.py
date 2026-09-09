"""Calibrated, two-sided shadow prediction pipeline (never places wagers)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from config import load_config, project_path
from execution.betting import apply_exposure_caps, no_vig_probabilities, select_opportunities
from features.pregame_features import build_pregame_features
from models.ensemble import PooledPropEnsemble
from models.hmm_minutes import predict_next_minutes
from storage.event_mapping import nba_team_aliases, nba_team_ids
from storage.dates import parse_game_dates
from storage.name_matching import normalize_name_series


SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_predictions (
    pred_id VARCHAR PRIMARY KEY, created_at TIMESTAMP NOT NULL,
    prediction_time TIMESTAMP, odds_asof_time TIMESTAMP, event_id VARCHAR, game_id VARCHAR,
    player_id INTEGER NOT NULL, player_name VARCHAR, stat VARCHAR NOT NULL,
    market VARCHAR, book VARCHAR NOT NULL, side VARCHAR,
    our_line FLOAT NOT NULL, over_odds INTEGER, under_odds INTEGER, price INTEGER,
    hmm_expected_min FLOAT, qrf_p_over FLOAT, calibrated_p_over FLOAT,
    model_probability FLOAT, book_p_over FLOAT, book_probability FLOAT,
    edge FLOAT, kelly_fraction FLOAT, stake FLOAT,
    signal VARCHAR, calibration_status VARCHAR,
    conformal_lower FLOAT, conformal_upper FLOAT, conformal_width FLOAT,
    close_line FLOAT, clv FLOAT, actual_result FLOAT, outcome FLOAT,
    feature_version VARCHAR, data_version VARCHAR, model_version VARCHAR,
    settled BOOLEAN DEFAULT FALSE
)
"""

MARKET_TO_STAT = {market: stat for stat, market in load_config()["statistics"].items()}


def ensure_schema(con) -> None:
    con.execute(SCHEMA)
    additions = {
        "prediction_time": "TIMESTAMP", "odds_asof_time": "TIMESTAMP",
        "event_id": "VARCHAR", "game_id": "VARCHAR",
        "market": "VARCHAR", "side": "VARCHAR", "price": "INTEGER",
        "calibrated_p_over": "FLOAT", "model_probability": "FLOAT",
        "book_probability": "FLOAT", "stake": "FLOAT", "calibration_status": "VARCHAR",
        "actual_result": "FLOAT", "outcome": "FLOAT", "feature_version": "VARCHAR",
        "data_version": "VARCHAR", "model_version": "VARCHAR",
    }
    for column, dtype in additions.items():
        con.execute(f"ALTER TABLE shadow_predictions ADD COLUMN IF NOT EXISTS {column} {dtype}")


def make_pred_id(event_id, player_id, stat, book, side, line, prediction_time) -> str:
    raw = "|".join(map(str, [event_id, player_id, stat, book, side, line, prediction_time]))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def load_prop_lines(con, prediction_time=None) -> pd.DataFrame:
    """Load each book's latest complete pair as of one decision timestamp."""
    cutoff = prediction_time or dt.datetime.utcnow()
    frame = con.execute("""
        SELECT * EXCLUDE (rn) FROM (
            SELECT event_id, player_name, market, line AS sportsbook_line,
                   over_price, under_price, bookmaker, asof_time AS odds_asof_time,
                   home_team, away_team, commence_time,
                   ROW_NUMBER() OVER (
                       PARTITION BY event_id, player_name, market, bookmaker
                       ORDER BY asof_time DESC, line_id DESC
                   ) rn
            FROM prop_lines
            WHERE asof_time <= ? AND TRY_CAST(commence_time AS TIMESTAMP) > ?
              AND over_price IS NOT NULL AND under_price IS NOT NULL
              AND market IN ('player_points', 'player_rebounds', 'player_assists')
        ) WHERE rn = 1
    """, [cutoff, cutoff]).fetchdf()
    frame["prediction_time"] = cutoff
    return frame


def _build_live_feature_rows(con, props: pd.DataFrame) -> pd.DataFrame:
    lookup = con.execute("SELECT player_id, full_name FROM player_lookup").fetchdf()
    lookup["name_key"] = normalize_name_series(lookup["full_name"])
    props = props.copy()
    props["name_key"] = normalize_name_series(props["player_name"])
    props = props.merge(lookup[["player_id", "name_key"]], on="name_key", how="inner")
    history = con.execute("""
        SELECT * EXCLUDE (row_rank) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY player_id, game_id
                ORDER BY asof_time DESC, feature_version DESC, feature_id DESC
            ) row_rank
            FROM player_game_features
        ) WHERE row_rank = 1
    """).fetchdf()
    history["prediction_time"] = parse_game_dates(history["game_date"])
    table_names = {
        row[0] for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    }
    team_games = con.execute("SELECT * FROM team_game_logs").fetchdf() if "team_game_logs" in table_names else None
    lineups = con.execute("SELECT * FROM starting_lineups").fetchdf() if "starting_lineups" in table_names else None
    availability = con.execute("SELECT * FROM player_availability").fetchdf() if "player_availability" in table_names else None
    opponent_allowed = con.execute("SELECT * FROM opponent_position_allowed").fetchdf() if "opponent_position_allowed" in table_names else None
    event_map = con.execute("SELECT event_id, game_id FROM event_game_map").fetchdf() if "event_game_map" in table_names else pd.DataFrame()
    event_to_game = dict(zip(event_map.get("event_id", []), event_map.get("game_id", [])))
    props["game_id"] = props["event_id"].map(event_to_game).fillna(props["event_id"]).astype(str)
    for context in (lineups, availability):
        if context is not None and not context.empty:
            mapped = context["event_id"].map(event_to_game)
            context["game_id"] = context["game_id"].fillna(mapped).fillna(context["event_id"]).astype(str)
    aliases = nba_team_aliases()
    abbreviation_ids = nba_team_ids()
    current, hmm_rows = [], []
    for _, prop in props.drop_duplicates(["event_id", "player_id"]).iterrows():
        tip_time = pd.to_datetime(prop["commence_time"], utc=True, errors="coerce")
        past = history[
            (history["player_id"] == prop["player_id"])
            & (parse_game_dates(history["game_date"]) < tip_time)
        ].sort_values("game_date")
        try:
            result = predict_next_minutes(int(prop["player_id"]), past["minutes"].dropna().to_numpy(dtype=float))
            state, means = result["next_state_probs"], result["state_means"]
            expected = float(result["expected_minutes"])
            uncertainty = float(np.sqrt(sum(state[name] * (means[name] - expected) ** 2 for name in state)))
            hmm_rows.append({
                "player_id": int(prop["player_id"]), "asof_time": prop["prediction_time"],
                "hmm_expected_minutes": expected, "hmm_p_dnp": state["DNP"],
                "hmm_p_limited": state["Limited"], "hmm_p_rotation": state["Rotation"],
                "hmm_p_featured": state["Featured"], "hmm_minutes_uncertainty": uncertainty,
            })
            latest = past.iloc[-1] if not past.empty else pd.Series(dtype=object)
            team_abbreviation = latest.get("team_abbreviation")
            home_abbreviation = aliases.get(str(prop["home_team"]).strip().lower())
            away_abbreviation = aliases.get(str(prop["away_team"]).strip().lower())
            opponent_abbreviation = away_abbreviation if team_abbreviation == home_abbreviation else home_abbreviation
            current.append({
                "player_id": int(prop["player_id"]), "game_id": str(prop["game_id"]),
                "game_date": prop["commence_time"], "prediction_time": prop["prediction_time"],
                "pts": np.nan, "reb": np.nan, "ast": np.nan, "minutes": np.nan,
                "matchup": f"{prop['away_team']} @ {prop['home_team']}",
                "team_id": latest.get("team_id"),
                "opponent_id": abbreviation_ids.get(opponent_abbreviation),
                "position": latest.get("position"),
            })
        except Exception:
            continue
    if not current:
        return pd.DataFrame()
    successful = {row["player_id"] for row in current}
    props = props[props["player_id"].isin(successful)]
    combined = pd.concat([history, pd.DataFrame(current)], ignore_index=True, sort=False)
    features = build_pregame_features(
        combined, team_games=team_games, hmm_predictions=pd.DataFrame(hmm_rows),
        lineup_predictions=lineups, availability=availability,
        opponent_allowed=opponent_allowed,
    )
    live = features[features["game_id"].isin(props["game_id"].astype(str))].drop(
        columns=["prediction_time"]
    )
    return props.merge(live, on=["player_id", "game_id"], how="inner")


def generate_shadow_predictions(con, artifact_dir: str | Path | None = None, bankroll: float = 1000.0) -> pd.DataFrame:
    ensure_schema(con)
    props = load_prop_lines(con)
    if props.empty:
        return pd.DataFrame()
    features = _build_live_feature_rows(con, props)
    directory = Path(artifact_dir) if artifact_dir else project_path("artifacts")
    outputs = []
    for market, group in features.groupby("market"):
        stat = MARKET_TO_STAT[market]
        artifact_path = directory / f"{stat}.pkl"
        if not artifact_path.exists():
            continue
        model = PooledPropEnsemble.load(artifact_path)
        distribution = model.predict_distribution(group, group["sportsbook_line"])
        prediction = pd.concat([group.reset_index(drop=True), distribution.reset_index(drop=True)], axis=1)
        prediction["stat"] = stat
        prediction["artifact_feature_version"] = model.artifact.feature_version
        prediction["artifact_data_version"] = model.artifact.data_version
        prediction["artifact_model_version"] = model.artifact.model_version
        outputs.append(prediction)
    if not outputs:
        return pd.DataFrame()
    all_predictions = pd.concat(outputs, ignore_index=True)
    ready = all_predictions[all_predictions["calibration_status"] == "READY"].copy()
    opportunities = apply_exposure_caps(select_opportunities(ready), bankroll)
    recommendation_keys = {
        (row.event_id, row.player_id, row.stat, row.bookmaker, row.side, float(row.sportsbook_line)): row
        for row in opportunities.itertuples()
    }
    logged, now = [], dt.datetime.utcnow()
    for row in all_predictions.itertuples():
        book_over, book_under = no_vig_probabilities(int(row.over_price), int(row.under_price))
        for side in ("OVER", "UNDER"):
            recommended = recommendation_keys.get(
                (row.event_id, row.player_id, row.stat, row.bookmaker, side, float(row.sportsbook_line))
            )
            signal = side if recommended is not None else ("NOT_READY" if row.calibration_status != "READY" else "PASS")
            price = int(row.over_price if side == "OVER" else row.under_price)
            model_probability = float(row.p_over if side == "OVER" else row.p_under)
            book_probability = float(book_over if side == "OVER" else book_under)
            record = {
                "pred_id": make_pred_id(row.event_id, row.player_id, row.stat, row.bookmaker, side, row.sportsbook_line, row.prediction_time),
                "created_at": now, "prediction_time": row.prediction_time,
                "odds_asof_time": row.odds_asof_time,
                "event_id": row.event_id, "game_id": row.game_id, "player_id": int(row.player_id),
                "player_name": row.player_name, "stat": row.stat, "market": row.market,
                "book": row.bookmaker, "side": side, "our_line": float(row.sportsbook_line),
                "over_odds": int(row.over_price), "under_odds": int(row.under_price), "price": price,
                "hmm_expected_min": float(row.hmm_expected_minutes) if pd.notna(row.hmm_expected_minutes) else None,
                "qrf_p_over": float(row.p_over_raw), "calibrated_p_over": float(row.p_over),
                "model_probability": model_probability, "book_p_over": float(book_over),
                "book_probability": book_probability,
                "edge": float(recommended.edge) if recommended is not None else None,
                "kelly_fraction": float(recommended.kelly_fraction) if recommended is not None else 0.0,
                "stake": float(recommended.stake) if recommended is not None else 0.0,
                "signal": signal, "calibration_status": row.calibration_status,
                "conformal_lower": float(row.conformal_lower), "conformal_upper": float(row.conformal_upper),
                "conformal_width": float(row.conformal_upper - row.conformal_lower),
                "feature_version": row.artifact_feature_version,
                "data_version": row.artifact_data_version,
                "model_version": row.artifact_model_version,
            }
            columns = list(record)
            con.execute(f"INSERT OR IGNORE INTO shadow_predictions ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})", list(record.values()))
            logged.append(record)
    return pd.DataFrame(logged)


def print_shadow_log(con) -> None:
    frame = con.execute("""
        SELECT prediction_time, event_id, player_name, stat, book, side, our_line,
               calibrated_p_over, edge, stake, signal, clv, settled
        FROM shadow_predictions ORDER BY created_at DESC LIMIT 100
    """).fetchdf()
    print(frame.to_string(index=False) if not frame.empty else "No shadow predictions logged yet.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "log"], nargs="?", default="run")
    parser.add_argument("--db", type=Path, default=project_path("database"))
    parser.add_argument("--artifact-dir", type=Path, default=project_path("artifacts"))
    parser.add_argument("--bankroll", type=float, default=1000.0)
    args = parser.parse_args(argv)
    with duckdb.connect(str(args.db)) as con:
        ensure_schema(con)
        if args.command == "log":
            print_shadow_log(con)
        else:
            result = generate_shadow_predictions(con, args.artifact_dir, args.bankroll)
            print(result.to_string(index=False) if not result.empty else "No predictions generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
