"""Leakage-free chronological replay against timestamped sportsbook lines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

from config import project_path
from execution.betting import american_to_decimal, select_opportunities
from features.pregame_features import apply_snapshot_context
from models.ensemble import SUPPORTED_STATS, train_pooled_ensemble
from models.train import load_training_frame


def _table_exists(con, table: str) -> bool:
    return bool(con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0])


def calibration_error(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        )
        if mask.any():
            error += mask.mean() * abs(probabilities[mask].mean() - outcomes[mask].mean())
    return float(error)


def maximum_drawdown(pnl: pd.Series) -> float:
    equity = pd.to_numeric(pnl, errors="coerce").fillna(0.0).cumsum()
    if equity.empty:
        return 0.0
    wealth = 1.0 + equity
    peak = wealth.cummax()
    return float(((peak - wealth) / peak.clip(lower=1e-9)).max())


def settle_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    settled = predictions.copy()
    settled["actual_over"] = (settled["actual"] > settled["sportsbook_line"]).astype(int)
    settled["is_push"] = settled["actual"] == settled["sportsbook_line"]
    settled["predicted_side"] = np.where(settled["p_over"] >= 0.5, "OVER", "UNDER")
    settled["direction_correct"] = np.where(
        settled["is_push"], np.nan,
        ((settled["predicted_side"] == "OVER") == (settled["actual_over"] == 1)).astype(float),
    )
    if "side" in settled:
        won = ((settled["side"] == "OVER") & (settled["actual_over"] == 1)) | (
            (settled["side"] == "UNDER") & (settled["actual_over"] == 0)
        )
        settled["win"] = np.where(settled["is_push"], np.nan, won.astype(float))
        settled["pnl"] = np.where(
            settled["is_push"], 0.0,
            np.where(won, settled["price"].map(american_to_decimal) - 1.0, -1.0),
        )
    return settled


def metric_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {"n": 0}
    non_push = frame[~frame["is_push"]] if "is_push" in frame else frame
    actual = frame["actual"].to_numpy(dtype=float)
    median = frame["median"].to_numpy(dtype=float)
    result: dict[str, Any] = {
        "n": int(len(frame)),
        "mae": float(mean_absolute_error(actual, median)),
        "rmse": float(mean_squared_error(actual, median) ** 0.5),
        "direction_accuracy": None,
        "brier_score": None,
        "log_loss": None,
        "calibration_error": None,
    }
    if not non_push.empty:
        probabilities = np.clip(non_push["p_over"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        outcomes = non_push["actual_over"].to_numpy(dtype=int)
        result.update({
            "direction_accuracy": float(non_push["direction_correct"].mean()),
            "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
            "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1])),
            "calibration_error": calibration_error(probabilities, outcomes),
        })
    if "win" in frame:
        bets = frame[frame["win"].notna()]
        result.update({
            "win_rate": float(bets["win"].mean()) if len(bets) else None,
            "roi_after_vig": float(bets["pnl"].sum() / len(bets)) if len(bets) else None,
            "clv": float(bets["clv"].mean()) if "clv" in bets and bets["clv"].notna().any() else None,
            "maximum_drawdown": maximum_drawdown(bets["pnl"]) if len(bets) else 0.0,
        })
    return result


def grouped_metrics(frame: pd.DataFrame) -> dict[str, list[dict]]:
    output = {}
    working = frame.copy()
    if "edge" in working:
        working["edge_bucket"] = pd.cut(
            working["edge"], [-np.inf, 0.025, 0.05, 0.10, np.inf],
            labels=["<2.5%", "2.5-5%", "5-10%", "10%+"],
        ).astype(str)
    groups = {"player": ["player_id"], "statistic": ["stat"], "edge_threshold": ["edge_bucket"], "sportsbook": ["bookmaker"]}
    for name, columns in groups.items():
        if all(column in working for column in columns):
            rows = []
            for key, group in working.groupby(columns, dropna=False):
                values = key if isinstance(key, tuple) else (key,)
                rows.append({**dict(zip(columns, values)), **metric_summary(group)})
            output[name] = rows
    return output


def run_walk_forward(frame: pd.DataFrame, *, minimum_history_dates: int = 40) -> pd.DataFrame:
    """Train/calibrate on earlier data and predict each next date as a batch."""
    required = {"prediction_time", "stat", "sportsbook_line", "over_price", "under_price", "actual"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Replay frame is missing: {missing}")
    data = frame.copy()
    data["prediction_time"] = pd.to_datetime(data["prediction_time"], utc=True, errors="coerce")
    dates = np.asarray(sorted(data["prediction_time"].dropna().dt.normalize().unique()))
    rows = []
    for date in dates[int(minimum_history_dates):]:
        historical = data[data["prediction_time"].dt.normalize() < date]
        future = data[data["prediction_time"].dt.normalize() == date]
        for stat in SUPPORTED_STATS:
            train = historical[historical["stat"] == stat].copy()
            predict = future[future["stat"] == stat].copy()
            if train.empty or predict.empty:
                continue
            # Base-model fitting is one observation per player/game. Multiple
            # books and intraday snapshots must not reweight the realized game.
            train = train.sort_values(
                ["prediction_time", "bookmaker", "event_id"], kind="mergesort"
            ).drop_duplicates(["player_id", "game_id", "stat"], keep="last")
            train[stat] = train["actual"]
            try:
                model = train_pooled_ensemble(train, stat)
            except ValueError:
                continue
            distribution = model.predict_distribution(predict, predict["sportsbook_line"])
            candidates = pd.concat([predict.reset_index(drop=True), distribution.reset_index(drop=True)], axis=1)
            candidates = candidates[candidates["calibration_status"] == "READY"]
            selected = select_opportunities(candidates)
            if not selected.empty:
                if "close_line" in selected:
                    selected["clv"] = np.where(
                        selected["side"] == "OVER",
                        selected["close_line"] - selected["sportsbook_line"],
                        selected["sportsbook_line"] - selected["close_line"],
                    )
                rows.append(selected)
    return settle_predictions(pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()


def load_replay_frame(con) -> pd.DataFrame:
    """Load only event-mapped historical snapshots and final results."""
    features = load_training_frame(con)
    has_close = bool(con.execute("""
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_name = 'close_snapshots' AND column_name = 'event_id'
    """).fetchone()[0])
    close_join = """
        LEFT JOIN (
            SELECT event_id, lower(trim(player_name)) player_key, market, bookmaker,
                   arg_max(close_line, snapshot_ts) close_line
            FROM close_snapshots
            GROUP BY event_id, player_key, market, bookmaker
        ) closing
          ON closing.event_id = lines.event_id
         AND closing.player_key = lower(trim(lines.player_name))
         AND closing.market = lines.market AND closing.bookmaker = lines.bookmaker
    """ if has_close else ""
    close_select = ", closing.close_line" if has_close else ", CAST(NULL AS FLOAT) AS close_line"
    odds = con.execute(f"""
        SELECT map.game_id, lines.event_id, lookup.player_id,
               CASE lines.market WHEN 'player_points' THEN 'pts'
                   WHEN 'player_rebounds' THEN 'reb' WHEN 'player_assists' THEN 'ast' END stat,
               lines.line sportsbook_line, lines.over_price, lines.under_price,
               lines.bookmaker, lines.asof_time line_snapshot_time, lines.line_id
               {close_select}
        FROM prop_lines lines
        JOIN event_game_map map ON map.event_id = lines.event_id
        JOIN player_lookup lookup ON lower(trim(lookup.full_name)) = lower(trim(lines.player_name))
        {close_join}
        WHERE lines.over_price IS NOT NULL AND lines.under_price IS NOT NULL
          AND lines.market IN ('player_points', 'player_rebounds', 'player_assists')
    """).fetchdf()
    if odds.empty:
        raise ValueError("No event-mapped, paired historical sportsbook snapshots are available")
    long_features = []
    for stat in SUPPORTED_STATS:
        copy = features.copy()
        copy["stat"] = stat
        copy["actual"] = copy[stat]
        long_features.append(copy)
    feature_long = pd.concat(long_features, ignore_index=True)
    replay = odds.merge(
        feature_long,
        on=["player_id", "game_id", "stat"],
        how="inner",
        validate="many_to_one",
    )
    replay["line_snapshot_time"] = pd.to_datetime(
        replay["line_snapshot_time"], utc=True, errors="coerce"
    )
    replay["prediction_time"] = pd.to_datetime(
        replay["prediction_time"], utc=True, errors="coerce"
    )
    replay = replay[
        replay["line_snapshot_time"].notna()
        & replay["prediction_time"].notna()
        & (replay["line_snapshot_time"] <= replay["prediction_time"])
    ].copy()
    # At one reproducible decision time per game, retain the newest complete
    # pair at each book. Earlier intraday snapshots are not independent bets
    # and must never be mixed into a later line-shopping decision.
    replay = replay.sort_values(
        ["event_id", "player_id", "stat", "bookmaker", "line_snapshot_time", "line_id"],
        ascending=[True, True, True, True, False, False],
        kind="mergesort",
    ).drop_duplicates(
        ["event_id", "player_id", "stat", "bookmaker"], keep="first"
    )
    # Re-evaluate genuinely time-varying context at the decision timestamp.
    lineups = con.execute("SELECT * FROM starting_lineups").fetchdf() if _table_exists(con, "starting_lineups") else None
    availability = con.execute("SELECT * FROM player_availability").fetchdf() if _table_exists(con, "player_availability") else None
    opponent_allowed = con.execute("SELECT * FROM opponent_position_allowed").fetchdf() if _table_exists(con, "opponent_position_allowed") else None
    return apply_snapshot_context(
        replay, lineup_predictions=lineups, availability=availability,
        opponent_allowed=opponent_allowed,
    ).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=project_path("database"))
    parser.add_argument("--output", type=Path, default=project_path("replay_output"))
    args = parser.parse_args(argv)
    try:
        with duckdb.connect(str(args.db), read_only=True) as con:
            source = load_replay_frame(con)
    except ValueError as error:
        print(f"NOT_READY: {error}")
        return 2
    replay = run_walk_forward(source)
    if replay.empty:
        print("No replay predictions were produced.")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    replay.to_csv(args.output, index=False)
    print(metric_summary(replay))
    print(grouped_metrics(replay))
    print(f"Saved {len(replay)} settled predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
