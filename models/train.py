"""CLI and orchestration for chronological pooled ensemble training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

from config import load_config, project_path
from storage.dates import parse_game_dates
from storage.name_matching import normalized_name_sql
from features.pregame_features import build_pregame_features
from models.ensemble import SUPPORTED_STATS, train_pooled_ensemble
from models.hmm_minutes import build_walk_forward_hmm_features
from models.hmm_minutes import fit_hmm, reorder_states, save_model


def _table_exists(con, table: str) -> bool:
    return bool(con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchone()[0])


def attach_historical_sportsbook_lines(con, features: pd.DataFrame) -> pd.DataFrame:
    """Attach one deterministic paired line per game/stat from before tipoff.

    Lines are calibration labels, not model features. The explicit timestamp
    filter prevents a closing or post-game materialization from leaking into a
    nominally pregame training row.
    """
    required_tables = ("player_lookup", "prop_lines", "event_game_map")
    if not all(_table_exists(con, table) for table in required_tables):
        return features
    lines = con.execute(f"""
        SELECT mapped.game_id, pl.market, pl.line, pl.asof_time,
               pl.bookmaker, pl.line_id, lookup.player_id
        FROM prop_lines pl
        JOIN event_game_map mapped ON mapped.event_id = pl.event_id
        JOIN player_lookup lookup
          ON {normalized_name_sql("pl.player_name")} = {normalized_name_sql("lookup.full_name")}
        WHERE pl.over_price IS NOT NULL AND pl.under_price IS NOT NULL
    """).fetchdf()
    if lines.empty:
        return features

    output = features.copy()
    output["prediction_time"] = pd.to_datetime(output["prediction_time"], utc=True, errors="coerce")
    lines["asof_time"] = pd.to_datetime(lines["asof_time"], utc=True, errors="coerce")
    reverse_markets = {market: stat for stat, market in load_config()["statistics"].items()}
    lines["stat"] = lines["market"].map(reverse_markets)
    candidates = output[["player_id", "game_id", "prediction_time"]].merge(
        lines, on=["player_id", "game_id"], how="inner"
    )
    candidates = candidates[
        candidates["stat"].notna()
        & candidates["asof_time"].notna()
        & (candidates["asof_time"] <= candidates["prediction_time"])
    ].copy()
    if candidates.empty:
        return output
    candidates = candidates.sort_values(
        ["player_id", "game_id", "stat", "asof_time", "bookmaker", "line_id"],
        ascending=[True, True, True, False, True, True],
        kind="mergesort",
    ).drop_duplicates(["player_id", "game_id", "stat"], keep="first")
    for stat in SUPPORTED_STATS:
        selected = candidates[candidates["stat"] == stat].rename(
            columns={"line": f"sportsbook_line_{stat}"}
        )
        if selected.empty:
            continue
        output = output.merge(
            selected[["player_id", "game_id", f"sportsbook_line_{stat}"]],
            on=["player_id", "game_id"], how="left", validate="one_to_one",
        )
    return output


def load_training_frame(con) -> pd.DataFrame:
    games = con.execute("""
        SELECT * EXCLUDE (row_rank) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY player_id, game_id
                ORDER BY asof_time DESC, feature_version DESC, feature_id DESC
            ) AS row_rank
            FROM player_game_features
        ) WHERE row_rank = 1
        ORDER BY player_id, game_date, game_id
    """).fetchdf()
    if games.empty:
        raise ValueError("player_game_features is empty; collect and materialize data first")
    if _table_exists(con, "game_schedule"):
        schedule = con.execute(
            "SELECT game_id, tip_time_utc, close_time_utc FROM game_schedule"
        ).fetchdf()
        games = games.merge(schedule, on="game_id", how="left")
        games["prediction_time"] = pd.to_datetime(
            games["close_time_utc"], utc=True, errors="coerce"
        )
        games["prediction_time"] = games["prediction_time"].fillna(
            pd.to_datetime(games["tip_time_utc"], utc=True, errors="coerce")
        )
        games = games.drop(columns=["tip_time_utc", "close_time_utc"])
    else:
        games["prediction_time"] = pd.NaT
    # Historical schedule rows may be unavailable. A mapped sportsbook event
    # still provides its scheduled tip; use five minutes before that event as
    # the decision time, never a postgame materialization timestamp.
    if all(_table_exists(con, table) for table in ("prop_lines", "event_game_map")):
        event_times = con.execute("""
            SELECT mapped.game_id,
                   MIN(TRY_CAST(pl.commence_time AS TIMESTAMP)) AS event_tip_time
            FROM prop_lines pl
            JOIN event_game_map mapped ON mapped.event_id = pl.event_id
            GROUP BY mapped.game_id
        """).fetchdf()
        if not event_times.empty:
            event_times["event_prediction_time"] = (
                pd.to_datetime(event_times["event_tip_time"], utc=True, errors="coerce")
                - pd.Timedelta(minutes=5)
            )
            games = games.merge(
                event_times[["game_id", "event_prediction_time"]],
                on="game_id",
                how="left",
            )
            games["prediction_time"] = pd.to_datetime(
                games["prediction_time"], utc=True, errors="coerce"
            ).fillna(games["event_prediction_time"])
            games = games.drop(columns="event_prediction_time")
    games["prediction_time"] = pd.to_datetime(
        games["prediction_time"], utc=True, errors="coerce"
    ).fillna(parse_game_dates(games["game_date"]))
    team_games = con.execute("SELECT * FROM team_game_logs").fetchdf() if _table_exists(con, "team_game_logs") else None
    if team_games is not None and not team_games.empty and "opponent_id" in games:
        team_ids = team_games[["game_id", "team_id"]].drop_duplicates()
        missing_opponent = games["opponent_id"].isna()
        if missing_opponent.any() and "team_id" in games:
            opponent_lookup = team_ids.merge(team_ids, on="game_id", suffixes=("", "_opponent"))
            opponent_lookup = opponent_lookup[opponent_lookup["team_id"] != opponent_lookup["team_id_opponent"]]
            opponent_lookup = opponent_lookup.rename(columns={"team_id_opponent": "derived_opponent_id"})
            games = games.merge(opponent_lookup, on=["game_id", "team_id"], how="left")
            games["opponent_id"] = games["opponent_id"].fillna(games["derived_opponent_id"])
            games = games.drop(columns="derived_opponent_id")
    hmm = con.execute(
        "SELECT *, prediction_time AS asof_time FROM hmm_pregame_predictions"
    ).fetchdf() if _table_exists(con, "hmm_pregame_predictions") else None
    if hmm is None or hmm.empty:
        hmm = build_walk_forward_hmm_features(games)
    lineups = con.execute("SELECT * FROM starting_lineups").fetchdf() if _table_exists(con, "starting_lineups") else None
    availability = con.execute("SELECT * FROM player_availability").fetchdf() if _table_exists(con, "player_availability") else None
    opponent_allowed = con.execute("SELECT * FROM opponent_position_allowed").fetchdf() if _table_exists(con, "opponent_position_allowed") else None
    features = build_pregame_features(
        games, team_games=team_games, hmm_predictions=hmm,
        lineup_predictions=lineups, availability=availability,
        opponent_allowed=opponent_allowed,
    )
    return attach_historical_sportsbook_lines(con, features)


def train_all(frame: pd.DataFrame, output_dir: str | Path | None = None) -> dict[str, Path]:
    destination = Path(output_dir) if output_dir else project_path("artifacts")
    paths: dict[str, Path] = {}
    for stat in SUPPORTED_STATS:
        stat_frame = frame.copy()
        stat_line = f"sportsbook_line_{stat}"
        if stat_line in stat_frame:
            stat_frame["sportsbook_line"] = stat_frame[stat_line]
        ensemble = train_pooled_ensemble(stat_frame, stat)
        path = ensemble.save(destination / f"{stat}.pkl")
        metadata = {
            "stat": stat,
            "feature_version": ensemble.artifact.feature_version,
            "data_version": ensemble.artifact.data_version,
            "model_version": ensemble.artifact.model_version,
            "feature_columns": ensemble.artifact.feature_columns,
            "split_boundaries": ensemble.artifact.split_boundaries,
            "validation_metrics": ensemble.artifact.validation_metrics,
            "test_metrics": ensemble.artifact.test_metrics,
            "calibration_status": "READY" if ensemble.artifact.calibrator is not None else "NOT_READY",
            "conformal_rows": int(ensemble.artifact.conformal_scores.size),
            "torch_quantile_enabled": ensemble.artifact.torch_quantile_model is not None,
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        paths[stat] = path
    return paths


def train_live_hmms(con, minimum_games: int = 30) -> int:
    """Save one current HMM per player for the shadow inference path."""
    games = con.execute("""
        SELECT player_id, game_id, minutes, game_date,
               ROW_NUMBER() OVER (PARTITION BY player_id, game_id ORDER BY asof_time DESC) row_rank
        FROM player_game_features WHERE minutes IS NOT NULL
        ORDER BY player_id, game_date
    """).fetchdf()
    games = games[games["row_rank"] == 1]
    count = 0
    for player_id, group in games.groupby("player_id", sort=False):
        values = pd.to_numeric(group["minutes"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < int(minimum_games):
            continue
        try:
            hmm_logger = logging.getLogger("hmmlearn.base")
            previous_level = hmm_logger.level
            hmm_logger.setLevel(logging.ERROR)
            try:
                model = reorder_states(fit_hmm(values, n_iter=100))
            finally:
                hmm_logger.setLevel(previous_level)
            save_model(model, int(player_id))
            count += 1
        except Exception:
            continue
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=project_path("database"))
    parser.add_argument("--output-dir", type=Path, default=project_path("artifacts"))
    args = parser.parse_args(argv)
    with duckdb.connect(str(args.db), read_only=True) as con:
        hmm_count = train_live_hmms(con)
        frame = load_training_frame(con)
    paths = train_all(frame, args.output_dir)
    for stat, path in paths.items():
        print(f"{stat}: {path}")
    print(f"hmm_players: {hmm_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
