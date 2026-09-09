"""Collect team context and ingest timestamped lineup/injury snapshots.

NBA team logs come from ``LeagueGameLog``. Lineup and availability feeds vary
by provider, so this module exposes a provider-neutral JSON ingestion command;
every record is timestamped and versioned before it can enter feature building.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from config import load_config, project_path
from storage.pipeline_schema import ensure_pipeline_schemas


def _utc_naive(value=None) -> datetime:
    parsed = pd.to_datetime(value or datetime.now(timezone.utc), utc=True).to_pydatetime()
    return parsed.replace(tzinfo=None)


def collect_team_game_logs(season: str) -> list[dict]:
    """Fetch team logs and derive pace/defense from paired game box scores."""
    response = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="T",
    ).get_dict()
    result = response.get("resultSets", [{}])[0]
    headers = result.get("headers", [])
    index = {name: position for position, name in enumerate(headers)}
    raw_rows = []
    for raw in result.get("rowSet", []):
        matchup = str(raw[index.get("MATCHUP", 0)] or "")
        possessions = (
            float(raw[index["FGA"]]) - float(raw[index["OREB"]])
            + float(raw[index["TOV"]]) + 0.44 * float(raw[index["FTA"]])
        )
        raw_rows.append({
            "team_id": int(raw[index["TEAM_ID"]]),
            "team_abbreviation": str(raw[index["TEAM_ABBREVIATION"]]),
            "game_id": str(raw[index["GAME_ID"]]),
            "game_date": str(raw[index["GAME_DATE"]]),
            "opponent_abbreviation": matchup.split()[-1] if matchup else None,
            "is_home": "@" not in matchup,
            "possessions": possessions,
            "points": float(raw[index["PTS"]]),
        })
    by_game_team = {(row["game_id"], row["team_abbreviation"]): row for row in raw_rows}
    rows = []
    for row in raw_rows:
        opponent = by_game_team.get((row["game_id"], row["opponent_abbreviation"]))
        game_possessions = np.mean([row["possessions"], opponent["possessions"]]) if opponent else row["possessions"]
        rows.append({
            **row,
            "pace": float(game_possessions),
            "def_rating": float(100.0 * opponent["points"] / max(game_possessions, 1.0)) if opponent else None,
        })
    return rows


def insert_team_game_logs(con, records: Iterable[dict], *, asof_time=None) -> int:
    ensure_pipeline_schemas(con)
    version = load_config()["versions"]["data"]
    timestamp = _utc_naive(asof_time)
    records = list(records)
    abbreviation_to_id = {
        str(record.get("team_abbreviation")): int(record["team_id"])
        for record in records if record.get("team_abbreviation") and record.get("team_id") is not None
    }
    inserted = 0
    for record in records:
        opponent_id = record.get("opponent_id") or abbreviation_to_id.get(str(record.get("opponent_abbreviation")))
        con.execute("""
            INSERT OR REPLACE INTO team_game_logs (
                team_id, game_id, game_date, opponent_id, is_home, pace,
                def_rating, asof_time, source_raw_id, data_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            int(record["team_id"]), str(record["game_id"]), record["game_date"],
            opponent_id, bool(record.get("is_home", False)), record.get("pace"),
            record.get("def_rating"), timestamp, record.get("source_raw_id"), version,
        ])
        inserted += 1
    return inserted


def ingest_snapshot_records(con, table: str, records: Iterable[dict]) -> int:
    """Ingest provider-neutral lineup, availability, or position-defense JSON."""
    if table not in {"starting_lineups", "player_availability", "opponent_position_allowed"}:
        raise ValueError("unsupported context snapshot table")
    ensure_pipeline_schemas(con)
    version = load_config()["versions"]["data"]
    inserted = 0
    for record in records:
        asof = _utc_naive(record.get("asof_time"))
        if table == "opponent_position_allowed":
            con.execute("""
                INSERT OR REPLACE INTO opponent_position_allowed VALUES (?, ?, ?, ?, ?, ?)
            """, [int(record["opponent_id"]), str(record["position"]),
                  float(record["opponent_allowed_position"]), asof,
                  record.get("source_raw_id"), version])
        elif table == "starting_lineups":
            common = [str(record["event_id"]), record.get("game_id"), int(record["player_id"])]
            con.execute("""
                INSERT OR REPLACE INTO starting_lineups VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [*common, record.get("starter_probability"), record.get("bench_probability"),
                  asof, record.get("source_raw_id"), version])
        else:
            common = [str(record["event_id"]), record.get("game_id"), int(record["player_id"])]
            con.execute("""
                INSERT OR REPLACE INTO player_availability VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [*common, record.get("status"), record.get("play_probability"),
                  record.get("teammate_absence_count"), record.get("teammate_absent_usage"),
                  record.get("usage_change_indicator"), asof, record.get("source_raw_id"), version])
        inserted += 1
    return inserted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=project_path("database"))
    parser.add_argument("--season", choices=load_config()["seasons"])
    parser.add_argument("--lineups-json", type=Path)
    parser.add_argument("--availability-json", type=Path)
    parser.add_argument("--position-defense-json", type=Path)
    args = parser.parse_args(argv)
    with duckdb.connect(str(args.db)) as con:
        if args.season:
            print(f"team_game_logs: {insert_team_game_logs(con, collect_team_game_logs(args.season))}")
        for table, path in [
            ("starting_lineups", args.lineups_json),
            ("player_availability", args.availability_json),
            ("opponent_position_allowed", args.position_defense_json),
        ]:
            if path:
                payload = json.loads(path.read_text(encoding="utf-8"))
                print(f"{table}: {ingest_snapshot_records(con, table, payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
