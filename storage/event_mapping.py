"""Deterministically map sportsbook event IDs to NBA game IDs."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from nba_api.stats.static import teams


def nba_team_aliases() -> dict[str, str]:
    aliases = {}
    for team in teams.get_teams():
        abbreviation = team["abbreviation"].upper()
        for name in (team["full_name"], team["nickname"], abbreviation):
            aliases[str(name).strip().lower()] = abbreviation
    return aliases


def nba_team_ids() -> dict[str, int]:
    return {team["abbreviation"].upper(): int(team["id"]) for team in teams.get_teams()}


def map_events_to_games(con, aliases: dict[str, str] | None = None) -> int:
    """Map on normalized teams and tip time, rejecting ambiguous matches."""
    aliases = aliases or nba_team_aliases()
    events = con.execute("""
        SELECT DISTINCT event_id, home_team, away_team, commence_time
        FROM prop_lines WHERE event_id IS NOT NULL
    """).fetchdf()
    schedule = con.execute("""
        SELECT game_id, home_team, away_team, tip_time_utc, game_date_et
        FROM game_schedule
    """).fetchdf()
    if events.empty or schedule.empty:
        return 0
    events["home_key"] = events["home_team"].astype(str).str.strip().str.lower().map(aliases)
    events["away_key"] = events["away_team"].astype(str).str.strip().str.lower().map(aliases)
    events["tip"] = pd.to_datetime(events["commence_time"], utc=True, errors="coerce")
    schedule["home_key"] = schedule["home_team"].astype(str).str.upper()
    schedule["away_key"] = schedule["away_team"].astype(str).str.upper()
    schedule["tip"] = pd.to_datetime(schedule["tip_time_utc"], utc=True, errors="coerce")
    # A game scraped after tipoff has no parseable tip time, and that loss is
    # permanent. Two given teams meet at most once on a calendar date, so the
    # Eastern game date is an unambiguous fallback key.
    schedule["date_key"] = pd.to_datetime(
        schedule["game_date_et"], errors="coerce"
    ).dt.date
    events["date_key"] = events["tip"].dt.tz_convert("America/New_York").dt.date
    inserted = 0
    for event in events.itertuples():
        same_teams = schedule[
            (schedule["home_key"] == event.home_key)
            & (schedule["away_key"] == event.away_key)
        ]
        candidates = same_teams[
            (same_teams["tip"] - event.tip).abs() <= pd.Timedelta(hours=2)
        ]
        source = "teams_and_tip"
        if candidates.empty:
            candidates = same_teams[
                same_teams["tip"].isna() & (same_teams["date_key"] == event.date_key)
            ]
            source = "teams_and_date"
        if len(candidates) != 1:
            continue
        con.execute(
            "INSERT OR REPLACE INTO event_game_map VALUES (?, ?, ?, ?)",
            [event.event_id, candidates.iloc[0]["game_id"], datetime.utcnow(), source],
        )
        inserted += 1
    return inserted


__all__ = ["map_events_to_games", "nba_team_aliases", "nba_team_ids"]
