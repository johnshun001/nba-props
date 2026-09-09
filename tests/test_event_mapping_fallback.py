"""Event mapping must survive the loss of a parseable tip time.

``schedule_scraper`` reads ``GAME_STATUS_TEXT``, which becomes ``Final`` or
``1st Qtr`` once a game starts and no longer parses to a timestamp. Mapping
previously required a non-null ``tip_time_utc``, so any game whose schedule was
first collected after tipoff could never be matched to its sportsbook event.
"""

from __future__ import annotations

import duckdb
import pytest

from storage.event_mapping import map_events_to_games


ALIASES = {"los angeles lakers": "LAL", "sacramento kings": "SAC",
           "boston celtics": "BOS", "philadelphia 76ers": "PHI"}


def _database(schedule_rows, line_rows):
    con = duckdb.connect(":memory:")
    con.execute("""CREATE TABLE game_schedule (
        game_id VARCHAR, home_team VARCHAR, away_team VARCHAR,
        tip_time_utc TIMESTAMP, game_date_et VARCHAR)""")
    con.executemany("INSERT INTO game_schedule VALUES (?, ?, ?, ?, ?)", schedule_rows)
    con.execute("""CREATE TABLE prop_lines (
        event_id VARCHAR, home_team VARCHAR, away_team VARCHAR, commence_time VARCHAR)""")
    con.executemany("INSERT INTO prop_lines VALUES (?, ?, ?, ?)", line_rows)
    con.execute("""CREATE TABLE event_game_map (
        event_id VARCHAR PRIMARY KEY, game_id VARCHAR, asof_time TIMESTAMP, source VARCHAR)""")
    return con


def test_maps_on_tip_time_when_available():
    con = _database(
        [("0022500878", "LAL", "SAC", "2026-03-02 02:30:00", "2026-03-01")],
        [("evt1", "Los Angeles Lakers", "Sacramento Kings", "2026-03-02T02:30:00Z")],
    )
    assert map_events_to_games(con, ALIASES) == 1
    row = con.execute("SELECT game_id, source FROM event_game_map").fetchone()
    assert row == ("0022500878", "teams_and_tip")


def test_maps_on_date_when_tip_time_was_lost():
    """The regression: schedule collected after tipoff has a NULL tip time."""
    con = _database(
        [("0022500874", "BOS", "PHI", None, "2026-03-01")],
        [("evt2", "Boston Celtics", "Philadelphia 76ers", "2026-03-02T01:00:00Z")],
    )
    assert map_events_to_games(con, ALIASES) == 1
    row = con.execute("SELECT game_id, source FROM event_game_map").fetchone()
    assert row == ("0022500874", "teams_and_date")


def test_date_fallback_respects_eastern_calendar_date():
    """A 01:00 UTC tip is still the previous evening in Eastern time."""
    con = _database(
        [("0022500874", "BOS", "PHI", None, "2026-03-02")],
        [("evt3", "Boston Celtics", "Philadelphia 76ers", "2026-03-02T01:00:00Z")],
    )
    # 2026-03-02 01:00 UTC is 2026-03-01 20:00 ET, so the 03-02 row must not match.
    assert map_events_to_games(con, ALIASES) == 0


def test_ambiguous_dates_are_rejected():
    """Never guess when two same-matchup rows share a date."""
    con = _database(
        [("0022500874", "BOS", "PHI", None, "2026-03-01"),
         ("0022500999", "BOS", "PHI", None, "2026-03-01")],
        [("evt4", "Boston Celtics", "Philadelphia 76ers", "2026-03-02T01:00:00Z")],
    )
    assert map_events_to_games(con, ALIASES) == 0
    assert con.execute("SELECT COUNT(*) FROM event_game_map").fetchone()[0] == 0


@pytest.mark.parametrize("home,away", [("Boston Celtics", "Los Angeles Lakers")])
def test_mismatched_teams_do_not_map(home, away):
    con = _database(
        [("0022500874", "BOS", "PHI", None, "2026-03-01")],
        [("evt5", home, away, "2026-03-02T01:00:00Z")],
    )
    assert map_events_to_games(con, ALIASES) == 0
