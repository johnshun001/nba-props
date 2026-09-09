# ScrapingHealthCheck — daily hard gate before any bets are generated.
# Rules from spec:
#   game_coverage < 0.90  → No-Trade Day
#   null_rate     > 0.02  → No-Trade Day
# A No-Trade Day produces ZERO bets regardless of model signals.

import datetime
import duckdb
from pathlib import Path

from storage.preflight import MissingPipelineData, require_tables

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")

GAME_COVERAGE_MIN = 0.90
NULL_RATE_MAX     = 0.02


def connect_db():
    return duckdb.connect(DB_PATH)


def check_game_coverage(con, today_str):
    """
    game_coverage = games with at least one odds snapshot today
                    / total games scheduled today
    """
    scheduled = con.execute("""
        SELECT COUNT(*) FROM game_schedule
        WHERE game_date_et = ?
    """, [today_str]).fetchone()[0]

    if scheduled == 0:
        print(f"  No games scheduled for {today_str}")
        return None

    scraped = con.execute("""
        SELECT COUNT(DISTINCT gs.game_id)
        FROM game_schedule gs
        JOIN event_game_map mapping ON mapping.game_id = gs.game_id
        JOIN prop_lines pl ON pl.event_id = mapping.event_id
        WHERE gs.game_date_et = ?
          AND pl.asof_time >= CAST(? AS TIMESTAMP)
          AND pl.asof_time <  CAST(? AS TIMESTAMP)
          AND pl.over_price IS NOT NULL AND pl.under_price IS NOT NULL
    """, [
        today_str,
        f"{today_str} 00:00:00",
        f"{today_str} 23:59:59"
    ]).fetchone()[0]

    coverage = scraped / scheduled if scheduled > 0 else 0.0
    print(f"  game_coverage: {scraped}/{scheduled} = {coverage:.2%}")
    return coverage


def check_prop_coverage(con, today_str):
    """
    prop_coverage = players with at least one prop line today
                    / total players in lookup table
    """
    total_players = con.execute("SELECT COUNT(*) FROM player_lookup").fetchone()[0]
    if total_players == 0:
        return None

    players_with_lines = con.execute("""
        SELECT COUNT(DISTINCT pl.player_name)
        FROM prop_lines pl
        JOIN player_lookup lk ON LOWER(pl.player_name) = LOWER(lk.full_name)
        WHERE pl.asof_time >= CAST(? AS TIMESTAMP)
    """, [f"{today_str} 00:00:00"]).fetchone()[0]

    coverage = players_with_lines / total_players
    print(f"  prop_coverage: {players_with_lines}/{total_players} tracked players = {coverage:.2%}")
    return coverage


def check_null_rate(con, today_str):
    """
    null_rate = rows with null payload or parse error
                / total rows scraped today
    """
    total = con.execute("""
        SELECT COUNT(*) FROM raw_api_responses
        WHERE ingestion_timestamp_utc >= CAST(? AS TIMESTAMP)
    """, [f"{today_str} 00:00:00"]).fetchone()[0]

    if total == 0:
        print("  null_rate: no rows scraped today")
        return None

    bad = con.execute("""
        SELECT COUNT(*) FROM raw_api_responses
        WHERE ingestion_timestamp_utc >= CAST(? AS TIMESTAMP)
          AND (parse_status != 'ok' OR raw_payload_json IS NULL OR raw_payload_json = '{}')
    """, [f"{today_str} 00:00:00"]).fetchone()[0]

    rate = bad / total
    print(f"  null_rate: {bad}/{total} bad rows = {rate:.2%}")
    return rate


def run_health_check(con=None):
    """
    Run all health checks. Returns True if safe to trade, False = No-Trade Day.
    """
    if con is None:
        con = connect_db()

    et_now    = datetime.datetime.utcnow()
    today_str = et_now.strftime("%Y-%m-%d")

    print(f"\n=== ScrapingHealthCheck {today_str} ===\n")

    game_cov  = check_game_coverage(con, today_str)
    check_prop_coverage(con, today_str)
    null_rate = check_null_rate(con, today_str)

    print()
    failures = []

    if game_cov is None:
        failures.append("No games scheduled — cannot assess coverage")
    elif game_cov < GAME_COVERAGE_MIN:
        failures.append(f"game_coverage {game_cov:.2%} < {GAME_COVERAGE_MIN:.0%} threshold")

    if null_rate is None:
        failures.append("No data scraped today")
    elif null_rate > NULL_RATE_MAX:
        failures.append(f"null_rate {null_rate:.2%} > {NULL_RATE_MAX:.0%} threshold")

    if failures:
        print("NO-TRADE DAY — reasons:")
        for f in failures:
            print(f"  ✗ {f}")
        print("\nZero bets will be generated today.")
        return False
    else:
        print("HEALTH CHECK PASSED — safe to run shadow tracker")
        return True


def main():
    con = connect_db()
    try:
        require_tables(con, ["game_schedule", "raw_api_responses"])
    except MissingPipelineData as error:
        print(f"NOT_READY: {error}")
        con.close()
        return 2
    passed = run_health_check(con)
    con.close()
    return 0 if passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
