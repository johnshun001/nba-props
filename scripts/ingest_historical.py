"""
Scrape 2023-24 historical gamelogs for all 10 players and materialize into
player_game_features. Safe to re-run — skips players/seasons already in DB.

Run:
    python -m scripts.ingest_historical
"""
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scrapers.nba_scraper import (
    fetch_player_gamelog,
    insert_raw_response,
    make_raw_id,
    stable_params_hash,
    connect_db,
    ensure_schema,
    PLAYER_IDS,
    SOURCE,
    ENDPOINT,
    DB_PATH,
    utc_now,
)

PAST_SEASONS    = ["2023-24"]   # 2024-25 already in DB
SLEEP_BETWEEN   = 1.5           # seconds between requests — nba_api rate limit
SCRAPE_WINDOW   = "historical_ingest_2023_24"


def already_ingested(con, params_hash: str) -> bool:
    """Return True if this exact params_hash already exists in raw_api_responses."""
    result = con.execute(
        "SELECT COUNT(*) FROM raw_api_responses WHERE params_hash = ?",
        [params_hash],
    ).fetchone()
    return int(result[0]) > 0


def ingest_season(con, season: str) -> int:
    """Scrape all players for one season. Returns count of newly inserted rows."""
    print(f"\n=== Season: {season} ===")
    inserted = 0
    for idx, player_id in enumerate(PLAYER_IDS, start=1):
        params      = {"player_id": str(player_id), "season": season}
        params_hash = stable_params_hash(params)

        if already_ingested(con, params_hash):
            print(f"  [{idx:>2}/{len(PLAYER_IDS)}] player={player_id} — already in DB, skipping")
            continue

        ingestion_ts                          = utc_now()
        raw_id                                = make_raw_id(SOURCE, ENDPOINT, params_hash, ingestion_ts)
        http_status, parse_status, payload_json, n_games = fetch_player_gamelog(player_id, season)

        ok = insert_raw_response(
            con=con,
            raw_id=raw_id,
            source=SOURCE,
            endpoint=ENDPOINT,
            params_hash=params_hash,
            ingestion_ts=ingestion_ts,
            scrape_window_id=SCRAPE_WINDOW,
            raw_payload_json=payload_json,
            http_status=http_status,
            parse_status=parse_status,
        )

        if ok and http_status == 200:
            print(f"  [{idx:>2}/{len(PLAYER_IDS)}] player={player_id} season={season} → {n_games} games ✓")
            inserted += 1
        else:
            print(f"  [{idx:>2}/{len(PLAYER_IDS)}] player={player_id} season={season} → FAILED (status={http_status})")

        if idx < len(PLAYER_IDS):
            time.sleep(SLEEP_BETWEEN)

    return inserted


def main() -> int:
    con = connect_db(DB_PATH)
    if con is None:
        print("ERROR: could not connect to DB")
        return 1

    try:
        if not ensure_schema(con):
            print("ERROR: could not ensure schema")
            return 1

        total = 0
        for season in PAST_SEASONS:
            total += ingest_season(con, season)

        print(f"\nDone. {total} new rows inserted across {PAST_SEASONS}.")

        if total > 0:
            print("\nRunning materialize.py to build player_game_features...")
            result = subprocess.run(
                [sys.executable, "-m", "storage.materialize"],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if result.returncode == 0:
                print("Materialize complete.")
            else:
                print("WARNING: materialize.py exited with non-zero status — check output above.")
        else:
            print("No new data — skipping materialize.")

    finally:
        try:
            con.close()
        except Exception:
            pass

    print("\nNext steps:")
    print("  python3 models/hmm_minutes.py")
    print("  python3 models/qrf_model.py")
    print("  python3 tests/replay.py")
    print("  python3 models/calibration.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
