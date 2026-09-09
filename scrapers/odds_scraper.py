import datetime
import hashlib
import json
import os
import time
import argparse
from pathlib import Path
import requests
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")
SCRAPE_WINDOW_ID = "init_odds_001"
SOURCE = "odds_api"
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0, tzinfo=None)

def sha256_hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_raw_id(source, endpoint, params_hash, ingestion_ts):
    return sha256_hex(f"{source}{endpoint}{params_hash}{ingestion_ts.isoformat()}")

def stable_params_hash(params):
    return sha256_hex(json.dumps(params, sort_keys=True, separators=(",", ":")))

def get_api_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ERROR: ODDS_API_KEY environment variable not set")
        print("Run: export ODDS_API_KEY=your_key_here")
        exit(1)
    return key

def connect_db():
    try:
        return duckdb.connect(DB_PATH)
    except Exception as e:
        print(f"Failed to connect to DuckDB: {e}")
        exit(1)

def insert_row(con, raw_id, endpoint, params_hash, ingestion_ts, payload_json, http_status, parse_status):
    try:
        con.execute("""
            INSERT OR IGNORE INTO raw_api_responses
                (raw_id, source, endpoint, params_hash, ingestion_timestamp_utc,
                 scrape_window_id, raw_payload_json, http_status, parse_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [raw_id, SOURCE, endpoint, params_hash, ingestion_ts,
              SCRAPE_WINDOW_ID, payload_json, http_status, parse_status])
        return True
    except Exception as e:
        print(f"Insert failed for raw_id={raw_id[:16]}...: {e}")
        return False

def fetch_events(api_key):
    url = f"{BASE_URL}/sports/{SPORT}/events"
    params = {"apiKey": api_key}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            print(f"Events fetch failed: HTTP {r.status_code} — {r.text[:200]}")
            return []
        return r.json()
    except Exception as e:
        print(f"Events fetch error: {e}")
        return []


def fetch_historical_events(api_key, snapshot_time):
    """Fetch the event slate as it existed at an ISO-8601 historical time."""
    url = f"{BASE_URL}/historical/sports/{SPORT}/events"
    try:
        response = requests.get(url, params={"apiKey": api_key, "date": snapshot_time}, timeout=15)
        if response.status_code != 200:
            print(f"Historical events fetch failed: HTTP {response.status_code} — {response.text[:200]}")
            return []
        payload = response.json()
        return payload.get("data", payload if isinstance(payload, list) else [])
    except Exception as error:
        print(f"Historical events fetch error: {error}")
        return []

def fetch_event_odds(api_key, event_id, con):
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    hash_params = {"event_id": event_id, "regions": "us",
                   "markets": "player_points,player_rebounds,player_assists",
                   "oddsFormat": "american"}
    params = {**hash_params, "apiKey": api_key}
    params_hash = stable_params_hash(hash_params)
    ingestion_ts = utc_now()
    raw_id = make_raw_id(SOURCE, "event_odds", params_hash, ingestion_ts)

    try:
        r = requests.get(url, params=params, timeout=10)
        http_status = r.status_code
        remaining = r.headers.get("x-requests-remaining", "?")
        used = r.headers.get("x-requests-used", "?")

        if http_status != 200:
            print(f"  Event {event_id} failed: HTTP {http_status}")
            insert_row(con, raw_id, "event_odds", params_hash, ingestion_ts,
                      r.text[:500], http_status, f"error: HTTP {http_status}")
            return 0, remaining, used

        data = r.json()
        payload_json = json.dumps(data)
        n_markets = len(data.get("bookmakers", []))
        insert_row(con, raw_id, "event_odds", params_hash, ingestion_ts,
                  payload_json, http_status, "ok")
        return n_markets, remaining, used

    except Exception as e:
        print(f"  Event {event_id} error: {e}")
        insert_row(con, raw_id, "event_odds", params_hash, ingestion_ts,
                  "{}", 0, f"error: {e}")
        return 0, "?", "?"


def fetch_historical_event_odds(api_key, event_id, snapshot_time, con):
    """Store an immutable historical player-prop snapshot in the raw layer."""
    url = f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds"
    hash_params = {
        "event_id": event_id, "date": snapshot_time, "regions": "us",
        "markets": "player_points,player_rebounds,player_assists", "oddsFormat": "american",
    }
    params_hash = stable_params_hash(hash_params)
    ingestion_ts = utc_now()
    raw_id = make_raw_id(SOURCE, "event_odds", params_hash, ingestion_ts)
    try:
        response = requests.get(url, params={**hash_params, "apiKey": api_key}, timeout=15)
        payload = response.json() if response.status_code == 200 else {}
        event_payload = payload.get("data", payload)
        if isinstance(event_payload, dict):
            event_payload["_snapshot_time"] = payload.get("timestamp", snapshot_time)
        status = "ok" if response.status_code == 200 else f"error: HTTP {response.status_code}"
        insert_row(con, raw_id, "event_odds", params_hash, ingestion_ts,
                   json.dumps(event_payload), response.status_code, status)
        return len(event_payload.get("bookmakers", [])) if isinstance(event_payload, dict) else 0
    except Exception as error:
        insert_row(con, raw_id, "event_odds", params_hash, ingestion_ts, "{}", 0, f"error: {error}")
        return 0

def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect current or historical NBA prop snapshots")
    parser.add_argument("--historical-date", help="ISO-8601 UTC snapshot time supported by The Odds API")
    args = parser.parse_args(argv)
    api_key = get_api_key()
    con = connect_db()

    print("Fetching NBA events...")
    events = fetch_historical_events(api_key, args.historical_date) if args.historical_date else fetch_events(api_key)
    if not events:
        print("No events returned. Check your API key or there may be no games today.")
        return

    print(f"Found {len(events)} events. Fetching odds...\n")
    success = 0
    remaining = "?"
    used = "?"

    for i, event in enumerate(events):
        event_id = event.get("id", "unknown")
        home = event.get("home_team", "?")
        away = event.get("away_team", "?")

        if args.historical_date:
            n_markets = fetch_historical_event_odds(api_key, event_id, args.historical_date, con)
            remaining, used = "?", "?"
        else:
            n_markets, remaining, used = fetch_event_odds(api_key, event_id, con)
        print(f"Scraped event {event_id[:8]}... {away} @ {home} — {n_markets} bookmakers found")

        if n_markets >= 0:
            success += 1

        if i < len(events) - 1:
            time.sleep(1)

    print(f"\nDone. {success}/{len(events)} events scraped successfully.")
    print(f"API credits used: {used} | Remaining: {remaining}")
    con.close()

if __name__ == "__main__":
    main()
