import datetime
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import duckdb
from nba_api.stats.endpoints import playergamelog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")
SCRAPE_WINDOW_ID = "init_scrape_001"
SOURCE = "nba_api"
ENDPOINT = "PlayerGameLog"
SEASON = os.environ.get("NBA_SEASON", "2024-25")
PLAYER_IDS = [
    1629029, 1628384, 1627759, 1629628, 1630224,
    1628973, 1629639, 1630162, 1628400, 1629685,
]

RAW_API_RESPONSES_DDL = """
CREATE TABLE IF NOT EXISTS raw_api_responses (
    raw_id                  VARCHAR PRIMARY KEY,
    source                  VARCHAR NOT NULL,
    endpoint                VARCHAR NOT NULL,
    params_hash             VARCHAR NOT NULL,
    ingestion_timestamp_utc TIMESTAMP NOT NULL,
    scrape_window_id        VARCHAR,
    raw_payload_json        TEXT NOT NULL,
    http_status             INTEGER,
    parse_status            VARCHAR
)
"""

def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_raw_id(source: str, endpoint: str, params_hash: str, ingestion_timestamp_utc: datetime.datetime) -> str:
    return sha256_hex(f"{source}{endpoint}{params_hash}{ingestion_timestamp_utc.isoformat()}")

def stable_params_hash(params: Dict[str, Any]) -> str:
    try:
        params_json = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256_hex(params_json)
    except TypeError as e:
        print(f"Failed to JSON-serialize params. Error: {e}. Params: {params}")
        return sha256_hex(str(params))

def connect_db(db_path: str) -> Optional[duckdb.DuckDBPyConnection]:
    try:
        return duckdb.connect(db_path)
    except Exception as e:
        print(f"Failed to connect to DuckDB at {db_path}. Error: {e}")
        return None

def ensure_schema(con: duckdb.DuckDBPyConnection) -> bool:
    try:
        con.execute(RAW_API_RESPONSES_DDL)
        return True
    except Exception as e:
        print(f"Failed to ensure schema. Error: {e}")
        return False

def fetch_player_gamelog(player_id: int, season: str) -> Tuple[int, str, str, int]:
    params = {"player_id": str(player_id), "season": season}
    params_hash = stable_params_hash(params)
    try:
        endpoint_obj = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        payload_dict = endpoint_obj.get_dict()
        payload_json = json.dumps(payload_dict, ensure_ascii=False)
        games = 0
        try:
            rs = payload_dict.get("resultSets")
            if isinstance(rs, list) and len(rs) > 0:
                row_set = rs[0].get("rowSet")
                if isinstance(row_set, list):
                    games = len(row_set)
        except Exception as e:
            print(f"Failed to count games for player_id={player_id}. Error: {e}")
        return 200, "ok", payload_json, games
    except Exception as e:
        msg = f"error: {e}"
        print(f"Failed to fetch PlayerGameLog for player_id={player_id}. Error: {e}")
        return 0, msg, "{}", 0

def insert_raw_response(con, raw_id, source, endpoint, params_hash,
                        ingestion_ts, scrape_window_id, raw_payload_json,
                        http_status, parse_status) -> bool:
    sql = """
        INSERT OR IGNORE INTO raw_api_responses (
            raw_id, source, endpoint, params_hash, ingestion_timestamp_utc,
            scrape_window_id, raw_payload_json, http_status, parse_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        con.execute(sql, [raw_id, source, endpoint, params_hash, ingestion_ts,
                          scrape_window_id, raw_payload_json, http_status, parse_status])
        return True
    except Exception as e:
        print(f"Failed inserting row raw_id={raw_id}. Error: {e}")
        return False

def scrape_all_players(con) -> int:
    success = 0
    for idx, player_id in enumerate(PLAYER_IDS, start=1):
        params = {"player_id": str(player_id), "season": SEASON}
        params_hash = stable_params_hash(params)
        ingestion_ts = utc_now()
        raw_id = make_raw_id(SOURCE, ENDPOINT, params_hash, ingestion_ts)
        http_status, parse_status, payload_json, n_games = fetch_player_gamelog(player_id, SEASON)
        inserted = insert_raw_response(
            con=con, raw_id=raw_id, source=SOURCE, endpoint=ENDPOINT,
            params_hash=params_hash, ingestion_ts=ingestion_ts,
            scrape_window_id=SCRAPE_WINDOW_ID, raw_payload_json=payload_json,
            http_status=http_status, parse_status=parse_status,
        )
        if inserted and http_status == 200 and parse_status == "ok":
            success += 1
            print(f"Scraped player {player_id} — {n_games} games found")
        else:
            print(f"FAILED player {player_id}")
        if idx < len(PLAYER_IDS):
            time.sleep(1)
    print(f"Done. {success}/10 players scraped successfully.")
    return success

def main() -> int:
    con = connect_db(DB_PATH)
    if con is None:
        return 1
    try:
        if not ensure_schema(con):
            return 1
        scrape_all_players(con)
        return 0
    finally:
        try:
            con.close()
        except Exception as e:
            print(f"Failed to close DuckDB connection. Error: {e}")

if __name__ == "__main__":
    raise SystemExit(main())
