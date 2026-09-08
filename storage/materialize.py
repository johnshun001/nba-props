import datetime
import hashlib
import json
from pathlib import Path
import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")

GAME_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS player_game_features (
    feature_id              VARCHAR PRIMARY KEY,
    player_id               INTEGER NOT NULL,
    game_id                 VARCHAR NOT NULL,
    game_date               VARCHAR NOT NULL,
    asof_time               TIMESTAMP NOT NULL,
    source_raw_id           VARCHAR NOT NULL,
    matchup                 VARCHAR,
    wl                      VARCHAR,
    minutes                 FLOAT,
    pts                     FLOAT,
    reb                     FLOAT,
    ast                     FLOAT,
    stl                     FLOAT,
    blk                     FLOAT,
    fga                     FLOAT,
    fgm                     FLOAT,
    fg_pct                  FLOAT,
    fg3a                    FLOAT,
    fg3m                    FLOAT,
    tov                     FLOAT,
    plus_minus              FLOAT,
    feature_version         VARCHAR NOT NULL,
    feature_hash            VARCHAR
)
"""

PROP_LINES_SCHEMA = """
CREATE TABLE IF NOT EXISTS prop_lines (
    line_id                 VARCHAR PRIMARY KEY,
    event_id                VARCHAR NOT NULL,
    home_team               VARCHAR,
    away_team               VARCHAR,
    commence_time           VARCHAR,
    bookmaker               VARCHAR NOT NULL,
    player_name             VARCHAR NOT NULL,
    market                  VARCHAR NOT NULL,
    line                    FLOAT NOT NULL,
    over_price              INTEGER,
    under_price             INTEGER,
    asof_time               TIMESTAMP NOT NULL,
    source_raw_id           VARCHAR NOT NULL,
    feature_version         VARCHAR NOT NULL
)
"""

FEATURE_VERSION = "v1.0"


def sha256_hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_feature_id(*args):
    return sha256_hex("".join(str(a) for a in args))


def make_feature_hash(r: dict) -> str:
    """
    Reproducibility hash: sha256(source_raw_id | feature_version | sorted_features_json).
    Deterministic: two materializations of the same raw data always produce the same hash.
    """
    STAT_KEYS = ["minutes", "pts", "reb", "ast", "stl", "blk",
                 "fga", "fgm", "fg_pct", "fg3a", "fg3m", "tov", "plus_minus"]
    features_dict  = {k: round(float(r.get(k, 0) or 0), 6) for k in STAT_KEYS}
    canonical_json = json.dumps(features_dict, sort_keys=True, separators=(",", ":"))
    raw = "|".join([
        str(r.get("source_raw_id", "")),
        str(r.get("feature_version", "")),
        canonical_json,
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def connect_db():
    try:
        return duckdb.connect(DB_PATH)
    except Exception as e:
        print(f"Failed to connect to DuckDB: {e}")
        exit(1)


def ensure_schemas(con):
    con.execute(GAME_LOG_SCHEMA)
    con.execute(PROP_LINES_SCHEMA)
    try:
        con.execute(
            "ALTER TABLE player_game_features ADD COLUMN IF NOT EXISTS feature_hash VARCHAR"
        )
    except Exception:
        pass
    print("OK: feature tables ready")


def load_raw_game_logs(con, asof_time):
    rows = con.execute("""
        SELECT raw_id, ingestion_timestamp_utc, raw_payload_json
        FROM raw_api_responses
        WHERE source = 'nba_api'
          AND endpoint = 'PlayerGameLog'
          AND parse_status = 'ok'
          AND ingestion_timestamp_utc < ?
    """, [asof_time]).fetchall()
    print(f"Found {len(rows)} raw game log rows with ingestion_ts < {asof_time}")
    return rows


def parse_game_log_row(raw_id, ingestion_ts, payload_json, asof_time):
    records = []
    try:
        payload     = json.loads(payload_json)
        result_sets = payload.get("resultSets", [])
        if not result_sets:
            return records
        headers = result_sets[0].get("headers", [])
        row_set = result_sets[0].get("rowSet", [])
        h = {v: i for i, v in enumerate(headers)}

        for row in row_set:
            try:
                player_id  = int(row[h["Player_ID"]])
                game_id    = str(row[h["Game_ID"]])
                game_date  = str(row[h["GAME_DATE"]])
                feature_id = make_feature_id(player_id, game_id, FEATURE_VERSION, raw_id)

                d = {
                    "feature_id":      feature_id,
                    "player_id":       player_id,
                    "game_id":         game_id,
                    "game_date":       game_date,
                    "asof_time":       asof_time,
                    "source_raw_id":   raw_id,
                    "matchup":         str(row[h.get("MATCHUP", 0)] or ""),
                    "wl":              str(row[h.get("WL", 0)] or ""),
                    "minutes":         float(row[h["MIN"]] or 0),
                    "pts":             float(row[h["PTS"]] or 0),
                    "reb":             float(row[h["REB"]] or 0),
                    "ast":             float(row[h["AST"]] or 0),
                    "stl":             float(row[h["STL"]] or 0),
                    "blk":             float(row[h["BLK"]] or 0),
                    "fga":             float(row[h["FGA"]] or 0),
                    "fgm":             float(row[h["FGM"]] or 0),
                    "fg_pct":          float(row[h["FG_PCT"]] or 0),
                    "fg3a":            float(row[h["FG3A"]] or 0),
                    "fg3m":            float(row[h["FG3M"]] or 0),
                    "tov":             float(row[h["TOV"]] or 0),
                    "plus_minus":      float(row[h["PLUS_MINUS"]] or 0),
                    "feature_version": FEATURE_VERSION,
                }
                d["feature_hash"] = make_feature_hash(d)
                records.append(d)
            except Exception as e:
                print(f"  Skipped row for game_id={game_id}: {e}")
    except Exception as e:
        print(f"  Failed to parse raw_id={raw_id[:16]}...: {e}")
    return records


def insert_game_features(con, records):
    inserted = 0
    for r in records:
        try:
            con.execute("""
                INSERT OR IGNORE INTO player_game_features VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, [
                r["feature_id"], r["player_id"], r["game_id"], r["game_date"],
                r["asof_time"], r["source_raw_id"], r["matchup"], r["wl"],
                r["minutes"], r["pts"], r["reb"], r["ast"], r["stl"], r["blk"],
                r["fga"], r["fgm"], r["fg_pct"], r["fg3a"], r["fg3m"],
                r["tov"], r["plus_minus"], r["feature_version"], r["feature_hash"]
            ])
            inserted += 1
        except Exception as e:
            print(f"  Insert failed for feature_id={r['feature_id'][:16]}...: {e}")
    return inserted


def load_raw_odds(con, asof_time):
    rows = con.execute("""
        SELECT raw_id, ingestion_timestamp_utc, raw_payload_json
        FROM raw_api_responses
        WHERE source = 'odds_api'
          AND endpoint = 'event_odds'
          AND parse_status = 'ok'
          AND ingestion_timestamp_utc < ?
    """, [asof_time]).fetchall()
    print(f"Found {len(rows)} raw odds rows with ingestion_ts < {asof_time}")
    return rows


def parse_odds_row(raw_id, ingestion_ts, payload_json, asof_time):
    records = []
    try:
        data          = json.loads(payload_json)
        event_id      = data.get("id", "")
        home_team     = data.get("home_team", "")
        away_team     = data.get("away_team", "")
        commence_time = data.get("commence_time", "")

        for bookmaker in data.get("bookmakers", []):
            book_key = bookmaker.get("key", "")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", outcome.get("name", ""))
                    line        = outcome.get("point", None)
                    price       = outcome.get("price", None)
                    side        = outcome.get("name", "")
                    if line is None:
                        continue
                    line_id     = make_feature_id(
                        event_id, book_key, market_key,
                        player_name, side, FEATURE_VERSION, raw_id
                    )
                    over_price  = price if side == "Over"  else None
                    under_price = price if side == "Under" else None
                    records.append({
                        "line_id":         line_id,
                        "event_id":        event_id,
                        "home_team":       home_team,
                        "away_team":       away_team,
                        "commence_time":   commence_time,
                        "bookmaker":       book_key,
                        "player_name":     player_name,
                        "market":          market_key,
                        "line":            float(line),
                        "over_price":      int(over_price)  if over_price  is not None else None,
                        "under_price":     int(under_price) if under_price is not None else None,
                        "asof_time":       asof_time,
                        "source_raw_id":   raw_id,
                        "feature_version": FEATURE_VERSION,
                    })
    except Exception as e:
        print(f"  Failed to parse odds raw_id={raw_id[:16]}...: {e}")
    return records


def insert_prop_lines(con, records):
    inserted = 0
    for r in records:
        try:
            con.execute("""
                INSERT OR IGNORE INTO prop_lines VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, [
                r["line_id"], r["event_id"], r["home_team"], r["away_team"],
                r["commence_time"], r["bookmaker"], r["player_name"], r["market"],
                r["line"], r["over_price"], r["under_price"], r["asof_time"],
                r["source_raw_id"], r["feature_version"]
            ])
            inserted += 1
        except Exception as e:
            print(f"  Insert failed for line_id={r['line_id'][:16]}...: {e}")
    return inserted


def print_summary(con):
    n_games   = con.execute("SELECT COUNT(*) FROM player_game_features").fetchone()[0]
    n_players = con.execute("SELECT COUNT(DISTINCT player_id) FROM player_game_features").fetchone()[0]
    n_lines   = con.execute("SELECT COUNT(*) FROM prop_lines").fetchone()[0]
    n_markets = con.execute("SELECT COUNT(DISTINCT market) FROM prop_lines").fetchone()[0]
    print(f"\n--- Materialization Summary ---")
    print(f"Game features:  {n_games} rows across {n_players} players")
    print(f"Prop lines:     {n_lines} rows across {n_markets} market types")
    print(f"Layer 1 materialization complete.")


def main():
    asof_time = datetime.datetime.utcnow()
    print(f"Materializing features as-of: {asof_time}\n")
    con = connect_db()
    ensure_schemas(con)

    raw_logs           = load_raw_game_logs(con, asof_time)
    total_game_records = 0
    for raw_id, ingestion_ts, payload_json in raw_logs:
        records = parse_game_log_row(raw_id, ingestion_ts, payload_json, asof_time)
        n       = insert_game_features(con, records)
        total_game_records += n
        print(f"  raw_id={raw_id[:16]}... -> {n} game rows inserted")

    raw_odds            = load_raw_odds(con, asof_time)
    total_line_records  = 0
    for raw_id, ingestion_ts, payload_json in raw_odds:
        records = parse_odds_row(raw_id, ingestion_ts, payload_json, asof_time)
        n       = insert_prop_lines(con, records)
        total_line_records += n
        print(f"  raw_id={raw_id[:16]}... -> {n} prop line rows inserted")

    print_summary(con)
    con.close()


if __name__ == "__main__":
    main()
