import duckdb
import time
from pathlib import Path
from nba_api.stats.endpoints import commonplayerinfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")

PLAYER_IDS = [
    1629029, 1628384, 1627759, 1629628, 1630224,
    1628973, 1629639, 1630162, 1628400, 1629685,
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_lookup (
    player_id       INTEGER PRIMARY KEY,
    full_name       VARCHAR NOT NULL,
    first_name      VARCHAR,
    last_name       VARCHAR,
    team            VARCHAR
)
"""

def connect_db():
    return duckdb.connect(DB_PATH)

def ensure_schema(con):
    con.execute(SCHEMA)
    print("OK: player_lookup table ready")

def fetch_player_info(player_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        data = info.get_dict()
        headers = data["resultSets"][0]["headers"]
        row = data["resultSets"][0]["rowSet"][0]
        h = {v: i for i, v in enumerate(headers)}
        first = str(row[h["FIRST_NAME"]])
        last = str(row[h["LAST_NAME"]])
        team = str(row[h["TEAM_ABBREVIATION"]])
        full_name = f"{first} {last}"
        return first, last, full_name, team
    except Exception as e:
        print(f"  Failed to fetch info for player_id={player_id}: {e}")
        return None, None, None, None

def insert_player(con, player_id, full_name, first_name, last_name, team):
    try:
        con.execute("""
            INSERT OR IGNORE INTO player_lookup
                (player_id, full_name, first_name, last_name, team)
            VALUES (?, ?, ?, ?, ?)
        """, [player_id, full_name, first_name, last_name, team])
        return True
    except Exception as e:
        print(f"  Insert failed for player_id={player_id}: {e}")
        return False

def main():
    con = connect_db()
    ensure_schema(con)

    print("\nFetching player info from nba_api...\n")
    success = 0
    for i, player_id in enumerate(PLAYER_IDS):
        first, last, full_name, team = fetch_player_info(player_id)
        if full_name:
            insert_player(con, player_id, full_name, first, last, team)
            print(f"  {player_id} → {full_name} ({team})")
            success += 1
        else:
            print(f"  {player_id} → FAILED")
        if i < len(PLAYER_IDS) - 1:
            time.sleep(1)

    print(f"\nDone. {success}/{len(PLAYER_IDS)} players in lookup table.")

    print("\n--- Current Lookup Table ---")
    rows = con.execute("SELECT player_id, full_name, team FROM player_lookup").fetchall()
    for row in rows:
        print(f"  {row[0]} | {row[1]} | {row[2]}")

    con.close()

if __name__ == "__main__":
    main()
