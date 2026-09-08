import datetime
import hashlib
import json
from pathlib import Path
import pytz
import duckdb
from nba_api.stats.endpoints import scoreboardv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")
ET = pytz.timezone("America/New_York")
UTC = pytz.utc

SCHEMA = """
CREATE TABLE IF NOT EXISTS game_schedule (
    game_id             VARCHAR PRIMARY KEY,
    game_date_et        VARCHAR NOT NULL,
    tip_time_et         VARCHAR,
    tip_time_utc        TIMESTAMP,
    close_time_utc      TIMESTAMP,
    home_team           VARCHAR,
    away_team           VARCHAR,
    home_team_id        INTEGER,
    away_team_id        INTEGER,
    ingestion_ts        TIMESTAMP NOT NULL
)
"""

def connect_db():
    return duckdb.connect(DB_PATH)

def ensure_schema(con):
    con.execute(SCHEMA)

def parse_tip_time_utc(game_date_str, tip_time_str):
    """
    Convert NBA tip time to UTC.
    game_date_str: 'YYYY-MM-DD'
    tip_time_str:  'HH:MMpm ET' or similar from nba_api
    Returns UTC datetime or None if unparseable.
    """
    if not tip_time_str or tip_time_str.strip() == "":
        return None
    try:
        # nba_api returns times like '7:30 pm ET'
        clean = tip_time_str.strip().replace("\u00a0", " ")
        dt_str = f"{game_date_str} {clean.replace(' ET', '').strip()}"
        dt_et = ET.localize(datetime.datetime.strptime(dt_str, "%Y-%m-%d %I:%M %p"))
        return dt_et.astimezone(UTC).replace(tzinfo=None)
    except Exception:
        try:
            # fallback: try 24h format
            dt_et = ET.localize(datetime.datetime.strptime(
                f"{game_date_str} {tip_time_str.strip()}", "%Y-%m-%d %H:%M"))
            return dt_et.astimezone(UTC).replace(tzinfo=None)
        except Exception:
            return None

def fetch_todays_schedule():
    """Fetch today's NBA schedule from nba_api ScoreboardV2."""
    today = datetime.datetime.now(ET).strftime("%m/%d/%Y")
    board = scoreboardv2.ScoreboardV2(game_date=today)
    data  = board.get_dict()

    games = []
    try:
        # GameHeader resultSet
        for rs in data.get("resultSets", []):
            if rs["name"] != "GameHeader":
                continue
            headers = rs["headers"]
            h = {v: i for i, v in enumerate(headers)}
            for row in rs["rowSet"]:
                game_id      = str(row[h["GAME_ID"]])
                game_date    = str(row[h["GAME_DATE_EST"]])[:10]
                tip_time_str = str(row[h.get("GAME_STATUS_TEXT", 0)] or "")
                home_id      = int(row[h["HOME_TEAM_ID"]])
                away_id      = int(row[h["VISITOR_TEAM_ID"]])
                games.append({
                    "game_id":       game_id,
                    "game_date_et":  game_date,
                    "tip_time_str":  tip_time_str,
                    "home_team_id":  home_id,
                    "away_team_id":  away_id,
                })
    except Exception as e:
        print(f"Failed to parse GameHeader: {e}")

    return games

def fetch_team_name_map():
    """Build team_id → abbreviation map from nba_api."""
    from nba_api.stats.static import teams
    all_teams = teams.get_teams()
    return {t["id"]: t["abbreviation"] for t in all_teams}

def insert_games(con, games, team_map):
    now = datetime.datetime.utcnow()
    inserted = 0
    for g in games:
        tip_utc   = parse_tip_time_utc(g["game_date_et"], g["tip_time_str"])
        close_utc = (tip_utc - datetime.timedelta(minutes=5)) if tip_utc else None
        home_abbr = team_map.get(g["home_team_id"], str(g["home_team_id"]))
        away_abbr = team_map.get(g["away_team_id"], str(g["away_team_id"]))

        try:
            con.execute("""
                INSERT OR IGNORE INTO game_schedule (
                    game_id, game_date_et, tip_time_et, tip_time_utc,
                    close_time_utc, home_team, away_team,
                    home_team_id, away_team_id, ingestion_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                g["game_id"], g["game_date_et"], g["tip_time_str"],
                tip_utc, close_utc, home_abbr, away_abbr,
                g["home_team_id"], g["away_team_id"], now
            ])
            inserted += 1
        except Exception as e:
            print(f"  Insert failed for game_id={g['game_id']}: {e}")

    return inserted

def get_todays_close_windows(con):
    """
    Return list of (game_id, home, away, tip_time_utc, close_time_utc)
    for games where close_time is in the future.
    Used by CloseSpec to know when to capture final odds snapshot.
    """
    now = datetime.datetime.utcnow()
    rows = con.execute("""
        SELECT game_id, home_team, away_team, tip_time_utc, close_time_utc
        FROM game_schedule
        WHERE close_time_utc IS NOT NULL
          AND tip_time_utc > ?
        ORDER BY tip_time_utc ASC
    """, [now]).fetchall()
    return rows

def main():
    con = connect_db()
    ensure_schema(con)

    print("Fetching today's NBA schedule...")
    games    = fetch_todays_schedule()
    team_map = fetch_team_name_map()

    if not games:
        print("No games found for today.")
        con.close()
        return

    inserted = insert_games(con, games, team_map)
    print(f"Inserted {inserted}/{len(games)} games into game_schedule\n")

    print("=== Today's Games + CloseSpec Windows ===\n")
    rows = get_todays_close_windows(con)
    if rows:
        for r in rows:
            print(f"  {r[1]} vs {r[2]} | tip={r[3]} UTC | close={r[4]} UTC")
    else:
        print("  No upcoming games with tip times parsed.")
        print("  Showing all scheduled games:")
        all_rows = con.execute(
            "SELECT game_id, home_team, away_team, tip_time_et FROM game_schedule"
        ).fetchall()
        for r in all_rows:
            print(f"  {r[2]} @ {r[1]} | tip_et={r[3]}")

    con.close()

if __name__ == "__main__":
    main()
