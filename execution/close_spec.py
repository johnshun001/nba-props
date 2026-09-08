# Internal CloseSpec — deterministic rule: close_time = tip_time - 5min
# For each game, finds the last odds snapshot captured before close_time
# and uses that as the benchmark line for CLV calculation.
# Hard gate: if no snapshot exists within the window, CLV = NULL (excluded).

import datetime
import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS close_snapshots (
    snap_id         VARCHAR PRIMARY KEY,
    game_id         VARCHAR NOT NULL,
    player_name     VARCHAR NOT NULL,
    market          VARCHAR NOT NULL,
    bookmaker       VARCHAR NOT NULL,
    close_line      FLOAT NOT NULL,
    close_over_price  INTEGER,
    close_under_price INTEGER,
    close_time_utc  TIMESTAMP NOT NULL,
    snapshot_ts     TIMESTAMP NOT NULL,
    ingestion_ts    TIMESTAMP NOT NULL
)
"""

def connect_db():
    return duckdb.connect(DB_PATH)

def ensure_schema(con):
    con.execute(SCHEMA)

def make_snap_id(game_id, player_name, market, bookmaker):
    import hashlib
    raw = f"{game_id}{player_name}{market}{bookmaker}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def get_games_needing_close(con):
    """Get games where close_time_utc has passed but we haven't captured a close snapshot yet."""
    now = datetime.datetime.utcnow()
    return con.execute("""
        SELECT game_id, home_team, away_team, close_time_utc, tip_time_utc
        FROM game_schedule
        WHERE close_time_utc IS NOT NULL
          AND close_time_utc <= ?
          AND game_id NOT IN (SELECT DISTINCT game_id FROM close_snapshots)
        ORDER BY close_time_utc ASC
    """, [now]).fetchall()

def get_last_odds_before_close(con, close_time_utc):
    """
    For each (player_name, market, bookmaker), find the last prop_line
    snapshot with asof_time <= close_time_utc.
    This is the CloseSpec line.
    """
    return con.execute("""
        SELECT
            p.player_name,
            p.market,
            p.bookmaker,
            p.line,
            p.over_price,
            p.under_price,
            MAX(p.asof_time) as snapshot_ts
        FROM prop_lines p
        WHERE p.asof_time <= ?
          AND p.market IN ('player_points', 'player_rebounds', 'player_assists')
        GROUP BY p.player_name, p.market, p.bookmaker, p.line, p.over_price, p.under_price
    """, [close_time_utc]).fetchall()

def capture_close_snapshots(con):
    """
    Main CloseSpec capture function.
    For each game past its close_time, find the last odds snapshot and store it.
    """
    ensure_schema(con)
    now = datetime.datetime.utcnow()
    games = get_games_needing_close(con)

    if not games:
        print("No games needing close snapshot right now.")
        return 0

    total = 0
    for game_id, home, away, close_time, tip_time in games:
        print(f"\n{away} @ {home} | close_time={close_time} UTC")
        rows = get_last_odds_before_close(con, close_time)

        if not rows:
            print(f"  WARNING: No odds snapshot found before close_time — game excluded from CLV")
            continue

        inserted = 0
        for player_name, market, bookmaker, line, over_price, under_price, snapshot_ts in rows:
            snap_id = make_snap_id(game_id, player_name, market, bookmaker)
            try:
                con.execute("""
                    INSERT OR IGNORE INTO close_snapshots (
                        snap_id, game_id, player_name, market, bookmaker,
                        close_line, close_over_price, close_under_price,
                        close_time_utc, snapshot_ts, ingestion_ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [snap_id, game_id, player_name, market, bookmaker,
                      line, over_price, under_price, close_time, snapshot_ts, now])
                inserted += 1
            except Exception as e:
                print(f"  Insert failed {player_name} {market}: {e}")

        print(f"  Captured {inserted} close snapshots")
        total += inserted

    return total

def settle_clv(con):
    """
    For each unsettled shadow prediction, find matching close snapshot
    and compute CLV = our_line - close_line (for overs).
    Updates shadow_predictions.clv and shadow_predictions.settled.
    Hard gate: if no close snapshot exists, CLV stays NULL.
    """
    rows = con.execute("""
        SELECT sp.pred_id, sp.player_name, sp.stat, sp.book, sp.our_line, sp.signal
        FROM shadow_predictions sp
        WHERE sp.settled = FALSE
    """).fetchall()

    if not rows:
        print("No unsettled predictions.")
        return

    settled = 0
    excluded = 0
    for pred_id, player_name, stat, book, our_line, signal in rows:
        market = f"player_{stat}"
        snap = con.execute("""
            SELECT close_line FROM close_snapshots
            WHERE player_name = ?
              AND market      = ?
              AND bookmaker   = ?
            LIMIT 1
        """, [player_name, market, book]).fetchone()

        if snap is None:
            excluded += 1
            continue

        close_line = snap[0]
        # CLV = our_line - close_line for overs (positive = we got better than close)
        clv = round(our_line - close_line, 3)

        con.execute("""
            UPDATE shadow_predictions
            SET clv = ?, settled = TRUE
            WHERE pred_id = ?
        """, [clv, pred_id])
        settled += 1

    print(f"Settled {settled} predictions | Excluded (no close snapshot): {excluded}")

def print_clv_summary(con):
    df = con.execute("""
        SELECT
            player_name,
            stat,
            COUNT(*) as n_bets,
            ROUND(AVG(clv), 4) as avg_clv,
            ROUND(SUM(clv), 4) as total_clv,
            SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive_clv
        FROM shadow_predictions
        WHERE settled = TRUE
        GROUP BY player_name, stat
        ORDER BY avg_clv DESC
    """).df()

    if df.empty:
        print("No settled CLV data yet.")
        return

    print(f"\n=== CLV Summary ===\n")
    print(f"{'PLAYER':<22} {'STAT':<6} {'N':>4} {'AVG_CLV':>9} {'TOT_CLV':>9} {'POS':>4}")
    print("-" * 60)
    for _, r in df.iterrows():
        print(f"{r['player_name']:<22} {r['stat']:<6} {r['n_bets']:>4} "
              f"{r['avg_clv']:>9.4f} {r['total_clv']:>9.4f} {r['positive_clv']:>4}")

    overall = con.execute("""
        SELECT COUNT(*) as n, ROUND(AVG(clv), 4) as avg_clv
        FROM shadow_predictions WHERE settled = TRUE
    """).fetchone()
    print(f"\nOverall: {overall[0]} settled bets | avg CLV = {overall[1]}")
    gate = "PASS ✓" if overall[1] and overall[1] > 0.01 else "FAIL — not ready for live capital"
    print(f"Shadow gate (CLV > +0.01): {gate}")

def main():
    import sys
    con = connect_db()
    ensure_schema(con)

    if len(sys.argv) > 1 and sys.argv[1] == "settle":
        settle_clv(con)
        print_clv_summary(con)
    else:
        n = capture_close_snapshots(con)
        print(f"\nTotal close snapshots captured: {n}")

    con.close()

if __name__ == "__main__":
    main()
