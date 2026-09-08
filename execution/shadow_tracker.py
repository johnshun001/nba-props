import sys
import datetime
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.shin_devig import devig_prop_line
from models.hmm_minutes import predict_next_minutes, load_model as load_hmm
from models.qrf_model import prob_over_line
from models.conformal import conformal_bet_gate

DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_predictions (
    pred_id             VARCHAR PRIMARY KEY,
    created_at          TIMESTAMP NOT NULL,
    player_id           INTEGER NOT NULL,
    player_name         VARCHAR,
    stat                VARCHAR NOT NULL,
    book                VARCHAR NOT NULL,
    our_line            FLOAT NOT NULL,
    over_odds           INTEGER,
    under_odds          INTEGER,
    hmm_expected_min    FLOAT,
    qrf_p_over          FLOAT,
    book_p_over         FLOAT,
    edge                FLOAT,
    kelly_fraction      FLOAT,
    signal              VARCHAR,
    conformal_lower     FLOAT,
    conformal_upper     FLOAT,
    conformal_width     FLOAT,
    conformal_signal    VARCHAR,
    close_line          FLOAT,
    clv                 FLOAT,
    settled             BOOLEAN DEFAULT FALSE
)
"""

MARKET_TO_STAT = {
    "player_points":   "pts",
    "player_rebounds": "reb",
    "player_assists":  "ast",
}


def connect_db():
    return duckdb.connect(DB_PATH)


def ensure_schema(con):
    con.execute(SCHEMA)
    for col, dtype in [
        ("conformal_lower",  "FLOAT"),
        ("conformal_upper",  "FLOAT"),
        ("conformal_width",  "FLOAT"),
        ("conformal_signal", "VARCHAR"),
    ]:
        try:
            con.execute(
                f"ALTER TABLE shadow_predictions ADD COLUMN IF NOT EXISTS {col} {dtype}"
            )
        except Exception:
            pass


def make_pred_id(player_id, stat, book, created_at):
    import hashlib
    raw = f"{player_id}{stat}{book}{created_at.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_prop_lines(con):
    return con.execute("""
        SELECT
            o.player_name,
            o.market,
            o.line,
            o.over_price,
            u.under_price,
            o.bookmaker
        FROM prop_lines o
        JOIN prop_lines u
          ON o.player_name = u.player_name
         AND o.market      = u.market
         AND o.line        = u.line
         AND o.bookmaker   = u.bookmaker
         AND o.over_price  IS NOT NULL
         AND u.under_price IS NOT NULL
        WHERE o.market IN ('player_points', 'player_rebounds', 'player_assists')
    """).df()


def load_lookup(con):
    rows = con.execute("SELECT player_id, full_name FROM player_lookup").fetchall()
    return {row[1].strip().lower(): row[0] for row in rows}


def load_recent_minutes(con, player_id, n=15):
    rows = con.execute("""
        SELECT minutes FROM player_game_features
        WHERE player_id = ? AND minutes > 0
        ORDER BY game_date DESC
        LIMIT ?
    """, [player_id, n]).fetchall()
    mins = np.array([r[0] for r in rows], dtype=float)
    return mins[::-1]


def load_nonconf_scores(con, player_id, stat, n=60):
    rows = con.execute(
        "SELECT " + stat + ", game_date FROM player_game_features "
        "WHERE player_id = ? AND " + stat + " IS NOT NULL "
        "ORDER BY game_date ASC",
        [player_id]
    ).fetchall()
    if len(rows) < 11:
        return np.array([])
    vals = np.array([r[0] for r in rows], dtype=float)
    scores = []
    for i in range(10, len(vals)):
        roll_avg = vals[i - 10:i].mean()
        scores.append(abs(vals[i] - roll_avg))
    arr = np.array(scores, dtype=float)
    return arr[-n:]


def run_shadow_pass(con):
    ensure_schema(con)
    prop_df  = load_prop_lines(con)
    name_map = load_lookup(con)
    now      = datetime.datetime.utcnow()

    if prop_df.empty:
        print("No prop lines found. Run odds_scraper.py first.")
        return

    print(f"Running shadow pass at {now.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Prop lines loaded: {len(prop_df)}\n")

    results = []
    for _, row in prop_df.iterrows():
        norm_name = row["player_name"].strip().lower()
        player_id = name_map.get(norm_name)
        if player_id is None:
            continue

        stat = MARKET_TO_STAT.get(row["market"])
        if stat is None:
            continue

        over_odds  = int(row["over_price"])
        under_odds = int(row["under_price"])
        line       = float(row["line"])
        book       = row["bookmaker"]

        # HMM minutes estimate
        try:
            minutes_seq = load_recent_minutes(con, player_id)
            if len(minutes_seq) < 3:
                continue
            hmm_result  = predict_next_minutes(player_id, minutes_seq)
            exp_minutes = hmm_result["expected_minutes"]
        except Exception as e:
            print(f"  HMM failed for {row['player_name']}: {e}")
            continue

        # QRF probability
        try:
            p_over = prob_over_line(player_id, stat, line, exp_minutes)
        except Exception as e:
            print(f"  QRF failed for {row['player_name']} {stat}: {e}")
            continue

        # Conformal gate
        try:
            nonconf_scores = load_nonconf_scores(con, player_id, stat)
            conf_result    = conformal_bet_gate(player_id, stat, line, line, nonconf_scores)
            conf_signal    = conf_result["conformal_signal"]
            conf_interval  = conf_result["interval"]
        except Exception as e:
            print(f"  Conformal failed for {row['player_name']} {stat}: {e}")
            conf_signal   = "PASS"
            conf_interval = {"lower": 0.0, "upper": 0.0, "width": float("inf")}

        # Shin de-vig + Kelly
        try:
            devig = devig_prop_line(over_odds, under_odds, p_over)
        except Exception as e:
            print(f"  Devig failed for {row['player_name']}: {e}")
            continue

        # Final signal: edge AND conformal must both approve
        final_signal = devig["bet_signal"] if conf_signal == "BET" else "PASS"

        pred_id = make_pred_id(player_id, stat, book, now)

        con.execute("""
            INSERT OR IGNORE INTO shadow_predictions (
                pred_id, created_at, player_id, player_name, stat, book,
                our_line, over_odds, under_odds, hmm_expected_min,
                qrf_p_over, book_p_over, edge, kelly_fraction, signal,
                conformal_lower, conformal_upper, conformal_width, conformal_signal,
                close_line, clv, settled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, FALSE)
        """, [
            pred_id, now, player_id, row["player_name"], stat, book,
            line, over_odds, under_odds, exp_minutes,
            devig["projected_p_over"], devig["book_p_over"],
            devig["edge"], devig["kelly_fraction"], final_signal,
            conf_interval["lower"], conf_interval["upper"],
            conf_interval["width"], conf_signal,
        ])

        results.append({
            "player":  row["player_name"],
            "stat":    stat.upper(),
            "book":    book,
            "line":    line,
            "exp_min": exp_minutes,
            "p_over":  round(devig["projected_p_over"], 3),
            "book_p":  round(devig["book_p_over"], 3),
            "edge":    round(devig["edge"], 3),
            "kelly":   devig["kelly_fraction"],
            "conf_w":  round(conf_interval["width"], 2),
            "signal":  final_signal,
        })

    if not results:
        print("No matches between prop lines and tracked players.")
        return

    df = pd.DataFrame(results).sort_values("edge", ascending=False)
    print(
        f"{'PLAYER':<22} {'STAT':<6} {'BOOK':<15} {'LINE':>5} "
        f"{'EXP_MIN':>7} {'P_OVER':>7} {'BOOK_P':>7} "
        f"{'EDGE':>6} {'KELLY':>6} {'CONF_W':>7} {'SIGNAL'}"
    )
    print("-" * 112)
    for _, r in df.iterrows():
        print(
            f"{r['player']:<22} {r['stat']:<6} {r['book']:<15} {r['line']:>5} "
            f"{r['exp_min']:>7.1f} {r['p_over']:>7.3f} {r['book_p']:>7.3f} "
            f"{r['edge']:>6.3f} {r['kelly']:>6.4f} {r['conf_w']:>7.2f} {r['signal']}"
        )

    bets = df[df["signal"] == "OVER"]
    print(f"\nShadow bets logged: {len(results)} predictions, {len(bets)} OVER signals")
    print("Run settle_clv.py after games complete to log CLV.")


def print_shadow_log(con):
    df = con.execute("""
        SELECT player_name, stat, book, our_line, qrf_p_over,
               book_p_over, edge, kelly_fraction, signal,
               conformal_width, conformal_signal,
               close_line, clv, settled, created_at
        FROM shadow_predictions
        ORDER BY created_at DESC
        LIMIT 50
    """).df()

    if df.empty:
        print("No shadow predictions logged yet.")
        return

    print(
        f"\n{'PLAYER':<22} {'STAT':<6} {'LINE':>5} {'EDGE':>6} "
        f"{'CONF_W':>7} {'SIGNAL':<8} {'CLV':>6} {'SETTLED'}"
    )
    print("-" * 80)
    for _, r in df.iterrows():
        clv_str     = f"{r['clv']:.3f}" if pd.notna(r["clv"]) else "pending"
        settled_str = "YES" if r["settled"] else "NO"
        conf_w_str  = f"{r['conformal_width']:.2f}" if pd.notna(r["conformal_width"]) else "n/a"
        print(
            f"{r['player_name']:<22} {r['stat']:<6} {r['our_line']:>5} "
            f"{r['edge']:>6.3f} {conf_w_str:>7} {r['signal']:<8} {clv_str:>6} {settled_str}"
        )


def main():
    con = connect_db()
    ensure_schema(con)

    if len(sys.argv) > 1 and sys.argv[1] == "log":
        print_shadow_log(con)
    else:
        run_shadow_pass(con)

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
