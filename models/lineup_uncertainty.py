import sys
from typing import Dict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH              = str(PROJECT_ROOT / "data" / "raw.db")
DNP_THRESHOLD        = 0.20
STARTER_MIN_MINUTES  = 28
DNP_MAX_MINUTES      = 2
LOOKBACK_GAMES       = 15


def load_recent_minutes(con, player_id: int, n: int = 15) -> np.ndarray:
    rows = con.execute(
        "SELECT minutes FROM player_game_features "
        "WHERE player_id = ? AND minutes IS NOT NULL "
        "ORDER BY game_date DESC LIMIT ?",
        [int(player_id), int(n)],
    ).fetchall()
    if not rows:
        return np.array([], dtype=float)
    vals = []
    for r in rows:
        try:
            vals.append(float(r[0]))
        except Exception:
            continue
    if not vals:
        return np.array([], dtype=float)
    vals.reverse()  # chronological order
    return np.asarray(vals, dtype=float)


def compute_lineup_probs(minutes_arr: np.ndarray) -> Dict[str, float]:
    m = np.asarray(minutes_arr, dtype=float).reshape(-1)
    if m.size == 0:
        return {"p_dnp": 0.5, "p_starter": 0.0, "p_rotation": 0.5}
    n          = float(m.size)
    p_dnp      = float(np.sum(m <= float(DNP_MAX_MINUTES)) / n)
    p_starter  = float(np.sum(m >= float(STARTER_MIN_MINUTES)) / n)
    p_rotation = float(max(0.0, 1.0 - p_dnp - p_starter))
    return {"p_dnp": p_dnp, "p_starter": p_starter, "p_rotation": p_rotation}


def lineup_gate(player_id: int, con) -> Dict[str, object]:
    minutes_arr = load_recent_minutes(con, int(player_id), n=LOOKBACK_GAMES)
    probs       = compute_lineup_probs(minutes_arr)
    p_dnp       = float(probs["p_dnp"])
    signal      = "PASS" if p_dnp >= float(DNP_THRESHOLD) else "BET"
    return {
        "player_id":          int(player_id),
        "p_dnp":              float(probs["p_dnp"]),
        "p_starter":          float(probs["p_starter"]),
        "p_rotation":         float(probs["p_rotation"]),
        "signal":             signal,
        "n_games_used":       int(minutes_arr.size),
        "dnp_threshold_used": float(DNP_THRESHOLD),
    }


def batch_lineup_report(con) -> pd.DataFrame:
    rows = con.execute(
        "SELECT player_id FROM player_lookup WHERE player_id IS NOT NULL"
    ).fetchall()
    results = []
    for r in rows:
        try:
            pid = int(r[0])
        except Exception:
            continue
        try:
            results.append(lineup_gate(pid, con))
        except Exception:
            results.append({
                "player_id":          pid,
                "p_dnp":              0.5,
                "p_starter":          0.0,
                "p_rotation":         0.5,
                "signal":             "PASS",
                "n_games_used":       0,
                "dnp_threshold_used": float(DNP_THRESHOLD),
            })
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values(
        ["p_dnp", "player_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return df


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df = batch_lineup_report(con)
    finally:
        con.close()

    if df is None or df.empty:
        print("No players found.")
        return 0

    show_cols = ["player_id", "p_dnp", "p_starter", "p_rotation", "signal", "n_games_used"]
    disp      = df[show_cols].copy()
    for c in ["p_dnp", "p_starter", "p_rotation"]:
        disp[c] = disp[c].map(lambda x: round(float(x), 3))

    header = ["player_id", "p_dnp", "p_starter", "p_rot", "signal", "n"]
    widths = [10, 7, 9, 6, 6, 3]
    hline  = " ".join(header[i].ljust(widths[i]) for i in range(len(header)))
    print(hline)
    print("-" * len(hline))

    for _, r in disp.iterrows():
        sig      = str(r["signal"])
        sig_disp = ("PASS*" if sig == "PASS" else "BET").ljust(widths[4])
        row = [
            str(int(r["player_id"])).rjust(widths[0]),
            f"{float(r['p_dnp']):.3f}".rjust(widths[1]),
            f"{float(r['p_starter']):.3f}".rjust(widths[2]),
            f"{float(r['p_rotation']):.3f}".rjust(widths[3]),
            sig_disp,
            str(int(r["n_games_used"])).rjust(widths[5]),
        ]
        print(" ".join(row))

    n_pass = int((df["signal"] == "PASS").sum())
    print(f"\nPASS* = p_dnp >= {DNP_THRESHOLD:.2f}.  Players flagged: {n_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
