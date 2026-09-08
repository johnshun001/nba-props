import sys
import time
import datetime
from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
from scipy.stats import gamma as gamma_dist
from scipy.signal import find_peaks
from nba_api.stats.endpoints import CommonPlayerInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")

PHASE_BOUNDS: Dict[int, str] = {
    0:  "Rookie",
    3:  "Developing",
    6:  "Prime",
    13: "Decline",
}


def get_career_start_year(player_id: int) -> int:
    try:
        df = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        time.sleep(0.6)
        if df is None or df.empty:
            return 2020
        val = df.loc[0, "FROM_YEAR"]
        return int(val)
    except Exception:
        try:
            time.sleep(0.6)
        except Exception:
            pass
        return 2020


def classify_phase(years_exp: int) -> str:
    y     = int(years_exp)
    label = PHASE_BOUNDS[0]
    for k in sorted(PHASE_BOUNDS.keys()):
        if y >= k:
            label = PHASE_BOUNDS[k]
    return label


def compute_trajectory_slope(stat_vals: np.ndarray, window: int = 10) -> float:
    v = np.asarray(stat_vals, dtype=float).reshape(-1)
    w = int(window)
    if v.size < (w + 5):
        return 0.0
    roll_means = []
    for i in range(w, v.size + 1):
        roll_means.append(float(np.mean(v[i - w:i])))
    rm = np.asarray(roll_means, dtype=float)
    if rm.size < 2:
        return 0.0
    x = np.arange(rm.size, dtype=float)
    try:
        slope = float(np.polyfit(x, rm, 1)[0])
        if not np.isfinite(slope):
            return 0.0
        return slope
    except Exception:
        return 0.0


def detect_usage_peaks(stat_vals: np.ndarray, window: int = 10,
                       prominence: float = 0.15) -> dict:
    """
    Detect distinct usage peaks in a player's rolling-mean stat series.
    Uses scipy.signal.find_peaks on normalized rolling means.
    Returns:
      n_peaks: int -- number of distinct peaks found
      peak_magnitudes: list[float] -- relative heights above median
      is_opportunistic: bool -- True if n_peaks >= 2 and any peak > 1.5x median
    """
    v = np.asarray(stat_vals, dtype=float).reshape(-1)
    w = int(window)
    if v.size < (w + 5):
        return {"n_peaks": 0, "peak_magnitudes": [], "is_opportunistic": False}

    roll_means = []
    for i in range(w, v.size + 1):
        roll_means.append(float(np.mean(v[i - w:i])))
    rm = np.asarray(roll_means, dtype=float)

    median_val = float(np.median(rm))
    if median_val <= 0.0:
        return {"n_peaks": 0, "peak_magnitudes": [], "is_opportunistic": False}

    peak_range = float(rm.max() - rm.min())
    min_prom   = max(peak_range * float(prominence), 0.5)
    peaks, _   = find_peaks(rm, prominence=min_prom, distance=5)

    magnitudes       = [float(rm[p] / median_val) for p in peaks]
    is_opportunistic = len(peaks) >= 2 and any(m > 1.5 for m in magnitudes)

    return {
        "n_peaks":          int(len(peaks)),
        "peak_magnitudes":  [round(m, 3) for m in magnitudes],
        "is_opportunistic": bool(is_opportunistic),
    }


def gamma_basis_weight(years_exp: int) -> float:
    shape = 8.0
    scale = 1.5
    y     = float(max(0, int(years_exp)))
    try:
        pdf_val = float(gamma_dist.pdf(y, a=shape, scale=scale))
        mode    = (shape - 1.0) * scale
        peak    = float(gamma_dist.pdf(mode, a=shape, scale=scale))
        norm    = peak if (peak > 0.0 and np.isfinite(peak)) else 1.0
        w       = pdf_val / norm
    except Exception:
        w = 1.0
    if not np.isfinite(w):
        w = 1.0
    return float(np.clip(w, 0.3, 1.0))


def ensure_cache_table(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS plc_cache (
            player_id INTEGER PRIMARY KEY,
            from_year INTEGER,
            cached_at TIMESTAMP
        )
    """)


def get_cached_from_year(con, player_id: int) -> Optional[int]:
    try:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=7)
        row = con.execute(
            "SELECT from_year FROM plc_cache WHERE player_id = ? AND cached_at > ?",
            [int(player_id), cutoff],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])
    except Exception:
        return None


def save_cached_from_year(con, player_id: int, from_year: int) -> None:
    con.execute(
        "INSERT OR REPLACE INTO plc_cache (player_id, from_year, cached_at) VALUES (?, ?, ?)",
        [int(player_id), int(from_year), datetime.datetime.utcnow()],
    )


def _load_player_stat_vals(con, player_id: int, stat: str) -> np.ndarray:
    rows = con.execute(
        "SELECT " + stat + " FROM player_game_features "
        "WHERE player_id = ? AND " + stat + " IS NOT NULL "
        "ORDER BY game_date ASC",
        [int(player_id)],
    ).fetchall()
    if not rows:
        return np.array([], dtype=float)
    vals = []
    for r in rows:
        if r[0] is None:
            continue
        try:
            vals.append(float(r[0]))
        except Exception:
            continue
    return np.asarray(vals, dtype=float)


def compute_plc_prior(player_id: int, stat: str, con) -> Dict[str, Any]:
    ensure_cache_table(con)
    stat_vals = _load_player_stat_vals(con, player_id, stat)
    from_year = get_cached_from_year(con, player_id)
    if from_year is None:
        from_year = get_career_start_year(player_id)
        try:
            save_cached_from_year(con, player_id, from_year)
        except Exception:
            pass

    current_year = int(datetime.datetime.now().year)
    years_exp    = int(max(0, current_year - int(from_year)))
    phase        = classify_phase(years_exp)
    slope        = compute_trajectory_slope(stat_vals, window=10)
    weight       = gamma_basis_weight(years_exp)

    if stat_vals.size == 0:
        peaks = {"n_peaks": 0, "is_opportunistic": False, "peak_magnitudes": []}
    else:
        peaks = detect_usage_peaks(stat_vals, window=10)

    if slope < -0.05 and phase == "Decline":
        recommendation = "FADE"
    elif slope > 0.05 and phase in ["Rookie", "Developing"]:
        recommendation = "BOOST"
    else:
        recommendation = "NEUTRAL"

    return {
        "player_id":        int(player_id),
        "stat":             str(stat),
        "phase":            str(phase),
        "years_exp":        int(years_exp),
        "trajectory_slope": float(slope),
        "gamma_weight":     float(weight),
        "recommendation":   str(recommendation),
        "n_usage_peaks":    int(peaks.get("n_peaks", 0)),
        "is_opportunistic": bool(peaks.get("is_opportunistic", False)),
        "peak_magnitudes":  list(peaks.get("peak_magnitudes", [])),
    }


def main() -> int:
    con = duckdb.connect(DB_PATH)
    try:
        ensure_cache_table(con)
        players = con.execute(
            "SELECT player_id, full_name FROM player_lookup "
            "WHERE player_id IS NOT NULL ORDER BY player_id"
        ).fetchall()

        header = ["player_id", "player", "stat", "phase", "yrs", "slope",
                  "gamma_w", "rec", "n_peaks", "opportunistic"]
        print(" | ".join(h.ljust(12) for h in header))

        for pid, name in players:
            player_id   = int(pid)
            player_name = str(name) if name is not None else str(player_id)
            for stat in ["pts", "reb", "ast"]:
                try:
                    r = compute_plc_prior(player_id, stat, con)
                except Exception as e:
                    print(f"{player_id} | {player_name} | {stat} | ERROR: {e}")
                    continue
                row = [
                    str(r["player_id"]).ljust(12),
                    player_name[:12].ljust(12),
                    str(r["stat"]).ljust(12),
                    str(r["phase"]).ljust(12),
                    str(r["years_exp"]).ljust(12),
                    f"{r['trajectory_slope']:.4f}".ljust(12),
                    f"{r['gamma_weight']:.3f}".ljust(12),
                    str(r["recommendation"]).ljust(12),
                    str(r["n_usage_peaks"]).ljust(12),
                    ("Y" if r["is_opportunistic"] else "N").ljust(12),
                ]
                print(" | ".join(row))
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
