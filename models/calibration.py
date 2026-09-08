"""
Isotonic regression calibration layer for QRF p_over outputs.

Usage:
    from models.calibration import fit_and_save_all, load_calibrator, calibrate_prob

    # After running replay.py:
    fit_and_save_all("data/replay_results.csv")

    # At prediction time:
    cal = load_calibrator("pts")
    p_calibrated = calibrate_prob(raw_p_over, cal)
"""
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn"], check=True)
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = str(PROJECT_ROOT / "models" / "calibration_store")
STATS           = ["pts", "reb", "ast"]


# ── core ──────────────────────────────────────────────────────────────────────

def _ece(p_arr: np.ndarray, y_arr: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n     = len(p_arr)
    err   = 0.0
    for i in range(n_bins):
        lo   = edges[i]
        hi   = edges[i + 1]
        mask = (p_arr >= lo) & (p_arr <= hi) if i == n_bins - 1 else (p_arr >= lo) & (p_arr < hi)
        cnt  = int(np.sum(mask))
        if cnt == 0:
            continue
        err += (cnt / n) * abs(float(np.mean(p_arr[mask])) - float(np.mean(y_arr[mask])))
    return float(err)


def fit_calibrator(p_over_arr: np.ndarray,
                   outcome_arr: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression on raw p_over → actual hit rate."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p_over_arr.reshape(-1), outcome_arr.reshape(-1))
    return ir


def calibrate_prob(p_over: float, calibrator) -> float:
    """Apply fitted calibrator to a single raw p_over value."""
    if calibrator is None:
        return float(p_over)
    return float(np.clip(calibrator.predict([float(p_over)])[0], 0.0, 1.0))


def calibrate_array(p_over_arr: np.ndarray, calibrator) -> np.ndarray:
    """Apply fitted calibrator to an array of raw p_over values."""
    if calibrator is None:
        return p_over_arr.copy()
    return np.clip(calibrator.predict(p_over_arr.reshape(-1)), 0.0, 1.0)


# ── persist ───────────────────────────────────────────────────────────────────

def save_calibrator(calibrator: IsotonicRegression, stat: str,
                    store_dir: str = CALIBRATION_DIR) -> None:
    os.makedirs(store_dir, exist_ok=True)
    path = os.path.join(store_dir, f"{stat}_calibrator.pkl")
    with open(path, "wb") as f:
        pickle.dump(calibrator, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_calibrator(stat: str,
                    store_dir: str = CALIBRATION_DIR) -> "IsotonicRegression | None":
    path = os.path.join(store_dir, f"{stat}_calibrator.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── fit from replay CSV ────────────────────────────────────────────────────────

def fit_and_save_all(replay_csv: str = "data/replay_results.csv",
                     stats: tuple = tuple(STATS),
                     min_samples: int = 50) -> dict:
    """
    Load replay_results.csv, fit one isotonic calibrator per stat, save to disk.
    Prints ECE before/after for each stat.
    """
    if not os.path.exists(replay_csv):
        print(f"Replay CSV not found: {replay_csv}")
        print("Run  python3 tests/replay.py  first.")
        return {}

    df = pd.read_csv(replay_csv)
    if df.empty:
        print("Replay CSV is empty.")
        return {}

    calibrators = {}
    print("=== Calibration Fit ===\n")
    print(f"{'stat':<6} {'n':>6} {'ECE before':>11} {'ECE after':>10}")
    print("-" * 38)

    for stat in stats:
        sub = df[df["stat"] == stat].copy()
        if len(sub) < min_samples:
            print(f"{stat:<6} {'--':>6}  (only {len(sub)} samples, need {min_samples})")
            continue

        p   = sub["p_over"].to_numpy(dtype=float)
        y   = sub["outcome"].to_numpy(dtype=float)

        # remove NaN
        mask = np.isfinite(p) & np.isfinite(y)
        p, y = p[mask], y[mask]
        if len(p) < min_samples:
            continue

        ece_before = _ece(p, y)
        cal        = fit_calibrator(p, y)
        p_cal      = calibrate_array(p, cal)
        ece_after  = _ece(p_cal, y)

        save_calibrator(cal, stat)
        calibrators[stat] = cal

        print(f"{stat:<6} {len(p):>6} {ece_before:>11.4f} {ece_after:>10.4f}")

    print(f"\nCalibrators saved → {CALIBRATION_DIR}/")
    return calibrators


# ── reliability diagram (text) ────────────────────────────────────────────────

def print_reliability_diagram(p_arr: np.ndarray, y_arr: np.ndarray,
                               label: str = "", n_bins: int = 5) -> None:
    """Print a simple ASCII reliability diagram."""
    frac_pos, mean_pred = calibration_curve(y_arr, p_arr, n_bins=n_bins, strategy="quantile")
    print(f"Reliability diagram {label}")
    print(f"  {'predicted':>10} {'actual':>8}  {'diff':>7}")
    for mp, fp in zip(mean_pred, frac_pos):
        diff = fp - mp
        bar  = "+" * int(abs(diff) * 40) if diff >= 0 else "-" * int(abs(diff) * 40)
        print(f"  {mp:>10.3f} {fp:>8.3f}  {diff:>+7.3f}  {bar}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    import sys
    replay_csv = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(PROJECT_ROOT / "data" / "replay_results.csv")
    )
    cals = fit_and_save_all(replay_csv)
    if not cals:
        return 1

    # Print reliability diagrams before/after for each stat
    df = pd.read_csv(replay_csv)
    for stat, cal in cals.items():
        sub = df[df["stat"] == stat]
        p   = sub["p_over"].to_numpy(dtype=float)
        y   = sub["outcome"].to_numpy(dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        p, y = p[mask], y[mask]
        p_cal = calibrate_array(p, cal)
        print()
        print_reliability_diagram(p, y, label=f"[{stat}] RAW")
        print_reliability_diagram(p_cal, y, label=f"[{stat}] CALIBRATED")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
