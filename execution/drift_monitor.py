import sys
import datetime
from typing import Any, Dict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from config import load_config, project_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH             = str(project_path("database"))
CLV_WINDOW_DAYS     = 30
CLV_STOP_THRESHOLD  = float(load_config()["thresholds"]["clv_stop_threshold"])
ECE_STOP_THRESHOLD  = float(load_config()["thresholds"]["maximum_ece"])
MIN_SETTLED_BETS    = int(load_config()["thresholds"]["minimum_drift_settlements"])
N_CALIBRATION_BINS  = 10
PROMOTION_MIN_BETS  = int(load_config()["thresholds"]["minimum_shadow_settlements"])
PROMOTION_CLV_GATE  = float(load_config()["thresholds"]["minimum_mean_clv"])


def load_settled_predictions(con, window_days: int = 30) -> pd.DataFrame:
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=int(window_days))
    try:
        df = con.execute(
            "SELECT CASE WHEN side = 'UNDER' "
            "THEN 1.0 - COALESCE(calibrated_p_over, qrf_p_over) "
            "ELSE COALESCE(calibrated_p_over, qrf_p_over) END AS model_probability, "
            "COALESCE(calibrated_p_over, qrf_p_over) AS calibrated_p_over, "
            "side, signal, clv, edge, outcome, created_at "
            "FROM shadow_predictions WHERE settled = TRUE AND created_at >= ? "
            "AND signal IN ('OVER', 'UNDER') AND calibration_status = 'READY' "
            "AND clv IS NOT NULL AND outcome IS NOT NULL",
            [cutoff],
        ).fetchdf()
    except Exception:
        return pd.DataFrame(columns=[
            "model_probability", "calibrated_p_over", "side", "signal",
            "clv", "edge", "outcome", "created_at",
        ])
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "model_probability", "calibrated_p_over", "side", "signal",
            "clv", "edge", "outcome", "created_at",
        ])
    return df


def compute_promotion_gate(con) -> Dict[str, Any]:
    """
    Shadow mode promotion gate.
    Queries shadow_predictions for all settled rows with CLV not null.
    Returns:
      status: "ELIGIBLE" | "NOT_ELIGIBLE" | "NOT_READY"
      n_settled: int
      mean_clv: float or None
      min_bets_required: int
      clv_threshold: float
    """
    try:
        rows = con.execute(
            "SELECT clv, CASE WHEN side = 'UNDER' "
            "THEN 1.0 - COALESCE(calibrated_p_over, qrf_p_over) "
            "ELSE COALESCE(calibrated_p_over, qrf_p_over) END, outcome "
            "FROM shadow_predictions WHERE settled = TRUE AND clv IS NOT NULL "
            "AND outcome IS NOT NULL AND signal IN ('OVER', 'UNDER') "
            "AND calibration_status = 'READY'"
        ).fetchall()
        n = int(len(rows))
        if n < int(PROMOTION_MIN_BETS):
            return {
                "status":            "NOT_READY",
                "n_settled":         n,
                "mean_clv":          None,
                "min_bets_required": int(PROMOTION_MIN_BETS),
                "clv_threshold":     float(PROMOTION_CLV_GATE),
            }
        vals = []
        for r in rows:
            try:
                vals.append(float(r[0]))
            except Exception:
                continue
        if not vals:
            return {
                "status":            "NOT_READY",
                "n_settled":         0,
                "mean_clv":          None,
                "min_bets_required": int(PROMOTION_MIN_BETS),
                "clv_threshold":     float(PROMOTION_CLV_GATE),
            }
        mean_clv = float(np.mean(np.asarray(vals, dtype=float)))
        probabilities = np.asarray([float(r[1]) for r in rows], dtype=float)
        outcomes = np.asarray([float(r[2]) for r in rows], dtype=float)
        calibration = compute_ece(pd.DataFrame({
            "clv": np.ones(n), "calibrated_p_over": probabilities, "outcome": outcomes,
        }))
        status = "ELIGIBLE" if (
            n >= int(PROMOTION_MIN_BETS)
            and mean_clv > float(PROMOTION_CLV_GATE)
            and calibration.get("status") == "OK"
        ) else "NOT_ELIGIBLE"
        return {
            "status":            status,
            "n_settled":         n,
            "mean_clv":          mean_clv,
            "calibration_error": calibration.get("ece"),
            "calibration_status": calibration.get("status"),
            "min_bets_required": int(PROMOTION_MIN_BETS),
            "clv_threshold":     float(PROMOTION_CLV_GATE),
        }
    except Exception:
        return {
            "status":            "NOT_READY",
            "n_settled":         0,
            "mean_clv":          None,
            "min_bets_required": int(PROMOTION_MIN_BETS),
            "clv_threshold":     float(PROMOTION_CLV_GATE),
        }


def compute_rolling_clv(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty or int(df.shape[0]) < MIN_SETTLED_BETS:
        return {"status": "NOT_READY", "mean_clv": None, "n": 0}
    d = df[df["clv"].notna()].copy()
    if d.empty or int(d.shape[0]) < MIN_SETTLED_BETS:
        return {"status": "NOT_READY", "mean_clv": None, "n": int(d.shape[0])}
    mean_clv = float(pd.to_numeric(d["clv"], errors="coerce").mean())
    gate     = "STOP" if mean_clv < float(CLV_STOP_THRESHOLD) else "OK"
    return {
        "status":    gate,
        "mean_clv":  mean_clv,
        "n":         int(d.shape[0]),
        "threshold": float(CLV_STOP_THRESHOLD),
    }


def compute_ece(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
    insuf = {"ece": None, "status": "NOT_READY",
             "n_bins_used": 0, "threshold": float(ECE_STOP_THRESHOLD)}
    n_bins = int(n_bins)
    if df is None or df.empty:
        return insuf
    probability_column = (
        "model_probability" if "model_probability" in df.columns
        else ("calibrated_p_over" if "calibrated_p_over" in df.columns else "qrf_p_over")
    )
    if "clv" not in df.columns or probability_column not in df.columns or "outcome" not in df.columns:
        return insuf
    d = df[df["clv"].notna()].copy()
    if d.empty or int(d.shape[0]) < MIN_SETTLED_BETS:
        return insuf
    d[probability_column] = pd.to_numeric(d[probability_column], errors="coerce")
    d["outcome"] = pd.to_numeric(d["outcome"], errors="coerce")
    d = d[d[probability_column].notna() & d["outcome"].notna()].copy()
    if d.empty:
        return insuf

    d[probability_column] = d[probability_column].clip(lower=0.0, upper=1.0)
    total_n         = int(d.shape[0])
    if total_n <= 0:
        return insuf

    edges     = np.linspace(0.0, 1.0, n_bins + 1)
    p         = d[probability_column].to_numpy(dtype=float)
    y         = d["outcome"].to_numpy(dtype=float)
    ece       = 0.0
    bins_used = 0

    for i in range(n_bins):
        lo   = edges[i]
        hi   = edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        cnt  = int(np.sum(mask))
        if cnt <= 0:
            continue
        bins_used  += 1
        mean_pred   = float(np.mean(p[mask]))
        frac_corr   = float(np.mean(y[mask]))
        ece        += (cnt / total_n) * abs(mean_pred - frac_corr)

    gate = "STOP" if float(ece) > float(ECE_STOP_THRESHOLD) else "OK"
    return {
        "ece":         float(ece),
        "status":      gate,
        "n_bins_used": int(bins_used),
        "threshold":   float(ECE_STOP_THRESHOLD),
    }


def check_all_gates(con) -> Dict[str, Any]:
    df           = load_settled_predictions(con, window_days=CLV_WINDOW_DAYS)
    clv_result   = compute_rolling_clv(df)
    ece_result   = compute_ece(df, n_bins=N_CALIBRATION_BINS)
    promo_result = compute_promotion_gate(con)
    statuses = {clv_result.get("status"), ece_result.get("status")}
    promotion_status = promo_result.get("status")
    if "STOP" in statuses or promotion_status == "NOT_ELIGIBLE":
        overall = "STOP"
    elif "NOT_READY" in statuses or promotion_status != "ELIGIBLE":
        overall = "NOT_READY"
    else:
        overall = "OK"
    return {
        "overall":    overall,
        "clv":        clv_result,
        "ece":        ece_result,
        "promotion":  promo_result,
        "checked_at": datetime.datetime.utcnow().isoformat(),
    }


def print_drift_report(result: Dict[str, Any]) -> None:
    overall    = result.get("overall", "UNKNOWN")
    checked_at = result.get("checked_at", "UNKNOWN")
    clv        = result.get("clv", {}) or {}
    ece        = result.get("ece", {}) or {}
    promo      = result.get("promotion", {}) or {}

    print("\n=== Drift Monitor Report ===")
    print(f"Checked at (UTC): {checked_at}")
    print(f"Overall gate:     {overall}\n")

    clv_mean = clv.get("mean_clv", None)
    print("Rolling CLV gate")
    print(f"  status:    {clv.get('status', 'UNKNOWN')}")
    print(f"  mean_clv:  {float(clv_mean):.6f}" if clv_mean is not None else "  mean_clv:  None")
    print(f"  n:         {int(clv.get('n', 0))}")
    print(f"  threshold: {float(clv.get('threshold', CLV_STOP_THRESHOLD)):.6f}\n")

    ece_val = ece.get("ece", None)
    print("Calibration (ECE) gate")
    print(f"  status:      {ece.get('status', 'UNKNOWN')}")
    print(f"  ece:         {float(ece_val):.6f}" if ece_val is not None else "  ece:         None")
    print(f"  n_bins_used: {int(ece.get('n_bins_used', 0))}")
    print(f"  threshold:   {float(ece.get('threshold', ECE_STOP_THRESHOLD)):.6f}\n")

    promo_mean = promo.get("mean_clv", None)
    promo_n    = int(promo.get("n_settled", 0) or 0)
    promo_min  = int(promo.get("min_bets_required", PROMOTION_MIN_BETS) or PROMOTION_MIN_BETS)
    promo_thr  = float(promo.get("clv_threshold", PROMOTION_CLV_GATE) or PROMOTION_CLV_GATE)
    print("Shadow Promotion Gate")
    print(f"  status:            {promo.get('status', 'UNKNOWN')}")
    print(f"  n_settled:         {promo_n}")
    print(f"  mean_clv:          {float(promo_mean):.6f}" if promo_mean is not None else "  mean_clv:          None")
    print(f"  min_bets_required: {promo_min}")
    print(f"  clv_threshold:     {promo_thr:.6f}")
    if promo.get("status") == "ELIGIBLE":
        print("*** ELIGIBLE FOR LIVE CAPITAL PROMOTION ***")
    print()

    if overall == "STOP":
        print("*** MODEL STOP TRIGGERED ***\n")


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        result = check_all_gates(con)
        print_drift_report(result)
        return 0 if result.get("overall") == "OK" else (1 if result.get("overall") == "STOP" else 2)
    finally:
        try:
            con.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
