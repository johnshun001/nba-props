import numpy as np
from datetime import datetime, timezone


def exponential_weights(n, discount=0.97):
    n = int(n)
    if n <= 0:
        return np.array([], dtype=float)
    w = np.power(float(discount), np.arange(n - 1, -1, -1, dtype=float))
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def weighted_quantile(values, weights, quantile):
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    v, w = v[mask], w[mask]
    w = w / w.sum()
    order = np.argsort(v, kind="mergesort")
    v, cw = v[order], np.cumsum(w[order])
    idx = min(int(np.searchsorted(cw, float(quantile), side="left")), len(v) - 1)
    return float(v[idx])


def compute_prediction_interval(nonconf_scores, predicted_median, alpha=0.10, discount=0.97):
    scores = np.asarray(nonconf_scores, dtype=float).reshape(-1)
    scores = scores[np.isfinite(scores)]
    n = int(scores.size)
    if n < 5:
        return {"lower": float(predicted_median), "upper": float(predicted_median),
                "width": float("inf"), "threshold": float("inf"), "n_cal": n}
    weights   = exponential_weights(n, discount)
    threshold = weighted_quantile(scores, weights, 1.0 - alpha)
    med       = float(predicted_median)
    return {"lower": med - threshold, "upper": med + threshold,
            "width": 2 * threshold, "threshold": threshold, "n_cal": n}


def interval_gate(interval, line, max_width=8.0):
    width = float(interval.get("width", float("inf")))
    n_cal = int(interval.get("n_cal", 0))
    if n_cal < 5 or not np.isfinite(width):
        return {"pass_gate": False, "reason": "insufficient_calibration_data"}
    if width > float(max_width):
        return {"pass_gate": False, "reason": "interval_too_wide"}
    return {"pass_gate": True, "reason": "passed"}


def validate_time_split(calibration_times, prediction_time):
    pt = datetime.fromisoformat(prediction_time) if isinstance(prediction_time, str) else prediction_time
    for t in calibration_times:
        tt = datetime.fromisoformat(t) if isinstance(t, str) else t
        assert tt < pt, f"DATA LEAKAGE: calibration {tt} >= prediction {pt}"


def conformal_bet_gate(player_id, stat, line, predicted_median, nonconf_scores,
                       alpha=0.10, max_width=8.0):
    interval = compute_prediction_interval(nonconf_scores, predicted_median, alpha)
    gate     = interval_gate(interval, line, max_width)
    return {"player_id": player_id, "stat": stat, "line": float(line),
            "predicted_median": float(predicted_median), "interval": interval,
            "gate": gate, "conformal_signal": "BET" if gate["pass_gate"] else "PASS"}


if __name__ == "__main__":
    # Test 1: tight scores → BET
    r1 = conformal_bet_gate(1, "pts", 12.5, 14.0,
                             np.array([0.4, 0.5, 0.45, 0.55, 0.6, 0.5, 0.48, 0.52]))
    assert r1["conformal_signal"] == "BET"
    print(f"Test 1 PASSED — signal={r1['conformal_signal']} width={r1['interval']['width']:.2f}")

    # Test 2: wide scores → PASS
    r2 = conformal_bet_gate(2, "reb", 7.5, 8.0,
                             np.array([1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.0, 15.0]))
    assert r2["conformal_signal"] == "PASS"
    print(f"Test 2 PASSED — signal={r2['conformal_signal']} width={r2['interval']['width']:.2f}")

    # Test 3: valid time split
    cal = [datetime(2026, 1, i, tzinfo=timezone.utc) for i in range(1, 4)]
    validate_time_split(cal, datetime(2026, 1, 4, tzinfo=timezone.utc))
    print("Test 3 PASSED — valid time split")

    # Test 4: leakage detection
    try:
        validate_time_split(cal, datetime(2026, 1, 2, tzinfo=timezone.utc))
        print("Test 4 FAILED — should have caught leakage")
    except AssertionError:
        print("Test 4 PASSED — leakage caught")

    print("\nconformal.py ready.")
