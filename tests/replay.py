"""
Legacy synthetic benchmark: for each game G, train HMM+QRF strictly on data
before G, simulate the book line as rolling 10-game average, predict
P(over), then compare to actual outcome.

Production sportsbook replay lives in :mod:`analysis.replay`; this module is
kept only as an offline synthetic regression test.

Reports:
  - Direction accuracy and calibration (ECE)
  - Minutes Variance Effect: low-minutes vs high-minutes games
"""
import os
import warnings
from typing import Any, Dict, List, Optional
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from models.hmm_minutes import (
    TruncatedStudentHMM,
    fit_hmm,
    reorder_states,
    current_state,
    next_game_probs,
    expected_minutes,
    N_STATES,
    STATE_LABELS,
)
from models.qrf_model import (
    _fit_qrf_for_player_stat,
    _predict_quantiles,
    _piecewise_linear_cdf_from_quantiles,
    _sample_minutes_from_hmm,
    PlayerStatQRF,
    FEATURE_COLS,
    QUANTILES,
)

warnings.filterwarnings("ignore")  # suppress hmmlearn convergence warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH       = str(PROJECT_ROOT / "data" / "raw.db")
RESULTS_CSV   = str(PROJECT_ROOT / "data" / "replay_results.csv")
WARMUP_GAMES  = 30          # min games before we start predicting
REFIT_EVERY   = 10          # refit HMM+QRF every N games
SIM_LINE_WIN  = 10          # rolling window for simulated book line
N_MONTE_CARLO = 10          # HMM minutes MC samples
STATS         = ["pts", "reb", "ast"]
MIN_GAMES_QRF = 20          # QRF requires this many rows to fit


# ── data loading ──────────────────────────────────────────────────────────────

def load_player_games(con, player_id: int, stat: str) -> pd.DataFrame:
    """Load all games for a player+stat, sorted by date. Returns DataFrame."""
    df = con.execute(
        "SELECT game_date, " + stat + " AS stat_val, minutes, "
        "fga, fg3a, tov, plus_minus "
        "FROM player_game_features "
        "WHERE player_id = ? "
        "AND " + stat + " IS NOT NULL "
        "AND minutes IS NOT NULL AND minutes >= 1 "
        "ORDER BY game_date ASC",
        [int(player_id)],
    ).fetchdf()
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"]).copy()
    df["stat_val"]   = pd.to_numeric(df["stat_val"],   errors="coerce")
    df["minutes"]    = pd.to_numeric(df["minutes"],    errors="coerce")
    df["fga"]        = pd.to_numeric(df["fga"],        errors="coerce").fillna(0.0)
    df["fg3a"]       = pd.to_numeric(df["fg3a"],       errors="coerce").fillna(0.0)
    df["tov"]        = pd.to_numeric(df["tov"],        errors="coerce").fillna(0.0)
    df["plus_minus"] = pd.to_numeric(df["plus_minus"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["stat_val", "minutes"]).reset_index(drop=True)
    return df


# ── book line simulation ───────────────────────────────────────────────────────

def simulate_line(stat_history: np.ndarray, window: int = SIM_LINE_WIN) -> Optional[float]:
    """Rolling average of past `window` games = simulated book line."""
    if stat_history.size < window:
        return None
    return float(np.mean(stat_history[-window:]))


# ── per-game prediction ────────────────────────────────────────────────────────

def replay_player_stat(
    df: pd.DataFrame,
    stat: str,
    player_id: int = 0,
    warmup: int = WARMUP_GAMES,
    refit_every: int = REFIT_EVERY,
) -> List[Dict[str, Any]]:
    """
    Walk-forward replay for one player+stat.

    Returns list of dicts with per-game prediction metadata.
    HMM and QRF are cached and refit every `refit_every` games.
    """
    records: List[Dict[str, Any]] = []
    n = len(df)
    if n < warmup + 1:
        return records

    cached_hmm: Optional[TruncatedStudentHMM] = None
    cached_qrf: Optional[PlayerStatQRF]       = None
    last_fit_idx: int = -1

    for i in range(warmup, n):
        train = df.iloc[:i].copy()
        game  = df.iloc[i]

        # refit HMM + QRF if cache is stale
        if cached_hmm is None or (i - last_fit_idx) >= refit_every:
            min_arr = train["minutes"].to_numpy(dtype=float)
            try:
                hmm_model = fit_hmm(min_arr)
                hmm_model = reorder_states(hmm_model)
                cached_hmm = hmm_model
            except Exception:
                cached_hmm = None

            # build QRF training frame
            train_qrf = train.rename(columns={"stat_val": stat}).copy()
            train_qrf["player_id"] = player_id
            try:
                mdl = _fit_qrf_for_player_stat(
                    train_qrf,
                    stat=stat,
                    feature_cols=FEATURE_COLS,
                )
                if mdl is not None:
                    cached_qrf = PlayerStatQRF(
                        player_id=0,
                        stat=stat,
                        model=mdl,
                        feature_cols=list(FEATURE_COLS),
                        quantiles=QUANTILES.copy(),
                    )
                else:
                    cached_qrf = None
            except Exception:
                cached_qrf = None

            last_fit_idx = i

        if cached_hmm is None or cached_qrf is None:
            continue

        # simulated book line
        stat_hist = train["stat_val"].to_numpy(dtype=float)
        line = simulate_line(stat_hist)
        if line is None:
            continue

        actual = float(game["stat_val"])
        outcome = 1 if actual > line else 0

        # HMM next-game distribution
        min_arr = train["minutes"].to_numpy(dtype=float)
        try:
            state       = current_state(cached_hmm, min_arr)
            probs_arr   = next_game_probs(cached_hmm, state)
            exp_min     = expected_minutes(cached_hmm, probs_arr)
            hmm_result  = {
                "next_state_probs": {STATE_LABELS[k]: float(probs_arr[k]) for k in range(N_STATES)},
                "expected_minutes": float(exp_min),
                "state_means":      {STATE_LABELS[k]: float(cached_hmm.truncated_state_means()[k])
                                     for k in range(N_STATES)},
                "nu_per_state":     {STATE_LABELS[k]: float(cached_hmm.nu_[k])
                                     for k in range(N_STATES)},
            }
        except Exception:
            continue

        # P(over line) via Monte Carlo over HMM minutes
        try:
            minutes_samples = _sample_minutes_from_hmm(hmm_result, n_samples=N_MONTE_CARLO)
            p_over_vals = []
            for m in minutes_samples:
                x_row  = np.array([float(m), 0.0, 0.0, 0.0, 0.0], dtype=float)
                q_vals = _predict_quantiles(cached_qrf.model, x_row, QUANTILES)
                cdf    = _piecewise_linear_cdf_from_quantiles(QUANTILES, q_vals, float(line))
                p_over_vals.append(float(np.clip(1.0 - cdf, 0.0, 1.0)))
            p_over = float(np.mean(p_over_vals))
        except Exception:
            # fallback: scalar expected_minutes
            try:
                x_row  = np.array([float(exp_min), 0.0, 0.0, 0.0, 0.0], dtype=float)
                q_vals = _predict_quantiles(cached_qrf.model, x_row, QUANTILES)
                cdf    = _piecewise_linear_cdf_from_quantiles(QUANTILES, q_vals, float(line))
                p_over = float(np.clip(1.0 - cdf, 0.0, 1.0))
            except Exception:
                continue

        records.append({
            "game_date":    str(game["game_date"].date()),
            "stat":         stat,
            "line":         float(line),
            "actual":       actual,
            "outcome":      outcome,
            "p_over":       p_over,
            "exp_minutes":  float(exp_min),
            "act_minutes":  float(game["minutes"]),
        })

    return records


# ── calibration ───────────────────────────────────────────────────────────────

def compute_calibration(records: List[Dict[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    """Expected Calibration Error (ECE) over all records."""
    if not records:
        return {"ece": None, "n": 0}
    p_arr = np.array([r["p_over"]  for r in records], dtype=float)
    y_arr = np.array([r["outcome"] for r in records], dtype=float)
    n     = len(p_arr)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    for i in range(n_bins):
        lo   = edges[i]
        hi   = edges[i + 1]
        mask = (p_arr >= lo) & (p_arr <= hi) if i == n_bins - 1 else (p_arr >= lo) & (p_arr < hi)
        cnt  = int(np.sum(mask))
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(float(np.mean(p_arr[mask])) - float(np.mean(y_arr[mask])))
    return {"ece": float(ece), "n": n}


# ── main replay loop ──────────────────────────────────────────────────────────

def run_replay(con) -> pd.DataFrame:
    players = con.execute(
        "SELECT player_id, full_name FROM player_lookup "
        "WHERE player_id IS NOT NULL ORDER BY player_id"
    ).fetchall()

    all_rows: List[Dict[str, Any]] = []
    for pid, name in players:
        player_id   = int(pid)
        player_name = str(name) if name is not None else str(player_id)
        for stat in STATS:
            df = load_player_games(con, player_id, stat)
            if df.empty:
                continue
            try:
                records = replay_player_stat(df, stat, player_id=player_id)
            except Exception as e:
                print(f"  replay failed player_id={player_id} stat={stat}: {e}")
                records = []
            for r in records:
                r["player_id"]   = player_id
                r["player_name"] = player_name
            all_rows.extend(records)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# ── reporting ─────────────────────────────────────────────────────────────────

def print_replay_report(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("No replay results — run materialize.py + hmm_minutes.py + qrf_model.py first.")
        return

    total_n   = int(len(df))
    direction_correct = ((df["p_over"] >= 0.5).astype(int) == df["outcome"].astype(int))
    direction_accuracy = float(direction_correct.mean())
    cal       = compute_calibration(df.to_dict("records"))
    ece       = cal.get("ece")

    print("=== Walk-Forward Replay Report ===\n")
    print(f"Total predictions : {total_n}")
    print(f"Direction accuracy: {direction_accuracy:.3f}")
    print(f"ECE               : {f'{ece:.4f}' if ece is not None else 'N/A'}\n")

    # Per-stat summary
    print(f"{'stat':<6} {'n':>6} {'accuracy':>9} {'ece':>7}")
    print("-" * 28)
    for stat, g in df.groupby("stat"):
        stat_n    = int(len(g))
        stat_hit  = float(((g["p_over"] >= 0.5).astype(int) == g["outcome"].astype(int)).mean())
        stat_cal  = compute_calibration(g.to_dict("records"))
        stat_ece  = stat_cal.get("ece")
        ece_str   = f"{stat_ece:.4f}" if stat_ece is not None else "   N/A"
        print(f"{stat:<6} {stat_n:>6} {stat_hit:>9.3f} {ece_str:>7}")

    # Minutes Variance Effect
    print("\n--- Minutes Variance Effect ---")
    print("(Split predictions by actual minutes vs expected minutes)\n")
    df["min_delta"] = df["act_minutes"] - df["exp_minutes"]
    lo = df[df["min_delta"] <  0.0]
    hi = df[df["min_delta"] >= 0.0]
    for label, grp in [("Low minutes (actual < expected)", lo),
                        ("High minutes (actual >= expected)", hi)]:
        if grp.empty:
            print(f"{label}: no data")
            continue
        g_hit = float(((grp["p_over"] >= 0.5).astype(int) == grp["outcome"].astype(int)).mean())
        g_n   = int(len(grp))
        print(f"{label}")
        print(f"  n={g_n}  direction_accuracy={g_hit:.3f}\n")

    # Per-player summary (top 20 by n)
    player_frame = df.assign(direction_correct=((df["p_over"] >= 0.5).astype(int) == df["outcome"].astype(int)))
    player_stats = (
        player_frame.groupby(["player_name", "stat"])
        .agg(n=("outcome", "count"), hit=("direction_correct", "mean"))
        .reset_index()
        .sort_values(["n", "hit"], ascending=[False, False])
        .head(20)
    )
    print("--- Top Players by Prediction Count ---\n")
    print(f"{'player':<24} {'stat':<5} {'n':>5} {'accuracy':>9}")
    print("-" * 42)
    for _, row in player_stats.iterrows():
        print(f"{str(row['player_name'])[:24]:<24} {row['stat']:<5} {int(row['n']):>5} {float(row['hit']):>6.3f}")

    # Outperformers — every game where actual beat the line
    df["margin"] = df["actual"] - df["line"]
    overs = (
        df[df["outcome"] == 1]
        .sort_values("margin", ascending=False)
        [["game_date", "player_name", "stat", "line", "actual", "margin",
          "exp_minutes", "act_minutes"]]
    )
    print(f"\n--- All Outperformances ({len(overs)} games) — sorted by margin ---\n")
    print(f"{'date':<12} {'player':<22} {'stat':<5} {'line':>6} {'actual':>7} {'margin':>7} "
          f"{'exp_min':>8} {'act_min':>8}")
    print("-" * 80)
    for _, r in overs.iterrows():
        print(f"{str(r['game_date']):<12} {str(r['player_name'])[:22]:<22} {r['stat']:<5} "
              f"{r['line']:>6.1f} {r['actual']:>7.1f} {r['margin']:>7.1f} "
              f"{r['exp_minutes']:>8.1f} {r['act_minutes']:>8.1f}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df = run_replay(con)
        print_replay_report(df)
        if df is not None and not df.empty:
            os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
            df.to_csv(RESULTS_CSV, index=False)
            print(f"\nResults saved → {RESULTS_CSV}")
    finally:
        try:
            con.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
