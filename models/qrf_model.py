import os
import sys
import pickle
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import duckdb
from scipy.stats import t as student_t


def _ensure_quantile_forest() -> None:
    try:
        from quantile_forest import RandomForestQuantileRegressor  # noqa: F401
        return
    except Exception:
        pass
    cmd = [sys.executable, "-m", "pip", "install", "quantile-forest"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to install quantile-forest (rc={proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


_ensure_quantile_forest()
from quantile_forest import RandomForestQuantileRegressor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH      = str(PROJECT_ROOT / "data" / "raw.db")
TABLE_NAME   = "player_game_features"
MODEL_DIR    = str(PROJECT_ROOT / "models" / "qrf_store")
FEATURE_COLS = ["minutes", "fga", "fg3a", "tov", "plus_minus"]
STATS        = ["pts", "reb", "ast"]
QUANTILES    = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=float)


@dataclass
class PlayerStatQRF:
    player_id:    int
    stat:         str
    model:        RandomForestQuantileRegressor
    feature_cols: List[str]
    quantiles:    np.ndarray


_MODELS: Dict[Tuple[int, str], PlayerStatQRF] = {}


def _load_player_game_features(db_path: str = DB_PATH,
                                table_name: str = TABLE_NAME) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(f"""
            SELECT player_id, game_date, minutes,
                   pts, reb, ast, fga, fgm, fg3a, tov, plus_minus
            FROM {table_name}
            WHERE player_id IS NOT NULL
              AND game_date IS NOT NULL
              AND minutes IS NOT NULL
        """).fetchdf()
    finally:
        con.close()
    df["player_id"] = df["player_id"].astype(int)
    df["minutes"]   = df["minutes"].astype(float)
    for c in ["pts", "reb", "ast", "fga", "fgm", "fg3a", "tov", "plus_minus"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"]).copy()
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return df


def _fit_qrf_for_player_stat(
    g: pd.DataFrame,
    stat: str,
    feature_cols: List[str] = FEATURE_COLS,
    random_state: int = 7,
) -> Optional[RandomForestQuantileRegressor]:
    cols_needed = ["player_id", "game_date", stat] + feature_cols
    missing = [c for c in cols_needed if c not in g.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    d = g[cols_needed].dropna(subset=[stat] + feature_cols).copy()
    if d.shape[0] < 20:
        return None
    X = d[feature_cols].to_numpy(dtype=float)
    y = d[stat].to_numpy(dtype=float)
    model = RandomForestQuantileRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _predict_quantiles(
    model: RandomForestQuantileRegressor,
    x_row: np.ndarray,
    quantiles: np.ndarray = QUANTILES,
) -> np.ndarray:
    x_row = np.asarray(x_row, dtype=float).reshape(1, -1)
    q     = model.predict(x_row, quantiles=quantiles.tolist())
    return np.asarray(q, dtype=float).reshape(-1)


def _sample_minutes_from_hmm(
    hmm_result: dict,
    n_samples: int = 500,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Monte Carlo sample of minutes from HMM distribution.
    Draws state from next_state_probs, then samples TruncStudentT(mu_k, sigma_k, nu_k; 0,48).
    Returns array of shape (n_samples,).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    labels     = ["DNP", "Limited", "Rotation", "Featured"]
    probs_dict = hmm_result.get("next_state_probs", {}) or {}
    means_dict = hmm_result.get("state_means", {}) or {}
    nu_dict    = hmm_result.get("nu_per_state", {}) or {}

    probs = np.array([float(probs_dict.get(k, 0.0)) for k in labels], dtype=float)
    s     = float(np.sum(probs))
    if not np.isfinite(s) or s <= 0.0:
        probs = np.full(len(labels), 1.0 / len(labels), dtype=float)
    else:
        probs = probs / s

    mus = np.array([float(means_dict.get(k, 0.0)) for k in labels], dtype=float)
    nus = np.array([float(nu_dict.get(k, 10.0)) for k in labels], dtype=float)

    out = np.zeros(int(n_samples), dtype=float)
    for i in range(int(n_samples)):
        k      = int(rng.choice(len(labels), p=probs))
        mu_k   = float(mus[k])
        nu_k   = float(nus[k]) if np.isfinite(nus[k]) else 10.0
        nu_k   = max(2.01, nu_k)
        sigma_k = max(mu_k * 0.25, 2.0)
        m      = mu_k
        for _attempt in range(20):
            draw = float(student_t.rvs(df=nu_k, random_state=rng))
            cand = draw * sigma_k + mu_k
            if 0.0 <= cand <= 48.0:
                m = cand
                break
        out[i] = float(np.clip(m, 0.0, 48.0))
    return out


def _piecewise_linear_cdf_from_quantiles(
    q_levels: np.ndarray,
    q_values: np.ndarray,
    x: float,
) -> float:
    q_levels = np.asarray(q_levels, dtype=float).reshape(-1)
    q_values = np.asarray(q_values, dtype=float).reshape(-1)
    order    = np.argsort(q_values)
    v        = q_values[order]
    p        = q_levels[order]
    if x <= v[0]:
        return float(p[0])
    if x >= v[-1]:
        return float(p[-1])
    j       = int(np.searchsorted(v, x, side="right") - 1)
    v0, v1  = float(v[j]), float(v[j + 1])
    p0, p1  = float(p[j]), float(p[j + 1])
    if v1 == v0:
        return float(max(p0, p1))
    w = (x - v0) / (v1 - v0)
    return float(p0 + w * (p1 - p0))


def train_all_players(
    df: pd.DataFrame,
    model_dir: str = MODEL_DIR,
    stats: List[str] = STATS,
    feature_cols: List[str] = FEATURE_COLS,
    quantiles: np.ndarray = QUANTILES,
) -> Dict[Tuple[int, str], PlayerStatQRF]:
    global _MODELS
    os.makedirs(model_dir, exist_ok=True)
    models: Dict[Tuple[int, str], PlayerStatQRF] = {}

    for player_id, g in df.groupby("player_id", sort=True):
        g = g.sort_values("game_date").reset_index(drop=True)
        for stat in stats:
            mdl = _fit_qrf_for_player_stat(g, stat=stat, feature_cols=feature_cols)
            if mdl is None:
                continue
            key     = (int(player_id), stat)
            wrapper = PlayerStatQRF(
                player_id=int(player_id),
                stat=stat,
                model=mdl,
                feature_cols=list(feature_cols),
                quantiles=np.asarray(quantiles, dtype=float),
            )
            models[key] = wrapper
            out_path = os.path.join(model_dir, f"{int(player_id)}_{stat}_qrf.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(wrapper, f, protocol=pickle.HIGHEST_PROTOCOL)

    _MODELS = models
    return models


def load_model_from_disk(player_id: int, stat: str,
                         model_dir: str = MODEL_DIR) -> PlayerStatQRF:
    path = os.path.join(model_dir, f"{int(player_id)}_{stat}_qrf.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, PlayerStatQRF):
        return obj
    # Backward compat: old format was raw RandomForestQuantileRegressor
    if isinstance(obj, RandomForestQuantileRegressor):
        return PlayerStatQRF(
            player_id=int(player_id),
            stat=str(stat),
            model=obj,
            feature_cols=list(FEATURE_COLS),
            quantiles=QUANTILES.copy(),
        )
    raise TypeError(f"Unexpected pickle type at {path}: {type(obj)}")


def prob_over_line(player_id: int, stat: str, line: float,
                   minutes_estimate: float,
                   model_dir: str = MODEL_DIR) -> float:
    """
    P(stat > line) given scalar minutes_estimate.
    Interpolates piecewise-linear CDF from QRF quantile predictions.
    """
    wrapper = _MODELS.get((int(player_id), stat))
    if wrapper is None:
        wrapper = load_model_from_disk(int(player_id), stat, model_dir=model_dir)
    x     = np.array([minutes_estimate, 0.0, 0.0, 0.0, 0.0], dtype=float)
    q_vals = _predict_quantiles(wrapper.model, x, wrapper.quantiles)
    cdf   = _piecewise_linear_cdf_from_quantiles(wrapper.quantiles, q_vals, float(line))
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def prob_over_line_hmm(
    player_id: int,
    stat: str,
    line: float,
    hmm_result: dict,
    n_samples: int = 500,
    model_dir: str = MODEL_DIR,
) -> float:
    """
    P(stat > line) integrating over HMM minutes distribution.
    Monte Carlo: sample N minutes values from HMM, compute QRF P(over) at each, return mean.
    Falls back to prob_over_line() using expected_minutes if sampling fails.
    """
    try:
        wrapper = _MODELS.get((int(player_id), stat))
        if wrapper is None:
            wrapper = load_model_from_disk(int(player_id), stat, model_dir=model_dir)

        minutes_samples = _sample_minutes_from_hmm(hmm_result, n_samples=int(n_samples))
        if minutes_samples.size == 0:
            raise ValueError("Empty minutes_samples")

        p_over_samples = np.zeros(minutes_samples.shape[0], dtype=float)
        for i, m in enumerate(minutes_samples):
            x      = np.array([float(m), 0.0, 0.0, 0.0, 0.0], dtype=float)
            q_vals = _predict_quantiles(wrapper.model, x, wrapper.quantiles)
            cdf    = _piecewise_linear_cdf_from_quantiles(wrapper.quantiles, q_vals, float(line))
            p_over_samples[i] = float(np.clip(1.0 - cdf, 0.0, 1.0))
        return float(np.mean(p_over_samples))
    except Exception:
        return prob_over_line(
            player_id=int(player_id),
            stat=str(stat),
            line=float(line),
            minutes_estimate=float(hmm_result.get("expected_minutes", 0.0)),
            model_dir=model_dir,
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        df = _load_player_game_features(DB_PATH, TABLE_NAME)
        if df.empty:
            raise ValueError("Empty features")
    except Exception:
        print("No features found. Run materialize.py first.")
        return 1

    print(f"Loaded {len(df)} rows for {df['player_id'].nunique()} players\n")
    models = train_all_players(df)

    print("\n=== Quantile Predictions (last game features) ===\n")
    for (pid, stat), wrapper in sorted(models.items(), key=lambda x: (x[0][0], x[0][1])):
        g     = df[df["player_id"] == pid].sort_values("game_date")
        x_row = g.iloc[-1][FEATURE_COLS].to_numpy(dtype=float)
        q_vals = _predict_quantiles(wrapper.model, x_row, wrapper.quantiles)
        q_str  = "  ".join([f"q{int(q*100)}={v:.1f}" for q, v in zip(wrapper.quantiles, q_vals)])
        print(f"Player {pid} | {stat.upper()}: {q_str}")

    if models:
        pid0, stat0 = next(iter(models))
        p = prob_over_line(pid0, stat0, line=15.5, minutes_estimate=28.0)
        print(f"\nprob_over_line example: player={pid0} stat={stat0} line=15.5 -> P(over)={p:.4f}")

    print(f"\nModels saved to {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
