"""Model player-stat upside during unusual minutes spikes.

A shock game is one where a player's minutes exceed the trailing rolling mean
by a configurable z-score. The module can train a separate quantile forest for
those games and blend it with the player's normal model at prediction time.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

from models.qrf_model import (
    FEATURE_COLS,
    MODEL_DIR,
    QUANTILES,
    PlayerStatQRF,
    _fit_qrf_for_player_stat,
    _piecewise_linear_cdf_from_quantiles,
    _predict_quantiles,
    load_model_from_disk,
)
from storage.dates import parse_game_dates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")
SHOCK_STORE = str(PROJECT_ROOT / "models" / "shock_store")
MIN_SHOCK_GAMES = 8
SHOCK_Z_THRESHOLD = 1.2
ROLLING_WINDOW = 10
SUPPORTED_STATS = {"pts", "reb", "ast"}


@dataclass
class ShockProfile:
    player_id: int
    stat: str
    normal_mean: float
    shock_mean: float
    shock_uplift: float
    shock_rate: float
    n_shock: int
    n_normal: int
    shock_qrf: Optional[object]
    feature_cols: List[str]
    quantiles: np.ndarray


def classify_shock_games(
    df: pd.DataFrame,
    z_threshold: float = SHOCK_Z_THRESHOLD,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Label games whose minutes spike relative to prior games only."""
    if df is None or df.empty:
        out = pd.DataFrame(columns=list(df.columns) if df is not None else [])
        out["rolling_mean_min"] = np.nan
        out["rolling_std_min"] = np.nan
        out["is_shock"] = False
        return out
    if "minutes" not in df.columns:
        raise ValueError("classify_shock_games requires a 'minutes' column")
    if window < 1:
        raise ValueError("window must be at least 1")

    result = df.copy()
    minutes = pd.to_numeric(result["minutes"], errors="coerce").astype(float)
    rolling = minutes.shift(1).rolling(window=window, min_periods=window)
    result["rolling_mean_min"] = rolling.mean()
    result["rolling_std_min"] = rolling.std(ddof=0)
    threshold = (
        result["rolling_mean_min"]
        + float(z_threshold) * result["rolling_std_min"]
    )
    result["is_shock"] = (
        (minutes > threshold)
        & result["rolling_mean_min"].notna()
        & result["rolling_std_min"].notna()
    ).fillna(False)
    return result


def _validate_stat(stat: str) -> str:
    normalized = str(stat).lower()
    if normalized not in SUPPORTED_STATS:
        raise ValueError(f"Unsupported stat: {stat!r}. Choose from {sorted(SUPPORTED_STATS)}")
    return normalized


def _load_player_stat_games(
    con: duckdb.DuckDBPyConnection,
    player_id: int,
    stat: str,
) -> pd.DataFrame:
    stat = _validate_stat(stat)
    query = f"""
        SELECT game_date, minutes, {stat} AS stat_val, fga, fg3a, tov, plus_minus
        FROM player_game_features
        WHERE player_id = ? AND minutes >= 1 AND {stat} IS NOT NULL
        ORDER BY game_date ASC
    """
    df = con.execute(query, [int(player_id)]).fetchdf()
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["game_date", "minutes", "stat_val", "fga", "fg3a", "tov", "plus_minus"]
        )

    for column in ["minutes", "stat_val", "fga", "fg3a", "tov", "plus_minus"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    df["game_date"] = parse_game_dates(df["game_date"], utc=False)
    df = df.dropna(subset=["game_date", "minutes", "stat_val"])
    df = df.sort_values("game_date").reset_index(drop=True)
    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")
    return df


def build_shock_profile(
    player_id: int,
    stat: str,
    con: duckdb.DuckDBPyConnection,
) -> Optional[ShockProfile]:
    """Build a player's normal-versus-shock minutes profile."""
    stat = _validate_stat(stat)
    df = _load_player_stat_games(con, player_id, stat)
    if df.empty:
        return None

    df = classify_shock_games(df)
    shock_df = df[df["is_shock"]].copy()
    normal_df = df[~df["is_shock"]].copy()
    n_shock = int(len(shock_df))
    n_normal = int(len(normal_df))
    if n_normal == 0:
        return None

    normal_mean = float(normal_df["minutes"].mean())
    shock_mean = float(shock_df["minutes"].mean()) if n_shock else normal_mean
    shock_uplift = shock_mean - normal_mean

    shock_qrf = None
    if n_shock >= MIN_SHOCK_GAMES:
        training_df = shock_df.rename(columns={"stat_val": stat})
        shock_qrf = _fit_qrf_for_player_stat(
            training_df,
            stat=stat,
            feature_cols=list(FEATURE_COLS),
        )

    return ShockProfile(
        player_id=int(player_id),
        stat=stat,
        normal_mean=normal_mean,
        shock_mean=shock_mean,
        shock_uplift=shock_uplift,
        shock_rate=float(n_shock / len(df)),
        n_shock=n_shock,
        n_normal=n_normal,
        shock_qrf=shock_qrf,
        feature_cols=list(FEATURE_COLS),
        quantiles=np.asarray(QUANTILES, dtype=float),
    )


def save_shock_profile(profile: ShockProfile) -> str:
    os.makedirs(SHOCK_STORE, exist_ok=True)
    path = os.path.join(SHOCK_STORE, f"{profile.player_id}_{profile.stat}_shock.pkl")
    with open(path, "wb") as file_handle:
        pickle.dump(profile, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_shock_profile(player_id: int, stat: str) -> Optional[ShockProfile]:
    path = os.path.join(SHOCK_STORE, f"{int(player_id)}_{_validate_stat(stat)}_shock.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as file_handle:
            profile = pickle.load(file_handle)
    except (OSError, pickle.PickleError):
        return None
    return profile if isinstance(profile, ShockProfile) else None


def load_normal_qrf(player_id: int, stat: str) -> Optional[PlayerStatQRF]:
    try:
        return load_model_from_disk(int(player_id), _validate_stat(stat), MODEL_DIR)
    except (FileNotFoundError, OSError, TypeError, pickle.PickleError):
        return None


def _p_over_from_qrf(
    model_obj: Any,
    quantiles: np.ndarray,
    x_row: np.ndarray,
    line: float,
) -> float:
    predictions = _predict_quantiles(
        model_obj,
        x_row=np.asarray(x_row, dtype=float),
        quantiles=np.asarray(quantiles, dtype=float),
    )
    cdf = _piecewise_linear_cdf_from_quantiles(quantiles, predictions, float(line))
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def shock_adjusted_prob(
    player_id: int,
    stat: str,
    line: float,
    minutes_estimate: float,
    shock_prior: float = 0.0,
) -> Dict[str, Any]:
    """Blend normal and minutes-shock probabilities using ``shock_prior``."""
    stat = _validate_stat(stat)
    profile = load_shock_profile(player_id, stat)
    normal_wrapper = load_normal_qrf(player_id, stat)
    features = np.array([float(minutes_estimate), 0.0, 0.0, 0.0, 0.0])

    p_normal = 0.5
    if normal_wrapper is not None:
        p_normal = _p_over_from_qrf(
            normal_wrapper.model,
            normal_wrapper.quantiles,
            features,
            line,
        )

    shock_uplift = profile.shock_uplift if profile is not None else 0.0
    p_shock = p_normal
    if profile is not None and profile.shock_qrf is not None:
        p_shock = _p_over_from_qrf(profile.shock_qrf, profile.quantiles, features, line)
    else:
        p_shock = float(np.clip(p_normal + np.clip(shock_uplift / 20.0, 0.0, 0.12), 0.0, 1.0))

    prior = float(np.clip(shock_prior, 0.0, 1.0))
    adjusted = (1.0 - prior) * p_normal + prior * p_shock
    signal = "SHOCK_EDGE" if prior > 0.3 and p_shock > p_normal + 0.05 else "NORMAL"
    return {
        "p_over_adjusted": float(np.clip(adjusted, 0.0, 1.0)),
        "p_normal": float(p_normal),
        "p_shock": float(p_shock),
        "shock_prior": prior,
        "shock_uplift": float(shock_uplift),
        "n_shock": profile.n_shock if profile is not None else 0,
        "signal": signal,
    }


def train_all_shock_profiles(con: duckdb.DuckDBPyConnection) -> int:
    """Train and save profiles for every player and supported stat."""
    os.makedirs(SHOCK_STORE, exist_ok=True)
    rows = con.execute(
        "SELECT player_id, full_name FROM player_lookup "
        "WHERE player_id IS NOT NULL ORDER BY player_id"
    ).fetchall()
    saved = 0
    for player_id, name in rows:
        for stat in sorted(SUPPORTED_STATS):
            profile = build_shock_profile(int(player_id), stat, con)
            if profile is None:
                continue
            save_shock_profile(profile)
            saved += 1
            print(
                f"{name or player_id} | {stat} | n_shock={profile.n_shock} "
                f"n_normal={profile.n_normal} shock_uplift={profile.shock_uplift:.2f}"
            )
    return saved


def main() -> None:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        count = train_all_shock_profiles(con)
        print(f"Saved {count} shock profiles -> {SHOCK_STORE}/")
    finally:
        con.close()


if __name__ == "__main__":
    main()
