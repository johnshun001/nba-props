"""Production pooled LightGBM/XGBoost/quantile-forest prop ensemble."""

from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from config import load_config, project_path
from features.pregame_features import FEATURE_COLUMNS


QUANTILE_LEVELS = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90], dtype=float)
QUANTILE_NAMES = ("q10", "q25", "median", "q75", "q90")
SUPPORTED_STATS = ("pts", "reb", "ast")


@dataclass(frozen=True)
class ChronologicalSplit:
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    boundaries: dict[str, str | None]


@dataclass
class EnsembleArtifact:
    stat: str
    feature_columns: list[str]
    feature_version: str
    data_version: str
    model_version: str
    trained_through: str
    imputer: SimpleImputer
    lightgbm_quantiles: dict[float, Any]
    xgboost_model: Any
    qrf_model: Any
    torch_quantile_model: Any | None = None
    calibrator: IsotonicRegression | None = None
    conformal_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    split_boundaries: dict[str, str | None] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)


def chronological_split(
    frame: pd.DataFrame,
    *,
    time_column: str = "prediction_time",
    validation_fraction: float = 0.15,
    calibration_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> ChronologicalSplit:
    """Split on whole timestamps, preserving train < val < cal < test."""
    if time_column not in frame:
        raise ValueError(f"Missing chronological split column: {time_column}")
    times = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    if times.isna().any():
        raise ValueError(f"{time_column} contains invalid timestamps")
    unique = np.asarray(sorted(times.unique()))
    if len(unique) < 4:
        raise ValueError("At least four distinct prediction timestamps are required")
    fractions = [float(validation_fraction), float(calibration_fraction), float(test_fraction)]
    if any(value <= 0 for value in fractions) or sum(fractions) >= 1.0:
        raise ValueError("validation/calibration/test fractions must be positive and sum below one")
    n = len(unique)
    n_val = max(1, int(math.floor(n * fractions[0])))
    n_cal = max(1, int(math.floor(n * fractions[1])))
    n_test = max(1, int(math.floor(n * fractions[2])))
    n_train = n - n_val - n_cal - n_test
    if n_train < 1:
        raise ValueError("Not enough distinct timestamps for requested split")
    train_end = unique[n_train - 1]
    val_end = unique[n_train + n_val - 1]
    cal_end = unique[n_train + n_val + n_cal - 1]
    arrays = {
        "train": np.flatnonzero((times <= train_end).to_numpy()),
        "validation": np.flatnonzero(((times > train_end) & (times <= val_end)).to_numpy()),
        "calibration": np.flatnonzero(((times > val_end) & (times <= cal_end)).to_numpy()),
        "test": np.flatnonzero((times > cal_end).to_numpy()),
    }
    boundaries = {
        "train_end": pd.Timestamp(train_end).isoformat(),
        "validation_end": pd.Timestamp(val_end).isoformat(),
        "calibration_end": pd.Timestamp(cal_end).isoformat(),
        "test_end": pd.Timestamp(unique[-1]).isoformat(),
    }
    return ChronologicalSplit(**arrays, boundaries=boundaries)


def _cdf_from_quantiles(values: np.ndarray, line: float) -> float:
    values = np.maximum.accumulate(np.asarray(values, dtype=float))
    if line <= values[0]:
        # Linear lower-tail extrapolation using the first two knots.
        width = max(values[1] - values[0], 1.0)
        return float(np.clip(QUANTILE_LEVELS[0] * math.exp((line - values[0]) / width), 1e-6, 1.0))
    if line >= values[-1]:
        width = max(values[-1] - values[-2], 1.0)
        survival = (1.0 - QUANTILE_LEVELS[-1]) * math.exp((values[-1] - line) / width)
        return float(np.clip(1.0 - survival, 0.0, 1.0 - 1e-6))
    return float(np.interp(line, values, QUANTILE_LEVELS))


class PooledPropEnsemble:
    """One pooled model per statistic; never consumes same-game actual minutes."""

    def __init__(self, artifact: EnsembleArtifact):
        self.artifact = artifact

    @property
    def stat(self) -> str:
        return self.artifact.stat

    def predict_quantiles(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.artifact.feature_columns if c not in frame]
        if missing:
            raise ValueError(f"Prediction frame is missing trained features: {missing}")
        x = self.artifact.imputer.transform(frame[self.artifact.feature_columns])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            lgb = np.column_stack(
                [self.artifact.lightgbm_quantiles[level].predict(x) for level in QUANTILE_LEVELS]
            )
        qrf = np.asarray(self.artifact.qrf_model.predict(x, quantiles=QUANTILE_LEVELS.tolist()))
        if qrf.ndim == 1:
            qrf = qrf.reshape(-1, len(QUANTILE_LEVELS))
        if self.artifact.torch_quantile_model is not None:
            torch_quantiles = self.artifact.torch_quantile_model.predict(x)
            quantiles = 0.47 * lgb + 0.38 * qrf + 0.15 * torch_quantiles
        else:
            quantiles = 0.55 * lgb + 0.45 * qrf
        xgb_median = np.asarray(self.artifact.xgboost_model.predict(x), dtype=float)
        quantiles[:, 2] = 0.4 * quantiles[:, 2] + 0.6 * xgb_median
        quantiles = np.maximum.accumulate(quantiles, axis=1)
        return pd.DataFrame(quantiles, columns=QUANTILE_NAMES, index=frame.index)

    def predict_distribution(self, frame: pd.DataFrame, lines: pd.Series | np.ndarray) -> pd.DataFrame:
        output = self.predict_quantiles(frame)
        line_values = np.asarray(lines, dtype=float).reshape(-1)
        if len(line_values) != len(output):
            raise ValueError("lines must have one value per prediction row")
        raw = np.asarray(
            [1.0 - _cdf_from_quantiles(row, line) for row, line in zip(output.to_numpy(), line_values)],
            dtype=float,
        )
        if self.artifact.calibrator is not None:
            calibrated = np.clip(self.artifact.calibrator.predict(raw), 0.0, 1.0)
            status = "READY"
        else:
            calibrated = raw
            status = "NOT_READY"
        output["line"] = line_values
        output["p_over_raw"] = raw
        output["p_over"] = calibrated
        output["p_under"] = 1.0 - calibrated
        output["calibration_status"] = status
        if self.artifact.conformal_scores.size:
            radius = float(np.quantile(self.artifact.conformal_scores, 0.90, method="higher"))
            output["conformal_lower"] = output["median"] - radius
            output["conformal_upper"] = output["median"] + radius
        else:
            output["conformal_lower"] = np.nan
            output["conformal_upper"] = np.nan
            output["calibration_status"] = "NOT_READY"
        return output

    def save(self, path: str | Path | None = None) -> Path:
        destination = Path(path) if path is not None else project_path("artifacts") / f"{self.stat}.pkl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self.artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PooledPropEnsemble":
        with Path(path).open("rb") as handle:
            artifact = pickle.load(handle)
        if not isinstance(artifact, EnsembleArtifact):
            raise TypeError(f"Unexpected ensemble artifact: {type(artifact)!r}")
        return cls(artifact)


def _metric_bundle(actual: np.ndarray, median: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, median)),
        "rmse": float(mean_squared_error(actual, median) ** 0.5),
    }


def _probability_metrics(actual: np.ndarray, lines: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    non_push = actual != lines
    if not non_push.any():
        return {}
    outcomes = (actual[non_push] > lines[non_push]).astype(int)
    probs = np.clip(probabilities[non_push], 1e-6, 1.0 - 1e-6)
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for index in range(10):
        mask = (probs >= edges[index]) & (probs <= edges[index + 1] if index == 9 else probs < edges[index + 1])
        if mask.any():
            ece += mask.mean() * abs(probs[mask].mean() - outcomes[mask].mean())
    return {
        "brier_score": float(np.mean((probs - outcomes) ** 2)),
        "log_loss": float(log_loss(outcomes, probs, labels=[0, 1])),
        "calibration_error": float(ece),
        "direction_accuracy": float(np.mean((probs >= 0.5) == outcomes)),
    }


def train_pooled_ensemble(
    frame: pd.DataFrame,
    stat: str,
    *,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
) -> PooledPropEnsemble:
    """Fit the three-model ensemble with disjoint chronological periods."""
    if stat not in SUPPORTED_STATS:
        raise ValueError(f"Unsupported stat: {stat}")
    config = load_config()
    columns = list(feature_columns or FEATURE_COLUMNS)
    required = [stat, "prediction_time", *columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Training frame is missing columns: {missing}")
    clean = frame.dropna(subset=[stat, "prediction_time"]).sort_values("prediction_time").reset_index(drop=True)
    split_config = config["splits"]
    split = chronological_split(
        clean,
        validation_fraction=split_config["validation_fraction"],
        calibration_fraction=split_config["calibration_fraction"],
        test_fraction=split_config["test_fraction"],
    )
    if len(split.train) < int(split_config["minimum_training_rows"]):
        raise ValueError(
            f"Need at least {split_config['minimum_training_rows']} chronological training rows; "
            f"received {len(split.train)}"
        )
    # Validation remains genuinely out of sample; calibration and test are later still.
    fit_idx = split.train
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_fit = imputer.fit_transform(clean.iloc[fit_idx][columns])
    y_fit = clean.iloc[fit_idx][stat].to_numpy(dtype=float)

    lgb_models: dict[float, LGBMRegressor] = {}
    for level in QUANTILE_LEVELS:
        model = LGBMRegressor(
            objective="quantile",
            alpha=float(level),
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=24,
            min_child_samples=15,
            random_state=random_state,
            verbosity=-1,
        )
        model.fit(x_fit, y_fit)
        lgb_models[float(level)] = model
    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=220,
        learning_rate=0.035,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=1,
    )
    xgb.fit(x_fit, y_fit)
    qrf = RandomForestQuantileRegressor(
        n_estimators=240,
        min_samples_leaf=6,
        random_state=random_state,
        n_jobs=-1,
    )
    qrf.fit(x_fit, y_fit)
    torch_model = None
    if config.get("torch_quantile_if_available", False):
        try:
            from models.torch_quantile import TorchQuantileRegressor
            torch_model = TorchQuantileRegressor(QUANTILE_LEVELS, random_state=random_state).fit(x_fit, y_fit)
        except (ImportError, RuntimeError):
            torch_model = None
    versions = config["versions"]
    artifact = EnsembleArtifact(
        stat=stat,
        feature_columns=columns,
        feature_version=versions["feature"],
        data_version=versions["data"],
        model_version=versions["model"],
        trained_through=str(pd.to_datetime(clean.iloc[fit_idx]["prediction_time"], utc=True).max()),
        imputer=imputer,
        lightgbm_quantiles=lgb_models,
        xgboost_model=xgb,
        qrf_model=qrf,
        torch_quantile_model=torch_model,
        split_boundaries=split.boundaries,
    )
    ensemble = PooledPropEnsemble(artifact)
    validation = clean.iloc[split.validation]
    artifact.validation_metrics = _metric_bundle(
        validation[stat].to_numpy(dtype=float), ensemble.predict_quantiles(validation)["median"].to_numpy()
    )

    calibration = clean.iloc[split.calibration]
    if "sportsbook_line" in calibration and len(calibration) >= int(config["thresholds"]["minimum_calibration_rows"]):
        usable = calibration.dropna(subset=["sportsbook_line"])
        if len(usable) >= int(config["thresholds"]["minimum_calibration_rows"]):
            q = ensemble.predict_quantiles(usable).to_numpy()
            lines = usable["sportsbook_line"].to_numpy(dtype=float)
            raw_p = np.asarray([1.0 - _cdf_from_quantiles(row, line) for row, line in zip(q, lines)])
            outcomes = (usable[stat].to_numpy(dtype=float) > lines).astype(float)
            non_push = usable[stat].to_numpy(dtype=float) != lines
            if non_push.sum() >= 2 and np.unique(outcomes[non_push]).size == 2:
                artifact.calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_p[non_push], outcomes[non_push])
    calibration_median = ensemble.predict_quantiles(calibration)["median"].to_numpy(dtype=float)
    artifact.conformal_scores = np.abs(calibration[stat].to_numpy(dtype=float) - calibration_median)
    test = clean.iloc[split.test]
    artifact.test_metrics = _metric_bundle(
        test[stat].to_numpy(dtype=float), ensemble.predict_quantiles(test)["median"].to_numpy()
    )
    if "sportsbook_line" in test and test["sportsbook_line"].notna().all():
        test_distribution = ensemble.predict_distribution(test, test["sportsbook_line"])
        artifact.test_metrics.update(_probability_metrics(
            test[stat].to_numpy(dtype=float), test["sportsbook_line"].to_numpy(dtype=float),
            test_distribution["p_over"].to_numpy(dtype=float),
        ))
    return ensemble


__all__ = [
    "ChronologicalSplit",
    "EnsembleArtifact",
    "PooledPropEnsemble",
    "QUANTILE_LEVELS",
    "chronological_split",
    "train_pooled_ensemble",
]
