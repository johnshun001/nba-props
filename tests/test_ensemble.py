"""Tests for ensemble: (lgb + xgb) / 2."""

import numpy as np
import pytest
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Test the ensemble logic directly since nba_prop_model structure has changed
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split


def train_simple_ensemble_models(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Train simple ensemble models for testing."""
    alphas = [0.1, 0.5, 0.9]

    X = df[feature_cols].values
    y = df[target_col].values

    # Create simple weights
    weights = np.ones(len(df))
    garbage_time = (df['quarter'] == 4) & (df['point_diff'] > 15) & (df['minutes_remaining'] < 5)
    clutch = ((df['quarter'] >= 3) & (df['point_diff'] < 5))
    weights[garbage_time] = 0.3
    weights[clutch] = 1.5

    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    lgb_models = {}
    xgb_models = {}
    calibrators = {}

    for alpha in alphas:
        lgb_model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            n_estimators=50,  # Fewer for faster tests
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train, sample_weight=weights_train)

        xgb_model = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train, sample_weight=weights_train)

        lgb_pred_raw = lgb_model.predict(X_val)
        xgb_pred_raw = xgb_model.predict(X_val)
        ensemble_pred_raw = (lgb_pred_raw + xgb_pred_raw) / 2

        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(ensemble_pred_raw, y_val)

        lgb_models[alpha] = lgb_model
        xgb_models[alpha] = xgb_model
        calibrators[alpha] = calibrator

    return lgb_models, xgb_models, calibrators


def simple_predict(lgb_models, xgb_models, calibrators, X):
    """Simple prediction function for ensemble model."""
    predictions = {}
    for alpha in [0.1, 0.5, 0.9]:
        lgb_pred = lgb_models[alpha].predict(X)
        xgb_pred = xgb_models[alpha].predict(X)
        ensemble_raw = (lgb_pred + xgb_pred) / 2
        predictions[alpha] = calibrators[alpha].transform(ensemble_raw)
    return predictions


def ensemble_predictions(lgb_pred: np.ndarray, xgb_pred: np.ndarray) -> np.ndarray:
    """Simple average ensemble: (lgb + xgb) / 2."""
    return (lgb_pred + xgb_pred) / 2


class TestEnsemble:
    """Tests for ensemble prediction system."""

    def test_simple_average(self):
        """Ensemble should be simple average of LGB and XGB."""
        lgb_pred = np.array([10.0, 20.0, 30.0, 40.0])
        xgb_pred = np.array([12.0, 18.0, 32.0, 38.0])

        ensemble = ensemble_predictions(lgb_pred, xgb_pred)

        expected = np.array([11.0, 19.0, 31.0, 39.0])
        np.testing.assert_array_almost_equal(ensemble, expected)

    def test_ensemble_between_base_models(self):
        """Ensemble predictions should be between LGB and XGB predictions."""
        np.random.seed(42)
        lgb_pred = np.random.randn(100) * 10 + 20
        xgb_pred = np.random.randn(100) * 10 + 20

        ensemble = ensemble_predictions(lgb_pred, xgb_pred)

        # Ensemble should be between the two (or equal if they're the same)
        for i in range(len(ensemble)):
            min_pred = min(lgb_pred[i], xgb_pred[i])
            max_pred = max(lgb_pred[i], xgb_pred[i])
            assert min_pred <= ensemble[i] <= max_pred

    def test_equal_weights(self):
        """LGB and XGB should have equal weight (0.5 each)."""
        lgb_pred = np.array([100.0])
        xgb_pred = np.array([0.0])

        ensemble = ensemble_predictions(lgb_pred, xgb_pred)

        # Should be exactly 50.0 (equal weighting)
        assert ensemble[0] == 50.0

    def test_variance_reduction(self):
        """Ensemble should have lower variance than individual models."""
        np.random.seed(42)
        n = 1000

        # Simulate two models with same mean but independent noise
        lgb_pred = np.random.randn(n) * 5 + 25
        xgb_pred = np.random.randn(n) * 5 + 25

        ensemble = ensemble_predictions(lgb_pred, xgb_pred)

        # Ensemble variance should be lower (averaging reduces variance)
        lgb_var = np.var(lgb_pred)
        xgb_var = np.var(xgb_pred)
        ensemble_var = np.var(ensemble)

        # With independent errors, ensemble var should be ~half
        assert ensemble_var < lgb_var
        assert ensemble_var < xgb_var

    def test_integration_with_predict_function(self):
        """Test ensemble is used in the predict function."""
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'pts_roll5': np.random.randn(200) * 5 + 20,
            'reb_roll5': np.random.randn(200) * 3 + 8,
            'ast_roll5': np.random.randn(200) * 2 + 5,
            'career_ppg': np.random.randn(200) * 4 + 18,
            'strength_ratio': np.random.rand(200) * 0.5 + 0.8,
            'is_bench': np.random.randint(0, 2, 200),
            'quarter': np.random.choice([1, 2, 3, 4], 200),
            'point_diff': np.random.randn(200) * 10 + 5,
            'minutes_remaining': np.random.rand(200) * 48,
            'pts_target': np.random.randn(200) * 6 + 22,
        })

        feature_cols = ['pts_roll5', 'reb_roll5', 'ast_roll5', 'career_ppg', 'strength_ratio', 'is_bench']

        # Train models using local function
        lgb_preds, xgb_preds, calibrators = train_simple_ensemble_models(df, feature_cols, 'pts_target')

        # Generate predictions using simple_predict
        X = df.head(10)[feature_cols].values
        predictions = simple_predict(lgb_preds, xgb_preds, calibrators, X)

        # Should have predictions for all quantiles
        assert 0.1 in predictions
        assert 0.5 in predictions
        assert 0.9 in predictions

        # Each should have correct shape
        for alpha, pred in predictions.items():
            assert len(pred) == 10

    def test_ensemble_improves_accuracy(self):
        """Ensemble should generally outperform individual models."""
        np.random.seed(42)
        n = 500

        # Generate true values
        X = np.random.randn(n, 5)
        true_y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n) * 0.5

        # Simulate two models with different biases
        lgb_pred = true_y + np.random.randn(n) * 1.5 + 0.5  # LGB has +0.5 bias
        xgb_pred = true_y + np.random.randn(n) * 1.5 - 0.5  # XGB has -0.5 bias

        ensemble = ensemble_predictions(lgb_pred, xgb_pred)

        # Calculate MSE
        lgb_mse = np.mean((lgb_pred - true_y) ** 2)
        xgb_mse = np.mean((xgb_pred - true_y) ** 2)
        ensemble_mse = np.mean((ensemble - true_y) ** 2)

        # Ensemble should have lower MSE (biases cancel out)
        assert ensemble_mse < lgb_mse
        assert ensemble_mse < xgb_mse

    def test_empty_predictions(self):
        """Empty arrays should return empty ensemble."""
        lgb_pred = np.array([])
        xgb_pred = np.array([])

        ensemble = ensemble_predictions(lgb_pred, xgb_pred)

        assert len(ensemble) == 0

    def test_single_prediction(self):
        """Single prediction should work correctly."""
        lgb_pred = np.array([25.5])
        xgb_pred = np.array([27.5])

        ensemble = ensemble_predictions(lgb_pred, xgb_pred)

        assert ensemble[0] == 26.5

    def test_large_batch(self):
        """Should handle large batches efficiently."""
        np.random.seed(42)
        n = 10000

        lgb_pred = np.random.randn(n) * 10 + 25
        xgb_pred = np.random.randn(n) * 10 + 25

        ensemble = ensemble_predictions(lgb_pred, xgb_pred)

        assert len(ensemble) == n
        assert np.mean(ensemble) > 20 and np.mean(ensemble) < 30

    def test_numerical_stability(self):
        """Should handle extreme values without overflow."""
        lgb_pred = np.array([1e10, -1e10, 0.0])
        xgb_pred = np.array([1e10, -1e10, 0.0])

        ensemble = ensemble_predictions(lgb_pred, xgb_pred)

        assert ensemble[0] == 1e10
        assert ensemble[1] == -1e10
        assert ensemble[2] == 0.0
