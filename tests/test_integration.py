"""Integration test wiring all terminals together.

Terminal 1 (data) → Terminal 2 (model) → Terminal 3 (bets)

Run on historical data (last week of 2023 playoffs games).
Verify: bets generated, P&L tracked, attribution working.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas.validation import PropLine, Bet, RevenueAttribution, validate_bet
from storage.bet_tracker import (
    init_db as init_bets_db, log_bet, get_all_bets,
    get_betting_summary, update_bet_outcome, clear_all_bets
)
from analysis.revenue_attribution import (
    init_db as init_attr_db, log_attribution,
    get_aggregate_attribution, AttributionCalculator
)
from analysis.backtest import BacktestConfig, Backtester
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split


def train_simple_ensemble_models(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Train simple ensemble models for testing."""
    alphas = [0.1, 0.5, 0.9]

    X = df[feature_cols].values
    y = df[target_col].values

    # Create leverage weights
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
            n_estimators=50,
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


# Test data simulating 2023 playoffs last week
def generate_playoffs_test_data(n_games: int = 50) -> pd.DataFrame:
    """Generate synthetic data simulating 2023 playoffs games."""
    np.random.seed(42)

    # Playoffs have higher intensity, more clutch situations
    data = {
        'pts_roll5': np.random.randn(n_games) * 4 + 24,  # Higher scoring
        'reb_roll5': np.random.randn(n_games) * 3 + 9,
        'ast_roll5': np.random.randn(n_games) * 2 + 6,
        'career_ppg': np.random.randn(n_games) * 3 + 22,  # Star players
        'strength_ratio': np.random.rand(n_games) * 0.3 + 0.9,  # Closer matchups
        'is_bench': np.random.randint(0, 2, n_games),
        'quarter': np.random.choice([1, 2, 3, 4], n_games, p=[0.2, 0.2, 0.25, 0.35]),
        'point_diff': np.random.randn(n_games) * 8 + 3,  # Closer games
        'minutes_remaining': np.random.rand(n_games) * 48,
        'pts_target': np.random.randn(n_games) * 5 + 26,
        'reb_target': np.random.randn(n_games) * 3 + 10,
        'ast_target': np.random.randn(n_games) * 2 + 7,
    }

    return pd.DataFrame(data)


def simulate_terminal_1_data() -> pd.DataFrame:
    """Terminal 1: Data layer - load/generate historical data."""
    return generate_playoffs_test_data(n_games=100)


def simulate_terminal_2_model(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Terminal 2: Model layer - train models and generate predictions."""
    # Train models for points
    lgb_pts, xgb_pts, cal_pts = train_simple_ensemble_models(
        df, feature_cols, 'pts_target'
    )

    # Generate predictions for sample data
    X = df[feature_cols].values[:10]
    predictions = simple_predict(lgb_pts, xgb_pts, cal_pts, X)

    return {
        'lgb_models': lgb_pts,
        'xgb_models': xgb_pts,
        'calibrators': cal_pts,
        'predictions': predictions,
        'feature_cols': feature_cols
    }


def simulate_terminal_3_bets(
    predictions: Dict,
    market_lines: List[Dict],
    bankroll: float = 100.0
) -> List[Dict]:
    """Terminal 3: Betting layer - generate bets from model predictions."""
    bets = []

    # Kelly sizing parameters
    kelly_fraction = 0.5
    max_bet_pct = 0.02

    for i, (alpha, pred_values) in enumerate(predictions['predictions'].items()):
        for j, pred_median in enumerate(pred_values[:5]):  # First 5 predictions
            # Get market line
            market_line = market_lines[j % len(market_lines)]

            # Calculate edge
            model_projection = pred_median
            market_projection = market_line['line']
            edge = (model_projection - market_projection) / market_projection

            # Only bet with positive edge
            if edge <= 0.02:  # Minimum 2% edge threshold
                continue

            # Kelly bet size
            if edge > 0:
                b = (market_line['odds'] / 100) if market_line['odds'] > 0 else (100 / abs(market_line['odds']))
                p = 0.5 + edge
                q = 1 - p
                kelly_pct = max(0, (b * p - q) / b) * kelly_fraction
                kelly_pct = min(kelly_pct, max_bet_pct)
                bet_size = round(bankroll * kelly_pct, 2)

                if bet_size >= 1.0:  # Minimum bet
                    bet_side = 'over' if model_projection > market_projection else 'under'

                    bet = {
                        'player_id': f'player_{j}',
                        'prop_type': 'points',
                        'line': market_line['line'],
                        'bet_size': bet_size,
                        'edge': round(edge, 4),
                        'bet_side': bet_side,
                        'sportsbook': market_line['sportsbook'],
                        'odds': market_line['odds'],
                        'timestamp': datetime.utcnow()
                    }
                    bets.append(bet)

    return bets


def simulate_bet_settlement(bets: List[Dict]) -> List[Dict]:
    """Simulate bet settlement with realistic outcomes."""
    results = []
    np.random.seed(42)

    for bet in bets:
        # Simulate outcome based on edge (higher edge = higher win probability)
        win_prob = 0.5 + bet['edge']
        outcome_roll = np.random.random()

        if outcome_roll < win_prob:
            outcome = 'win'
            if bet['odds'] > 0:
                pnl = bet['bet_size'] * (bet['odds'] / 100)
            else:
                pnl = bet['bet_size'] * (100 / abs(bet['odds']))
        else:
            outcome = 'loss'
            pnl = -bet['bet_size']

        results.append({
            'bet': bet,
            'outcome': outcome,
            'pnl': round(pnl, 2)
        })

    return results


def calculate_attribution_for_bets(bets: List[Dict], results: List[Dict]) -> List[RevenueAttribution]:
    """Calculate revenue attribution for all bets."""
    attributions = []
    calc = AttributionCalculator()

    for i, result in enumerate(results):
        bet = result['bet']

        # Simulate market lines for line shopping
        market_lines = [
            {'line': bet['line'] - 0.5},
            {'line': bet['line']},
            {'line': bet['line'] + 0.5}
        ]

        attribution = calc.calculate(
            bet_id=f"bet_{i:04d}",
            selected_line=bet['line'],
            market_lines=market_lines,
            kelly_bet=bet['bet_size'],
            flat_bet=5.0,  # Baseline flat bet
            model_projection=bet['line'] * (1 + bet['edge']),
            market_projection=bet['line'],
            matchup_adjustment=0.5
        )
        attributions.append(attribution)

    return attributions


class TestTerminalIntegration:
    """Integration tests for full terminal pipeline."""

    @pytest.fixture
    def temp_bets_db(self):
        """Create temporary bets database."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        init_bets_db(path)
        yield path
        os.unlink(path)

    @pytest.fixture
    def temp_attr_db(self):
        """Create temporary attribution database."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        init_attr_db(path)
        yield path
        os.unlink(path)

    def test_terminal_1_data_generation(self):
        """Terminal 1: Generate/load historical data."""
        df = simulate_terminal_1_data()

        assert len(df) == 100
        assert 'pts_roll5' in df.columns
        assert 'reb_roll5' in df.columns
        assert 'ast_roll5' in df.columns
        assert 'pts_target' in df.columns

    def test_terminal_2_model_training(self):
        """Terminal 2: Train models and generate predictions."""
        df = simulate_terminal_1_data()
        feature_cols = ['pts_roll5', 'reb_roll5', 'ast_roll5', 'career_ppg', 'strength_ratio', 'is_bench']

        model_output = simulate_terminal_2_model(df, feature_cols)

        assert 'lgb_models' in model_output
        assert 'xgb_models' in model_output
        assert 'calibrators' in model_output
        assert 'predictions' in model_output

        # Check predictions
        predictions = model_output['predictions']
        assert 0.1 in predictions
        assert 0.5 in predictions
        assert 0.9 in predictions

    def test_terminal_3_bet_generation(self):
        """Terminal 3: Generate bets from model predictions."""
        df = simulate_terminal_1_data()
        feature_cols = ['pts_roll5', 'reb_roll5', 'ast_roll5', 'career_ppg', 'strength_ratio', 'is_bench']

        model_output = simulate_terminal_2_model(df, feature_cols)

        market_lines = [
            {'line': 24.5, 'odds': -110, 'sportsbook': 'draftkings'},
            {'line': 25.0, 'odds': -115, 'sportsbook': 'fanduel'},
            {'line': 25.5, 'odds': -105, 'sportsbook': 'betmgm'},
        ]

        bets = simulate_terminal_3_bets(model_output, market_lines, bankroll=100.0)

        # Should generate some bets with positive edge
        assert len(bets) > 0

        # Validate each bet
        for bet in bets:
            validation = validate_bet(bet)
            assert validation.valid is True, f"Bet validation failed: {validation.errors}"

    def test_full_pipeline_bets_generated(self, temp_bets_db):
        """Full pipeline: Verify bets are generated and logged."""
        # Terminal 1: Data
        df = simulate_terminal_1_data()
        feature_cols = ['pts_roll5', 'reb_roll5', 'ast_roll5', 'career_ppg', 'strength_ratio', 'is_bench']

        # Terminal 2: Model
        model_output = simulate_terminal_2_model(df, feature_cols)

        # Terminal 3: Bets
        market_lines = [
            {'line': 24.5, 'odds': -110, 'sportsbook': 'draftkings'},
            {'line': 25.0, 'odds': -115, 'sportsbook': 'fanduel'},
            {'line': 25.5, 'odds': -105, 'sportsbook': 'betmgm'},
        ]

        bets = simulate_terminal_3_bets(model_output, market_lines, bankroll=100.0)

        # Log bets to database
        for bet_data in bets:
            bet = Bet(**bet_data)
            log_bet(bet, temp_bets_db)

        # Verify bets logged
        all_bets = get_all_bets(temp_bets_db)
        assert len(all_bets) == len(bets)

    def test_full_pipeline_pnl_tracked(self, temp_bets_db):
        """Full pipeline: Verify P&L is tracked correctly."""
        # Generate and log bets
        df = simulate_terminal_1_data()
        feature_cols = ['pts_roll5', 'reb_roll5', 'ast_roll5', 'career_ppg', 'strength_ratio', 'is_bench']
        model_output = simulate_terminal_2_model(df, feature_cols)

        market_lines = [
            {'line': 24.5, 'odds': -110, 'sportsbook': 'draftkings'},
        ]

        bets = simulate_terminal_3_bets(model_output, market_lines, bankroll=100.0)

        # Log and settle bets
        bet_ids = []
        for bet_data in bets[:10]:  # First 10 bets
            bet = Bet(**bet_data)
            bet_id = log_bet(bet, temp_bets_db)
            bet_ids.append((bet_id, bet_data))

        # Settle bets
        for bet_id, bet_data in bet_ids:
            # Simulate outcome
            win_prob = 0.5 + bet_data['edge']
            outcome = 'win' if np.random.random() < win_prob else 'loss'
            pnl = bet_data['bet_size'] * 0.91 if outcome == 'win' else -bet_data['bet_size']
            update_bet_outcome(bet_id, outcome, pnl, temp_bets_db)

        # Verify P&L tracking
        summary = get_betting_summary(temp_bets_db)
        assert summary['total_bets'] == len(bet_ids)
        assert 'total_pnl' in summary

    def test_full_pipeline_attribution_working(self, temp_attr_db):
        """Full pipeline: Verify attribution is working."""
        # Generate bets and results
        df = simulate_terminal_1_data()
        feature_cols = ['pts_roll5', 'reb_roll5', 'ast_roll5', 'career_ppg', 'strength_ratio', 'is_bench']
        model_output = simulate_terminal_2_model(df, feature_cols)

        market_lines = [
            {'line': 24.5, 'odds': -110, 'sportsbook': 'draftkings'},
        ]

        bets = simulate_terminal_3_bets(model_output, market_lines, bankroll=100.0)
        results = simulate_bet_settlement(bets[:50])  # 50 bets

        # Calculate and log attribution
        attributions = calculate_attribution_for_bets(bets, results)

        for attr in attributions:
            log_attribution(attr, temp_attr_db)

        # Verify attribution logged
        aggregate = get_aggregate_attribution(min_bets=10, db_path=temp_attr_db)

        assert aggregate is not None
        assert aggregate['total_bets'] >= 10
        assert 'total_line_shopping' in aggregate
        assert 'total_model_edge' in aggregate

    def test_end_to_end_integration(self, temp_bets_db, temp_attr_db):
        """End-to-end test: Full pipeline from data to attribution."""
        # ===== Terminal 1: Data =====
        print("\n=== Terminal 1: Data Layer ===")
        df = simulate_terminal_1_data()
        print(f"Generated {len(df)} games of historical data")

        feature_cols = ['pts_roll5', 'reb_roll5', 'ast_roll5', 'career_ppg', 'strength_ratio', 'is_bench']

        # ===== Terminal 2: Model =====
        print("\n=== Terminal 2: Model Layer ===")
        model_output = simulate_terminal_2_model(df, feature_cols)
        print(f"Trained LGB, XGB, and calibrators for quantiles: 0.1, 0.5, 0.9")

        # ===== Terminal 3: Bets =====
        print("\n=== Terminal 3: Betting Layer ===")
        market_lines = [
            {'line': 24.5, 'odds': -110, 'sportsbook': 'draftkings'},
            {'line': 25.0, 'odds': -115, 'sportsbook': 'fanduel'},
            {'line': 25.5, 'odds': -105, 'sportsbook': 'betmgm'},
        ]

        bets = simulate_terminal_3_bets(model_output, market_lines, bankroll=100.0)
        print(f"Generated {len(bets)} bets with positive edge")

        # Log bets
        bet_results = []
        for i, bet_data in enumerate(bets):
            bet = Bet(**bet_data)
            bet_id = log_bet(bet, temp_bets_db)

            # Simulate settlement
            win_prob = 0.5 + bet_data['edge']
            outcome = 'win' if np.random.random() < win_prob else 'loss'
            pnl = bet_data['bet_size'] * 0.91 if outcome == 'win' else -bet_data['bet_size']
            update_bet_outcome(bet_id, outcome, pnl, temp_bets_db)

            bet_results.append({
                'bet_id': bet_id,
                'bet': bet_data,
                'outcome': outcome,
                'pnl': pnl
            })

        # ===== Revenue Attribution =====
        print("\n=== Revenue Attribution ===")
        attributions = calculate_attribution_for_bets(
            [r['bet'] for r in bet_results],
            bet_results
        )

        for attr in attributions:
            log_attribution(attr, temp_attr_db)

        print(f"Logged {len(attributions)} attribution records")

        # ===== Verification =====
        print("\n=== Verification ===")

        # Check bets database
        bets_summary = get_betting_summary(temp_bets_db)
        print(f"Bets Summary:")
        print(f"  Total bets: {bets_summary['total_bets']}")
        print(f"  Wins: {bets_summary['wins']}")
        print(f"  Losses: {bets_summary['losses']}")
        print(f"  Win rate: {bets_summary['win_rate']:.2%}")
        print(f"  Total P&L: ${bets_summary['total_pnl']:.2f}")
        print(f"  ROI: {bets_summary['roi']:.2f}%")

        # Check attribution database
        attr_summary = get_aggregate_attribution(min_bets=5, db_path=temp_attr_db)
        if attr_summary:
            print(f"\nAttribution Summary:")
            print(f"  Total bets: {attr_summary['total_bets']}")
            print(f"  Line shopping value: {attr_summary['total_line_shopping']:.3f}")
            print(f"  Model edge: {attr_summary['total_model_edge']:.3f}")
            print(f"  Kelly edge: {attr_summary['total_kelly_edge']:.3f}")
            print(f"  Matchup edge: {attr_summary['total_matchup_edge']:.3f}")

        # Assertions
        assert bets_summary['total_bets'] > 0, "No bets generated"
        assert attr_summary is not None, "Attribution not working"

        print("\n=== All Integration Tests Passed ===")
