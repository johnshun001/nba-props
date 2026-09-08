"""Tests for advanced backtest and analytics framework."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.backtest import (
    BacktestConfig, BacktestBet, BacktestResult, Backtester,
    run_parameter_sweep
)
from analysis.monte_carlo import (
    MonteCarloConfig, SimulationResult, MonteCarloSimulator,
    run_quick_simulation
)
from analysis.ab_testing import (
    ABTestConfig, ABTestResult, ABTester, StrategyComparator
)
from analysis.cohort_analysis import (
    CohortMetrics, CohortAnalyzer, MarketEfficiencyTracker
)


# =============================================================================
# Backtest Framework Tests
# =============================================================================

class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.initial_bankroll == 1000.0
        assert config.kelly_fraction == 0.5
        assert config.min_edge == 0.03
        assert config.max_bet_pct == 0.02

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_bankroll=5000.0,
            kelly_fraction=0.25,
            min_edge=0.05
        )

        assert config.initial_bankroll == 5000.0
        assert config.kelly_fraction == 0.25
        assert config.min_edge == 0.05


class TestBacktester:
    """Tests for Backtester class."""

    @pytest.fixture
    def sample_historical_data(self):
        """Generate sample historical data for testing."""
        np.random.seed(42)
        n_games = 100

        dates = [datetime(2023, 12, 1) + timedelta(days=i) for i in range(n_games)]

        data = {
            'date': dates,
            'player_id': [f'player_{i % 10}' for i in range(n_games)],
            'prop_type': ['points'] * n_games,
            'line': [25.5] * n_games,
            'model_projection': np.random.randn(n_games) * 3 + 27,
            'actual': np.random.randn(n_games) * 5 + 26,
        }

        return pd.DataFrame(data)

    def test_backtester_initialization(self):
        """Test Backtester initializes correctly."""
        backtester = Backtester()
        assert backtester is not None

    def test_run_backtest(self, sample_historical_data):
        """Test running a backtest."""
        config = BacktestConfig(initial_bankroll=1000.0)
        backtester = Backtester()

        result = backtester.run(sample_historical_data, config)

        assert isinstance(result, BacktestResult)
        assert result.total_bets > 0
        assert result.final_bankroll > 0

    def test_backtest_metrics(self, sample_historical_data):
        """Test backtest calculates all metrics."""
        config = BacktestConfig()
        backtester = Backtester()

        result = backtester.run(sample_historical_data, config)

        # Check all metrics are calculated
        assert result.win_rate >= 0 and result.win_rate <= 1
        assert result.roi is not None
        assert result.sharpe_ratio is not None
        assert result.max_drawdown >= 0
        assert result.calmar_ratio is not None

    def test_bankroll_trajectory(self, sample_historical_data):
        """Test bankroll trajectory is tracked."""
        config = BacktestConfig(initial_bankroll=1000.0)
        backtester = Backtester()

        result = backtester.run(sample_historical_data, config)

        assert len(result.bankroll_trajectory) > 0
        assert result.bankroll_trajectory[0] == config.initial_bankroll
        assert result.bankroll_trajectory[-1] == result.final_bankroll

    def test_edge_distribution(self, sample_historical_data):
        """Test edge distribution is tracked."""
        config = BacktestConfig()
        backtester = Backtester()

        result = backtester.run(sample_historical_data, config)

        assert '0-3%' in result.edge_distribution
        assert '3-5%' in result.edge_distribution
        assert '5-10%' in result.edge_distribution
        assert '10%+' in result.edge_distribution

    def test_date_range_filter(self, sample_historical_data):
        """Test date range filtering."""
        config = BacktestConfig(
            start_date=datetime(2023, 12, 10),
            end_date=datetime(2023, 12, 20)
        )
        backtester = Backtester()

        result = backtester.run(sample_historical_data, config)

        # Should have fewer bets with date filter
        assert result.total_bets < len(sample_historical_data)

    def test_generate_report(self, sample_historical_data):
        """Test report generation."""
        config = BacktestConfig()
        backtester = Backtester()

        result = backtester.run(sample_historical_data, config)
        report = backtester.generate_report(result)

        assert isinstance(report, str)
        assert 'BACKTEST REPORT' in report
        assert 'Sharpe Ratio' in report
        assert 'Max Drawdown' in report

    def test_result_to_dict(self, sample_historical_data):
        """Test result serialization."""
        config = BacktestConfig()
        backtester = Backtester()

        result = backtester.run(sample_historical_data, config)
        result_dict = result.to_dict()

        assert 'total_bets' in result_dict
        assert 'win_rate' in result_dict
        assert 'sharpe_ratio' in result_dict
        assert 'max_drawdown' in result_dict


class TestParameterSweep:
    """Tests for parameter sweep functionality."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        n_games = 200

        data = {
            'date': [datetime(2023, 12, 1) + timedelta(days=i) for i in range(n_games)],
            'player_id': [f'player_{i % 10}' for i in range(n_games)],
            'prop_type': ['points'] * n_games,
            'line': [25.5] * n_games,
            'model_projection': np.random.randn(n_games) * 3 + 27,
            'actual': np.random.randn(n_games) * 5 + 26,
        }

        return pd.DataFrame(data)

    def test_parameter_sweep(self, sample_data):
        """Test parameter sweep runs all combinations."""
        results = run_parameter_sweep(
            sample_data,
            kelly_fractions=[0.25, 0.5],
            min_edges=[0.03, 0.05]
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 4  # 2x2 combinations
        assert 'kelly_fraction' in results.columns
        assert 'sharpe_ratio' in results.columns


# =============================================================================
# Monte Carlo Simulation Tests
# =============================================================================

class TestMonteCarloConfig:
    """Tests for MonteCarloConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MonteCarloConfig()

        assert config.initial_bankroll == 1000.0
        assert config.num_simulations == 10000
        assert config.num_bets_per_season == 500


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    def test_simulator_initialization(self):
        """Test simulator initializes correctly."""
        sim = MonteCarloSimulator(seed=42)
        assert sim is not None

    def test_run_simulation(self):
        """Test running Monte Carlo simulation."""
        config = MonteCarloConfig(
            num_simulations=1000  # Fewer for faster tests
        )
        sim = MonteCarloSimulator(seed=42)

        result = sim.run(config)

        assert isinstance(result, SimulationResult)
        assert len(result.final_bankrolls) == config.num_simulations

    def test_ruin_probability(self):
        """Test ruin probability is calculated."""
        config = MonteCarloConfig(
            num_simulations=1000,
            win_rate=0.45,  # Losing strategy
            kelly_fraction=1.0,  # Aggressive
        )
        sim = MonteCarloSimulator(seed=42)

        result = sim.run(config)

        assert result.ruin_probability >= 0
        assert result.ruin_probability <= 1

    def test_median_bankroll(self):
        """Test median bankroll is calculated."""
        config = MonteCarloConfig(
            num_simulations=1000,
            win_rate=0.55,  # Winning strategy
        )
        sim = MonteCarloSimulator(seed=42)

        result = sim.run(config)

        assert result.median_bankroll > 0

    def test_confidence_intervals(self):
        """Test confidence intervals are calculated."""
        config = MonteCarloConfig(num_simulations=1000)
        sim = MonteCarloSimulator(seed=42)

        result = sim.run(config)

        assert result.percentile_5 <= result.percentile_25
        assert result.percentile_25 <= result.median_bankroll
        assert result.median_bankroll <= result.percentile_75
        assert result.percentile_75 <= result.percentile_95

    def test_result_to_dict(self):
        """Test result serialization."""
        config = MonteCarloConfig(num_simulations=100)
        sim = MonteCarloSimulator(seed=42)

        result = sim.run(config)
        result_dict = result.to_dict()

        assert 'ruin_probability' in result_dict
        assert 'median_bankroll' in result_dict
        assert 'interpretation' in result_dict

    def test_strategy_comparison(self):
        """Test Kelly fraction comparison."""
        base_config = MonteCarloConfig(num_simulations=500, kelly_fraction=0.5)
        sim = MonteCarloSimulator(seed=42)

        comparison = sim.run_strategy_comparison(
            base_config,
            kelly_fractions=[0.25, 0.5, 1.0]
        )

        assert 'strategies' in comparison
        assert 'optimal_kelly' in comparison

    def test_quick_simulation(self):
        """Test quick simulation helper function."""
        result = run_quick_simulation(
            initial_bankroll=1000.0,
            win_rate=0.55,
            num_simulations=100
        )

        assert 'ruin_probability' in result
        assert 'median_bankroll' in result


# =============================================================================
# A/B Testing Framework Tests
# =============================================================================

class TestABTestConfig:
    """Tests for ABTestConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ABTestConfig(
            name="test",
            variant_a_name="A",
            variant_b_name="B",
            metric="roi"
        )

        assert config.significance_level == 0.05
        assert config.power == 0.80


class TestABTester:
    """Tests for ABTester class."""

    def test_tester_initialization(self):
        """Test ABTester initializes correctly."""
        tester = ABTester()
        assert tester is not None

    def test_run_test_significant(self):
        """Test A/B test with significant difference."""
        np.random.seed(42)
        samples_a = np.random.randn(100) * 0.1 + 0.05  # Mean 5%
        samples_b = np.random.randn(100) * 0.1 + 0.15  # Mean 15%

        config = ABTestConfig(
            name="test",
            variant_a_name="A",
            variant_b_name="B",
            metric="roi"
        )
        tester = ABTester()

        result = tester.run_test(config, samples_a.tolist(), samples_b.tolist())

        assert result.is_significant == True
        assert result.p_value < 0.05
        assert result.effect_size > 0

    def test_run_test_not_significant(self):
        """Test A/B test with no significant difference."""
        np.random.seed(42)
        samples_a = np.random.randn(100) * 0.1 + 0.10
        samples_b = np.random.randn(100) * 0.1 + 0.10

        config = ABTestConfig(
            name="test",
            variant_a_name="A",
            variant_b_name="B",
            metric="roi"
        )
        tester = ABTester()

        result = tester.run_test(config, samples_a.tolist(), samples_b.tolist())

        assert result.is_significant == False
        assert result.p_value >= 0.05

    def test_calculate_sample_size(self):
        """Test sample size calculation."""
        tester = ABTester()

        n = tester.calculate_sample_size(
            minimum_effect=0.05,
            significance_level=0.05,
            power=0.80
        )

        assert n > 0
        assert n < 10000  # Reasonable upper bound

    def test_result_to_dict(self):
        """Test result serialization."""
        np.random.seed(42)
        samples_a = np.random.randn(50) * 0.1 + 0.10
        samples_b = np.random.randn(50) * 0.1 + 0.15

        config = ABTestConfig(
            name="test",
            variant_a_name="A",
            variant_b_name="B",
            metric="roi"
        )
        tester = ABTester()
        result = tester.run_test(config, samples_a.tolist(), samples_b.tolist())

        result_dict = result.to_dict()

        assert 't_statistic' in result_dict
        assert 'p_value' in result_dict
        assert 'effect_size' in result_dict
        assert 'recommendation' in result_dict


class TestStrategyComparator:
    """Tests for StrategyComparator class."""

    def test_rank_strategies(self):
        """Test strategy ranking."""
        np.random.seed(42)

        strategy_results = {
            'strategy_a': np.random.randn(100) * 0.1 + 0.10,
            'strategy_b': np.random.randn(100) * 0.1 + 0.15,
            'strategy_c': np.random.randn(100) * 0.1 + 0.05,
        }

        comparator = StrategyComparator()
        rankings = comparator.rank_strategies(strategy_results)

        assert len(rankings) == 3
        # Strategy B should be ranked first (highest mean)
        assert rankings[0][0] == 'strategy_b'


# =============================================================================
# Cohort Analysis Tests
# =============================================================================

class TestCohortAnalyzer:
    """Tests for CohortAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test CohortAnalyzer initializes correctly."""
        analyzer = CohortAnalyzer()
        assert analyzer is not None

    def test_add_bet(self):
        """Test adding bets to analyzer."""
        analyzer = CohortAnalyzer()

        bet = {
            'date': datetime.now(),
            'player_id': 'lebron_james',
            'prop_type': 'points',
            'bet_side': 'over',
            'bet_size': 10.0,
            'odds': -110,
            'edge': 0.05,
            'outcome': 'win',
            'pnl': 9.09,
            'sportsbook': 'draftkings',
            'player_type': 'star'
        }

        analyzer.add_bet(bet)

        assert len(analyzer.bets) == 1

    def test_add_bets_batch(self):
        """Test adding multiple bets."""
        analyzer = CohortAnalyzer()

        bets = [
            {
                'date': datetime.now(),
                'player_id': 'lebron_james',
                'prop_type': 'points',
                'bet_side': 'over',
                'bet_size': 10.0,
                'odds': -110,
                'edge': 0.05,
                'outcome': 'win',
                'pnl': 9.09,
                'sportsbook': 'draftkings',
                'player_type': 'star'
            }
            for _ in range(10)
        ]

        analyzer.add_bets_batch(bets)

        assert len(analyzer.bets) == 10

    def test_generate_report(self):
        """Test report generation."""
        analyzer = CohortAnalyzer(min_bets_for_significance=5)

        # Add bets for multiple cohorts
        for i in range(50):
            bet = {
                'date': datetime.now(),
                'player_id': f'player_{i % 5}',
                'prop_type': ['points', 'rebounds', 'assists'][i % 3],
                'bet_side': 'over',
                'bet_size': 10.0,
                'odds': -110,
                'edge': 0.05,
                'outcome': 'win' if i % 2 == 0 else 'loss',
                'pnl': 9.09 if i % 2 == 0 else -10.0,
                'sportsbook': ['draftkings', 'fanduel'][i % 2],
                'player_type': ['star', 'role_player'][i % 2]
            }
            analyzer.add_bet(bet)

        report = analyzer.generate_report()

        assert 'summary' in report
        assert 'by_cohort_type' in report
        assert 'top_performers' in report
        assert 'worst_performers' in report

    def test_get_cohort_performance(self):
        """Test getting specific cohort performance."""
        analyzer = CohortAnalyzer()

        for _ in range(10):
            bet = {
                'date': datetime.now(),
                'player_id': 'lebron_james',
                'prop_type': 'points',
                'bet_side': 'over',
                'bet_size': 10.0,
                'odds': -110,
                'edge': 0.05,
                'outcome': 'win',
                'pnl': 9.09,
                'sportsbook': 'draftkings',
                'player_type': 'star'
            }
            analyzer.add_bet(bet)

        metrics = analyzer.get_cohort_performance('player_type', 'star')

        assert metrics is not None
        assert metrics.total_bets == 10


class TestMarketEfficiencyTracker:
    """Tests for MarketEfficiencyTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        tracker = MarketEfficiencyTracker()
        assert tracker is not None

    def test_add_observation(self):
        """Test adding edge observations."""
        tracker = MarketEfficiencyTracker()

        tracker.add_observation(
            market_type='player_points',
            sportsbook='draftkings',
            edge=0.03
        )

        assert 'player_points' in tracker.market_edges
        assert 'draftkings' in tracker.sportsbook_edges

    def test_soft_market_alert(self):
        """Test soft market alert triggering."""
        tracker = MarketEfficiencyTracker()

        # Add observation with high edge (soft market)
        tracker.add_observation(
            market_type='player_points',
            sportsbook='draftkings',
            edge=0.08  # > 5% threshold
        )

        assert len(tracker.alerts) > 0
        assert 'soft market' in tracker.alerts[0]['message'].lower()

    def test_get_market_efficiency(self):
        """Test market efficiency metrics."""
        tracker = MarketEfficiencyTracker()

        for _ in range(20):
            tracker.add_observation(
                market_type='player_points',
                sportsbook='draftkings',
                edge=np.random.randn() * 0.02
            )

        efficiency = tracker.get_market_efficiency('player_points')

        assert efficiency['market_type'] == 'player_points'
        assert efficiency['observations'] == 20
        assert 'mean_edge' in efficiency
        assert 'std_edge' in efficiency

    def test_generate_report(self):
        """Test efficiency report generation."""
        tracker = MarketEfficiencyTracker()

        for market in ['player_points', 'player_rebounds']:
            for _ in range(10):
                tracker.add_observation(
                    market_type=market,
                    sportsbook='draftkings',
                    edge=np.random.randn() * 0.02
                )

        report = tracker.generate_report()

        assert 'markets' in report
        assert 'sportsbooks' in report
        assert 'alerts' in report


# =============================================================================
# Integration Tests
# =============================================================================

class TestAdvancedAnalyticsIntegration:
    """Integration tests for advanced analytics framework."""

    @pytest.fixture
    def sample_backtest_data(self):
        """Generate sample data for integration tests."""
        np.random.seed(42)
        n_games = 500

        dates = [datetime(2023, 12, 1) + timedelta(days=i) for i in range(n_games)]
        prop_types = [['points', 'rebounds', 'assists'][i % 3] for i in range(n_games)]

        data = {
            'date': dates,
            'player_id': [f'player_{i % 20}' for i in range(n_games)],
            'prop_type': prop_types,
            'line': [25.5] * n_games,
            'model_projection': np.random.randn(n_games) * 3 + 27,
            'actual': np.random.randn(n_games) * 5 + 26,
        }

        return pd.DataFrame(data)

    def test_full_backtest_to_cohort_pipeline(self, sample_backtest_data):
        """Test full pipeline: backtest -> cohort analysis."""
        # Run backtest
        config = BacktestConfig(initial_bankroll=1000.0)
        backtester = Backtester()
        result = backtester.run(sample_backtest_data, config)

        # Convert to cohort format
        analyzer = CohortAnalyzer(min_bets_for_significance=10)

        for bet in result.bets:
            cohort_bet = {
                'date': bet.date,
                'player_id': bet.player_id,
                'prop_type': bet.prop_type,
                'bet_side': bet.bet_side,
                'bet_size': bet.bet_size,
                'odds': bet.odds,
                'edge': bet.edge,
                'outcome': 'win' if bet.actual_outcome == 1 else ('loss' if bet.actual_outcome == 0 else 'push'),
                'pnl': bet.pnl,
                'sportsbook': 'draftkings',
                'player_type': 'star'
            }
            analyzer.add_bet(cohort_bet)

        report = analyzer.generate_report()

        assert report['summary']['total_bets'] == len(result.bets)

    def test_monte_carlo_informs_backtest(self, sample_backtest_data):
        """Test Monte Carlo simulation informs backtest expectations."""
        # Run backtest
        config = BacktestConfig()
        backtester = Backtester()
        result = backtester.run(sample_backtest_data, config)

        # Run Monte Carlo with similar parameters
        mc_config = MonteCarloConfig(
            initial_bankroll=config.initial_bankroll,
            win_rate=result.win_rate,
            num_simulations=1000,
            num_bets_per_season=result.total_bets
        )
        sim = MonteCarloSimulator(seed=42)
        mc_result = sim.run(mc_config)

        # Compare results
        assert mc_result.median_bankroll > 0
        assert result.final_bankroll > 0
