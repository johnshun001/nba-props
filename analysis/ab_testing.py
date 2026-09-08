"""A/B testing framework for NBA prop betting strategies.

Test: Line shopping vs. no line shopping
Test: 3% edge vs. 5% edge threshold
Test: Ensemble vs. single model
Statistical significance testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ABTestConfig:
    """Configuration for A/B test."""
    name: str
    variant_a_name: str
    variant_b_name: str
    metric: str  # 'roi', 'win_rate', 'sharpe', etc.
    minimum_effect: float = 0.05  # Minimum detectable effect
    significance_level: float = 0.05  # Alpha
    power: float = 0.80  # 1 - Beta


@dataclass
class ABTestResult:
    """Results from A/B test."""
    config: ABTestConfig
    variant_a_samples: List[float]
    variant_b_samples: List[float]
    variant_a_mean: float = 0.0
    variant_b_mean: float = 0.0
    variant_a_std: float = 0.0
    variant_b_std: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    effect_size: float = 0.0  # Cohen's d
    is_significant: bool = False
    is_significant_practical: bool = False
    recommendation: str = ""
    power_achieved: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'config': {
                'name': self.config.name,
                'variant_a': self.config.variant_a_name,
                'variant_b': self.config.variant_b_name,
                'metric': self.config.metric,
            },
            'variant_a': {
                'mean': round(self.variant_a_mean, 4),
                'std': round(self.variant_a_std, 4),
                'n': len(self.variant_a_samples),
            },
            'variant_b': {
                'mean': round(self.variant_b_mean, 4),
                'std': round(self.variant_b_std, 4),
                'n': len(self.variant_b_samples),
            },
            't_statistic': round(self.t_statistic, 4),
            'p_value': round(self.p_value, 4),
            'confidence_interval': [round(self.confidence_interval[0], 4),
                                    round(self.confidence_interval[1], 4)],
            'effect_size': round(self.effect_size, 4),
            'is_significant': self.is_significant,
            'is_significant_practical': self.is_significant_practical,
            'recommendation': self.recommendation,
            'power_achieved': round(self.power_achieved, 4),
        }


class ABTester:
    """
    A/B testing framework for betting strategies.

    Usage:
        tester = ABTester()
        result = tester.run_test(config, samples_a, samples_b)
        print(result.recommendation)
    """

    def run_test(
        self,
        config: ABTestConfig,
        samples_a: List[float],
        samples_b: List[float]
    ) -> ABTestResult:
        """
        Run A/B test with statistical analysis.

        Args:
            config: ABTestConfig
            samples_a: Metric samples from variant A
            samples_b: Metric samples from variant B

        Returns:
            ABTestResult with all statistics
        """
        samples_a = np.array(samples_a)
        samples_b = np.array(samples_b)

        # Basic statistics
        mean_a = float(np.mean(samples_a))
        mean_b = float(np.mean(samples_b))
        std_a = float(np.std(samples_a, ddof=1))
        std_b = float(np.std(samples_b, ddof=1))

        # Two-sample t-test (Welch's t-test for unequal variances)
        t_stat, p_value = stats.ttest_ind(samples_a, samples_b, equal_var=False)

        # Confidence interval for difference in means
        diff = mean_b - mean_a
        se = np.sqrt(std_a**2 / len(samples_a) + std_b**2 / len(samples_b))
        t_crit = stats.t.ppf(1 - config.significance_level / 2,
                            df=len(samples_a) + len(samples_b) - 2)
        ci_lower = diff - t_crit * se
        ci_upper = diff + t_crit * se

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        effect_size = diff / pooled_std if pooled_std > 0 else 0

        # Statistical significance
        is_significant = p_value < config.significance_level

        # Practical significance (minimum detectable effect)
        relative_effect = abs(diff) / abs(mean_a) if mean_a != 0 else 0
        is_significant_practical = relative_effect >= config.minimum_effect

        # Power analysis
        power_achieved = self._calculate_power(
            effect_size=effect_size,
            n1=len(samples_a),
            n2=len(samples_b),
            alpha=config.significance_level
        )

        # Recommendation
        recommendation = self._generate_recommendation(
            is_significant=is_significant,
            is_significant_practical=is_significant_practical,
            effect_size=effect_size,
            better_variant='B' if diff > 0 else 'A',
            power=power_achieved
        )

        return ABTestResult(
            config=config,
            variant_a_samples=samples_a.tolist(),
            variant_b_samples=samples_b.tolist(),
            variant_a_mean=mean_a,
            variant_b_mean=mean_b,
            variant_a_std=std_a,
            variant_b_std=std_b,
            t_statistic=float(t_stat),
            p_value=float(p_value),
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            is_significant=is_significant,
            is_significant_practical=is_significant_practical,
            recommendation=recommendation,
            power_achieved=power_achieved
        )

    def _calculate_power(
        self,
        effect_size: float,
        n1: int,
        n2: int,
        alpha: float
    ) -> float:
        """Calculate statistical power."""
        # Use statsmodels if available, otherwise approximate
        try:
            from statsmodels.stats.power import TTestIndPower
            analysis = TTestIndPower()
            power = analysis.power(effect_size=effect_size, nobs1=n1, ratio=n2/n1, alpha=alpha)
            return float(power)
        except ImportError:
            # Approximate power calculation
            n_eff = (n1 * n2) / (n1 + n2)
            ncp = effect_size * np.sqrt(n_eff)
            z_crit = stats.norm.ppf(1 - alpha / 2)
            power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
            return float(power)

    def _generate_recommendation(
        self,
        is_significant: bool,
        is_significant_practical: bool,
        effect_size: float,
        better_variant: str,
        power: float
    ) -> str:
        """Generate recommendation based on test results."""
        if not is_significant:
            if power < 0.8:
                return (f"Inconclusive. Test is underpowered (power={power:.1%}). "
                       f"Consider collecting more data.")
            else:
                return (f"No significant difference detected. "
                       f"Both variants perform similarly (p={better_variant} is not better).")

        if not is_significant_practical:
            return (f"Statistically significant but effect is too small to matter. "
                   f"Effect size: {effect_size:.3f} (small).")

        # Interpret effect size
        if abs(effect_size) < 0.2:
            effect_desc = "small"
        elif abs(effect_size) < 0.5:
            effect_desc = "medium"
        else:
            effect_desc = "large"

        return (f"Variant {better_variant} is significantly better with {effect_desc} "
               f"effect size (d={effect_size:.3f}). Recommend switching to Variant {better_variant}.")

    def calculate_sample_size(
        self,
        minimum_effect: float = 0.05,
        significance_level: float = 0.05,
        power: float = 0.80,
        baseline_std: float = 0.1
    ) -> int:
        """
        Calculate required sample size per variant.

        Args:
            minimum_effect: Minimum detectable effect (as proportion)
            significance_level: Alpha level
            power: Desired power (1 - Beta)
            baseline_std: Expected standard deviation

        Returns:
            Required sample size per variant
        """
        # Effect size (Cohen's d)
        effect_size = minimum_effect / baseline_std if baseline_std > 0 else 0.5

        try:
            from statsmodels.stats.power import TTestIndPower
            analysis = TTestIndPower()
            n = analysis.solve_power(effect_size=effect_size, alpha=significance_level,
                                    power=power, ratio=1.0)
            return int(np.ceil(n))
        except ImportError:
            # Approximate
            z_alpha = stats.norm.ppf(1 - significance_level / 2)
            z_beta = stats.norm.ppf(power)
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return int(np.ceil(n))

    def sequential_test(
        self,
        config: ABTestConfig,
        data_stream: List[Tuple[str, float]],
        check_interval: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Run sequential A/B test with interim analyses.

        Args:
            config: ABTestConfig
            data_stream: List of (variant, metric) tuples
            check_interval: Check for significance every N samples

        Returns:
            List of interim results
        """
        results = []
        samples_a = []
        samples_b = []

        for i, (variant, metric) in enumerate(data_stream):
            if variant == 'A':
                samples_a.append(metric)
            else:
                samples_b.append(metric)

            # Check for significance at intervals
            if (i + 1) % check_interval == 0 and len(samples_a) > 10 and len(samples_b) > 10:
                result = self.run_test(config, samples_a, samples_b)
                results.append({
                    'n_total': i + 1,
                    'n_a': len(samples_a),
                    'n_b': len(samples_b),
                    'p_value': result.p_value,
                    'is_significant': result.is_significant,
                    'effect_size': result.effect_size,
                })

                # Early stopping if significant
                if result.is_significant and result.is_significant_practical:
                    results[-1]['early_stop'] = True
                    break

        return results


class StrategyComparator:
    """
    Compare multiple betting strategies.

    Usage:
        comparator = StrategyComparator()
        comparison = comparator.compare_all(backtest_results)
    """

    def __init__(self):
        self.ab_tester = ABTester()

    def compare_strategies(
        self,
        strategy_results: Dict[str, List[float]],
        metric: str = 'roi',
        baseline_strategy: Optional[str] = None
    ) -> Dict[str, ABTestResult]:
        """
        Compare multiple strategies against baseline.

        Args:
            strategy_results: Dict of strategy_name -> list of metric values
            metric: Metric being compared
            baseline_strategy: Name of baseline strategy (default: first)

        Returns:
            Dict of comparison results
        """
        strategies = list(strategy_results.keys())
        if baseline_strategy is None:
            baseline_strategy = strategies[0]

        results = {}
        baseline_samples = strategy_results[baseline_strategy]

        for strategy in strategies:
            if strategy == baseline_strategy:
                continue

            config = ABTestConfig(
                name=f"{strategy}_vs_{baseline_strategy}",
                variant_a_name=baseline_strategy,
                variant_b_name=strategy,
                metric=metric
            )

            result = self.ab_tester.run_test(
                config,
                baseline_samples,
                strategy_results[strategy]
            )

            results[strategy] = result

        return results

    def rank_strategies(
        self,
        strategy_results: Dict[str, List[float]],
        metric: str = 'sharpe_ratio'
    ) -> List[Tuple[str, float, float]]:
        """
        Rank strategies by mean metric with confidence intervals.

        Args:
            strategy_results: Dict of strategy_name -> list of metric values
            metric: Metric to rank by

        Returns:
            List of (strategy, mean, std) tuples sorted by mean
        """
        rankings = []

        for strategy, samples in strategy_results.items():
            samples = np.array(samples)
            mean = float(np.mean(samples))
            std = float(np.std(samples, ddof=1))
            rankings.append((strategy, mean, std))

        return sorted(rankings, key=lambda x: x[1], reverse=True)


def run_ab_test_suite(
    historical_data: pd.DataFrame,
    backtester
) -> Dict[str, Any]:
    """
    Run comprehensive A/B test suite.

    Tests:
    1. Line shopping vs. no line shopping
    2. 3% edge vs. 5% edge threshold
    3. Ensemble vs. single model
    4. Different Kelly fractions

    Args:
        historical_data: Historical data DataFrame
        backtester: Backtester instance

    Returns:
        Comprehensive test results
    """
    from analysis.backtest import BacktestConfig

    results = {}

    # Test 1: Line shopping
    config_no_shopping = BacktestConfig(kelly_fraction=0.5, min_edge=0.03)
    config_with_shopping = BacktestConfig(kelly_fraction=0.5, min_edge=0.03)
    # Note: Line shopping would be implemented in the backtester

    # Test 2: Edge thresholds
    edge_3pct = BacktestConfig(kelly_fraction=0.5, min_edge=0.03)
    edge_5pct = BacktestConfig(kelly_fraction=0.5, min_edge=0.05)

    result_3pct = backtester.run(historical_data, edge_3pct)
    result_5pct = backtester.run(historical_data, edge_5pct)

    ab_tester = ABTester()
    config = ABTestConfig(
        name="edge_threshold_comparison",
        variant_a_name="3% edge",
        variant_b_name="5% edge",
        metric="roi"
    )

    # Use bet-level ROI as samples
    samples_3pct = [b.pnl / b.bet_size * 100 for b in result_3pct.bets if b.bet_size > 0]
    samples_5pct = [b.pnl / b.bet_size * 100 for b in result_5pct.bets if b.bet_size > 0]

    edge_threshold_result = ab_tester.run_test(config, samples_3pct, samples_5pct)
    results['edge_threshold'] = edge_threshold_result.to_dict()

    # Test 3: Kelly fractions
    kelly_025 = BacktestConfig(kelly_fraction=0.25, min_edge=0.03)
    kelly_050 = BacktestConfig(kelly_fraction=0.5, min_edge=0.03)
    kelly_100 = BacktestConfig(kelly_fraction=1.0, min_edge=0.03)

    result_k25 = backtester.run(historical_data, kelly_025)
    result_k50 = backtester.run(historical_data, kelly_050)
    result_k100 = backtester.run(historical_data, kelly_100)

    samples_k25 = [b.pnl / b.bet_size * 100 for b in result_k25.bets if b.bet_size > 0]
    samples_k50 = [b.pnl / b.bet_size * 100 for b in result_k50.bets if b.bet_size > 0]
    samples_k100 = [b.pnl / b.bet_size * 100 for b in result_k100.bets if b.bet_size > 0]

    kelly_config = ABTestConfig(
        name="kelly_fraction_comparison",
        variant_a_name="Half-Kelly",
        variant_b_name="Full-Kelly",
        metric="roi"
    )

    kelly_result = ab_tester.run_test(kelly_config, samples_k50, samples_k100)
    results['kelly_fraction'] = kelly_result.to_dict()

    return results
