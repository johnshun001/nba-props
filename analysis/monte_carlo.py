"""Monte Carlo simulation for NBA prop betting.

Simulate 10,000 seasons with current edge.
Output: ruin probability, median bankroll, confidence intervals.
Answer: "What's my risk of going broke?"
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    initial_bankroll: float = 1000.0
    num_simulations: int = 10000
    num_bets_per_season: int = 500
    win_rate: float = 0.55  # Expected win rate
    avg_odds: float = -110  # Average odds
    kelly_fraction: float = 0.5
    max_bet_pct: float = 0.02
    min_bet: float = 1.0
    ruin_threshold: float = 100.0  # Bankroll level considered "ruin"


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    config: MonteCarloConfig
    final_bankrolls: List[float]
    ruin_count: int = 0
    ruin_probability: float = 0.0
    median_bankroll: float = 0.0
    mean_bankroll: float = 0.0
    std_bankroll: float = 0.0
    percentile_5: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0
    best_case: float = 0.0
    worst_case: float = 0.0
    max_drawdown_avg: float = 0.0
    max_drawdown_worst: float = 0.0
    seasons_profitable: int = 0
    profit_probability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'config': {
                'initial_bankroll': self.config.initial_bankroll,
                'num_simulations': self.config.num_simulations,
                'num_bets_per_season': self.config.num_bets_per_season,
                'win_rate': self.config.win_rate,
                'kelly_fraction': self.config.kelly_fraction,
            },
            'ruin_probability': round(self.ruin_probability, 4),
            'median_bankroll': round(self.median_bankroll, 2),
            'mean_bankroll': round(self.mean_bankroll, 2),
            'std_bankroll': round(self.std_bankroll, 2),
            'percentile_5': round(self.percentile_5, 2),
            'percentile_25': round(self.percentile_25, 2),
            'percentile_75': round(self.percentile_75, 2),
            'percentile_95': round(self.percentile_95, 2),
            'best_case': round(self.best_case, 2),
            'worst_case': round(self.worst_case, 2),
            'max_drawdown_avg': round(self.max_drawdown_avg, 2),
            'max_drawdown_worst': round(self.max_drawdown_worst, 2),
            'profit_probability': round(self.profit_probability, 4),
            'interpretation': self._interpret()
        }

    def _interpret(self) -> str:
        """Generate interpretation of results."""
        if self.ruin_probability > 0.5:
            risk_level = "EXTREME"
        elif self.ruin_probability > 0.2:
            risk_level = "HIGH"
        elif self.ruin_probability > 0.1:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return (
            f"Risk of ruin: {risk_level} ({self.ruin_probability:.1%}). "
            f"Median bankroll after {self.config.num_bets_per_season} bets: "
            f"${self.median_bankroll:.2f} ({(self.median_bankroll/self.config.initial_bankroll - 1)*100:+.1f}%). "
            f"{self.profit_probability:.1%} chance of being profitable."
        )


class MonteCarloSimulator:
    """
    Monte Carlo simulation for betting strategy analysis.

    Usage:
        sim = MonteCarloSimulator()
        result = sim.run(config)
        print(result.to_dict())
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

    def run(self, config: MonteCarloConfig) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            config: MonteCarloConfig object

        Returns:
            SimulationResult with all metrics
        """
        final_bankrolls = []
        max_drawdowns = []

        for _ in range(config.num_simulations):
            bankroll, max_dd = self._simulate_season(config)
            final_bankrolls.append(bankroll)
            max_drawdowns.append(max_dd)

        final_bankrolls = np.array(final_bankrolls)
        max_drawdowns = np.array(max_drawdowns)

        # Calculate metrics
        ruin_count = int(np.sum(final_bankrolls < config.ruin_threshold))
        ruin_probability = ruin_count / config.num_simulations

        median_bankroll = float(np.median(final_bankrolls))
        mean_bankroll = float(np.mean(final_bankrolls))
        std_bankroll = float(np.std(final_bankrolls))

        percentile_5 = float(np.percentile(final_bankrolls, 5))
        percentile_25 = float(np.percentile(final_bankrolls, 25))
        percentile_75 = float(np.percentile(final_bankrolls, 75))
        percentile_95 = float(np.percentile(final_bankrolls, 95))

        best_case = float(np.max(final_bankrolls))
        worst_case = float(np.min(final_bankrolls))

        max_drawdown_avg = float(np.mean(max_drawdowns))
        max_drawdown_worst = float(np.max(max_drawdowns))

        seasons_profitable = int(np.sum(final_bankrolls > config.initial_bankroll))
        profit_probability = seasons_profitable / config.num_simulations

        return SimulationResult(
            config=config,
            final_bankrolls=final_bankrolls.tolist(),
            ruin_count=ruin_count,
            ruin_probability=ruin_probability,
            median_bankroll=median_bankroll,
            mean_bankroll=mean_bankroll,
            std_bankroll=std_bankroll,
            percentile_5=percentile_5,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            percentile_95=percentile_95,
            best_case=best_case,
            worst_case=worst_case,
            max_drawdown_avg=max_drawdown_avg,
            max_drawdown_worst=max_drawdown_worst,
            seasons_profitable=seasons_profitable,
            profit_probability=profit_probability
        )

    def _simulate_season(self, config: MonteCarloConfig) -> Tuple[float, float]:
        """
        Simulate one season of betting.

        Returns:
            Tuple of (final_bankroll, max_drawdown)
        """
        bankroll = config.initial_bankroll
        peak_bankroll = bankroll
        max_drawdown = 0.0

        # Convert American odds to decimal
        if config.avg_odds > 0:
            odds_decimal = 1 + config.avg_odds / 100
        else:
            odds_decimal = 1 + 100 / abs(config.avg_odds)

        for _ in range(config.num_bets_per_season):
            # Calculate bet size (Kelly)
            edge = config.win_rate - 0.5  # Implied edge from win rate
            b = odds_decimal - 1
            p = config.win_rate
            q = 1 - p

            kelly_pct = (b * p - q) / b
            kelly_pct = max(0, kelly_pct) * config.kelly_fraction
            kelly_pct = min(kelly_pct, config.max_bet_pct)

            bet_size = bankroll * kelly_pct
            bet_size = max(config.min_bet, bet_size) if bet_size >= config.min_bet * 0.5 else config.min_bet

            # Simulate bet outcome
            if np.random.random() < config.win_rate:
                # Win
                if config.avg_odds > 0:
                    pnl = bet_size * (config.avg_odds / 100)
                else:
                    pnl = bet_size * (100 / abs(config.avg_odds))
            else:
                # Loss
                pnl = -bet_size

            bankroll += pnl

            # Track drawdown
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            else:
                drawdown = (peak_bankroll - bankroll) / peak_bankroll
                max_drawdown = max(max_drawdown, drawdown)

            # Check for ruin
            if bankroll < config.ruin_threshold:
                return bankroll, max_drawdown

        return bankroll, max_drawdown

    def run_strategy_comparison(
        self,
        base_config: MonteCarloConfig,
        kelly_fractions: List[float] = [0.25, 0.5, 0.75, 1.0]
    ) -> Dict[str, Any]:
        """
        Compare different Kelly fractions.

        Args:
            base_config: Base configuration (kelly_fraction will be overridden)
            kelly_fractions: List of Kelly fractions to test

        Returns:
            Comparison results
        """
        results = {}

        # Get base config dict without kelly_fraction
        base_dict = {k: v for k, v in base_config.__dict__.items() if k != 'kelly_fraction'}

        for kelly in kelly_fractions:
            config = MonteCarloConfig(**base_dict, kelly_fraction=kelly)
            result = self.run(config)
            results[f'kelly_{kelly}'] = result.to_dict()

        # Find optimal Kelly
        optimal = max(results.items(), key=lambda x: x[1]['median_bankroll'])

        return {
            'strategies': results,
            'optimal_kelly': float(optimal[0].split('_')[1]),
            'optimal_median_bankroll': optimal[1]['median_bankroll'],
        }

    def calculate_confidence_interval(
        self,
        result: SimulationResult,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence interval for final bankroll.

        Args:
            result: SimulationResult
            confidence: Confidence level (default 95%)

        Returns:
            Dictionary with CI bounds
        """
        bankrolls = np.array(result.final_bankrolls)
        alpha = 1 - confidence

        lower = float(np.percentile(bankrolls, alpha / 2 * 100))
        upper = float(np.percentile(bankrolls, (1 - alpha / 2) * 100))

        return {
            'confidence_level': confidence,
            'lower_bound': lower,
            'upper_bound': upper,
            'median': result.median_bankroll,
        }

    def answer_risk_question(
        self,
        result: SimulationResult,
        question: str = "What's my risk of going broke?"
    ) -> Dict[str, Any]:
        """
        Answer specific risk questions.

        Args:
            result: SimulationResult
            question: Risk question to answer

        Returns:
            Answer with explanation
        """
        answers = {
            "ruin": {
                "question": "What's my risk of going broke?",
                "answer": f"Based on {result.config.num_simulations:,} simulations, "
                         f"your risk of ruin (bankroll < ${result.config.ruin_threshold}) is "
                         f"{result.ruin_probability:.1%}. "
                         f"This means {result.ruin_count:,} out of {result.config.num_simulations:,} "
                         f"simulated seasons ended in ruin.",
                "risk_level": "HIGH" if result.ruin_probability > 0.2 else "MODERATE" if result.ruin_probability > 0.1 else "LOW",
            },
            "profitable": {
                "question": "What's my chance of being profitable?",
                "answer": f"You have a {result.profit_probability:.1%} chance of ending "
                         f"the season profitable. In {result.seasons_profitable:,} out of "
                         f"{result.config.num_simulations:,} simulations, you ended with "
                         f"more than your starting bankroll.",
            },
            "median": {
                "question": "What's my expected bankroll?",
                "answer": f"The median final bankroll is ${result.median_bankroll:.2f}, "
                         f"representing a {(result.median_bankroll/result.config.initial_bankroll - 1) * 100:+.1f}% "
                         f"return. The mean is ${result.mean_bankroll:.2f} with std dev ${result.std_bankroll:.2f}.",
            },
            "worst_case": {
                "question": "What's the worst that could happen?",
                "answer": f"In the worst 5% of cases, you'll end with ${result.percentile_5:.2f} or less. "
                         f"The absolute worst case was ${result.worst_case:.2f}. "
                         f"Average max drawdown was {result.max_drawdown_avg:.1%}.",
            },
            "best_case": {
                "question": "What's the best that could happen?",
                "answer": f"In the best 5% of cases, you'll end with ${result.percentile_95:.2f} or more. "
                         f"The absolute best case was ${result.best_case:.2f}.",
            },
        }

        # Match question to answer
        question_lower = question.lower()
        for key, answer_data in answers.items():
            if key in question_lower:
                return answer_data

        # Default: return all answers
        return {
            "question": question,
            "all_answers": answers,
            "summary": result._interpret()
        }


def run_quick_simulation(
    initial_bankroll: float = 1000.0,
    win_rate: float = 0.55,
    kelly_fraction: float = 0.5,
    num_bets: int = 500,
    num_simulations: int = 10000
) -> Dict[str, Any]:
    """
    Run a quick Monte Carlo simulation with default parameters.

    Args:
        initial_bankroll: Starting bankroll
        win_rate: Expected win rate
        kelly_fraction: Kelly fraction to use
        num_bets: Number of bets per season
        num_simulations: Number of simulations

    Returns:
        Simulation results as dictionary
    """
    config = MonteCarloConfig(
        initial_bankroll=initial_bankroll,
        num_simulations=num_simulations,
        num_bets_per_season=num_bets,
        win_rate=win_rate,
        kelly_fraction=kelly_fraction
    )

    sim = MonteCarloSimulator()
    result = sim.run(config)

    return result.to_dict()
