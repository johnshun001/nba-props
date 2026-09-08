"""Advanced backtest framework for NBA prop betting.

Run on 30+ days of historical data.
Track: win rate, ROI, Sharpe, max drawdown, Calmar ratio
Compare: different Kelly fractions, edge thresholds
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""
    initial_bankroll: float = 1000.0
    kelly_fraction: float = 0.5  # Half-Kelly
    min_edge: float = 0.03  # 3% minimum edge
    max_bet_pct: float = 0.02  # 2% max of bankroll
    min_bet: float = 1.0
    odds: float = -110  # Standard odds
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class BacktestBet:
    """Single bet in backtest."""
    bet_id: str
    date: datetime
    player_id: str
    prop_type: str
    line: float
    bet_side: str
    bet_size: float
    odds: float
    edge: float
    predicted_prob: float
    actual_outcome: Optional[float] = None  # 1=win, 0=loss
    pnl: float = 0.0
    bankroll_after: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    config: BacktestConfig
    bets: List[BacktestBet] = field(default_factory=list)
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_wagered: float = 0.0
    roi: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # In bets
    calmar_ratio: float = 0.0
    final_bankroll: float = 0.0
    bankroll_trajectory: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    edge_distribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'config': {
                'initial_bankroll': self.config.initial_bankroll,
                'kelly_fraction': self.config.kelly_fraction,
                'min_edge': self.config.min_edge,
                'max_bet_pct': self.config.max_bet_pct,
            },
            'total_bets': self.total_bets,
            'wins': self.wins,
            'losses': self.losses,
            'pushes': self.pushes,
            'win_rate': self.win_rate,
            'total_pnl': round(self.total_pnl, 2),
            'total_wagered': round(self.total_wagered, 2),
            'roi': round(self.roi, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_duration': self.max_drawdown_duration,
            'calmar_ratio': round(self.calmar_ratio, 3),
            'final_bankroll': round(self.final_bankroll, 2),
            'edge_distribution': self.edge_distribution,
        }


class Backtester:
    """
    Advanced backtest framework for NBA prop betting.

    Usage:
        backtester = Backtester()
        result = backtester.run(historical_data, config)
        print(result.to_dict())
    """

    def __init__(self):
        self.risk_free_rate = 0.02  # Annual risk-free rate

    def run(self, historical_data: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            historical_data: DataFrame with columns:
                - date: datetime
                - player_id: str
                - prop_type: str
                - line: float
                - actual: float (actual stat value)
                - model_projection: float
                - market_lines: List[Dict] (available lines from sportsbooks)
            config: BacktestConfig object

        Returns:
            BacktestResult with all metrics
        """
        bankroll = config.initial_bankroll
        bankroll_trajectory = [bankroll]
        daily_pnl = {}
        bets = []
        peak_bankroll = bankroll
        max_drawdown = 0.0
        drawdown_start = None
        max_drawdown_duration = 0
        current_drawdown_duration = 0

        # Filter by date range if specified
        df = historical_data.copy()
        if config.start_date:
            df = df[df['date'] >= config.start_date]
        if config.end_date:
            df = df[df['date'] <= config.end_date]

        df = df.sort_values('date').reset_index(drop=True)

        for idx, row in df.iterrows():
            # Calculate edge
            model_proj = row['model_projection']
            line = row['line']
            edge = (model_proj - line) / line

            # Skip if edge below threshold
            if edge < config.min_edge:
                continue

            # Calculate Kelly bet size
            bet_size = self._calculate_kelly_bet(
                bankroll=bankroll,
                edge=edge,
                odds=config.odds,
                kelly_fraction=config.kelly_fraction,
                max_bet_pct=config.max_bet_pct,
                min_bet=config.min_bet
            )

            if bet_size <= 0:
                continue

            # Determine bet side
            bet_side = 'over' if model_proj > line else 'under'

            # Calculate outcome
            actual = row['actual']
            if bet_side == 'over':
                won = actual > line
            else:
                won = actual < line

            # Handle push
            if actual == line:
                pnl = 0.0
                outcome = 0  # Push
            elif won:
                # Win
                if config.odds > 0:
                    pnl = bet_size * (config.odds / 100)
                else:
                    pnl = bet_size * (100 / abs(config.odds))
                outcome = 1
            else:
                # Loss
                pnl = -bet_size
                outcome = 0

            # Update bankroll
            bankroll += pnl

            # Track drawdown
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
                current_drawdown_duration = 0
            else:
                drawdown = (peak_bankroll - bankroll) / peak_bankroll
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                current_drawdown_duration += 1
                if current_drawdown_duration > max_drawdown_duration:
                    max_drawdown_duration = current_drawdown_duration

            # Track daily P&L
            date_key = row['date'].strftime('%Y-%m-%d')
            if date_key not in daily_pnl:
                daily_pnl[date_key] = 0.0
            daily_pnl[date_key] += pnl

            # Create bet record
            bet = BacktestBet(
                bet_id=f"bet_{idx:06d}",
                date=row['date'],
                player_id=row.get('player_id', 'unknown'),
                prop_type=row.get('prop_type', 'unknown'),
                line=line,
                bet_side=bet_side,
                bet_size=bet_size,
                odds=config.odds,
                edge=edge,
                predicted_prob=0.5 + edge,
                actual_outcome=outcome,
                pnl=pnl,
                bankroll_after=bankroll
            )
            bets.append(bet)
            bankroll_trajectory.append(bankroll)

        # Calculate metrics
        wins = sum(1 for b in bets if b.actual_outcome == 1)
        losses = sum(1 for b in bets if b.actual_outcome == 0)
        pushes = sum(1 for b in bets if b.actual_outcome is None or (b.actual_outcome == 0 and b.pnl == 0))

        decisive_bets = wins + losses
        win_rate = wins / decisive_bets if decisive_bets > 0 else 0.0

        total_wagered = sum(b.bet_size for b in bets)
        total_pnl = bankroll - config.initial_bankroll
        roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0.0

        # Calculate daily returns
        daily_returns = []
        prev_bankroll = config.initial_bankroll
        for date_key in sorted(daily_pnl.keys()):
            pnl = daily_pnl[date_key]
            daily_ret = pnl / prev_bankroll
            daily_returns.append(daily_ret)
            prev_bankroll += pnl

        # Sharpe ratio (annualized)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)

        # Calmar ratio (annualized return / max drawdown)
        annualized_return = (bankroll / config.initial_bankroll) ** (365 / max(len(daily_returns), 1)) - 1
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

        # Edge distribution
        edge_buckets = {'0-3%': 0, '3-5%': 0, '5-10%': 0, '10%+': 0}
        for bet in bets:
            edge_pct = bet.edge * 100
            if edge_pct < 3:
                edge_buckets['0-3%'] += 1
            elif edge_pct < 5:
                edge_buckets['3-5%'] += 1
            elif edge_pct < 10:
                edge_buckets['5-10%'] += 1
            else:
                edge_buckets['10%+'] += 1

        return BacktestResult(
            config=config,
            bets=bets,
            total_bets=len(bets),
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_wagered=total_wagered,
            roi=roi,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown * 100,  # As percentage
            max_drawdown_duration=max_drawdown_duration,
            calmar_ratio=calmar_ratio,
            final_bankroll=bankroll,
            bankroll_trajectory=bankroll_trajectory,
            daily_returns=daily_returns,
            edge_distribution=edge_buckets
        )

    def _calculate_kelly_bet(
        self,
        bankroll: float,
        edge: float,
        odds: float,
        kelly_fraction: float,
        max_bet_pct: float,
        min_bet: float
    ) -> float:
        """Calculate Kelly bet size."""
        if edge <= 0:
            return 0.0

        # Convert American odds to decimal
        if odds > 0:
            b = odds / 100
        else:
            b = 100 / abs(odds)

        p = 0.5 + edge
        q = 1 - p

        kelly_pct = (b * p - q) / b
        kelly_pct = max(0, kelly_pct)
        kelly_pct = kelly_pct * kelly_fraction
        kelly_pct = min(kelly_pct, max_bet_pct)

        bet_size = bankroll * kelly_pct
        bet_size = max(min_bet, bet_size) if bet_size >= min_bet * 0.5 else 0.0

        return round(bet_size, 2)

    def _calculate_sharpe_ratio(self, daily_returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(daily_returns) < 2:
            return 0.0

        returns = np.array(daily_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * 252  # Trading days
        annualized_std = std_return * np.sqrt(252)

        sharpe = (annualized_return - self.risk_free_rate) / annualized_std
        return sharpe

    def compare_strategies(
        self,
        historical_data: pd.DataFrame,
        configs: List[BacktestConfig]
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies.

        Args:
            historical_data: Historical data DataFrame
            configs: List of BacktestConfig objects with different parameters

        Returns:
            Comparison results
        """
        results = {}
        for config in configs:
            result = self.run(historical_data, config)
            key = f"kelly_{config.kelly_fraction}_minedge_{config.min_edge}"
            results[key] = result.to_dict()

        # Find best strategy by Sharpe ratio
        best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
        best_roi = max(results.items(), key=lambda x: x[1]['roi'])
        best_calmar = max(results.items(), key=lambda x: x[1]['calmar_ratio'])

        return {
            'strategies': results,
            'best_by_sharpe': best_sharpe[0],
            'best_by_roi': best_roi[0],
            'best_by_calmar': best_calmar[0],
        }

    def generate_report(self, result: BacktestResult) -> str:
        """Generate human-readable backtest report."""
        report = []
        report.append("=" * 60)
        report.append("BACKTEST REPORT")
        report.append("=" * 60)
        report.append(f"\nConfiguration:")
        report.append(f"  Initial Bankroll: ${result.config.initial_bankroll:,.2f}")
        report.append(f"  Kelly Fraction: {result.config.kelly_fraction:.0%}")
        report.append(f"  Min Edge: {result.config.min_edge:.1%}")
        report.append(f"  Max Bet: {result.config.max_bet_pct:.1%} of bankroll")

        report.append(f"\nPerformance Summary:")
        report.append(f"  Total Bets: {result.total_bets}")
        report.append(f"  Record: {result.wins}-{result.losses} ({result.win_rate:.1%})")
        report.append(f"  Total P&L: ${result.total_pnl:+,.2f}")
        report.append(f"  ROI: {result.roi:.2f}%")
        report.append(f"  Final Bankroll: ${result.final_bankroll:,.2f}")

        report.append(f"\nRisk Metrics:")
        report.append(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        report.append(f"  Max Drawdown: {result.max_drawdown:.2f}%")
        report.append(f"  Drawdown Duration: {result.max_drawdown_duration} bets")
        report.append(f"  Calmar Ratio: {result.calmar_ratio:.3f}")

        report.append(f"\nEdge Distribution:")
        for bucket, count in result.edge_distribution.items():
            report.append(f"  {bucket}: {count} bets")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def run_parameter_sweep(
    historical_data: pd.DataFrame,
    kelly_fractions: List[float] = [0.25, 0.5, 0.75, 1.0],
    min_edges: List[float] = [0.02, 0.03, 0.05, 0.07]
) -> pd.DataFrame:
    """
    Run parameter sweep over Kelly fractions and edge thresholds.

    Returns:
        DataFrame with results for each parameter combination
    """
    backtester = Backtester()
    results = []

    for kelly in kelly_fractions:
        for min_edge in min_edges:
            config = BacktestConfig(
                kelly_fraction=kelly,
                min_edge=min_edge
            )
            result = backtester.run(historical_data, config)

            results.append({
                'kelly_fraction': kelly,
                'min_edge': min_edge,
                'total_bets': result.total_bets,
                'win_rate': result.win_rate,
                'roi': result.roi,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'calmar_ratio': result.calmar_ratio,
                'final_bankroll': result.final_bankroll,
            })

    return pd.DataFrame(results)
