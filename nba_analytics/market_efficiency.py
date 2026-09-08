"""NBA Market Efficiency Analysis.

Track: which prop types are most efficient (pts vs. reb vs. ast)
Track: which books are softest on each prop type
Allocate: more to softer markets
Output: market_efficiency_report.json
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NBA_ANALYTICS_DIR = str(PROJECT_ROOT / "nba_analytics")


class PropType(Enum):
    """NBA prop types."""
    POINTS = "points"
    REBOUNDS = "rebounds"
    ASSISTS = "assists"
    THREES = "threes"
    STEALS = "steals"
    BLOCKS = "blocks"
    TURNOWERS = "turnovers"
    PRA = "points_rebounds_assists"
    PR = "points_rebounds"
    PA = "points_assists"
    RA = "rebounds_assists"


class EfficiencyRating(Enum):
    """Market efficiency rating."""
    VERY_SOFT = "very_soft"  # High edge available
    SOFT = "soft"  # Moderate edge
    FAIR = "fair"  # Normal efficiency
    EFFICIENT = "efficient"  # Low edge
    VERY_EFFICIENT = "very_efficient"  # No edge


@dataclass
class PropMarketMetrics:
    """Metrics for a specific prop market."""
    prop_type: str
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    win_rate: float = 0.0
    avg_edge_found: float = 0.0  # Average edge at bet time
    avg_closing_edge: float = 0.0  # Edge at closing line
    clv_avg: float = 0.0  # Closing line value
    roi: float = 0.0
    sharpe_ratio: float = 0.0
    kelly_growth: float = 0.0  # Expected Kelly growth rate
    market_efficiency_score: float = 0.0  # 0-100, lower = softer

    def to_dict(self) -> Dict[str, Any]:
        decisive = self.wins + self.losses
        return {
            'prop_type': self.prop_type,
            'activity': {
                'total_bets': self.total_bets,
                'record': f"{self.wins}-{self.losses}",
                'pushes': self.pushes,
                'win_rate': round(self.win_rate, 4) if decisive > 0 else 0,
            },
            'edge_metrics': {
                'avg_edge_found': round(self.avg_edge_found, 2),
                'avg_closing_edge': round(self.avg_closing_edge, 2),
                'clv_avg': round(self.clv_avg, 3),
            },
            'performance': {
                'roi': round(self.roi, 2),
                'sharpe_ratio': round(self.sharpe_ratio, 3),
                'kelly_growth': round(self.kelly_growth, 4),
            },
            'efficiency': {
                'score': round(self.market_efficiency_score, 1),
                'rating': self._get_efficiency_rating(),
            },
        }

    def _get_efficiency_rating(self) -> str:
        if self.market_efficiency_score >= 80:
            return EfficiencyRating.VERY_EFFICIENT.value
        elif self.market_efficiency_score >= 60:
            return EfficiencyRating.EFFICIENT.value
        elif self.market_efficiency_score >= 40:
            return EfficiencyRating.FAIR.value
        elif self.market_efficiency_score >= 20:
            return EfficiencyRating.SOFT.value
        else:
            return EfficiencyRating.VERY_SOFT.value


@dataclass
class SportsbookMetrics:
    """Metrics for a specific sportsbook."""
    sportsbook: str
    total_bets: int = 0
    by_prop_type: Dict[str, int] = field(default_factory=dict)
    avg_hold: float = 0.0  # Estimated hold percentage
    line_movement_freq: float = 0.0  # How often lines move after bet
    sharp_money_indicator: float = 0.0  # 0-100, higher = more sharp action
    softness_score: float = 0.0  # 0-100, higher = softer book

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sportsbook': self.sportsbook,
            'activity': {
                'total_bets': self.total_bets,
                'by_prop_type': self.by_prop_type,
            },
            'metrics': {
                'avg_hold': round(self.avg_hold, 2),
                'line_movement_freq': round(self.line_movement_freq, 3),
                'sharp_money_indicator': round(self.sharp_money_indicator, 1),
            },
            'softness': {
                'score': round(self.softness_score, 1),
                'rating': self._get_softness_rating(),
            },
        }

    def _get_softness_rating(self) -> str:
        if self.softness_score >= 80:
            return "very_soft"
        elif self.softness_score >= 60:
            return "soft"
        elif self.softness_score >= 40:
            return "moderate"
        elif self.softness_score >= 20:
            return "sharp"
        else:
            return "very_sharp"


@dataclass
class MarketEfficiencyReport:
    """Complete market efficiency report."""
    report_date: str
    period_start: str
    period_end: str
    prop_metrics: Dict[str, PropMarketMetrics] = field(default_factory=dict)
    sportsbook_metrics: Dict[str, SportsbookMetrics] = field(default_factory=dict)
    allocation_recommendations: Dict[str, float] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metadata': {
                'report_date': self.report_date,
                'period': {
                    'start': self.period_start,
                    'end': self.period_end,
                },
            },
            'prop_markets': {k: v.to_dict() for k, v in self.prop_metrics.items()},
            'sportsbooks': {k: v.to_dict() for k, v in self.sportsbook_metrics.items()},
            'allocation_recommendations': self.allocation_recommendations,
            'summary': self.summary,
        }


class NBAMarketAnalyzer:
    """
    Analyze NBA prop market efficiency.

    Usage:
        analyzer = NBAMarketAnalyzer()
        analyzer.add_bet(bet_data)
        report = analyzer.generate_report()
        analyzer.save_report(report)
    """

    def __init__(self, analytics_dir: str = NBA_ANALYTICS_DIR):
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(parents=True, exist_ok=True)

        self.prop_metrics: Dict[str, PropMarketMetrics] = {}
        self.sportsbook_metrics: Dict[str, SportsbookMetrics] = {}
        self.bet_history: List[Dict[str, Any]] = []

        self._initialize_metrics()
        self._load_existing_data()

    def _initialize_metrics(self):
        """Initialize metrics for all prop types and sportsbooks."""
        for prop_type in PropType:
            self.prop_metrics[prop_type.value] = PropMarketMetrics(prop_type=prop_type.value)

    def _load_existing_data(self):
        """Load existing market data."""
        data_file = self.analytics_dir / "market_efficiency_data.json"
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                self.bet_history = data.get('bet_history', [])
            except Exception as e:
                print(f"Error loading market data: {e}")

    def add_bet(
        self,
        prop_type: str,
        sportsbook: str,
        edge_found: float,
        closing_edge: float,
        outcome: str,
        pnl: float,
        bet_size: float,
        clv: float = 0.0,
        line_movement: float = 0.0,
    ):
        """
        Add bet to market efficiency tracking.

        Args:
            prop_type: Type of prop (points, rebounds, assists, etc.)
            sportsbook: Sportsbook where bet was placed
            edge_found: Edge at bet time
            closing_edge: Edge at closing line
            outcome: win, loss, or push
            pnl: Profit/loss
            bet_size: Bet size
            clv: Closing line value
            line_movement: Line movement after bet
        """
        bet_record = {
            'date': datetime.now().isoformat(),
            'prop_type': prop_type,
            'sportsbook': sportsbook,
            'edge_found': edge_found,
            'closing_edge': closing_edge,
            'outcome': outcome,
            'pnl': pnl,
            'bet_size': bet_size,
            'clv': clv,
            'line_movement': line_movement,
        }

        self.bet_history.append(bet_record)

        # Update prop metrics
        if prop_type in self.prop_metrics:
            self._update_prop_metrics(prop_type, bet_record)

        # Update sportsbook metrics
        if sportsbook not in self.sportsbook_metrics:
            self.sportsbook_metrics[sportsbook] = SportsbookMetrics(sportsbook=sportsbook)
        self._update_sportsbook_metrics(sportsbook, bet_record)

        # Save data
        self._save_data()

    def _update_prop_metrics(self, prop_type: str, bet: Dict[str, Any]):
        """Update prop type metrics."""
        metrics = self.prop_metrics[prop_type]

        metrics.total_bets += 1

        if bet['outcome'] == 'win':
            metrics.wins += 1
        elif bet['outcome'] == 'loss':
            metrics.losses += 1
        else:
            metrics.pushes += 1

        # Update averages (running average)
        n = metrics.total_bets
        metrics.avg_edge_found = ((metrics.avg_edge_found * (n - 1)) + bet['edge_found']) / n
        metrics.avg_closing_edge = ((metrics.avg_closing_edge * (n - 1)) + bet['closing_edge']) / n
        metrics.clv_avg = ((metrics.clv_avg * (n - 1)) + bet['clv']) / n

        # Calculate win rate
        decisive = metrics.wins + metrics.losses
        metrics.win_rate = metrics.wins / decisive if decisive > 0 else 0

        # Calculate ROI
        # Simplified: total pnl / total wagered
        total_wagered = sum(b['bet_size'] for b in self.bet_history if b['prop_type'] == prop_type)
        total_pnl = sum(b['pnl'] for b in self.bet_history if b['prop_type'] == prop_type)
        metrics.roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0

        # Calculate market efficiency score
        # Lower score = softer market (more edge available)
        # Based on: avg edge found, CLV, ROI
        edge_component = max(0, metrics.avg_edge_found * 10)  # Higher edge = softer
        clv_component = max(0, metrics.clv_avg * 100)  # Positive CLV = softer
        roi_component = max(0, metrics.roi / 5)  # Positive ROI = softer

        # Efficiency score: 0-100, lower = softer
        metrics.market_efficiency_score = max(0, 100 - (edge_component + clv_component + roi_component))

    def _update_sportsbook_metrics(self, sportsbook: str, bet: Dict[str, Any]):
        """Update sportsbook metrics."""
        metrics = self.sportsbook_metrics[sportsbook]

        metrics.total_bets += 1

        # Track by prop type
        prop_type = bet['prop_type']
        metrics.by_prop_type[prop_type] = metrics.by_prop_type.get(prop_type, 0) + 1

        # Estimate hold from line movement
        # More line movement = more sharp action = lower hold
        if abs(bet['line_movement']) > 0.5:
            metrics.avg_hold = (metrics.avg_hold * (metrics.total_bets - 1) + 4.0) / metrics.total_bets
        else:
            metrics.avg_hold = (metrics.avg_hold * (metrics.total_bets - 1) + 5.5) / metrics.total_bets

        # Line movement frequency
        if abs(bet['line_movement']) > 0.3:
            metrics.line_movement_freq = (
                (metrics.line_movement_freq * (metrics.total_bets - 1)) + 1
            ) / metrics.total_bets

        # Sharp money indicator (based on CLV)
        if bet['clv'] > 0:
            metrics.sharp_money_indicator = min(100, metrics.sharp_money_indicator + 2)
        else:
            metrics.sharp_money_indicator = max(0, metrics.sharp_money_indicator - 1)

        # Softness score: higher = softer book
        # Based on: hold, line movement, sharp indicator
        hold_component = max(0, (6.0 - metrics.avg_hold) * 10)  # Lower hold = softer
        movement_component = metrics.line_movement_freq * 20  # More movement = softer
        sharp_component = (100 - metrics.sharp_money_indicator) * 0.2  # Less sharp = softer

        metrics.softness_score = min(100, hold_component + movement_component + sharp_component)

    def _save_data(self):
        """Save market data to file."""
        data_file = self.analytics_dir / "market_efficiency_data.json"
        data = {
            'bet_history': self.bet_history[-1000:],  # Keep last 1000 bets
            'last_updated': datetime.now().isoformat(),
        }
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_report(
        self,
        period_days: int = 30
    ) -> MarketEfficiencyReport:
        """
        Generate market efficiency report.

        Args:
            period_days: Number of days to include

        Returns:
            MarketEfficiencyReport
        """
        cutoff = datetime.now() - timedelta(days=period_days)

        # Filter recent bets
        recent_bets = [
            b for b in self.bet_history
            if datetime.fromisoformat(b['date']) > cutoff
        ]

        # Recalculate metrics for period
        period_prop_metrics = {}
        period_sportsbook_metrics = {}

        for bet in recent_bets:
            prop_type = bet['prop_type']
            sportsbook = bet['sportsbook']

            if prop_type not in period_prop_metrics:
                period_prop_metrics[prop_type] = PropMarketMetrics(prop_type=prop_type)
            self._update_temp_metrics(period_prop_metrics[prop_type], bet)

            if sportsbook not in period_sportsbook_metrics:
                period_sportsbook_metrics[sportsbook] = SportsbookMetrics(sportsbook=sportsbook)
            self._update_temp_metrics_sb(period_sportsbook_metrics[sportsbook], bet)

        # Generate allocation recommendations
        allocations = self._calculate_allocations(period_prop_metrics)

        # Generate summary
        summary = self._generate_summary(
            period_prop_metrics,
            period_sportsbook_metrics,
            recent_bets
        )

        now = datetime.now()
        return MarketEfficiencyReport(
            report_date=now.isoformat(),
            period_start=(now - timedelta(days=period_days)).isoformat(),
            period_end=now.isoformat(),
            prop_metrics=period_prop_metrics,
            sportsbook_metrics=period_sportsbook_metrics,
            allocation_recommendations=allocations,
            summary=summary,
        )

    def _update_temp_metrics(self, metrics: PropMarketMetrics, bet: Dict[str, Any]):
        """Update temporary metrics for report period."""
        metrics.total_bets += 1

        if bet['outcome'] == 'win':
            metrics.wins += 1
        elif bet['outcome'] == 'loss':
            metrics.losses += 1
        else:
            metrics.pushes += 1

        n = metrics.total_bets
        metrics.avg_edge_found = ((metrics.avg_edge_found * (n - 1)) + bet['edge_found']) / n
        metrics.clv_avg = ((metrics.clv_avg * (n - 1)) + bet['clv']) / n

        decisive = metrics.wins + metrics.losses
        metrics.win_rate = metrics.wins / decisive if decisive > 0 else 0

        total_wagered = sum(b['bet_size'] for b in self.bet_history if b['prop_type'] == metrics.prop_type)
        total_pnl = sum(b['pnl'] for b in self.bet_history if b['prop_type'] == metrics.prop_type)
        metrics.roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0

        edge_component = max(0, metrics.avg_edge_found * 10)
        clv_component = max(0, metrics.clv_avg * 100)
        roi_component = max(0, metrics.roi / 5)
        metrics.market_efficiency_score = max(0, 100 - (edge_component + clv_component + roi_component))

    def _update_temp_metrics_sb(self, metrics: SportsbookMetrics, bet: Dict[str, Any]):
        """Update temporary sportsbook metrics."""
        metrics.total_bets += 1
        metrics.by_prop_type[bet['prop_type']] = metrics.by_prop_type.get(bet['prop_type'], 0) + 1

        if abs(bet['line_movement']) > 0.3:
            metrics.line_movement_freq = (
                (metrics.line_movement_freq * (metrics.total_bets - 1)) + 1
            ) / metrics.total_bets

        hold_component = max(0, (6.0 - metrics.avg_hold) * 10)
        movement_component = metrics.line_movement_freq * 20
        sharp_component = (100 - metrics.sharp_money_indicator) * 0.2
        metrics.softness_score = min(100, hold_component + movement_component + sharp_component)

    def _calculate_allocations(
        self,
        prop_metrics: Dict[str, PropMarketMetrics]
    ) -> Dict[str, float]:
        """Calculate recommended allocation by prop type."""
        allocations = {}
        total_score = 0

        for prop_type, metrics in prop_metrics.items():
            # Score based on softness (inverse of efficiency)
            softness_score = 100 - metrics.market_efficiency_score
            if metrics.total_bets >= 10:  # Minimum sample
                allocations[prop_type] = softness_score
                total_score += softness_score
            else:
                allocations[prop_type] = 0

        # Normalize to percentages
        if total_score > 0:
            allocations = {k: round(v / total_score * 100, 1) for k, v in allocations.items()}

        return allocations

    def _generate_summary(
        self,
        prop_metrics: Dict[str, PropMarketMetrics],
        sportsbook_metrics: Dict[str, SportsbookMetrics],
        bets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate report summary."""
        # Find softest prop market
        softest_prop = min(
            [(k, v.market_efficiency_score) for k, v in prop_metrics.items() if v.total_bets >= 10],
            key=lambda x: x[1],
            default=('unknown', 100)
        )

        # Find softest sportsbook
        softest_book = max(
            [(k, v.softness_score) for k, v in sportsbook_metrics.items() if v.total_bets >= 10],
            key=lambda x: x[1],
            default=('unknown', 0)
        )

        # Calculate total edge available
        avg_edge = np.mean([b['edge_found'] for b in bets]) if bets else 0

        # Best ROI prop
        best_roi_prop = max(
            [(k, v.roi) for k, v in prop_metrics.items() if v.total_bets >= 10],
            key=lambda x: x[1],
            default=('unknown', 0)
        )

        return {
            'softest_prop_type': softest_prop[0],
            'softest_prop_efficiency': round(softest_prop[1], 1),
            'softest_sportsbook': softest_book[0],
            'softest_sportsbook_score': round(softest_book[1], 1),
            'avg_edge_available': round(avg_edge, 2),
            'best_roi_prop': best_roi_prop[0],
            'best_roi': round(best_roi_prop[1], 2),
            'total_bets_analyzed': len(bets),
            'recommendation': self._generate_recommendation(softest_prop, softest_book),
        }

    def _generate_recommendation(
        self,
        softest_prop: Tuple[str, float],
        softest_book: Tuple[str, float]
    ) -> str:
        """Generate actionable recommendation."""
        if softest_prop[1] < 30:
            return (
                f"Focus on {softest_prop[0]} props at {softest_book[0]}. "
                f"This market shows significant inefficiency with efficiency score {softest_prop[1]:.1f}. "
                f"Recommend increasing allocation by 20%."
            )
        elif softest_prop[1] < 50:
            return (
                f"{softest_prop[0]} props at {softest_book[0]} show moderate inefficiency. "
                f"Maintain current allocation with slight increase."
            )
        else:
            return (
                "All markets showing normal efficiency. "
                "Maintain diversified allocation across prop types."
            )

    def save_report(self, report: MarketEfficiencyReport):
        """Save report to file."""
        filename = f"market_efficiency_{datetime.now().strftime('%Y%m%d')}.json"
        file_path = self.analytics_dir / filename

        with open(file_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"Market efficiency report saved to: {file_path}")

    def get_soft_markets(self, threshold: float = 40) -> List[Dict[str, Any]]:
        """Get markets below efficiency threshold (softer markets)."""
        soft_markets = []

        for prop_type, metrics in self.prop_metrics.items():
            if metrics.market_efficiency_score < threshold and metrics.total_bets >= 10:
                soft_markets.append({
                    'prop_type': prop_type,
                    'efficiency_score': round(metrics.market_efficiency_score, 1),
                    'avg_edge': round(metrics.avg_edge_found, 2),
                    'roi': round(metrics.roi, 2),
                    'rating': metrics._get_efficiency_rating(),
                })

        # Sort by efficiency score (lowest = softest)
        soft_markets.sort(key=lambda x: x['efficiency_score'])

        return soft_markets
