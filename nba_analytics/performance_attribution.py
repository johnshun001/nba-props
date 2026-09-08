"""NBA Performance Attribution System.

Track: edge from each source (line shopping, model, injuries, etc.)
Know: what's actually driving profits
Double down: on what works, cut what doesn't
Output: attribution/{month}_report.json
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION_DIR = str(PROJECT_ROOT / "attribution")


@dataclass
class EdgeSource:
    """Edge source attribution."""
    source_name: str
    description: str
    total_bets: int = 0
    total_edge_contrib: float = 0.0  # Total edge contributed
    avg_edge_per_bet: float = 0.0
    win_rate: float = 0.0
    roi: float = 0.0
    pnl_contribution: float = 0.0
    confidence: float = 0.0  # Statistical confidence in this source

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_name': self.source_name,
            'description': self.description,
            'metrics': {
                'total_bets': self.total_bets,
                'total_edge_contrib': round(self.total_edge_contrib, 3),
                'avg_edge_per_bet': round(self.avg_edge_per_bet, 4),
                'win_rate': round(self.win_rate, 4),
                'roi': round(self.roi, 2),
                'pnl_contribution': round(self.pnl_contribution, 2),
                'confidence': round(self.confidence, 4),
            },
            'attribution_pct': round(self._calc_attribution_pct(), 1),
        }

    def _calc_attribution_pct(self) -> float:
        """Calculate percentage of total edge from this source."""
        return (self.total_edge_contrib / max(0.001, sum([
            self.total_edge_contrib
        ]))) * 100


@dataclass
class MonthlyAttribution:
    """Monthly attribution report."""
    month: str
    total_bets: int = 0
    total_pnl: float = 0.0
    total_edge: float = 0.0
    by_source: Dict[str, EdgeSource] = field(default_factory=dict)
    by_prop_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_sportsbook: Dict[str, Dict[str, float]] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'month': self.month,
            'summary': {
                'total_bets': self.total_bets,
                'total_pnl': round(self.total_pnl, 2),
                'total_edge': round(self.total_edge, 3),
                'roi': round(self.total_pnl / max(1, self.total_bets * 100) * 100, 2),
            },
            'by_source': {k: v.to_dict() for k, v in self.by_source.items()},
            'by_prop_type': self.by_prop_type,
            'by_sportsbook': self.by_sportsbook,
            'insights': self.insights,
            'recommendations': self.recommendations,
        }


class PerformanceAttributor:
    """
    Attribute performance to edge sources.

    Usage:
        attributor = PerformanceAttributor()
        attributor.add_bet(bet_data)
        report = attributor.generate_monthly_report()
    """

    def __init__(self, attribution_dir: str = ATTRIBUTION_DIR):
        self.attribution_dir = Path(attribution_dir)
        self.attribution_dir.mkdir(parents=True, exist_ok=True)

        self.bets: List[Dict[str, Any]] = []
        self.edge_sources: Dict[str, EdgeSource] = self._initialize_sources()

        self._load_existing_data()

    def _initialize_sources(self) -> Dict[str, EdgeSource]:
        """Initialize edge sources."""
        return {
            'line_shopping': EdgeSource(
                source_name='line_shopping',
                description='Edge from finding best available line across sportsbooks',
            ),
            'model_prediction': EdgeSource(
                source_name='model_prediction',
                description='Edge from model prediction accuracy',
            ),
            'injury_information': EdgeSource(
                source_name='injury_information',
                description='Edge from faster/better injury information',
            ),
            'matchup_analysis': EdgeSource(
                source_name='matchup_analysis',
                description='Edge from matchup-specific analysis',
            ),
            'timing_advantage': EdgeSource(
                source_name='timing_advantage',
                description='Edge from betting at optimal time',
            ),
            'market_overreaction': EdgeSource(
                source_name='market_overreaction',
                description='Edge from public overreaction to news',
            ),
        }

    def _load_existing_data(self):
        """Load existing attribution data."""
        data_file = self.attribution_dir / "attribution_data.json"
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                self.bets = data.get('bets', [])
            except Exception as e:
                print(f"Error loading attribution data: {e}")

    def add_bet(
        self,
        prop_type: str,
        sportsbook: str,
        edge_total: float,
        edge_line_shopping: float,
        edge_model: float,
        edge_injury: float,
        edge_matchup: float,
        edge_timing: float,
        edge_market: float,
        outcome: str,
        pnl: float,
        bet_size: float,
    ):
        """
        Add bet with edge breakdown.

        Args:
            prop_type: Type of prop
            sportsbook: Sportsbook used
            edge_total: Total edge found
            edge_line_shopping: Edge from line shopping
            edge_model: Edge from model prediction
            edge_injury: Edge from injury info
            edge_matchup: Edge from matchup analysis
            edge_timing: Edge from timing
            edge_market: Edge from market overreaction
            outcome: win, loss, or push
            pnl: Profit/loss
            bet_size: Bet size
        """
        bet_record = {
            'date': datetime.now().isoformat(),
            'prop_type': prop_type,
            'sportsbook': sportsbook,
            'edge_total': edge_total,
            'edge_breakdown': {
                'line_shopping': edge_line_shopping,
                'model_prediction': edge_model,
                'injury_information': edge_injury,
                'matchup_analysis': edge_matchup,
                'timing_advantage': edge_timing,
                'market_overreaction': edge_market,
            },
            'outcome': outcome,
            'pnl': pnl,
            'bet_size': bet_size,
        }

        self.bets.append(bet_record)
        self._update_edge_sources(bet_record)
        self._save_data()

    def _update_edge_sources(self, bet: Dict[str, Any]):
        """Update edge source attribution."""
        for source_name, edge_contrib in bet['edge_breakdown'].items():
            if source_name in self.edge_sources:
                source = self.edge_sources[source_name]
                source.total_bets += 1
                source.total_edge_contrib += edge_contrib
                source.avg_edge_per_bet = source.total_edge_contrib / source.total_bets
                source.pnl_contribution += bet['pnl'] * (edge_contrib / max(0.001, bet['edge_total']))

                # Update win rate
                if bet['outcome'] == 'win':
                    source.win_rate = (
                        (source.win_rate * (source.total_bets - 1) + 1) /
                        source.total_bets
                    )

                # Update ROI
                total_wagered = sum(
                    b['bet_size'] for b in self.bets
                    if b['edge_breakdown'].get(source_name, 0) > 0
                )
                source.roi = (source.pnl_contribution / max(1, total_wagered)) * 100

                # Calculate confidence (based on sample size)
                source.confidence = min(0.95, source.total_bets / 100)

    def _save_data(self):
        """Save attribution data."""
        data_file = self.attribution_dir / "attribution_data.json"
        data = {
            'bets': self.bets[-500:],  # Keep last 500 bets
            'last_updated': datetime.now().isoformat(),
        }
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_monthly_report(
        self,
        month: Optional[str] = None
    ) -> MonthlyAttribution:
        """
        Generate monthly attribution report.

        Args:
            month: Month in YYYY-MM format (default: current month)

        Returns:
            MonthlyAttribution report
        """
        if month is None:
            month = datetime.now().strftime('%Y-%m')

        # Filter bets for month
        month_bets = [
            b for b in self.bets
            if b['date'].startswith(month)
        ]

        if not month_bets:
            return MonthlyAttribution(month=month)

        # Calculate totals
        total_pnl = sum(b['pnl'] for b in month_bets)
        total_edge = sum(b['edge_total'] for b in month_bets)

        # Calculate by-source attribution for month
        by_source = self._calculate_monthly_source_attribution(month_bets)

        # Calculate by prop type
        by_prop_type = self._calculate_by_prop_type(month_bets)

        # Calculate by sportsbook
        by_sportsbook = self._calculate_by_sportsbook(month_bets)

        # Generate insights and recommendations
        insights = self._generate_insights(by_source)
        recommendations = self._generate_recommendations(by_source)

        return MonthlyAttribution(
            month=month,
            total_bets=len(month_bets),
            total_pnl=total_pnl,
            total_edge=total_edge,
            by_source=by_source,
            by_prop_type=by_prop_type,
            by_sportsbook=by_sportsbook,
            insights=insights,
            recommendations=recommendations,
        )

    def _calculate_monthly_source_attribution(
        self,
        bets: List[Dict[str, Any]]
    ) -> Dict[str, EdgeSource]:
        """Calculate source attribution for specific period."""
        monthly_sources = {}

        for source_name, base_source in self.edge_sources.items():
            monthly_sources[source_name] = EdgeSource(
                source_name=source_name,
                description=base_source.description,
            )

        for bet in bets:
            for source_name, edge_contrib in bet['edge_breakdown'].items():
                if source_name in monthly_sources:
                    source = monthly_sources[source_name]
                    source.total_bets += 1
                    source.total_edge_contrib += edge_contrib
                    source.pnl_contribution += bet['pnl'] * (
                        edge_contrib / max(0.001, bet['edge_total'])
                    )

        # Calculate derived metrics
        for source in monthly_sources.values():
            if source.total_bets > 0:
                source.avg_edge_per_bet = source.total_edge_contrib / source.total_bets
                source.win_rate = sum(
                    1 for b in bets
                    if b['edge_breakdown'].get(source.source_name, 0) > 0
                    and b['outcome'] == 'win'
                ) / source.total_bets
                source.confidence = min(0.95, source.total_bets / 50)

        return monthly_sources

    def _calculate_by_prop_type(
        self,
        bets: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate attribution by prop type."""
        by_prop = {}

        for bet in bets:
            prop = bet['prop_type']
            if prop not in by_prop:
                by_prop[prop] = {
                    'bets': 0,
                    'pnl': 0.0,
                    'avg_edge': 0.0,
                }

            by_prop[prop]['bets'] += 1
            by_prop[prop]['pnl'] += bet['pnl']
            by_prop[prop]['avg_edge'] = (
                (by_prop[prop]['avg_edge'] * (by_prop[prop]['bets'] - 1) +
                 bet['edge_total']) / by_prop[prop]['bets']
            )

        # Calculate ROI
        for prop in by_prop.values():
            prop['roi'] = (prop['pnl'] / max(1, prop['bets'] * 100)) * 100

        return by_prop

    def _calculate_by_sportsbook(
        self,
        bets: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate attribution by sportsbook."""
        by_book = {}

        for bet in bets:
            book = bet['sportsbook']
            if book not in by_book:
                by_book[book] = {
                    'bets': 0,
                    'pnl': 0.0,
                    'avg_edge': 0.0,
                }

            by_book[book]['bets'] += 1
            by_book[book]['pnl'] += bet['pnl']
            by_book[book]['avg_edge'] = (
                (by_book[book]['avg_edge'] * (by_book[book]['bets'] - 1) +
                 bet['edge_total']) / by_book[book]['bets']
            )

        for book in by_book.values():
            book['roi'] = (book['pnl'] / max(1, book['bets'] * 100)) * 100

        return by_book

    def _generate_insights(
        self,
        by_source: Dict[str, EdgeSource]
    ) -> List[str]:
        """Generate insights from attribution data."""
        insights = []

        # Find top contributing source
        if by_source:
            top_source = max(by_source.values(), key=lambda x: x.total_edge_contrib)
            insights.append(
                f"{top_source.source_name.replace('_', ' ').title()} is the top edge source, "
                f"contributing {top_source.total_edge_contrib:.2f} edge points."
            )

        # Find highest ROI source
        high_roi_sources = [
            s for s in by_source.values()
            if s.total_bets >= 10 and s.roi > 10
        ]
        if high_roi_sources:
            best = max(high_roi_sources, key=lambda x: x.roi)
            insights.append(
                f"{best.source_name.replace('_', ' ').title()} shows highest ROI at {best.roi:.1f}%."
            )

        # Find underperforming sources
        low_roi_sources = [
            s for s in by_source.values()
            if s.total_bets >= 20 and s.roi < -5
        ]
        if low_roi_sources:
            worst = min(low_roi_sources, key=lambda x: x.roi)
            insights.append(
                f"{worst.source_name.replace('_', ' ').title()} is underperforming with {worst.roi:.1f}% ROI. "
                f"Consider reducing allocation."
            )

        return insights

    def _generate_recommendations(
        self,
        by_source: Dict[str, EdgeSource]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Double down on high-confidence, high-ROI sources
        strong_sources = [
            s for s in by_source.values()
            if s.confidence > 0.5 and s.roi > 5
        ]
        if strong_sources:
            best = max(strong_sources, key=lambda x: x.roi)
            recommendations.append(
                f"Increase focus on {best.source_name.replace('_', ' ')} - "
                f"showing {best.roi:.1f}% ROI with {best.confidence:.0%} confidence."
            )

        # Cut low-confidence or negative ROI sources
        weak_sources = [
            s for s in by_source.values()
            if s.total_bets >= 30 and s.roi < 0
        ]
        if weak_sources:
            worst = min(weak_sources, key=lambda x: x.roi)
            recommendations.append(
                f"Review {worst.source_name.replace('_', ' ')} strategy - "
                f"{worst.total_bets} bets with {worst.roi:.1f}% ROI."
            )

        # Diversification recommendation
        if len(by_source) > 0:
            concentration = max(s.total_edge_contrib for s in by_source.values())
            total_edge = sum(s.total_edge_contrib for s in by_source.values())
            if total_edge > 0 and concentration / total_edge > 0.6:
                recommendations.append(
                    "Edge concentration is high. Consider diversifying edge sources."
                )

        return recommendations

    def save_report(self, report: MonthlyAttribution):
        """Save monthly report to file."""
        filename = f"{report.month}_attribution_report.json"
        file_path = self.attribution_dir / filename

        with open(file_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"Attribution report saved to: {file_path}")

    def get_edge_source_ranking(self) -> List[Dict[str, Any]]:
        """Get ranking of edge sources by contribution."""
        ranking = []

        for source in self.edge_sources.values():
            if source.total_bets >= 5:
                ranking.append({
                    'source': source.source_name,
                    'total_edge': round(source.total_edge_contrib, 3),
                    'avg_edge': round(source.avg_edge_per_bet, 4),
                    'roi': round(source.roi, 2),
                    'confidence': round(source.confidence, 2),
                    'bets': source.total_bets,
                })

        ranking.sort(key=lambda x: x['total_edge'], reverse=True)
        return ranking
