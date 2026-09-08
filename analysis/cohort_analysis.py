"""Cohort analysis for NBA prop betting.

Group bets by: player type, market type, sportsbook, time of day
Find which cohorts are most profitable
Alert when a cohort stops working
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class CohortMetrics:
    """Metrics for a single cohort."""
    cohort_name: str
    cohort_type: str  # player_type, market_type, sportsbook, time_of_day
    cohort_value: str
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_wagered: float = 0.0
    roi: float = 0.0
    avg_bet_size: float = 0.0
    avg_edge: float = 0.0
    sharpe_ratio: float = 0.0
    is_profitable: bool = False
    is_significant: bool = False  # Statistically significant
    alert_triggered: bool = False
    alert_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cohort_name': self.cohort_name,
            'cohort_type': self.cohort_type,
            'cohort_value': self.cohort_value,
            'total_bets': self.total_bets,
            'record': f"{self.wins}-{self.losses}",
            'win_rate': round(self.win_rate, 4),
            'total_pnl': round(self.total_pnl, 2),
            'total_wagered': round(self.total_wagered, 2),
            'roi': round(self.roi, 2),
            'avg_bet_size': round(self.avg_bet_size, 2),
            'avg_edge': round(self.avg_edge, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'is_profitable': self.is_profitable,
            'is_significant': self.is_significant,
            'alert_triggered': self.alert_triggered,
            'alert_reason': self.alert_reason,
        }


@dataclass
class CohortAlert:
    """Alert when a cohort stops working."""
    alert_id: str
    timestamp: datetime
    cohort_type: str
    cohort_value: str
    alert_type: str  # 'stopped_working', 'started_working', 'anomaly'
    previous_roi: float
    current_roi: float
    severity: str  # 'low', 'medium', 'high'
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'cohort_type': self.cohort_type,
            'cohort_value': self.cohort_value,
            'alert_type': self.alert_type,
            'previous_roi': round(self.previous_roi, 2),
            'current_roi': round(self.current_roi, 2),
            'severity': self.severity,
            'message': self.message,
        }


class CohortAnalyzer:
    """
    Cohort analysis for betting performance.

    Usage:
        analyzer = CohortAnalyzer()
        analyzer.add_bet(bet_data)
        report = analyzer.generate_report()
    """

    def __init__(self, min_bets_for_significance: int = 30):
        self.bets: List[Dict[str, Any]] = []
        self.cohort_metrics: Dict[str, Dict[str, CohortMetrics]] = defaultdict(dict)
        self.alerts: List[CohortAlert] = []
        self.min_bets = min_bets_for_significance
        self._alert_counter = 0

    def add_bet(self, bet: Dict[str, Any]):
        """
        Add a bet to the analysis.

        Args:
            bet: Dict with keys:
                - date: datetime
                - player_id: str
                - prop_type: str
                - bet_side: str
                - bet_size: float
                - odds: float
                - edge: float
                - outcome: str ('win', 'loss', 'push')
                - pnl: float
                - sportsbook: str
                - player_type: str (star, role_player, bench)
        """
        self.bets.append(bet)
        self._update_cohorts(bet)

    def add_bets_batch(self, bets: List[Dict[str, Any]]):
        """Add multiple bets at once."""
        for bet in bets:
            self.add_bet(bet)

    def _update_cohorts(self, bet: Dict[str, Any]):
        """Update cohort metrics with new bet."""
        # Define cohorts
        cohorts = {
            ('player_type', bet.get('player_type', 'unknown')): bet,
            ('market_type', bet.get('prop_type', 'unknown')): bet,
            ('sportsbook', bet.get('sportsbook', 'unknown')): bet,
            ('bet_side', bet.get('bet_side', 'unknown')): bet,
            ('time_of_day', self._categorize_time(bet.get('date'))): bet,
            ('edge_bucket', self._categorize_edge(bet.get('edge', 0))): bet,
            ('day_of_week', bet.get('date', datetime.now()).strftime('%A')): bet,
        }

        for (cohort_type, cohort_value), bet_data in cohorts.items():
            cohort_key = f"{cohort_type}_{cohort_value}"

            if cohort_value not in self.cohort_metrics[cohort_type]:
                self.cohort_metrics[cohort_type][cohort_value] = CohortMetrics(
                    cohort_name=cohort_key,
                    cohort_type=cohort_type,
                    cohort_value=cohort_value
                )

            metrics = self.cohort_metrics[cohort_type][cohort_value]
            self._update_metrics(metrics, bet_data)

    def _update_metrics(self, metrics: CohortMetrics, bet: Dict[str, Any]):
        """Update cohort metrics with single bet."""
        metrics.total_bets += 1

        outcome = bet.get('outcome', '')
        if outcome == 'win':
            metrics.wins += 1
        elif outcome == 'loss':
            metrics.losses += 1
        else:
            metrics.pushes += 1

        metrics.total_pnl += bet.get('pnl', 0)
        metrics.total_wagered += bet.get('bet_size', 0)

        # Recalculate derived metrics
        decisive = metrics.wins + metrics.losses
        metrics.win_rate = metrics.wins / decisive if decisive > 0 else 0.0
        metrics.roi = (metrics.total_pnl / metrics.total_wagered * 100) if metrics.total_wagered > 0 else 0.0
        metrics.avg_bet_size = metrics.total_wagered / metrics.total_bets if metrics.total_bets > 0 else 0.0
        metrics.avg_edge = (metrics.avg_edge * (metrics.total_bets - 1) + bet.get('edge', 0)) / metrics.total_bets
        metrics.is_profitable = metrics.total_pnl > 0
        metrics.is_significant = metrics.total_bets >= self.min_bets

        # Check for alerts
        self._check_alerts(metrics)

    def _categorize_time(self, date: Optional[datetime]) -> str:
        """Categorize bet by time of day."""
        if date is None:
            return 'unknown'

        hour = date.hour
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    def _categorize_edge(self, edge: float) -> str:
        """Categorize bet by edge size."""
        edge_pct = abs(edge) * 100
        if edge_pct < 3:
            return '0-3%'
        elif edge_pct < 5:
            return '3-5%'
        elif edge_pct < 10:
            return '5-10%'
        else:
            return '10%+'

    def _check_alerts(self, metrics: CohortMetrics):
        """Check if cohort triggers alert."""
        if metrics.total_bets < self.min_bets * 2:
            return

        # Check if profitable cohort stopped working
        if metrics.is_significant and metrics.roi < -5:
            if not metrics.alert_triggered:
                self._trigger_alert(
                    cohort_type=metrics.cohort_type,
                    cohort_value=metrics.cohort_value,
                    alert_type='stopped_working',
                    previous_roi=5.0,  # Was profitable
                    current_roi=metrics.roi,
                    severity='high' if metrics.roi < -10 else 'medium',
                    message=f"Cohort {metrics.cohort_value} has stopped working. ROI: {metrics.roi:.1f}%"
                )
                metrics.alert_triggered = True
                metrics.alert_reason = 'stopped_working'

        # Check if unprofitable cohort started working
        if metrics.is_significant and metrics.roi > 10:
            if metrics.alert_triggered and metrics.alert_reason == 'stopped_working':
                self._trigger_alert(
                    cohort_type=metrics.cohort_type,
                    cohort_value=metrics.cohort_value,
                    alert_type='started_working',
                    previous_roi=metrics.roi - 15,
                    current_roi=metrics.roi,
                    severity='medium',
                    message=f"Cohort {metrics.cohort_value} has started working again. ROI: {metrics.roi:.1f}%"
                )
                metrics.alert_triggered = False
                metrics.alert_reason = ''

    def _trigger_alert(
        self,
        cohort_type: str,
        cohort_value: str,
        alert_type: str,
        previous_roi: float,
        current_roi: float,
        severity: str,
        message: str
    ):
        """Create and store an alert."""
        self._alert_counter += 1
        alert = CohortAlert(
            alert_id=f"alert_{self._alert_counter:04d}",
            timestamp=datetime.now(),
            cohort_type=cohort_type,
            cohort_value=cohort_value,
            alert_type=alert_type,
            previous_roi=previous_roi,
            current_roi=current_roi,
            severity=severity,
            message=message
        )
        self.alerts.append(alert)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive cohort analysis report."""
        report = {
            'summary': {
                'total_bets': len(self.bets),
                'total_cohorts': sum(len(v) for v in self.cohort_metrics.values()),
                'active_alerts': len([a for a in self.alerts if not a.alert_type == 'resolved']),
            },
            'by_cohort_type': {},
            'top_performers': [],
            'worst_performers': [],
            'alerts': [a.to_dict() for a in self.alerts[-10:]],  # Last 10 alerts
        }

        all_cohorts = []

        for cohort_type, cohorts in self.cohort_metrics.items():
            type_report = {
                'cohorts': [],
                'best': None,
                'worst': None,
            }

            best_roi = float('-inf')
            worst_roi = float('inf')

            for cohort_value, metrics in cohorts.items():
                cohort_dict = metrics.to_dict()
                type_report['cohorts'].append(cohort_dict)
                all_cohorts.append(cohort_dict)

                if metrics.roi > best_roi and metrics.total_bets >= self.min_bets:
                    best_roi = metrics.roi
                    type_report['best'] = cohort_dict

                if metrics.roi < worst_roi and metrics.total_bets >= self.min_bets:
                    worst_roi = metrics.roi
                    type_report['worst'] = cohort_dict

            report['by_cohort_type'][cohort_type] = type_report

        # Sort all cohorts by ROI
        all_cohorts_sorted = sorted(
            [c for c in all_cohorts if c['total_bets'] >= self.min_bets],
            key=lambda x: x['roi'],
            reverse=True
        )

        report['top_performers'] = all_cohorts_sorted[:5]
        report['worst_performers'] = all_cohorts_sorted[-5:]

        return report

    def get_cohort_performance(
        self,
        cohort_type: str,
        cohort_value: str
    ) -> Optional[CohortMetrics]:
        """Get metrics for specific cohort."""
        if cohort_type in self.cohort_metrics:
            return self.cohort_metrics[cohort_type].get(cohort_value)
        return None

    def export_to_json(self, filepath: str):
        """Export report to JSON file."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)


class MarketEfficiencyTracker:
    """
    Track market efficiency by market type and sportsbook.

    Alert when a market gets soft (edge > 5%)
    """

    def __init__(self):
        self.market_edges: Dict[str, List[float]] = defaultdict(list)
        self.sportsbook_edges: Dict[str, List[float]] = defaultdict(list)
        self.alerts: List[Dict[str, Any]] = []
        self.soft_market_threshold = 0.05  # 5% edge

    def add_observation(
        self,
        market_type: str,
        sportsbook: str,
        edge: float,
        timestamp: Optional[datetime] = None
    ):
        """Add edge observation."""
        self.market_edges[market_type].append(edge)
        self.sportsbook_edges[sportsbook].append(edge)

        # Check for soft market
        if abs(edge) > self.soft_market_threshold:
            self._trigger_soft_market_alert(market_type, sportsbook, edge, timestamp)

    def _trigger_soft_market_alert(
        self,
        market_type: str,
        sportsbook: str,
        edge: float,
        timestamp: Optional[datetime] = None
    ):
        """Alert when market is soft."""
        alert = {
            'alert_id': f"soft_market_{len(self.alerts):04d}",
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'market_type': market_type,
            'sportsbook': sportsbook,
            'edge': edge,
            'severity': 'high' if abs(edge) > 0.10 else 'medium',
            'message': f"Soft market detected: {market_type} at {sportsbook} with {edge:.1%} edge"
        }
        self.alerts.append(alert)

    def get_market_efficiency(self, market_type: str) -> Dict[str, Any]:
        """Get efficiency metrics for a market."""
        edges = self.market_edges.get(market_type, [])

        if not edges:
            return {'market_type': market_type, 'observations': 0}

        edges = np.array(edges)
        return {
            'market_type': market_type,
            'observations': len(edges),
            'mean_edge': float(np.mean(edges)),
            'std_edge': float(np.std(edges)),
            'max_edge': float(np.max(edges)),
            'min_edge': float(np.min(edges)),
            'positive_edge_pct': float(np.mean(edges > 0) * 100),
            'soft_market_count': int(np.sum(np.abs(edges) > self.soft_market_threshold)),
            'is_efficient': np.mean(np.abs(edges)) < 0.03,  # Market is efficient if avg edge < 3%
        }

    def get_sportsbook_efficiency(self, sportsbook: str) -> Dict[str, Any]:
        """Get efficiency metrics for a sportsbook."""
        edges = self.sportsbook_edges.get(sportsbook, [])

        if not edges:
            return {'sportsbook': sportsbook, 'observations': 0}

        edges = np.array(edges)
        return {
            'sportsbook': sportsbook,
            'observations': len(edges),
            'mean_edge': float(np.mean(edges)),
            'std_edge': float(np.std(edges)),
            'max_edge': float(np.max(edges)),
            'min_edge': float(np.min(edges)),
            'is_soft': np.mean(np.abs(edges)) > self.soft_market_threshold,
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate market efficiency report."""
        return {
            'markets': {m: self.get_market_efficiency(m) for m in self.market_edges.keys()},
            'sportsbooks': {s: self.get_sportsbook_efficiency(s) for s in self.sportsbook_edges.keys()},
            'alerts': self.alerts[-20:],  # Last 20 alerts
        }
