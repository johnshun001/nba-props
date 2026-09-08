"""Edge Decay Monitoring for NBA props.

Track: when edges start closing (market learning)
Alert: when a market gets too efficient
Pivot: to new markets before edge disappears
Output: edge_decay_alerts.json
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
EDGE_DECAY_DIR = str(PROJECT_ROOT / "edge_decay")


class DecayStatus(Enum):
    """Edge decay status."""
    GROWING = "growing"  # Edge increasing
    STABLE = "stable"  # Edge stable
    DECLINING = "declining"  # Edge declining
    CRITICAL = "critical"  # Edge collapsing


@dataclass
class EdgeDecayMetrics:
    """Metrics for tracking edge decay."""
    market_name: str
    current_edge: float = 0.0
    peak_edge: float = 0.0
    edge_30d_avg: float = 0.0
    edge_7d_avg: float = 0.0
    decay_rate: float = 0.0  # Rate of edge decline
    half_life: float = 0.0  # Days until edge halves at current rate
    status: str = "stable"
    days_tracked: int = 0
    alert_threshold: float = 0.5  # Alert when edge drops below this % of peak

    def to_dict(self) -> Dict[str, Any]:
        return {
            'market_name': self.market_name,
            'current_edge': round(self.current_edge, 4),
            'peak_edge': round(self.peak_edge, 4),
            'edge_30d_avg': round(self.edge_30d_avg, 4),
            'edge_7d_avg': round(self.edge_7d_avg, 4),
            'decay_rate': round(self.decay_rate, 4),
            'half_life_days': round(self.half_life, 1),
            'status': self.status,
            'days_tracked': self.days_tracked,
            'edge_vs_peak_pct': round((self.current_edge / max(0.001, self.peak_edge)) * 100, 1),
        }


@dataclass
class EdgeDecayAlert:
    """Alert for edge decay."""
    alert_id: str
    timestamp: str
    market_name: str
    alert_type: str  # decay_detected, critical_level, market_efficient
    severity: str  # low, medium, high, critical
    current_edge: float
    peak_edge: float
    decay_rate: float
    message: str
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp,
            'market_name': self.market_name,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'metrics': {
                'current_edge': round(self.current_edge, 4),
                'peak_edge': round(self.peak_edge, 4),
                'decay_rate': round(self.decay_rate, 4),
            },
            'message': self.message,
            'recommended_action': self.recommended_action,
        }


class EdgeDecayMonitor:
    """
    Monitor edge decay across NBA prop markets.

    Usage:
        monitor = EdgeDecayMonitor()
        monitor.update_edge(market, edge_value)
        alerts = monitor.check_for_decay()
        monitor.generate_report()
    """

    def __init__(self, decay_dir: str = EDGE_DECAY_DIR):
        self.decay_dir = Path(decay_dir)
        self.decay_dir.mkdir(parents=True, exist_ok=True)

        self.edge_history: Dict[str, List[Dict[str, Any]]] = {}
        self.metrics: Dict[str, EdgeDecayMetrics] = {}
        self.alerts: List[EdgeDecayAlert] = []
        self._alert_counter = 0

        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing edge decay data."""
        data_file = self.decay_dir / "edge_decay_data.json"
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                self.edge_history = data.get('edge_history', {})
                self.alerts = [
                    EdgeDecayAlert(**a) for a in data.get('alerts', [])
                ]
                self._alert_counter = len(self.alerts)
            except Exception as e:
                print(f"Error loading edge decay data: {e}")

    def update_edge(
        self,
        market_name: str,
        edge_value: float,
        edge_details: Optional[Dict[str, Any]] = None
    ):
        """
        Update edge value for a market.

        Args:
            market_name: Name of market (e.g., "player_points_draftkings")
            edge_value: Current edge value (as decimal, e.g., 0.05 for 5%)
            edge_details: Optional additional details
        """
        now = datetime.now()

        # Initialize market if new
        if market_name not in self.edge_history:
            self.edge_history[market_name] = []
            self.metrics[market_name] = EdgeDecayMetrics(market_name=market_name)

        # Add edge observation
        observation = {
            'timestamp': now.isoformat(),
            'edge': edge_value,
            'details': edge_details or {},
        }
        self.edge_history[market_name].append(observation)

        # Keep only last 90 days of data
        cutoff = now - timedelta(days=90)
        self.edge_history[market_name] = [
            o for o in self.edge_history[market_name]
            if datetime.fromisoformat(o['timestamp']) > cutoff
        ]

        # Update metrics
        self._update_metrics(market_name)

        # Check for decay alerts
        self._check_decay(market_name)

        # Save data
        self._save_data()

    def _update_metrics(self, market_name: str):
        """Update decay metrics for a market."""
        history = self.edge_history[market_name]
        metrics = self.metrics[market_name]

        if not history:
            return

        metrics.days_tracked = len(history)

        # Current edge (most recent)
        metrics.current_edge = history[-1]['edge']

        # Peak edge
        all_edges = [o['edge'] for o in history]
        metrics.peak_edge = max(all_edges)

        # 30-day average
        now = datetime.now()
        cutoff_30d = now - timedelta(days=30)
        edges_30d = [
            o['edge'] for o in history
            if datetime.fromisoformat(o['timestamp']) > cutoff_30d
        ]
        metrics.edge_30d_avg = np.mean(edges_30d) if edges_30d else 0

        # 7-day average
        cutoff_7d = now - timedelta(days=7)
        edges_7d = [
            o['edge'] for o in history
            if datetime.fromisoformat(o['timestamp']) > cutoff_7d
        ]
        metrics.edge_7d_avg = np.mean(edges_7d) if edges_7d else 0

        # Calculate decay rate (edge change per day)
        if len(history) >= 7:
            early_edges = [o['edge'] for o in history[:7]]
            recent_edges = [o['edge'] for o in history[-7:]]
            early_avg = np.mean(early_edges)
            recent_avg = np.mean(recent_edges)
            days_between = len(history)
            if days_between > 0:
                metrics.decay_rate = (recent_avg - early_avg) / days_between

        # Calculate half-life
        if metrics.decay_rate < 0 and metrics.current_edge > 0:
            metrics.half_life = abs(metrics.current_edge / metrics.decay_rate)
        else:
            metrics.half_life = float('inf')

        # Determine status
        edge_vs_peak = metrics.current_edge / max(0.001, metrics.peak_edge)

        if metrics.decay_rate > 0.001:
            metrics.status = DecayStatus.GROWING.value
        elif metrics.decay_rate < -0.001:
            if edge_vs_peak < 0.3:
                metrics.status = DecayStatus.CRITICAL.value
            else:
                metrics.status = DecayStatus.DECLINING.value
        else:
            metrics.status = DecayStatus.STABLE.value

    def _check_decay(self, market_name: str):
        """Check for decay alerts."""
        metrics = self.metrics[market_name]

        edge_vs_peak = metrics.current_edge / max(0.001, metrics.peak_edge)

        # Critical: edge dropped below 30% of peak
        if edge_vs_peak < 0.3 and metrics.status == DecayStatus.CRITICAL.value:
            self._create_alert(
                market_name=market_name,
                alert_type='critical_level',
                severity='critical',
                message=f"Edge for {market_name} has dropped to {edge_vs_peak*100:.1f}% of peak value",
                recommended_action='Immediately reduce allocation or exit market',
            )

        # High: edge declining rapidly
        elif metrics.decay_rate < -0.005:
            self._create_alert(
                market_name=market_name,
                alert_type='decay_detected',
                severity='high',
                message=f"Rapid edge decay detected: {metrics.decay_rate*1000:.2f}% per day",
                recommended_action='Reduce bet sizes, prepare to pivot',
            )

        # Medium: edge below threshold
        elif edge_vs_peak < metrics.alert_threshold:
            self._create_alert(
                market_name=market_name,
                alert_type='market_efficient',
                severity='medium',
                message=f"Market efficiency increasing. Edge at {edge_vs_peak*100:.1f}% of peak",
                recommended_action='Monitor closely, consider diversifying',
            )

    def _create_alert(
        self,
        market_name: str,
        alert_type: str,
        severity: str,
        message: str,
        recommended_action: str
    ):
        """Create a new alert."""
        # Check if we already have a recent similar alert
        now = datetime.now()
        for alert in self.alerts[-10:]:
            if (alert.market_name == market_name and
                alert.alert_type == alert_type and
                datetime.fromisoformat(alert.timestamp) > now - timedelta(hours=24)):
                return  # Don't duplicate alerts within 24 hours

        self._alert_counter += 1
        metrics = self.metrics.get(market_name, EdgeDecayMetrics(market_name=market_name))

        alert = EdgeDecayAlert(
            alert_id=f"decay_{self._alert_counter:04d}",
            timestamp=now.isoformat(),
            market_name=market_name,
            alert_type=alert_type,
            severity=severity,
            current_edge=metrics.current_edge,
            peak_edge=metrics.peak_edge,
            decay_rate=metrics.decay_rate,
            message=message,
            recommended_action=recommended_action,
        )

        self.alerts.append(alert)

    def _save_data(self):
        """Save edge decay data."""
        data_file = self.decay_dir / "edge_decay_data.json"
        data = {
            'edge_history': {
                k: v[-50:] for k, v in self.edge_history.items()  # Keep last 50 per market
            },
            'alerts': [a.to_dict() for a in self.alerts[-100:]],
            'last_updated': datetime.now().isoformat(),
        }
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def check_for_decay(self) -> List[EdgeDecayAlert]:
        """Check all markets for decay and return new alerts."""
        new_alerts = []

        for market_name in self.metrics.keys():
            prev_alert_count = len(self.alerts)
            self._check_decay(market_name)
            if len(self.alerts) > prev_alert_count:
                new_alerts.append(self.alerts[-1])

        return new_alerts

    def get_decay_summary(self) -> Dict[str, Any]:
        """Get summary of edge decay across all markets."""
        summary = {
            'total_markets_tracked': len(self.metrics),
            'markets_by_status': {},
            'critical_markets': [],
            'declining_markets': [],
            'healthy_markets': [],
            'recent_alerts': [],
        }

        for market_name, metrics in self.metrics.items():
            status = metrics.status

            if status not in summary['markets_by_status']:
                summary['markets_by_status'][status] = 0
            summary['markets_by_status'][status] += 1

            market_data = {
                'name': market_name,
                'current_edge': round(metrics.current_edge, 4),
                'vs_peak_pct': round((metrics.current_edge / max(0.001, metrics.peak_edge)) * 100, 1),
                'decay_rate': round(metrics.decay_rate * 1000, 3),
            }

            if status == DecayStatus.CRITICAL.value:
                summary['critical_markets'].append(market_data)
            elif status == DecayStatus.DECLINING.value:
                summary['declining_markets'].append(market_data)
            elif status == DecayStatus.GROWING.value:
                summary['healthy_markets'].append(market_data)

        # Recent alerts
        summary['recent_alerts'] = [
            a.to_dict() for a in self.alerts[-10:]
        ]

        return summary

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive edge decay report."""
        summary = self.get_decay_summary()

        report = {
            'report_date': datetime.now().isoformat(),
            'summary': summary,
            'market_details': {
                k: v.to_dict() for k, v in self.metrics.items()
            },
            'action_items': self._generate_action_items(),
        }

        return report

    def _generate_action_items(self) -> List[Dict[str, str]]:
        """Generate actionable items based on decay analysis."""
        action_items = []

        # Critical markets need immediate action
        for market in self.metrics.values():
            if market.status == DecayStatus.CRITICAL.value:
                action_items.append({
                    'priority': 'critical',
                    'market': market.market_name,
                    'action': f"Exit or significantly reduce exposure to {market.market_name}",
                    'reason': f"Edge at {market.current_edge*100:.2f}%, down from peak of {market.peak_edge*100:.2f}%",
                })

        # Declining markets need monitoring
        for market in self.metrics.values():
            if market.status == DecayStatus.DECLINING.value:
                action_items.append({
                    'priority': 'high',
                    'market': market.market_name,
                    'action': f"Reduce bet sizes for {market.market_name} by 50%",
                    'reason': f"Edge declining at {market.decay_rate*1000:.2f}% per day",
                })

        # Growing markets - opportunity
        for market in self.metrics.values():
            if market.status == DecayStatus.GROWING.value:
                action_items.append({
                    'priority': 'opportunity',
                    'market': market.market_name,
                    'action': f"Consider increasing allocation to {market.market_name}",
                    'reason': f"Edge growing, currently at {market.current_edge*100:.2f}%",
                })

        return action_items

    def save_report(self):
        """Save edge decay report to file."""
        report = self.generate_report()
        filename = f"edge_decay_{datetime.now().strftime('%Y%m%d')}.json"
        file_path = self.decay_dir / filename

        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Edge decay report saved to: {file_path}")

        # Also save alerts separately
        alerts_file = self.decay_dir / "edge_decay_alerts.json"
        with open(alerts_file, 'w') as f:
            json.dump([a.to_dict() for a in self.alerts], f, indent=2)
