"""Performance track record system for investors/partners.

Verified track record with third-party audit capability.
Monthly reports: win rate, ROI, Sharpe, max drawdown.
Build: 12+ month audited track record.
Output: track_record/{month}_report.json
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_RECORD_DIR = str(PROJECT_ROOT / "track_record")


@dataclass
class MonthlyPerformance:
    """Monthly performance metrics for track record."""
    month: str  # YYYY-MM format
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
    calmar_ratio: float = 0.0
    starting_bankroll: float = 0.0
    ending_bankroll: float = 0.0
    best_bet: float = 0.0
    worst_bet: float = 0.0
    avg_bet_size: float = 0.0
    avg_edge: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    bets_by_sport: Dict[str, int] = field(default_factory=dict)
    bets_by_market: Dict[str, int] = field(default_factory=dict)
    verified: bool = False
    verification_source: str = ""
    verification_url: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'month': self.month,
            'metrics': {
                'total_bets': self.total_bets,
                'record': f"{self.wins}-{self.losses}",
                'win_rate': round(self.win_rate, 4),
                'total_pnl': round(self.total_pnl, 2),
                'total_wagered': round(self.total_wagered, 2),
                'roi': round(self.roi, 2),
                'sharpe_ratio': round(self.sharpe_ratio, 3),
                'max_drawdown': round(self.max_drawdown, 2),
                'calmar_ratio': round(self.calmar_ratio, 3),
            },
            'bankroll': {
                'starting': round(self.starting_bankroll, 2),
                'ending': round(self.ending_bankroll, 2),
                'return_pct': round((self.ending_bankroll / self.starting_bankroll - 1) * 100, 2) if self.starting_bankroll > 0 else 0,
            },
            'betting_stats': {
                'avg_bet_size': round(self.avg_bet_size, 2),
                'avg_edge': round(self.avg_edge, 4),
                'best_bet': round(self.best_bet, 2),
                'worst_bet': round(self.worst_bet, 2),
            },
            'breakdown': {
                'by_sport': self.bets_by_sport,
                'by_market': self.bets_by_market,
            },
            'verification': {
                'verified': self.verified,
                'source': self.verification_source,
                'url': self.verification_url,
            },
            'notes': self.notes,
        }


@dataclass
class TrackRecordReport:
    """Complete track record report for investors."""
    generated_at: str
    report_type: str  # 'monthly', 'quarterly', 'annual', 'lifetime'
    periods: List[MonthlyPerformance] = field(default_factory=list)
    lifetime_metrics: Dict[str, Any] = field(default_factory=dict)
    audit_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': {
                'generated_at': self.generated_at,
                'report_type': self.report_type,
            },
            'periods': [p.to_dict() for p in self.periods],
            'lifetime_metrics': self.lifetime_metrics,
            'audit_info': self.audit_info,
        }


class TrackRecordManager:
    """
    Manage performance track record for investors/partners.

    Usage:
        manager = TrackRecordManager()
        manager.add_monthly_performance(performance)
        report = manager.generate_report('annual')
        manager.save_report(report)
    """

    def __init__(self, track_record_dir: str = TRACK_RECORD_DIR):
        self.track_record_dir = Path(track_record_dir)
        self.track_record_dir.mkdir(parents=True, exist_ok=True)
        self.monthly_records: Dict[str, MonthlyPerformance] = {}
        self._load_existing_records()

    def _load_existing_records(self):
        """Load existing track record files."""
        for file_path in self.track_record_dir.glob("*_report.json"):
            if file_path.name.startswith("lifetime"):
                continue

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                month = data.get('month', '')
                if month:
                    self.monthly_records[month] = self._dict_to_performance(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def _dict_to_performance(self, data: Dict[str, Any]) -> MonthlyPerformance:
        """Convert dictionary to MonthlyPerformance."""
        return MonthlyPerformance(
            month=data.get('month', ''),
            total_bets=data.get('total_bets', 0),
            wins=data.get('wins', 0),
            losses=data.get('losses', 0),
            pushes=data.get('pushes', 0),
            win_rate=data.get('win_rate', 0.0),
            total_pnl=data.get('total_pnl', 0.0),
            total_wagered=data.get('total_wagered', 0.0),
            roi=data.get('roi', 0.0),
            sharpe_ratio=data.get('sharpe_ratio', 0.0),
            max_drawdown=data.get('max_drawdown', 0.0),
            calmar_ratio=data.get('calmar_ratio', 0.0),
            starting_bankroll=data.get('starting_bankroll', 0.0),
            ending_bankroll=data.get('ending_bankroll', 0.0),
            best_bet=data.get('best_bet', 0.0),
            worst_bet=data.get('worst_bet', 0.0),
            avg_bet_size=data.get('avg_bet_size', 0.0),
            avg_edge=data.get('avg_edge', 0.0),
            daily_returns=data.get('daily_returns', []),
            bets_by_sport=data.get('bets_by_sport', {}),
            bets_by_market=data.get('bets_by_market', {}),
            verified=data.get('verified', False),
            verification_source=data.get('verification_source', ''),
            verification_url=data.get('verification_url', ''),
            notes=data.get('notes', ''),
        )

    def add_monthly_performance(self, performance: MonthlyPerformance):
        """Add monthly performance record."""
        self.monthly_records[performance.month] = performance
        self._save_monthly_report(performance)

    def _save_monthly_report(self, performance: MonthlyPerformance):
        """Save monthly performance to file."""
        file_path = self.track_record_dir / f"{performance.month}_report.json"

        with open(file_path, 'w') as f:
            json.dump(performance.to_dict(), f, indent=2)

    def verify_performance(
        self,
        month: str,
        source: str,
        url: str = ""
    ) -> bool:
        """
        Mark performance as verified by third party.

        Args:
            month: Month in YYYY-MM format
            source: Verification source (e.g., 'BetStamp', 'Pikkit')
            url: URL to verification page

        Returns:
            True if verification was successful
        """
        if month not in self.monthly_records:
            return False

        record = self.monthly_records[month]
        record.verified = True
        record.verification_source = source
        record.verification_url = url

        self._save_monthly_report(record)
        return True

    def calculate_lifetime_metrics(self) -> Dict[str, Any]:
        """Calculate lifetime track record metrics."""
        if not self.monthly_records:
            return {}

        sorted_months = sorted(self.monthly_records.keys())

        total_bets = sum(r.total_bets for r in self.monthly_records.values())
        total_wins = sum(r.wins for r in self.monthly_records.values())
        total_losses = sum(r.losses for r in self.monthly_records.values())
        total_pnl = sum(r.total_pnl for r in self.monthly_records.values())
        total_wagered = sum(r.total_wagered for r in self.monthly_records.values())

        # Calculate compound return
        if sorted_months:
            first_record = self.monthly_records[sorted_months[0]]
            last_record = self.monthly_records[sorted_months[-1]]
            compound_return = (last_record.ending_bankroll / first_record.starting_bankroll - 1) * 100 if first_record.starting_bankroll > 0 else 0
        else:
            compound_return = 0

        # Calculate lifetime Sharpe (annualized)
        all_daily_returns = []
        for record in self.monthly_records.values():
            all_daily_returns.extend(record.daily_returns)

        if len(all_daily_returns) > 1:
            sharpe = self._calculate_sharpe(all_daily_returns)
        else:
            sharpe = 0

        # Calculate max drawdown across all periods
        max_dd = max(r.max_drawdown for r in self.monthly_records.values()) if self.monthly_records else 0

        # Calculate Calmar ratio
        annualized_return = compound_return / len(sorted_months) * 12 if sorted_months else 0
        calmar = annualized_return / max_dd if max_dd > 0 else 0

        decisive_bets = total_wins + total_losses
        lifetime_win_rate = total_wins / decisive_bets if decisive_bets > 0 else 0

        return {
            'total_months': len(sorted_months),
            'total_bets': total_bets,
            'record': f"{total_wins}-{total_losses}",
            'win_rate': round(lifetime_win_rate, 4),
            'total_pnl': round(total_pnl, 2),
            'total_wagered': round(total_wagered, 2),
            'roi': round(total_pnl / total_wagered * 100, 2) if total_wagered > 0 else 0,
            'compound_return_pct': round(compound_return, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(max_dd, 2),
            'calmar_ratio': round(calmar, 3),
            'avg_monthly_roi': round(total_pnl / len(sorted_months), 2) if sorted_months else 0,
            'profitable_months': sum(1 for r in self.monthly_records.values() if r.total_pnl > 0),
            'best_month': max(self.monthly_records.values(), key=lambda x: x.roi).month if self.monthly_records else '',
            'worst_month': min(self.monthly_records.values(), key=lambda x: x.roi).month if self.monthly_records else '',
        }

    def _calculate_sharpe(self, daily_returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(daily_returns) < 2:
            return 0.0

        returns = np.array(daily_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * 252
        annualized_std = std_return * np.sqrt(252)

        risk_free_rate = 0.02
        sharpe = (annualized_return - risk_free_rate) / annualized_std
        return sharpe

    def generate_report(self, report_type: str = 'lifetime') -> TrackRecordReport:
        """
        Generate track record report.

        Args:
            report_type: 'monthly', 'quarterly', 'annual', 'lifetime'

        Returns:
            TrackRecordReport object
        """
        now = datetime.now().isoformat()

        if report_type == 'lifetime':
            periods = [self.monthly_records[m] for m in sorted(self.monthly_records.keys())]
            lifetime_metrics = self.calculate_lifetime_metrics()
        elif report_type == 'annual':
            current_year = datetime.now().year
            periods = [r for m, r in self.monthly_records.items() if m.startswith(str(current_year))]
            lifetime_metrics = self._calculate_period_metrics(periods)
        elif report_type == 'quarterly':
            current_month = datetime.now().month
            current_quarter = (current_month - 1) // 3
            periods = []
            for m, r in self.monthly_records.items():
                month_num = int(m.split('-')[1])
                if (month_num - 1) // 3 == current_quarter:
                    periods.append(r)
            lifetime_metrics = self._calculate_period_metrics(periods)
        else:  # monthly
            current_month = datetime.now().strftime('%Y-%m')
            periods = [self.monthly_records[current_month]] if current_month in self.monthly_records else []
            lifetime_metrics = self._calculate_period_metrics(periods)

        audit_info = {
            'verified_months': sum(1 for p in periods if p.verified),
            'verification_sources': list(set(p.verification_source for p in periods if p.verified)),
            'data_completeness': len(periods) / max(1, len(self.monthly_records)) * 100,
        }

        return TrackRecordReport(
            generated_at=now,
            report_type=report_type,
            periods=periods,
            lifetime_metrics=lifetime_metrics,
            audit_info=audit_info,
        )

    def _calculate_period_metrics(self, periods: List[MonthlyPerformance]) -> Dict[str, Any]:
        """Calculate metrics for a subset of periods."""
        if not periods:
            return {}

        total_bets = sum(p.total_bets for p in periods)
        total_wins = sum(p.wins for p in periods)
        total_losses = sum(p.losses for p in periods)
        total_pnl = sum(p.total_pnl for p in periods)
        total_wagered = sum(p.total_wagered for p in periods)

        decisive = total_wins + total_losses
        win_rate = total_wins / decisive if decisive > 0 else 0

        return {
            'periods': len(periods),
            'total_bets': total_bets,
            'win_rate': round(win_rate, 4),
            'total_pnl': round(total_pnl, 2),
            'roi': round(total_pnl / total_wagered * 100, 2) if total_wagered > 0 else 0,
        }

    def save_report(self, report: TrackRecordReport):
        """Save report to file."""
        filename = f"{report.report_type}_{datetime.now().strftime('%Y%m%d')}_report.json"
        file_path = self.track_record_dir / filename

        with open(file_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"Report saved to: {file_path}")

    def get_verification_status(self) -> Dict[str, Any]:
        """Get verification status summary."""
        total = len(self.monthly_records)
        verified = sum(1 for r in self.monthly_records.values() if r.verified)

        sources = {}
        for r in self.monthly_records.values():
            if r.verified and r.verification_source:
                sources[r.verification_source] = sources.get(r.verification_source, 0) + 1

        return {
            'total_months': total,
            'verified_months': verified,
            'verification_pct': round(verified / total * 100, 1) if total > 0 else 0,
            'verification_sources': sources,
            'audit_ready': verified >= total * 0.8,  # 80% verified = audit ready
        }


def create_monthly_performance_from_bets(
    month: str,
    bets: List[Dict[str, Any]],
    starting_bankroll: float = 1000.0
) -> MonthlyPerformance:
    """
    Create MonthlyPerformance from list of bets.

    Args:
        month: Month in YYYY-MM format
        bets: List of bet dictionaries with pnl, edge, etc.
        starting_bankroll: Starting bankroll for the month

    Returns:
        MonthlyPerformance object
    """
    if not bets:
        return MonthlyPerformance(month=month, starting_bankroll=starting_bankroll)

    wins = sum(1 for b in bets if b.get('outcome') == 'win')
    losses = sum(1 for b in bets if b.get('outcome') == 'loss')
    pushes = sum(1 for b in bets if b.get('outcome') == 'push')
    total_pnl = sum(b.get('pnl', 0) for b in bets)
    total_wagered = sum(b.get('bet_size', 0) for b in bets)

    decisive = wins + losses
    win_rate = wins / decisive if decisive > 0 else 0
    roi = total_pnl / total_wagered * 100 if total_wagered > 0 else 0

    # Calculate daily returns
    daily_pnl = {}
    for bet in bets:
        date = bet.get('date', '')[:10]  # YYYY-MM-DD
        if date:
            daily_pnl[date] = daily_pnl.get(date, 0) + bet.get('pnl', 0)

    daily_returns = []
    bankroll = starting_bankroll
    for date in sorted(daily_pnl.keys()):
        ret = daily_pnl[date] / bankroll
        daily_returns.append(ret)
        bankroll += daily_pnl[date]

    # Calculate Sharpe
    if len(daily_returns) > 1:
        sharpe = float(np.mean(daily_returns) * 252 / (np.std(daily_returns, ddof=1) * np.sqrt(252))) if np.std(daily_returns) > 0 else 0
    else:
        sharpe = 0

    # Calculate max drawdown
    peak = starting_bankroll
    max_dd = 0
    for ret in daily_returns:
        bankroll *= (1 + ret)
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak
        max_dd = max(max_dd, dd)

    ending_bankroll = starting_bankroll + total_pnl

    # Bets by sport/market
    bets_by_sport = {}
    bets_by_market = {}
    for bet in bets:
        sport = bet.get('sport', 'unknown')
        market = bet.get('market', 'unknown')
        bets_by_sport[sport] = bets_by_sport.get(sport, 0) + 1
        bets_by_market[market] = bets_by_market.get(market, 0) + 1

    pnls = [b.get('pnl', 0) for b in bets]
    bet_sizes = [b.get('bet_size', 0) for b in bets]
    edges = [b.get('edge', 0) for b in bets]

    return MonthlyPerformance(
        month=month,
        total_bets=len(bets),
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_wagered=total_wagered,
        roi=roi,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd * 100,
        calmar_ratio=(roi / 12) / max_dd if max_dd > 0 else 0,
        starting_bankroll=starting_bankroll,
        ending_bankroll=ending_bankroll,
        best_bet=max(pnls) if pnls else 0,
        worst_bet=min(pnls) if pnls else 0,
        avg_bet_size=np.mean(bet_sizes) if bet_sizes else 0,
        avg_edge=np.mean(edges) if edges else 0,
        daily_returns=daily_returns,
        bets_by_sport=bets_by_sport,
        bets_by_market=bets_by_market,
    )
