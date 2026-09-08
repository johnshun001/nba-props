"""Signal distribution system for monetization.

Package best bets as subscription service.
Delivery: Discord, Telegram, or private website.
Pricing: $50-200/month depending on edge.
Output: signals/{date}_picks.json
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNALS_DIR = str(PROJECT_ROOT / "signals")


class SignalTier(Enum):
    """Subscription tiers for signal distribution."""
    FREE = "free"  # Basic picks, delayed
    STANDARD = "standard"  # All picks, real-time
    PREMIUM = "premium"  # All picks + analysis + chat
    VIP = "vip"  # Everything + 1-on-1 consultation


class DeliveryChannel(Enum):
    """Delivery channels for signals."""
    DISCORD = "discord"
    TELEGRAM = "telegram"
    EMAIL = "email"
    WEBSITE = "website"
    SMS = "sms"


@dataclass
class SignalPick:
    """Individual signal pick for distribution."""
    signal_id: str
    sport: str
    league: str
    event_date: str
    event_time: str
    team_a: str
    team_b: str
    market: str
    selection: str  # e.g., "Lakers -5.5", "Over 225.5"
    odds: float
    stake_pct: float  # Percentage of bankroll to bet
    edge_pct: float  # Model edge percentage
    confidence: str  # "low", "medium", "high", "max"
    analysis: str  # Brief explanation
    status: str = "pending"  # pending, active, won, lost, pushed
    result: Optional[float] = None
    posted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    tier_access: List[str] = field(default_factory=lambda: ["standard", "premium", "vip"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'signal_id': self.signal_id,
            'event': {
                'sport': self.sport,
                'league': self.league,
                'date': self.event_date,
                'time': self.event_time,
                'matchup': f"{self.team_a} @ {self.team_b}",
            },
            'pick': {
                'market': self.market,
                'selection': self.selection,
                'odds': self.odds,
                'stake_pct': self.stake_pct,
            },
            'analysis': {
                'edge_pct': round(self.edge_pct, 2),
                'confidence': self.confidence,
                'reasoning': self.analysis,
            },
            'status': self.status,
            'result': self.result,
            'timing': {
                'posted_at': self.posted_at,
                'expires_at': self.expires_at,
            },
            'access': {
                'tiers': self.tier_access,
            },
        }


@dataclass
class DailyPicks:
    """Collection of daily picks for distribution."""
    date: str
    picks: List[SignalPick] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    performance_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'date': self.date,
            'picks': [p.to_dict() for p in self.picks],
            'summary': self.summary,
            'performance_note': self.performance_note,
            'metadata': {
                'total_picks': len(self.picks),
                'by_confidence': self._count_by_confidence(),
                'by_tier': self._count_by_tier(),
            },
        }

    def _count_by_confidence(self) -> Dict[str, int]:
        """Count picks by confidence level."""
        counts = {}
        for pick in self.picks:
            counts[pick.confidence] = counts.get(pick.confidence, 0) + 1
        return counts

    def _count_by_tier(self) -> Dict[str, int]:
        """Count picks accessible by tier."""
        counts = {'free': 0, 'standard': 0, 'premium': 0, 'vip': 0}
        for pick in self.picks:
            for tier in pick.tier_access:
                counts[tier] = counts.get(tier, 0) + 1
        return counts


class SignalDistributor:
    """
    Distribute betting signals to subscribers.

    Usage:
        distributor = SignalDistributor()
        pick = distributor.create_pick(...)
        distributor.publish_pick(pick)
        distributor.settle_pick(signal_id, result)
    """

    def __init__(self, signals_dir: str = SIGNALS_DIR):
        self.signals_dir = Path(signals_dir)
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.pending_picks: Dict[str, SignalPick] = {}
        self.settled_picks: Dict[str, SignalPick] = {}
        self._load_existing_picks()

    def _load_existing_picks(self):
        """Load existing signal files."""
        for file_path in self.signals_dir.glob("*_picks.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                for pick_data in data.get('picks', []):
                    pick = self._dict_to_pick(pick_data)
                    if pick.status == 'pending':
                        self.pending_picks[pick.signal_id] = pick
                    else:
                        self.settled_picks[pick.signal_id] = pick
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def _dict_to_pick(self, data: Dict[str, Any]) -> SignalPick:
        """Convert dictionary to SignalPick."""
        return SignalPick(
            signal_id=data.get('signal_id', ''),
            sport=data.get('event', {}).get('sport', ''),
            league=data.get('event', {}).get('league', ''),
            event_date=data.get('event', {}).get('date', ''),
            event_time=data.get('event', {}).get('time', ''),
            team_a=data.get('event', {}).get('matchup', '').split(' @ ')[0] if ' @ ' in data.get('event', {}).get('matchup', '') else '',
            team_b=data.get('event', {}).get('matchup', '').split(' @ ')[1] if ' @ ' in data.get('event', {}).get('matchup', '') else '',
            market=data.get('pick', {}).get('market', ''),
            selection=data.get('pick', {}).get('selection', ''),
            odds=data.get('pick', {}).get('odds', 0),
            stake_pct=data.get('pick', {}).get('stake_pct', 0),
            edge_pct=data.get('analysis', {}).get('edge_pct', 0),
            confidence=data.get('analysis', {}).get('confidence', 'medium'),
            analysis=data.get('analysis', {}).get('reasoning', ''),
            status=data.get('status', 'pending'),
            result=data.get('result'),
            posted_at=data.get('timing', {}).get('posted_at', ''),
            expires_at=data.get('timing', {}).get('expires_at'),
            tier_access=data.get('access', {}).get('tiers', ['standard']),
        )

    def create_pick(
        self,
        sport: str,
        league: str,
        event_date: str,
        event_time: str,
        team_a: str,
        team_b: str,
        market: str,
        selection: str,
        odds: float,
        edge_pct: float,
        analysis: str,
        confidence: str = "medium",
        stake_pct: Optional[float] = None,
        tier_access: Optional[List[str]] = None,
    ) -> SignalPick:
        """
        Create a new signal pick.

        Args:
            sport: Sport name
            league: League name
            event_date: Event date (YYYY-MM-DD)
            event_time: Event time (HH:MM)
            team_a: Away team
            team_b: Home team
            market: Bet market
            selection: Specific selection
            odds: American odds
            edge_pct: Model edge percentage
            analysis: Brief explanation
            confidence: Confidence level
            stake_pct: Percentage of bankroll (auto-calculated if None)
            tier_access: Which tiers get access

        Returns:
            SignalPick object
        """
        # Auto-calculate stake based on edge (Kelly-inspired)
        if stake_pct is None:
            stake_pct = min(edge_pct / 100 * 0.5, 2.0)  # Half-Kelly, max 2%

        # Auto-set tier access based on confidence
        if tier_access is None:
            if confidence == "max":
                tier_access = ["premium", "vip"]
            elif confidence == "high":
                tier_access = ["standard", "premium", "vip"]
            else:
                tier_access = ["standard", "premium", "vip"]

        # Generate unique signal ID
        signal_data = f"{event_date}_{team_a}_{team_b}_{market}_{selection}"
        signal_id = hashlib.sha256(signal_data.encode()).hexdigest()[:12]

        # Set expiration (1 hour before event)
        try:
            event_dt = datetime.strptime(f"{event_date} {event_time}", "%Y-%m-%d %H:%M")
            expires_at = (event_dt - timedelta(hours=1)).isoformat()
        except:
            expires_at = None

        pick = SignalPick(
            signal_id=signal_id,
            sport=sport,
            league=league,
            event_date=event_date,
            event_time=event_time,
            team_a=team_a,
            team_b=team_b,
            market=market,
            selection=selection,
            odds=odds,
            stake_pct=round(stake_pct, 2),
            edge_pct=round(edge_pct, 2),
            confidence=confidence,
            analysis=analysis,
            expires_at=expires_at,
            tier_access=tier_access,
        )

        self.pending_picks[signal_id] = pick
        return pick

    def publish_pick(self, pick: SignalPick, channels: Optional[List[str]] = None):
        """
        Publish pick to distribution channels.

        Args:
            pick: SignalPick to publish
            channels: List of channels to publish to
        """
        if channels is None:
            channels = ['discord', 'telegram', 'website']

        # Save to daily picks file
        date = pick.event_date
        self._save_daily_picks(date)

        # Format message for each channel
        for channel in channels:
            message = self._format_message(pick, channel)
            # In production, would send to actual channel API
            print(f"[{channel.upper()}] {message}")

    def _format_message(self, pick: SignalPick, channel: str) -> str:
        """Format pick message for specific channel."""
        confidence_emoji = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'max': '🔴'
        }.get(pick.confidence, '⚪')

        if channel == 'discord':
            return (
                f"{confidence_emoji} **{pick.sport.upper()} PICK** {confidence_emoji}\n"
                f"📅 {pick.event_date} {pick.event_time}\n"
                f"🏟️ {pick.team_a} @ {pick.team_b}\n"
                f"📊 {pick.market}: {pick.selection}\n"
                f"💰 Odds: {pick.odds} | Stake: {pick.stake_pct}%\n"
                f"📈 Edge: {pick.edge_pct}%\n"
                f"💡 {pick.analysis}"
            )
        elif channel == 'telegram':
            return (
                f"{confidence_emoji} {pick.sport.upper()} PICK\n\n"
                f"{pick.event_date} | {pick.event_time}\n"
                f"{pick.team_a} @ {pick.team_b}\n\n"
                f"{pick.market}: {pick.selection}\n"
                f"Odds: {pick.odds} | Stake: {pick.stake_pct}%\n"
                f"Edge: {pick.edge_pct}%\n\n"
                f"_{pick.analysis}_"
            )
        else:  # website
            return f"{pick.sport}: {pick.selection} at {pick.odds} ({pick.edge_pct}% edge)"

    def _save_daily_picks(self, date: str):
        """Save all picks for a date to file."""
        # Collect all picks for date
        picks_for_date = []
        for pick in list(self.pending_picks.values()) + list(self.settled_picks.values()):
            if pick.event_date == date:
                picks_for_date.append(pick)

        if not picks_for_date:
            return

        # Create daily picks object
        daily = DailyPicks(date=date, picks=picks_for_date)
        daily.summary = {
            'total_picks': len(picks_for_date),
            'pending': sum(1 for p in picks_for_date if p.status == 'pending'),
            'settled': sum(1 for p in picks_for_date if p.status != 'pending'),
        }

        # Save to file
        file_path = self.signals_dir / f"{date}_picks.json"
        with open(file_path, 'w') as f:
            json.dump(daily.to_dict(), f, indent=2)

    def settle_pick(
        self,
        signal_id: str,
        result: float,
        status: str = "won"
    ) -> bool:
        """
        Settle a pick with result.

        Args:
            signal_id: ID of pick to settle
            result: P&L result
            status: won, lost, or pushed

        Returns:
            True if settlement was successful
        """
        if signal_id not in self.pending_picks:
            return False

        pick = self.pending_picks[signal_id]
        pick.result = result
        pick.status = status

        # Move to settled
        del self.pending_picks[signal_id]
        self.settled_picks[signal_id] = pick

        # Update daily file
        self._save_daily_picks(pick.event_date)

        return True

    def get_pick(self, signal_id: str) -> Optional[SignalPick]:
        """Get pick by ID."""
        return self.pending_picks.get(signal_id) or self.settled_picks.get(signal_id)

    def get_daily_picks(self, date: str) -> List[SignalPick]:
        """Get all picks for a date."""
        picks = []
        for pick in list(self.pending_picks.values()) + list(self.settled_picks.values()):
            if pick.event_date == date:
                picks.append(pick)
        return picks

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for recent picks."""
        cutoff = datetime.now() - timedelta(days=days)

        settled = [p for p in self.settled_picks.values()
                   if datetime.fromisoformat(p.posted_at) > cutoff]

        if not settled:
            return {'error': 'No settled picks in period'}

        wins = sum(1 for p in settled if p.status == 'won')
        losses = sum(1 for p in settled if p.status == 'lost')
        pushes = sum(1 for p in settled if p.status == 'pushed')
        total_pnl = sum(p.result for p in settled if p.result is not None)

        decisive = wins + losses
        win_rate = wins / decisive if decisive > 0 else 0

        return {
            'period_days': days,
            'total_picks': len(settled),
            'record': f"{wins}-{losses}",
            'pushes': pushes,
            'win_rate': round(win_rate, 4),
            'total_pnl': round(total_pnl, 2),
            'avg_pick_pnl': round(total_pnl / len(settled), 2),
            'roi': round(total_pnl / sum(p.stake_pct for p in settled) * 100, 2) if settled else 0,
        }

    def export_for_delivery(
        self,
        date: str,
        tier: str = "standard",
        channel: str = "discord"
    ) -> str:
        """
        Export picks formatted for specific tier and channel.

        Args:
            date: Date to export
            tier: Tier level (free, standard, premium, vip)
            channel: Channel to format for

        Returns:
            Formatted message string
        """
        picks = self.get_daily_picks(date)

        # Filter by tier access
        accessible = [p for p in picks if tier in p.tier_access]

        if not accessible:
            return f"No picks available for {tier} tier on {date}"

        # Format for channel
        messages = []
        for pick in accessible:
            messages.append(self._format_message(pick, channel))

        return "\n\n".join(messages)


def create_subscription_package(
    tier: str,
    price_monthly: float,
    features: List[str],
    delivery_channels: List[str]
) -> Dict[str, Any]:
    """
    Create subscription package definition.

    Args:
        tier: Tier name
        price_monthly: Monthly price
        features: List of features
        delivery_channels: Available delivery channels

    Returns:
        Package definition dictionary
    """
    return {
        'tier': tier,
        'pricing': {
            'monthly': price_monthly,
            'quarterly': price_monthly * 3 * 0.9,  # 10% discount
            'annual': price_monthly * 12 * 0.8,  # 20% discount
        },
        'features': features,
        'delivery_channels': delivery_channels,
        'access_level': {
            'free': tier == 'free',
            'standard': tier in ['standard', 'premium', 'vip'],
            'premium': tier in ['premium', 'vip'],
            'vip': tier == 'vip',
        },
    }


# Pre-defined subscription packages
SUBSCRIPTION_PACKAGES = {
    'free': create_subscription_package(
        tier='free',
        price_monthly=0,
        features=[
            '1-2 picks per week',
            '24-hour delayed posting',
            'Basic analysis',
            'Public Discord access',
        ],
        delivery_channels=['discord'],
    ),
    'standard': create_subscription_package(
        tier='standard',
        price_monthly=50,
        features=[
            'All daily picks',
            'Real-time posting',
            'Full analysis',
            'Discord + Telegram',
            'Performance tracking',
        ],
        delivery_channels=['discord', 'telegram'],
    ),
    'premium': create_subscription_package(
        tier='premium',
        price_monthly=100,
        features=[
            'Everything in Standard',
            'Max confidence picks',
            'Live chat access',
            'Weekly Q&A calls',
            'Bankroll management guide',
        ],
        delivery_channels=['discord', 'telegram', 'email'],
    ),
    'vip': create_subscription_package(
        tier='vip',
        price_monthly=200,
        features=[
            'Everything in Premium',
            '1-on-1 consultation (monthly)',
            'Custom bet sizing',
            'Direct access to analysts',
            'Early access to new features',
        ],
        delivery_channels=['discord', 'telegram', 'email', 'sms'],
    ),
}
