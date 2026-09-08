"""Syndicate formation and book relationship management.

Syndicate: Pool capital with trusted partners.
Book relationships: Track which books allow action.
Output: syndicate/{month}_allocations.json, book_relationships.json
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNDICATE_DIR = str(PROJECT_ROOT / "syndicate")


class PartnerStatus(Enum):
    """Status of syndicate partner."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class BookStatus(Enum):
    """Status of sportsbook relationship."""
    GOOD = "good"  # Allows full action
    LIMITED = "limited"  # Reduced limits
    RESTRICTED = "restricted"  # Very limited
    BANNED = "banned"  # No action allowed


@dataclass
class SyndicatePartner:
    """Partner in betting syndicate."""
    partner_id: str
    name: str
    contribution: float  # Capital contributed
    profit_share_pct: float  # Percentage of profits
    status: str = "active"
    join_date: str = field(default_factory=lambda: datetime.now().isoformat())
    total_allocated: float = 0.0
    total_returned: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'partner_id': self.partner_id,
            'name': self.name,
            'contribution': round(self.contribution, 2),
            'profit_share_pct': round(self.profit_share_pct, 2),
            'status': self.status,
            'join_date': self.join_date,
            'total_allocated': round(self.total_allocated, 2),
            'total_returned': round(self.total_returned, 2),
            'net_pnl': round(self.total_returned - self.contribution, 2),
            'roi': round((self.total_returned / self.contribution - 1) * 100, 2) if self.contribution > 0 else 0,
            'notes': self.notes,
        }


@dataclass
class SyndicateAllocation:
    """Monthly allocation for syndicate."""
    month: str
    total_capital: float
    allocations: Dict[str, float]  # partner_id -> amount
    bets_placed: int = 0
    total_pnl: float = 0.0
    management_fee_pct: float = 0.0
    performance_fee_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        partner_pnl = {}
        for partner_id, amount in self.allocations.items():
            share = amount / self.total_capital if self.total_capital > 0 else 0
            partner_pnl[partner_id] = round(self.total_pnl * share, 2)

        return {
            'month': self.month,
            'total_capital': round(self.total_capital, 2),
            'allocations': {k: round(v, 2) for k, v in self.allocations.items()},
            'activity': {
                'bets_placed': self.bets_placed,
                'total_pnl': round(self.total_pnl, 2),
                'roi': round(self.total_pnl / self.total_capital * 100, 2) if self.total_capital > 0 else 0,
            },
            'fees': {
                'management_fee_pct': self.management_fee_pct,
                'performance_fee_pct': self.performance_fee_pct,
                'management_fee_amount': round(self.total_capital * self.management_fee_pct / 100, 2),
                'performance_fee_amount': round(max(0, self.total_pnl) * self.performance_fee_pct / 100, 2),
            },
            'partner_pnl': partner_pnl,
        }


@dataclass
class SportsbookRelationship:
    """Relationship tracking with sportsbook."""
    sportsbook: str
    status: str = "good"
    max_bet: float = 0.0
    current_limit: float = 0.0
    limit_history: List[Dict[str, Any]] = field(default_factory=list)
    vip_host: str = ""
    vip_host_contact: str = ""
    notes: str = ""
    last_bet_date: Optional[str] = None
    total_wagered: float = 0.0
    total_pnl: float = 0.0
    sharps_score: float = 0.0  # 0-100, higher = more likely to be limited

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sportsbook': self.sportsbook,
            'status': self.status,
            'limits': {
                'max_bet': round(self.max_bet, 2),
                'current_limit': round(self.current_limit, 2),
                'utilization_pct': round(self.current_limit / self.max_bet * 100, 1) if self.max_bet > 0 else 0,
            },
            'vip': {
                'host': self.vip_host,
                'contact': self.vip_host_contact,
            },
            'activity': {
                'last_bet_date': self.last_bet_date,
                'total_wagered': round(self.total_wagered, 2),
                'total_pnl': round(self.total_pnl, 2),
                'roi': round(self.total_pnl / self.total_wagered * 100, 2) if self.total_wagered > 0 else 0,
            },
            'risk': {
                'sharps_score': round(self.sharps_score, 1),
                'risk_level': self._get_risk_level(),
            },
            'notes': self.notes,
        }

    def _get_risk_level(self) -> str:
        if self.sharps_score >= 80:
            return "high"
        elif self.sharps_score >= 50:
            return "medium"
        else:
            return "low"


class SyndicateManager:
    """
    Manage betting syndicate operations.

    Usage:
        manager = SyndicateManager()
        manager.add_partner(partner)
        manager.create_monthly_allocation(month, total_capital)
        manager.generate_report()
    """

    def __init__(self, syndicate_dir: str = SYNDICATE_DIR):
        self.syndicate_dir = Path(syndicate_dir)
        self.syndicate_dir.mkdir(parents=True, exist_ok=True)
        self.partners: Dict[str, SyndicatePartner] = {}
        self.allocations: Dict[str, SyndicateAllocation] = {}
        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing syndicate data."""
        # Load partners
        partners_file = self.syndicate_dir / "partners.json"
        if partners_file.exists():
            with open(partners_file, 'r') as f:
                data = json.load(f)
            for p in data.get('partners', []):
                self.partners[p['partner_id']] = SyndicatePartner(**p)

        # Load allocations
        for alloc_file in self.syndicate_dir.glob("*_allocation.json"):
            try:
                with open(alloc_file, 'r') as f:
                    data = json.load(f)
                month = data.get('month', '')
                if month:
                    self.allocations[month] = SyndicateAllocation(**data)
            except Exception as e:
                print(f"Error loading {alloc_file}: {e}")

    def add_partner(self, partner: SyndicatePartner):
        """Add syndicate partner."""
        self.partners[partner.partner_id] = partner
        self._save_partners()

    def remove_partner(self, partner_id: str):
        """Remove syndicate partner."""
        if partner_id in self.partners:
            partner = self.partners[partner_id]
            partner.status = "inactive"
            self._save_partners()

    def _save_partners(self):
        """Save partners to file."""
        partners_file = self.syndicate_dir / "partners.json"
        data = {
            'partners': [p.to_dict() for p in self.partners.values()],
            'summary': {
                'total_partners': len(self.partners),
                'active_partners': sum(1 for p in self.partners.values() if p.status == 'active'),
                'total_capital': sum(p.contribution for p in self.partners.values() if p.status == 'active'),
            },
        }
        with open(partners_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_monthly_allocation(
        self,
        month: str,
        total_capital: float,
        management_fee_pct: float = 1.0,
        performance_fee_pct: float = 20.0
    ) -> SyndicateAllocation:
        """Create monthly allocation for syndicate."""
        active_partners = [p for p in self.partners.values() if p.status == 'active']

        # Calculate allocations pro-rata
        total_contribution = sum(p.contribution for p in active_partners)
        allocations = {}

        for partner in active_partners:
            share = partner.contribution / total_contribution if total_contribution > 0 else 0
            allocations[partner.partner_id] = total_capital * share
            partner.total_allocated += allocations[partner.partner_id]

        allocation = SyndicateAllocation(
            month=month,
            total_capital=total_capital,
            allocations=allocations,
            management_fee_pct=management_fee_pct,
            performance_fee_pct=performance_fee_pct,
        )

        self.allocations[month] = allocation
        self._save_allocation(allocation)
        self._save_partners()

        return allocation

    def _save_allocation(self, allocation: SyndicateAllocation):
        """Save allocation to file."""
        alloc_file = self.syndicate_dir / f"{allocation.month}_allocation.json"
        with open(alloc_file, 'w') as f:
            json.dump(allocation.to_dict(), f, indent=2)

    def update_allocation_pnl(self, month: str, pnl: float, bets_placed: int = 0):
        """Update allocation with P&L results."""
        if month not in self.allocations:
            return False

        allocation = self.allocations[month]
        allocation.total_pnl = pnl
        allocation.bets_placed = bets_placed

        # Update partner returns
        for partner_id, alloc_amount in allocation.allocations.items():
            if partner_id in self.partners:
                share = alloc_amount / allocation.total_capital if allocation.total_capital > 0 else 0
                partner_pnl = pnl * share
                self.partners[partner_id].total_returned += partner_pnl

        self._save_allocation(allocation)
        self._save_partners()
        return True

    def generate_report(self) -> Dict[str, Any]:
        """Generate syndicate report."""
        active_partners = [p for p in self.partners.values() if p.status == 'active']

        total_capital = sum(p.contribution for p in active_partners)
        total_returned = sum(p.total_returned for p in active_partners)

        return {
            'summary': {
                'total_partners': len(active_partners),
                'total_capital': round(total_capital, 2),
                'total_returned': round(total_returned, 2),
                'total_pnl': round(total_returned - total_capital, 2),
                'roi': round((total_returned / total_capital - 1) * 100, 2) if total_capital > 0 else 0,
            },
            'partners': [p.to_dict() for p in active_partners],
            'allocations': [a.to_dict() for a in sorted(self.allocations.values(),
                                                        key=lambda x: x.month, reverse=True)[:6]],
        }


class BookRelationshipManager:
    """
    Manage relationships with sportsbooks.

    Usage:
        manager = BookRelationshipManager()
        manager.add_sportsbook(book)
        manager.update_sharps_score(sportsbook, win_rate)
        manager.get_recommended_books()
    """

    def __init__(self):
        self.books: Dict[str, SportsbookRelationship] = {}
        self.syndicate_dir = Path(SYNDICATE_DIR)
        self.syndicate_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing book relationship data."""
        books_file = Path(SYNDICATE_DIR) / "book_relationships.json"
        if books_file.exists():
            try:
                with open(books_file, 'r') as f:
                    data = json.load(f)
                for b in data.get('books', []):
                    self.books[b['sportsbook']] = SportsbookRelationship(**b)
            except Exception as e:
                print(f"Error loading book relationships: {e}")

    def add_sportsbook(self, book: SportsbookRelationship):
        """Add sportsbook relationship."""
        self.books[book.sportsbook] = book
        self._save_books()

    def update_bet_record(
        self,
        sportsbook: str,
        wagered: float,
        pnl: float,
        bet_limit: Optional[float] = None
    ):
        """Update record for sportsbook after bet."""
        if sportsbook not in self.books:
            return

        book = self.books[sportsbook]
        book.total_wagered += wagered
        book.total_pnl += pnl
        book.last_bet_date = datetime.now().isoformat()

        if bet_limit is not None:
            book.current_limit = bet_limit
            book.limit_history.append({
                'date': datetime.now().isoformat(),
                'limit': bet_limit,
            })

        # Update Sharps score based on performance
        self._update_sharps_score(book)
        self._save_books()

    def _update_sharps_score(self, book: SportsbookRelationship):
        """Update Sharps score based on betting patterns."""
        score = 0

        # Factor 1: Win rate (higher = more likely to be limited)
        if book.total_wagered > 0:
            roi = book.total_pnl / book.total_wagered
            if roi > 0.10:  # >10% ROI
                score += 30
            elif roi > 0.05:
                score += 20
            elif roi > 0:
                score += 10

        # Factor 2: Bet sizing patterns
        if book.limit_history:
            limits = [h['limit'] for h in book.limit_history]
            if len(limits) >= 2 and limits[-1] < limits[0] * 0.5:
                score += 40  # Limit was cut significantly

        # Factor 3: Status
        status_scores = {
            'good': 0,
            'limited': 30,
            'restricted': 60,
            'banned': 100,
        }
        score += status_scores.get(book.status, 0)

        book.sharps_score = min(100, score)

    def get_recommended_books(self, min_limit: float = 1000) -> List[Dict[str, Any]]:
        """Get recommended sportsbooks for action."""
        recommended = []

        for book in self.books.values():
            if book.status in ['banned', 'restricted']:
                continue
            if book.current_limit < min_limit:
                continue

            recommended.append({
                'sportsbook': book.sportsbook,
                'current_limit': book.current_limit,
                'status': book.status,
                'sharps_score': book.sharps_score,
                'vip_contact': book.vip_host if book.vip_host else None,
            })

        # Sort by limit (descending) and sharps score (ascending)
        recommended.sort(key=lambda x: (-x['current_limit'], x['sharps_score']))

        return recommended

    def _save_books(self):
        """Save book relationships to file."""
        books_file = Path(SYNDICATE_DIR) / "book_relationships.json"
        data = {
            'books': [b.to_dict() for b in self.books.values()],
            'summary': {
                'total_books': len(self.books),
                'good_standing': sum(1 for b in self.books.values() if b.status == 'good'),
                'limited': sum(1 for b in self.books.values() if b.status == 'limited'),
                'restricted': sum(1 for b in self.books.values() if b.status == 'restricted'),
                'banned': sum(1 for b in self.books.values() if b.status == 'banned'),
            },
        }
        with open(books_file, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_report(self) -> Dict[str, Any]:
        """Generate book relationship report."""
        return {
            'books': [b.to_dict() for b in self.books.values()],
            'recommendations': self.get_recommended_books(),
            'alerts': self._generate_alerts(),
        }

    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts for book relationships."""
        alerts = []

        for book in self.books.values():
            if book.sharps_score >= 80:
                alerts.append({
                    'type': 'high_risk',
                    'sportsbook': book.sportsbook,
                    'message': f"High risk of limitation at {book.sportsbook} (Sharps score: {book.sharps_score})",
                    'action': 'Reduce bet sizes, vary timing',
                })

            if book.status == 'limited':
                alerts.append({
                    'type': 'limited',
                    'sportsbook': book.sportsbook,
                    'message': f"Limits reduced at {book.sportsbook}",
                    'action': 'Contact VIP host, consider reducing action',
                })

        return alerts


def create_syndicate_operating_agreement(
    syndicate_name: str,
    management_fee_pct: float,
    performance_fee_pct: float,
    high_water_mark: bool = True
) -> Dict[str, Any]:
    """
    Create syndicate operating agreement template.

    Args:
        syndicate_name: Name of syndicate
        management_fee_pct: Annual management fee
        performance_fee_pct: Performance fee on profits
        high_water_mark: Whether to use high water mark

    Returns:
        Operating agreement template
    """
    return {
        'syndicate_name': syndicate_name,
        'structure': 'LLC',
        'fees': {
            'management_fee': {
                'percentage': management_fee_pct,
                'frequency': 'monthly',
                'description': f"{management_fee_pct}% of assets under management annually",
            },
            'performance_fee': {
                'percentage': performance_fee_pct,
                'high_water_mark': high_water_mark,
                'description': f"{performance_fee_pct}% of profits above high water mark",
            },
        },
        'terms': {
            'minimum_investment': 10000,
            'lockup_period': '30 days',
            'withdrawal_notice': '14 days',
            'distribution_frequency': 'monthly',
        },
        'risk_disclosures': [
            'Past performance does not guarantee future results',
            'Sports betting involves substantial risk of loss',
            'No guarantee of profit or avoidance of loss',
            'Limited liquidity during lockup periods',
        ],
        'compliance': {
            'jurisdiction': 'To be determined by legal counsel',
            'accredited_investors_only': True,
            'max_investors': 99,  # For LLC structure
        },
    }
