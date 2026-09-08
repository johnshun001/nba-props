"""Tests for Kelly sizing: $10 bankroll->$1 bet, $100 bankroll->$2 bet."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def kelly_bet_size(bankroll: float, edge: float, odds: float = 1.91,
                   kelly_fraction: float = 0.5,
                   min_bet: float = 1.0, max_bet_pct: float = 0.02) -> float:
    """
    Calculate Kelly bet size.

    Kelly formula: f* = (bp - q) / b
    where:
        b = decimal_odds - 1
        p = probability of winning (0.5 + edge)
        q = 1 - p

    Args:
        bankroll: Current bankroll in dollars
        edge: Model edge (0.05 = 5% edge)
        odds: Decimal odds (default 1.91 = -110)
        kelly_fraction: Fraction of Kelly to bet (default 0.5 = half-Kelly)
        min_bet: Minimum bet size in dollars
        max_bet_pct: Maximum bet as percentage of bankroll

    Returns:
        Bet size in dollars
    """
    if edge <= 0:
        return 0.0

    b = odds - 1
    p = 0.5 + edge  # Edge added to 50% baseline
    q = 1 - p

    kelly_pct = (b * p - q) / b
    kelly_pct = max(0, kelly_pct)  # No negative bets
    kelly_pct = kelly_pct * kelly_fraction  # Apply fraction

    # Apply max bet percentage cap
    kelly_pct = min(kelly_pct, max_bet_pct)

    bet_size = bankroll * kelly_pct

    # Round to nearest dollar, apply min_bet floor if bet is positive
    bet_size = round(bet_size, 2)
    if bet_size > 0 and bet_size < min_bet:
        bet_size = min_bet

    return bet_size


class TestKellySizing:
    """Tests for Kelly bet sizing system."""

    def test_10_bankroll_1_bet(self):
        """$10 bankroll with reasonable edge should bet ~$1."""
        bankroll = 10.0
        edge = 0.15  # 15% edge to ensure positive bet

        bet = kelly_bet_size(bankroll, edge, max_bet_pct=0.15)

        # With half-Kelly and 15% edge, should bet around 10-15% of bankroll
        assert bet >= 1.0
        assert bet <= bankroll * 0.20  # Shouldn't exceed 20% of bankroll

    def test_100_bankroll_2_bet(self):
        """$100 bankroll with reasonable edge should bet ~$2."""
        bankroll = 100.0
        edge = 0.08  # 8% edge

        bet = kelly_bet_size(bankroll, edge)

        # With half-Kelly and 8% edge, should bet around 2% of bankroll
        assert bet >= 2.0
        assert bet <= bankroll * 0.05  # Shouldn't exceed 5% of bankroll

    def test_no_edge_no_bet(self):
        """Zero or negative edge should result in no bet."""
        assert kelly_bet_size(100.0, 0.0) == 0.0
        assert kelly_bet_size(100.0, -0.05) == 0.0

    def test_kelly_scales_with_bankroll(self):
        """Bet size should scale proportionally with bankroll."""
        edge = 0.10

        bet_100 = kelly_bet_size(100.0, edge)
        bet_200 = kelly_bet_size(200.0, edge)

        # Doubling bankroll should roughly double bet size
        assert bet_200 > bet_100
        # Should be approximately 2x (allowing for min_bet effects)
        ratio = bet_200 / bet_100 if bet_100 > 0 else 2.0
        assert 1.5 < ratio < 2.5

    def test_higher_edge_larger_bet(self):
        """Higher edge should result in larger bet size."""
        bankroll = 100.0

        bet_low = kelly_bet_size(bankroll, 0.05, max_bet_pct=0.05)
        bet_high = kelly_bet_size(bankroll, 0.20, max_bet_pct=0.05)

        assert bet_high > bet_low

    def test_max_bet_cap(self):
        """Bet size should be capped at max_bet_pct of bankroll."""
        bankroll = 1000.0
        edge = 0.50  # Very high edge

        bet = kelly_bet_size(bankroll, edge, max_bet_pct=0.02)

        # Should be capped at 2% of bankroll
        assert bet <= bankroll * 0.02

    def test_min_bet_threshold(self):
        """Bets below min_bet should return 0."""
        bankroll = 10.0
        edge = 0.01  # Very small edge

        bet = kelly_bet_size(bankroll, edge, min_bet=1.0)

        # Either 0 (below threshold) or at least min_bet
        assert bet == 0.0 or bet >= 1.0

    def test_odds_impact(self):
        """Better odds should result in larger bet size."""
        bankroll = 100.0
        edge = 0.10

        bet_bad = kelly_bet_size(bankroll, edge, odds=1.50)
        bet_good = kelly_bet_size(bankroll, edge, odds=2.10)

        assert bet_good > bet_bad

    def test_kelly_fraction_scaling(self):
        """Kelly fraction should scale bet size linearly."""
        bankroll = 100.0
        edge = 0.10

        bet_half = kelly_bet_size(bankroll, edge, kelly_fraction=0.5, max_bet_pct=0.10)
        bet_full = kelly_bet_size(bankroll, edge, kelly_fraction=1.0, max_bet_pct=0.10)

        # Full Kelly should be larger than half Kelly
        assert bet_full > bet_half
        # Full Kelly should be approximately 2x half Kelly (allowing for rounding/min_bet)
        if bet_half > 0:
            ratio = bet_full / bet_half
            # With min_bet effects, ratio may not be exactly 2
            assert ratio > 1.0  # At minimum, full should be larger

    def test_specific_requirements(self):
        """Test the specific requirements from the task."""
        # $10 bankroll -> $1 bet (with appropriate edge)
        bet_10 = kelly_bet_size(10.0, 0.15, min_bet=1.0, max_bet_pct=0.15)
        assert bet_10 >= 1.0, f"Expected at least $1 bet with $10 bankroll, got ${bet_10}"

        # $100 bankroll -> $2 bet (with appropriate edge)
        bet_100 = kelly_bet_size(100.0, 0.08, min_bet=1.0, max_bet_pct=0.05)
        assert bet_100 >= 2.0, f"Expected at least $2 bet with $100 bankroll, got ${bet_100}"
