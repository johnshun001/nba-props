"""Tests for pydantic validation schemas."""

import pytest
from datetime import datetime
from schemas.validation import (
    PropLine, Bet, BetResult, RevenueAttribution,
    validate_prop_line, validate_bet, validate_bet_result,
    ValidationResponse
)


class TestPropLine:
    """Tests for PropLine schema."""

    def test_valid_prop_line(self):
        """Valid prop line should pass validation."""
        prop = PropLine(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            over_odds=-110,
            under_odds=-110,
            sportsbook="draftkings"
        )

        assert prop.player_id == "lebron_james"
        assert prop.prop_type == "points"
        assert prop.line == 25.5
        assert prop.over_odds == -110
        assert prop.under_odds == -110
        assert prop.sportsbook == "draftkings"

    def test_prop_type_normalization(self):
        """Prop type should be normalized to lowercase."""
        prop = PropLine(
            player_id="lebron_james",
            prop_type="POINTS",
            line=25.5,
            over_odds=-110,
            under_odds=-110,
            sportsbook="draftkings"
        )

        assert prop.prop_type == "points"

    def test_invalid_prop_type(self):
        """Invalid prop type should fail validation."""
        with pytest.raises(ValueError, match="Invalid prop_type"):
            PropLine(
                player_id="lebron_james",
                prop_type="invalid_prop",
                line=25.5,
                over_odds=-110,
                under_odds=-110,
                sportsbook="draftkings"
            )

    def test_invalid_sportsbook(self):
        """Invalid sportsbook should fail validation."""
        with pytest.raises(ValueError, match="Invalid sportsbook"):
            PropLine(
                player_id="lebron_james",
                prop_type="points",
                line=25.5,
                over_odds=-110,
                under_odds=-110,
                sportsbook="invalid_book"
            )

    def test_line_must_be_positive(self):
        """Line must be positive."""
        with pytest.raises(ValueError):
            PropLine(
                player_id="lebron_james",
                prop_type="points",
                line=-5.5,
                over_odds=-110,
                under_odds=-110,
                sportsbook="draftkings"
            )

    def test_odds_range_validation(self):
        """Odds must be within valid range."""
        with pytest.raises(ValueError):
            PropLine(
                player_id="lebron_james",
                prop_type="points",
                line=25.5,
                over_odds=-1500,
                under_odds=-110,
                sportsbook="draftkings"
            )

    def test_both_odds_heavy_negative(self):
        """Both odds cannot be heavily negative."""
        with pytest.raises(ValueError, match="cannot be less than -200"):
            PropLine(
                player_id="lebron_james",
                prop_type="points",
                line=25.5,
                over_odds=-250,
                under_odds=-250,
                sportsbook="draftkings"
            )

    def test_all_valid_prop_types(self):
        """Test all valid prop types."""
        valid_types = ['points', 'rebounds', 'assists', 'threes', 'steals', 'blocks',
                       'turnovers', 'points_rebounds_assists', 'points_rebounds',
                       'points_assists', 'rebounds_assists']

        for prop_type in valid_types:
            prop = PropLine(
                player_id="player_1",
                prop_type=prop_type,
                line=1.5,
                over_odds=-110,
                under_odds=-110,
                sportsbook="fanduel"
            )
            assert prop.prop_type == prop_type


class TestBet:
    """Tests for Bet schema."""

    def test_valid_bet(self):
        """Valid bet should pass validation."""
        bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110
        )

        assert bet.player_id == "lebron_james"
        assert bet.bet_size == 10.0
        assert bet.edge == 0.08
        assert bet.bet_side == "over"

    def test_bet_side_normalization(self):
        """Bet side should be normalized to lowercase."""
        bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="OVER",
            sportsbook="draftkings",
            odds=-110
        )

        assert bet.bet_side == "over"

    def test_edge_must_be_positive(self):
        """Edge must be positive to place a bet."""
        with pytest.raises(ValueError, match="Edge must be positive"):
            Bet(
                player_id="lebron_james",
                prop_type="points",
                line=25.5,
                bet_size=10.0,
                edge=-0.05,
                bet_side="over",
                sportsbook="draftkings",
                odds=-110
            )

    def test_bet_size_range(self):
        """Bet size must be in valid range."""
        with pytest.raises(ValueError):
            Bet(
                player_id="lebron_james",
                prop_type="points",
                line=25.5,
                bet_size=1500,  # Over max
                edge=0.08,
                bet_side="over",
                sportsbook="draftkings",
                odds=-110
            )

    def test_bet_size_vs_edge_validation(self):
        """Bet size should be reasonable relative to edge."""
        with pytest.raises(ValueError, match="too large"):
            Bet(
                player_id="lebron_james",
                prop_type="points",
                line=25.5,
                bet_size=100,  # Too large for 1% edge
                edge=0.01,
                bet_side="over",
                sportsbook="draftkings",
                odds=-110
            )

    def test_timestamp_default(self):
        """Timestamp should default to current UTC time."""
        bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110
        )

        assert isinstance(bet.timestamp, datetime)


class TestBetResult:
    """Tests for BetResult schema."""

    def test_valid_bet_result_win(self):
        """Valid winning bet result."""
        bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110
        )

        result = BetResult(
            bet=bet,
            outcome="win",
            pnl=9.09  # $10 bet at -110 wins $9.09
        )

        assert result.outcome == "win"
        assert abs(result.pnl - 9.09) < 0.01

    def test_valid_bet_result_loss(self):
        """Valid losing bet result."""
        bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110
        )

        result = BetResult(
            bet=bet,
            outcome="loss",
            pnl=-10.0  # Lose entire bet
        )

        assert result.outcome == "loss"
        assert result.pnl == -10.0

    def test_valid_bet_result_push(self):
        """Valid push bet result."""
        bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110
        )

        result = BetResult(
            bet=bet,
            outcome="push",
            pnl=0.0  # Push = no P&L
        )

        assert result.outcome == "push"
        assert result.pnl == 0.0

    def test_invalid_outcome(self):
        """Invalid outcome should fail validation."""
        bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110
        )

        with pytest.raises(ValueError, match="Invalid outcome"):
            BetResult(
                bet=bet,
                outcome="invalid",
                pnl=0.0
            )

    def test_pnl_mismatch_win(self):
        """P&L should match expected win amount."""
        bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110
        )

        with pytest.raises(ValueError, match="doesn't match expected win"):
            BetResult(
                bet=bet,
                outcome="win",
                pnl=50.0  # Wrong P&L
            )

    def test_pnl_mismatch_loss(self):
        """P&L should match expected loss amount."""
        bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110
        )

        with pytest.raises(ValueError, match="doesn't match expected loss"):
            BetResult(
                bet=bet,
                outcome="loss",
                pnl=-5.0  # Wrong P&L
            )


class TestRevenueAttribution:
    """Tests for RevenueAttribution schema."""

    def test_valid_attribution(self):
        """Valid revenue attribution."""
        attr = RevenueAttribution(
            bet_id="bet_123",
            line_shopping_value=0.5,
            kelly_edge=0.02,
            model_edge=0.05,
            matchup_edge=0.01,
            total_edge=0.13  # 0.5/10 + 0.02 + 0.05 + 0.01 = 0.13
        )

        assert attr.bet_id == "bet_123"
        assert attr.line_shopping_value == 0.5
        assert abs(attr.total_edge - 0.13) < 0.01

    def test_total_edge_validation(self):
        """Total edge must equal sum of components."""
        with pytest.raises(ValueError, match="doesn't match sum"):
            RevenueAttribution(
                bet_id="bet_123",
                line_shopping_value=0.5,
                kelly_edge=0.02,
                model_edge=0.05,
                matchup_edge=0.01,
                total_edge=0.50  # Wrong total
            )

    def test_line_shopping_value_range(self):
        """Line shopping value must be in valid range."""
        with pytest.raises(ValueError):
            RevenueAttribution(
                bet_id="bet_123",
                line_shopping_value=15.0,  # Out of range
                kelly_edge=0.02,
                model_edge=0.05,
                matchup_edge=0.01,
                total_edge=0.13
            )


class TestValidationFunctions:
    """Tests for validation helper functions."""

    def test_validate_prop_line_success(self):
        """validate_prop_line should return valid response."""
        data = {
            "player_id": "lebron_james",
            "prop_type": "points",
            "line": 25.5,
            "over_odds": -110,
            "under_odds": -110,
            "sportsbook": "draftkings"
        }

        result = validate_prop_line(data)

        assert result.valid is True
        assert len(result.errors) == 0
        assert result.data is not None

    def test_validate_prop_line_failure(self):
        """validate_prop_line should return invalid response for bad data."""
        data = {
            "player_id": "lebron_james",
            "prop_type": "invalid_prop",
            "line": 25.5,
            "over_odds": -110,
            "under_odds": -110,
            "sportsbook": "draftkings"
        }

        result = validate_prop_line(data)

        assert result.valid is False
        assert len(result.errors) > 0
        assert result.data is None

    def test_validate_bet_success(self):
        """validate_bet should return valid response."""
        data = {
            "player_id": "lebron_james",
            "prop_type": "points",
            "line": 25.5,
            "bet_size": 10.0,
            "edge": 0.08,
            "bet_side": "over",
            "sportsbook": "draftkings",
            "odds": -110
        }

        result = validate_bet(data)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_bet_result_success(self):
        """validate_bet_result should return valid response."""
        bet_data = {
            "player_id": "lebron_james",
            "prop_type": "points",
            "line": 25.5,
            "bet_size": 10.0,
            "edge": 0.08,
            "bet_side": "over",
            "sportsbook": "draftkings",
            "odds": -110
        }
        data = {
            "bet": bet_data,
            "outcome": "win",
            "pnl": 9.09
        }

        result = validate_bet_result(data)

        assert result.valid is True
        assert len(result.errors) == 0
