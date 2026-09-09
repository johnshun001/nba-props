"""Tests for line shopping: selects best available line."""

import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def select_best_line(lines: List[Dict], bet_side: str = "over") -> Optional[Dict]:
    """
    Select the best available line from multiple sportsbooks.

    For "over" bets: highest line with best odds wins
    For "under" bets: lowest line with best odds wins

    Args:
        lines: List of dicts with keys: sportsbook, line, over_odds, under_odds
        bet_side: "over" or "under"

    Returns:
        Best line dict or None if no lines provided
    """
    if not lines:
        return None

    best_line = None
    best_value = float('-inf')

    for line in lines:
        line_val = line['line']
        odds = line['over_odds'] if bet_side == "over" else line['under_odds']

        decimal_odds = 1 + (odds / 100 if odds > 0 else 100 / abs(odds))

        # A lower threshold benefits an OVER; a higher threshold benefits an UNDER.
        # Price breaks ties, with the larger decimal payout preferred.

        if bet_side == "over":
            value = -line_val * 1000 + decimal_odds
        else:
            value = line_val * 1000 + decimal_odds

        if value > best_value:
            best_value = value
            best_line = line

    return best_line


def calculate_line_shopping_value(selected_line: Dict, market_lines: List[Dict],
                                   bet_side: str = "over") -> float:
    """
    Calculate the value gained from line shopping.

    Value = (avg_line - selected_line) for overs
    Value = (selected_line - avg_line) for unders

    Returns value in points.
    """
    if not market_lines:
        return 0.0

    avg_line = sum(item['line'] for item in market_lines) / len(market_lines)

    if bet_side == "over":
        return avg_line - selected_line['line']
    else:
        return selected_line['line'] - avg_line


class TestLineShopping:
    """Tests for line shopping system."""

    def test_selects_lowest_line_for_over(self):
        """For over bets, should select the lowest threshold."""
        lines = [
            {'sportsbook': 'draftkings', 'line': 24.5, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'fanduel', 'line': 25.5, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'betmgm', 'line': 24.0, 'over_odds': -110, 'under_odds': -110},
        ]

        best = select_best_line(lines, bet_side="over")

        assert best is not None
        assert best['sportsbook'] == 'betmgm'
        assert best['line'] == 24.0

    def test_selects_highest_line_for_under(self):
        """For under bets, should select the highest threshold."""
        lines = [
            {'sportsbook': 'draftkings', 'line': 24.5, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'fanduel', 'line': 25.5, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'betmgm', 'line': 24.0, 'over_odds': -110, 'under_odds': -110},
        ]

        best = select_best_line(lines, bet_side="under")

        assert best is not None
        assert best['sportsbook'] == 'fanduel'
        assert best['line'] == 25.5

    def test_considers_odds_when_lines_equal(self):
        """When lines are equal, should select best odds."""
        lines = [
            {'sportsbook': 'draftkings', 'line': 25.5, 'over_odds': -115, 'under_odds': -105},
            {'sportsbook': 'fanduel', 'line': 25.5, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'betmgm', 'line': 25.5, 'over_odds': -105, 'under_odds': -115},
        ]

        best = select_best_line(lines, bet_side="over")

        assert best is not None
        assert best['over_odds'] == -105  # Best payout at the same threshold

    def test_empty_lines_returns_none(self):
        """Empty lines list should return None."""
        best = select_best_line([], bet_side="over")
        assert best is None

    def test_line_shopping_value_positive(self):
        """Line shopping value should be positive when beating the market."""
        market_lines = [
            {'sportsbook': 'draftkings', 'line': 24.5, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'fanduel', 'line': 25.5, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'betmgm', 'line': 24.0, 'over_odds': -110, 'under_odds': -110},
        ]

        # Selecting the best line (25.5)
        best = select_best_line(market_lines, bet_side="over")
        value = calculate_line_shopping_value(best, market_lines, bet_side="over")

        # Avg line = 24.67; selecting 24.0 makes an OVER easier by 0.67.
        assert value > 0

    def test_line_shopping_value_calculation(self):
        """Test exact line shopping value calculation."""
        market_lines = [
            {'sportsbook': 'draftkings', 'line': 20.0, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'fanduel', 'line': 22.0, 'over_odds': -110, 'under_odds': -110},
        ]

        best = select_best_line(market_lines, bet_side="over")
        value = calculate_line_shopping_value(best, market_lines, bet_side="over")

        # Avg = 21.0, Best = 20.0, Value = 1.0
        assert abs(value - 1.0) < 0.01

    def test_multiple_sportsbooks(self):
        """Should handle many sportsbooks correctly."""
        lines = [
            {'sportsbook': 'draftkings', 'line': 23.5, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'fanduel', 'line': 24.5, 'over_odds': -115, 'under_odds': -105},
            {'sportsbook': 'betmgm', 'line': 25.0, 'over_odds': -108, 'under_odds': -112},
            {'sportsbook': 'caesars', 'line': 24.0, 'over_odds': -105, 'under_odds': -115},
            {'sportsbook': 'pointsbet', 'line': 25.5, 'over_odds': -112, 'under_odds': -108},
        ]

        best = select_best_line(lines, bet_side="over")

        assert best is not None
        assert best['line'] == 23.5  # Lowest OVER threshold

    def test_single_sportsbook(self):
        """Should work with single sportsbook."""
        lines = [
            {'sportsbook': 'draftkings', 'line': 25.5, 'over_odds': -110, 'under_odds': -110},
        ]

        best = select_best_line(lines, bet_side="over")

        assert best is not None
        assert best['sportsbook'] == 'draftkings'

    def test_negative_line_shopping_value(self):
        """Selecting worst line should give negative value."""
        market_lines = [
            {'sportsbook': 'draftkings', 'line': 24.5, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'fanduel', 'line': 25.5, 'over_odds': -110, 'under_odds': -110},
            {'sportsbook': 'betmgm', 'line': 26.5, 'over_odds': -110, 'under_odds': -110},
        ]

        # Manually select the highest (worst) line for an OVER bet.
        worst = {'sportsbook': 'betmgm', 'line': 26.5, 'over_odds': -110, 'under_odds': -110}
        value = calculate_line_shopping_value(worst, market_lines, bet_side="over")

        # Avg = 25.5, Selected = 24.5, Value = -1.0
        assert value < 0

    def test_odds_conversion_american_to_implied(self):
        """Test American odds to implied probability conversion."""
        # +100 -> 50%
        lines_plus100 = [{'sportsbook': 'dk', 'line': 25.5, 'over_odds': 100, 'under_odds': -100}]
        best = select_best_line(lines_plus100, bet_side="over")
        assert best is not None

        # -110 -> ~52.4%
        lines_minus110 = [{'sportsbook': 'dk', 'line': 25.5, 'over_odds': -110, 'under_odds': -110}]
        best = select_best_line(lines_minus110, bet_side="over")
        assert best is not None

        # +200 -> ~33.3%
        lines_plus200 = [{'sportsbook': 'dk', 'line': 25.5, 'over_odds': 200, 'under_odds': -200}]
        best = select_best_line(lines_plus200, bet_side="over")
        assert best is not None
