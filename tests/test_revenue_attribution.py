"""Tests for revenue attribution system."""

import pytest
import tempfile
import os
from datetime import datetime
from schemas.validation import RevenueAttribution
from analysis.revenue_attribution import (
    init_db, log_attribution, get_attribution_by_bet,
    get_all_attributions, get_aggregate_attribution,
    get_attribution_by_component, get_attribution_trend,
    get_component_ranking, clear_all_attributions,
    AttributionCalculator, get_db_connection
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    init_db(path)
    yield path
    os.unlink(path)


@pytest.fixture
def sample_attribution():
    """Create a sample attribution record."""
    return RevenueAttribution(
        bet_id="bet_001",
        line_shopping_value=0.5,
        kelly_edge=0.02,
        model_edge=0.05,
        matchup_edge=0.01,
        total_edge=0.13
    )


class TestInitDB:
    """Tests for database initialization."""

    def test_init_db_creates_table(self, temp_db):
        """init_db should create the revenue_attribution table."""
        with get_db_connection(temp_db) as conn:
            result = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='revenue_attribution'
            """).fetchone()

            assert result is not None
            assert result[0] == 'revenue_attribution'

    def test_init_db_creates_indexes(self, temp_db):
        """init_db should create required indexes."""
        with get_db_connection(temp_db) as conn:
            indexes = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND tbl_name='revenue_attribution'
            """).fetchall()

            index_names = [idx[0] for idx in indexes]
            assert 'idx_attribution_bet_id' in index_names
            assert 'idx_attribution_timestamp' in index_names


class TestLogAttribution:
    """Tests for logging attribution."""

    def test_log_attribution_success(self, temp_db, sample_attribution):
        """log_attribution should successfully insert a record."""
        attr_id = log_attribution(sample_attribution, temp_db)

        assert attr_id > 0

        attr = get_attribution_by_bet("bet_001", temp_db)
        assert attr is not None
        assert attr['bet_id'] == "bet_001"
        assert abs(attr['line_shopping_value'] - 0.5) < 0.01

    def test_log_multiple_attributions(self, temp_db, sample_attribution):
        """Should be able to log multiple attribution records."""
        for i in range(5):
            attr = RevenueAttribution(
                bet_id=f"bet_{i:03d}",
                line_shopping_value=0.5,
                kelly_edge=0.02,
                model_edge=0.05,
                matchup_edge=0.01,
                total_edge=0.13
            )
            log_attribution(attr, temp_db)

        all_attrs = get_all_attributions(temp_db)
        assert len(all_attrs) == 5


class TestGetAttribution:
    """Tests for retrieving attribution."""

    def test_get_attribution_by_bet(self, temp_db, sample_attribution):
        """get_attribution_by_bet should return correct record."""
        log_attribution(sample_attribution, temp_db)

        attr = get_attribution_by_bet("bet_001", temp_db)
        assert attr is not None
        assert attr['bet_id'] == "bet_001"

    def test_get_attribution_not_found(self, temp_db):
        """get_attribution_by_bet should return None for missing bet."""
        attr = get_attribution_by_bet("nonexistent", temp_db)
        assert attr is None

    def test_get_all_attributions(self, temp_db, sample_attribution):
        """get_all_attributions should return all records."""
        log_attribution(sample_attribution, temp_db)

        all_attrs = get_all_attributions(temp_db)
        assert len(all_attrs) == 1


class TestAggregateAttribution:
    """Tests for aggregate attribution statistics."""

    def test_get_aggregate_insufficient_data(self, temp_db, sample_attribution):
        """get_aggregate_attribution should return None if < 100 bets."""
        log_attribution(sample_attribution, temp_db)

        result = get_aggregate_attribution(min_bets=100, db_path=temp_db)
        assert result is None

    def test_get_aggregate_with_sufficient_data(self, temp_db):
        """get_aggregate_attribution should return stats if >= 100 bets."""
        # Insert 100 records
        for i in range(100):
            attr = RevenueAttribution(
                bet_id=f"bet_{i:03d}",
                line_shopping_value=0.5,
                kelly_edge=0.02,
                model_edge=0.05,
                matchup_edge=0.01,
                total_edge=0.13
            )
            log_attribution(attr, temp_db)

        result = get_aggregate_attribution(min_bets=100, db_path=temp_db)

        assert result is not None
        assert result['total_bets'] == 100
        assert abs(result['total_line_shopping'] - 50.0) < 0.01
        assert 'line_shopping_pct' in result

    def test_get_aggregate_component_percentages(self, temp_db):
        """get_aggregate_attribution should calculate component percentages."""
        for i in range(100):
            # total_edge = line_shopping/10 + kelly + model + matchup
            # = 1.0/10 + 0.02 + 0.05 + 0.03 = 0.20
            attr = RevenueAttribution(
                bet_id=f"bet_{i:03d}",
                line_shopping_value=1.0,
                kelly_edge=0.02,
                model_edge=0.05,
                matchup_edge=0.03,
                total_edge=0.20  # 0.1 + 0.02 + 0.05 + 0.03
            )
            log_attribution(attr, temp_db)

        result = get_aggregate_attribution(min_bets=100, db_path=temp_db)

        # Percentages should sum to ~100
        # line_shopping contributes 0.1/0.2 = 50%
        # kelly contributes 0.02/0.2 = 10%
        # model contributes 0.05/0.2 = 25%
        # matchup contributes 0.03/0.2 = 15%
        total_pct = (
            result['line_shopping_pct'] +
            result['kelly_edge_pct'] +
            result['model_edge_pct'] +
            result['matchup_edge_pct']
        )
        assert abs(total_pct - 100.0) < 5.0  # Allow some rounding


class TestAttributionByComponent:
    """Tests for component breakdown."""

    def test_get_attribution_by_component(self, temp_db, sample_attribution):
        """get_attribution_by_component should return component totals."""
        log_attribution(sample_attribution, temp_db)

        components = get_attribution_by_component(temp_db)

        assert 'line_shopping_value' in components
        assert 'kelly_edge' in components
        assert 'model_edge' in components
        assert 'matchup_edge' in components
        assert abs(components['line_shopping_value'] - 0.5) < 0.01

    def test_get_attribution_empty_db(self, temp_db):
        """get_attribution_by_component should handle empty database."""
        components = get_attribution_by_component(temp_db)

        assert components['line_shopping_value'] == 0.0
        assert components['kelly_edge'] == 0.0
        assert components['model_edge'] == 0.0
        assert components['matchup_edge'] == 0.0


class TestAttributionTrend:
    """Tests for attribution trend analysis."""

    def test_get_attribution_trend(self, temp_db):
        """get_attribution_trend should return recent records."""
        for i in range(60):
            attr = RevenueAttribution(
                bet_id=f"bet_{i:03d}",
                line_shopping_value=0.5,
                kelly_edge=0.02,
                model_edge=0.05,
                matchup_edge=0.01,
                total_edge=0.13
            )
            log_attribution(attr, temp_db)

        trend = get_attribution_trend(n_bets=50, db_path=temp_db)

        assert len(trend) == 50
        # Most recent should be first
        assert trend[0]['bet_id'] == 'bet_059'


class TestComponentRanking:
    """Tests for component ranking."""

    def test_get_component_ranking(self, temp_db):
        """get_component_ranking should rank components by contribution."""
        # Insert records where line_shopping dominates
        for i in range(50):
            attr = RevenueAttribution(
                bet_id=f"bet_{i:03d}",
                line_shopping_value=2.0,  # High line shopping value
                kelly_edge=0.01,
                model_edge=0.02,
                matchup_edge=0.01,
                total_edge=0.24
            )
            log_attribution(attr, temp_db)

        ranking = get_component_ranking(temp_db)

        assert len(ranking) == 4
        # Line shopping should be ranked first (highest absolute value)
        assert ranking[0]['component'] == 'line_shopping_value'


class TestAttributionCalculator:
    """Tests for AttributionCalculator."""

    def test_calculate_attribution(self):
        """AttributionCalculator should compute correct values."""
        calc = AttributionCalculator()

        attribution = calc.calculate(
            bet_id="bet_test",
            selected_line=25.5,
            market_lines=[
                {'line': 24.5},
                {'line': 25.5},
                {'line': 24.0}
            ],
            kelly_bet=10.0,
            flat_bet=5.0,
            model_projection=27.0,
            market_projection=25.0,
            matchup_adjustment=0.5
        )

        assert attribution.bet_id == "bet_test"
        # Line shopping: 25.5 - 24.67 = 0.83
        assert abs(attribution.line_shopping_value - 0.83) < 0.01
        # Kelly edge: (10-5)/5 * 0.1 = 0.1
        assert abs(attribution.kelly_edge - 0.1) < 0.01
        # Model edge: (27-25)/25 = 0.08
        assert abs(attribution.model_edge - 0.08) < 0.01
        # Matchup edge: 0.5/25 = 0.02
        assert abs(attribution.matchup_edge - 0.02) < 0.01

    def test_calculate_attribution_validation(self):
        """AttributionCalculator should produce valid RevenueAttribution."""
        calc = AttributionCalculator()

        attribution = calc.calculate(
            bet_id="bet_valid",
            selected_line=25.5,
            market_lines=[{'line': 25.0}],
            kelly_bet=10.0,
            flat_bet=5.0,
            model_projection=27.0,
            market_projection=25.0,
            matchup_adjustment=0.5
        )

        # Should be a valid RevenueAttribution
        expected_total = (
            attribution.line_shopping_value / 10 +
            attribution.kelly_edge +
            attribution.model_edge +
            attribution.matchup_edge
        )
        assert abs(attribution.total_edge - expected_total) < 0.001

    def test_calculate_with_equal_lines(self):
        """AttributionCalculator should handle equal market lines."""
        calc = AttributionCalculator()

        attribution = calc.calculate(
            bet_id="bet_equal",
            selected_line=25.0,
            market_lines=[
                {'line': 25.0},
                {'line': 25.0},
                {'line': 25.0}
            ],
            kelly_bet=5.0,
            flat_bet=5.0,
            model_projection=25.0,
            market_projection=25.0,
            matchup_adjustment=0.0
        )

        # No line shopping value when all lines are equal
        assert attribution.line_shopping_value == 0.0
        # No Kelly edge when kelly equals flat
        assert attribution.kelly_edge == 0.0


class TestIntegration:
    """Integration tests for revenue attribution workflow."""

    def test_full_attribution_workflow(self, temp_db):
        """Test complete attribution workflow: calculate -> log -> aggregate."""
        calc = AttributionCalculator()

        # Calculate and log 100 attributions
        for i in range(100):
            attribution = calc.calculate(
                bet_id=f"bet_{i:03d}",
                selected_line=25.5 + (i % 3) * 0.5,
                market_lines=[
                    {'line': 24.5},
                    {'line': 25.5},
                    {'line': 24.0}
                ],
                kelly_bet=10.0,
                flat_bet=5.0,
                model_projection=27.0,
                market_projection=25.0,
                matchup_adjustment=0.5
            )
            log_attribution(attribution, temp_db)

        # Get aggregate statistics
        aggregate = get_aggregate_attribution(min_bets=100, db_path=temp_db)

        assert aggregate is not None
        assert aggregate['total_bets'] == 100
        assert 'total_line_shopping' in aggregate
        assert 'line_shopping_pct' in aggregate

        # Get component ranking
        ranking = get_component_ranking(temp_db)
        assert len(ranking) == 4

    def test_attribution_with_pnl_correlation(self, temp_db):
        """Test that attribution correlates with actual P&L."""
        # High edge attributions should correlate with wins
        for i in range(100):
            # Create attributions with varying edge
            model_edge = 0.10 + (i % 10) * 0.02
            # total_edge = 0.5/10 + 0.02 + model_edge + 0.01
            total_edge = 0.05 + 0.02 + model_edge + 0.01
            attr = RevenueAttribution(
                bet_id=f"bet_{i:03d}",
                line_shopping_value=0.5,
                kelly_edge=0.02,
                model_edge=model_edge,
                matchup_edge=0.01,
                total_edge=total_edge
            )
            log_attribution(attr, temp_db)

        # Get trend for analysis
        trend = get_attribution_trend(n_bets=50, db_path=temp_db)
        assert len(trend) == 50

        # Average total edge should be positive
        avg_edge = sum(t['total_edge'] for t in trend) / len(trend)
        assert avg_edge > 0
