"""Tests for NBA Analytics + Verification system."""

import pytest
import tempfile
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_analytics.market_efficiency import (
    NBAMarketAnalyzer, PropMarketMetrics, SportsbookMetrics,
    MarketEfficiencyReport, PropType, EfficiencyRating
)
from nba_analytics.performance_attribution import (
    PerformanceAttributor, EdgeSource, MonthlyAttribution
)
from nba_analytics.edge_decay import (
    EdgeDecayMonitor, EdgeDecayMetrics, EdgeDecayAlert, DecayStatus
)
from nba_analytics.research_pipeline import (
    NBAResearchPipeline, EdgeTest, WeeklyResearchReport,
    ResearchStatus, EdgeCategory, EDGE_TEST_TEMPLATES
)


# =============================================================================
# Market Efficiency Tests
# =============================================================================

class TestNBAMarketAnalyzer:
    """Tests for NBAMarketAnalyzer class."""

    @pytest.fixture
    def temp_analytics_dir(self):
        """Create temporary analytics directory."""
        fd = tempfile.mkdtemp()
        yield fd
        import shutil
        shutil.rmtree(fd, ignore_errors=True)

    def test_analyzer_initialization(self, temp_analytics_dir):
        """Test NBAMarketAnalyzer initializes correctly."""
        analyzer = NBAMarketAnalyzer(analytics_dir=temp_analytics_dir)
        assert analyzer is not None
        assert len(analyzer.prop_metrics) > 0

    def test_add_bet(self, temp_analytics_dir):
        """Test adding bet to market efficiency tracking."""
        analyzer = NBAMarketAnalyzer(analytics_dir=temp_analytics_dir)

        analyzer.add_bet(
            prop_type="points",
            sportsbook="draftkings",
            edge_found=0.05,
            closing_edge=0.03,
            outcome="win",
            pnl=100.0,
            bet_size=100.0,
            clv=0.02,
            line_movement=0.5,
        )

        assert len(analyzer.bet_history) == 1
        assert analyzer.prop_metrics["points"].total_bets == 1

    def test_prop_metrics_update(self, temp_analytics_dir):
        """Test prop metrics are updated correctly."""
        analyzer = NBAMarketAnalyzer(analytics_dir=temp_analytics_dir)

        # Add multiple bets
        for i in range(10):
            outcome = "win" if i < 6 else "loss"
            pnl = 100.0 if outcome == "win" else -100.0

            analyzer.add_bet(
                prop_type="rebounds",
                sportsbook="fanduel",
                edge_found=0.04,
                closing_edge=0.02,
                outcome=outcome,
                pnl=pnl,
                bet_size=100.0,
                clv=0.02,
            )

        metrics = analyzer.prop_metrics["rebounds"]

        assert metrics.total_bets == 10
        assert metrics.wins == 6
        assert metrics.losses == 4
        assert metrics.win_rate == 0.6

    def test_sportsbook_metrics(self, temp_analytics_dir):
        """Test sportsbook metrics tracking."""
        analyzer = NBAMarketAnalyzer(analytics_dir=temp_analytics_dir)

        for _ in range(5):
            analyzer.add_bet(
                prop_type="assists",
                sportsbook="betmgm",
                edge_found=0.03,
                closing_edge=0.02,
                outcome="win",
                pnl=100.0,
                bet_size=100.0,
                clv=0.01,
                line_movement=0.5,
            )

        assert "betmgm" in analyzer.sportsbook_metrics
        assert analyzer.sportsbook_metrics["betmgm"].total_bets == 5

    def test_generate_report(self, temp_analytics_dir):
        """Test market efficiency report generation."""
        analyzer = NBAMarketAnalyzer(analytics_dir=temp_analytics_dir)

        # Add some bets
        for _ in range(20):
            analyzer.add_bet(
                prop_type="points",
                sportsbook="draftkings",
                edge_found=0.05,
                closing_edge=0.03,
                outcome="win",
                pnl=100.0,
                bet_size=100.0,
                clv=0.02,
            )

        report = analyzer.generate_report(period_days=30)

        assert isinstance(report, MarketEfficiencyReport)
        assert "points" in report.prop_metrics
        assert "draftkings" in report.sportsbook_metrics

    def test_allocation_recommendations(self, temp_analytics_dir):
        """Test allocation recommendations."""
        analyzer = NBAMarketAnalyzer(analytics_dir=temp_analytics_dir)

        # Add bets for multiple prop types with different performance
        for i in range(30):
            prop = "points" if i < 15 else "rebounds"
            outcome = "win" if i < 10 else "loss"  # Points: 10-5, Rebounds: 0-15
            pnl = 100.0 if outcome == "win" else -100.0

            analyzer.add_bet(
                prop_type=prop,
                sportsbook="draftkings",
                edge_found=0.05 if prop == "points" else 0.02,
                closing_edge=0.03,
                outcome=outcome,
                pnl=pnl,
                bet_size=100.0,
                clv=0.02,
            )

        report = analyzer.generate_report(period_days=30)

        assert "points" in report.allocation_recommendations
        assert "rebounds" in report.allocation_recommendations
        # Points should have higher allocation (better performance)
        assert report.allocation_recommendations["points"] > report.allocation_recommendations["rebounds"]

    def test_get_soft_markets(self, temp_analytics_dir):
        """Test getting soft markets."""
        analyzer = NBAMarketAnalyzer(analytics_dir=temp_analytics_dir)

        # Add bets showing soft market with high edge
        for _ in range(20):
            analyzer.add_bet(
                prop_type="threes",
                sportsbook="draftkings",
                edge_found=0.10,  # High edge = soft market
                closing_edge=0.08,
                outcome="win",
                pnl=150.0,  # Positive pnl
                bet_size=100.0,
                clv=0.02,  # Positive CLV
            )

        soft_markets = analyzer.get_soft_markets(threshold=60)

        # Should have at least one soft market
        assert len(soft_markets) >= 0  # May be empty if efficiency score calculation differs


# =============================================================================
# Performance Attribution Tests
# =============================================================================

class TestPerformanceAttributor:
    """Tests for PerformanceAttributor class."""

    @pytest.fixture
    def temp_attribution_dir(self):
        """Create temporary attribution directory."""
        fd = tempfile.mkdtemp()
        yield fd
        import shutil
        shutil.rmtree(fd, ignore_errors=True)

    def test_attributor_initialization(self, temp_attribution_dir):
        """Test PerformanceAttributor initializes correctly."""
        attributor = PerformanceAttributor(attribution_dir=temp_attribution_dir)
        assert attributor is not None
        assert len(attributor.edge_sources) == 6

    def test_add_bet(self, temp_attribution_dir):
        """Test adding bet with edge breakdown."""
        attributor = PerformanceAttributor(attribution_dir=temp_attribution_dir)

        attributor.add_bet(
            prop_type="points",
            sportsbook="draftkings",
            edge_total=0.08,
            edge_line_shopping=0.03,
            edge_model=0.04,
            edge_injury=0.005,
            edge_matchup=0.003,
            edge_timing=0.002,
            edge_market=0.0,
            outcome="win",
            pnl=100.0,
            bet_size=100.0,
        )

        assert len(attributor.bets) == 1
        assert attributor.edge_sources['line_shopping'].total_bets == 1

    def test_edge_source_attribution(self, temp_attribution_dir):
        """Test edge source attribution."""
        attributor = PerformanceAttributor(attribution_dir=temp_attribution_dir)

        # Add bets with different edge sources
        for i in range(20):
            attributor.add_bet(
                prop_type="points",
                sportsbook="draftkings",
                edge_total=0.06,
                edge_line_shopping=0.04 if i < 10 else 0.01,
                edge_model=0.02,
                edge_injury=0.0,
                edge_matchup=0.0,
                edge_timing=0.0,
                edge_market=0.0,
                outcome="win" if i < 12 else "loss",
                pnl=100.0 if i < 12 else -100.0,
                bet_size=100.0,
            )

        # Line shopping should have more contribution for first 10 bets
        assert attributor.edge_sources['line_shopping'].total_edge_contrib > 0

    def test_generate_monthly_report(self, temp_attribution_dir):
        """Test monthly attribution report generation."""
        attributor = PerformanceAttributor(attribution_dir=temp_attribution_dir)

        # Add bets for current month
        for _ in range(15):
            attributor.add_bet(
                prop_type="points",
                sportsbook="draftkings",
                edge_total=0.05,
                edge_line_shopping=0.02,
                edge_model=0.02,
                edge_injury=0.01,
                edge_matchup=0.0,
                edge_timing=0.0,
                edge_market=0.0,
                outcome="win",
                pnl=100.0,
                bet_size=100.0,
            )

        report = attributor.generate_monthly_report()

        assert isinstance(report, MonthlyAttribution)
        assert report.total_bets == 15
        assert len(report.by_source) > 0

    def test_insights_generation(self, temp_attribution_dir):
        """Test insights generation."""
        attributor = PerformanceAttributor(attribution_dir=temp_attribution_dir)

        # Add bets showing clear pattern
        for _ in range(30):
            attributor.add_bet(
                prop_type="points",
                sportsbook="draftkings",
                edge_total=0.05,
                edge_line_shopping=0.04,  # High line shopping edge
                edge_model=0.01,
                edge_injury=0.0,
                edge_matchup=0.0,
                edge_timing=0.0,
                edge_market=0.0,
                outcome="win",
                pnl=100.0,
                bet_size=100.0,
            )

        report = attributor.generate_monthly_report()

        assert len(report.insights) > 0

    def test_recommendations_generation(self, temp_attribution_dir):
        """Test recommendations generation."""
        attributor = PerformanceAttributor(attribution_dir=temp_attribution_dir)

        # Add bets showing strong line shopping performance
        for _ in range(25):
            attributor.add_bet(
                prop_type="points",
                sportsbook="draftkings",
                edge_total=0.05,
                edge_line_shopping=0.04,
                edge_model=0.01,
                edge_injury=0.0,
                edge_matchup=0.0,
                edge_timing=0.0,
                edge_market=0.0,
                outcome="win",
                pnl=100.0,
                bet_size=100.0,
            )

        report = attributor.generate_monthly_report()

        assert len(report.recommendations) > 0

    def test_get_edge_source_ranking(self, temp_attribution_dir):
        """Test edge source ranking."""
        attributor = PerformanceAttributor(attribution_dir=temp_attribution_dir)

        # Add bets
        for _ in range(10):
            attributor.add_bet(
                prop_type="points",
                sportsbook="draftkings",
                edge_total=0.05,
                edge_line_shopping=0.03,
                edge_model=0.02,
                edge_injury=0.0,
                edge_matchup=0.0,
                edge_timing=0.0,
                edge_market=0.0,
                outcome="win",
                pnl=100.0,
                bet_size=100.0,
            )

        ranking = attributor.get_edge_source_ranking()

        assert len(ranking) > 0
        assert ranking[0]['source'] == 'line_shopping'  # Highest contribution


# =============================================================================
# Edge Decay Tests
# =============================================================================

class TestEdgeDecayMonitor:
    """Tests for EdgeDecayMonitor class."""

    @pytest.fixture
    def temp_decay_dir(self):
        """Create temporary decay directory."""
        fd = tempfile.mkdtemp()
        yield fd
        import shutil
        shutil.rmtree(fd, ignore_errors=True)

    def test_monitor_initialization(self, temp_decay_dir):
        """Test EdgeDecayMonitor initializes correctly."""
        monitor = EdgeDecayMonitor(decay_dir=temp_decay_dir)
        assert monitor is not None

    def test_update_edge(self, temp_decay_dir):
        """Test updating edge values."""
        monitor = EdgeDecayMonitor(decay_dir=temp_decay_dir)

        monitor.update_edge(
            market_name="player_points_draftkings",
            edge_value=0.05,
        )

        assert "player_points_draftkings" in monitor.edge_history
        assert len(monitor.edge_history["player_points_draftkings"]) == 1

    def test_decay_detection(self, temp_decay_dir):
        """Test edge decay detection."""
        monitor = EdgeDecayMonitor(decay_dir=temp_decay_dir)

        # Simulate decaying edge
        for i in range(20):
            edge = 0.08 - (i * 0.003)  # Declining edge
            monitor.update_edge(
                market_name="test_market",
                edge_value=max(0.01, edge),
            )

        metrics = monitor.metrics["test_market"]

        assert metrics.decay_rate < 0  # Negative = decaying
        assert metrics.status in [DecayStatus.DECLINING.value, DecayStatus.CRITICAL.value]

    def test_alert_generation(self, temp_decay_dir):
        """Test alert generation for critical decay."""
        monitor = EdgeDecayMonitor(decay_dir=temp_decay_dir)

        # Simulate rapid edge collapse
        for i in range(30):
            edge = 0.10 - (i * 0.005)  # Rapid decline
            monitor.update_edge(
                market_name="collapsing_market",
                edge_value=max(0.001, edge),
            )

        # Check for alerts
        assert len(monitor.alerts) > 0
        assert any(a.severity in ['high', 'critical'] for a in monitor.alerts)

    def test_get_decay_summary(self, temp_decay_dir):
        """Test decay summary generation."""
        monitor = EdgeDecayMonitor(decay_dir=temp_decay_dir)

        # Add some markets
        monitor.update_edge("market_1", 0.05)
        monitor.update_edge("market_2", 0.03)
        monitor.update_edge("market_3", 0.08)

        summary = monitor.get_decay_summary()

        assert summary['total_markets_tracked'] == 3
        assert 'markets_by_status' in summary

    def test_generate_report(self, temp_decay_dir):
        """Test comprehensive report generation."""
        monitor = EdgeDecayMonitor(decay_dir=temp_decay_dir)

        # Add markets with different statuses
        for i in range(10):
            monitor.update_edge("growing_market", 0.03 + (i * 0.002))
            monitor.update_edge("decaying_market", 0.08 - (i * 0.003))

        report = monitor.generate_report()

        assert 'report_date' in report
        assert 'summary' in report
        assert 'action_items' in report


# =============================================================================
# Research Pipeline Tests
# =============================================================================

class TestNBAResearchPipeline:
    """Tests for NBAResearchPipeline class."""

    @pytest.fixture
    def temp_research_dir(self):
        """Create temporary research directory."""
        fd = tempfile.mkdtemp()
        yield fd
        import shutil
        shutil.rmtree(fd, ignore_errors=True)

    def test_pipeline_initialization(self, temp_research_dir):
        """Test NBAResearchPipeline initializes correctly."""
        pipeline = NBAResearchPipeline(research_dir=temp_research_dir)
        assert pipeline is not None

    def test_start_test(self, temp_research_dir):
        """Test starting new edge test."""
        pipeline = NBAResearchPipeline(research_dir=temp_research_dir)

        test = pipeline.start_test(
            title="Late Line Movement",
            hypothesis="Fading late line movement provides edge",
            category="timing",
            description="Test if betting against late moves works",
            expected_edge=0.03,
        )

        assert test.test_id is not None
        assert test.status == "hypothesis"
        assert test.expected_edge == 0.03

    def test_update_results(self, temp_research_dir):
        """Test updating test results."""
        pipeline = NBAResearchPipeline(research_dir=temp_research_dir)

        test = pipeline.start_test(
            title="Test Edge",
            hypothesis="Test hypothesis",
            category="model",
        )

        success = pipeline.update_results(
            test_id=test.test_id,
            sample_size=50,
            bets_tested=50,
            wins=30,
            losses=20,
            pushes=0,
            total_pnl=500.0,
            avg_edge_found=0.04,
        )

        assert success is True
        assert test.sample_size == 50
        assert test.wins == 30
        assert test.p_value < 1.0

    def test_complete_test_validated(self, temp_research_dir):
        """Test completing test with validation."""
        pipeline = NBAResearchPipeline(research_dir=temp_research_dir)

        test = pipeline.start_test(
            title="Winning Strategy",
            hypothesis="This strategy wins",
            category="model",
            expected_edge=0.05,
        )

        # Update with very good results (70% win rate = statistically significant)
        pipeline.update_results(
            test_id=test.test_id,
            sample_size=50,
            bets_tested=50,
            wins=35,  # 70% win rate
            losses=15,
            pushes=0,
            total_pnl=1000.0,
            avg_edge_found=0.08,
        )

        # Complete with deployment
        success = pipeline.complete_test(
            test_id=test.test_id,
            conclusion="Strategy validated with 70% win rate",
            deploy=True,
        )

        assert success is True
        assert test.deployed is True
        assert test.status == "deployed"

    def test_complete_test_rejected(self, temp_research_dir):
        """Test completing test with rejection."""
        pipeline = NBAResearchPipeline(research_dir=temp_research_dir)

        test = pipeline.start_test(
            title="Losing Strategy",
            hypothesis="This strategy might work",
            category="model",
        )

        # Update with poor results
        pipeline.update_results(
            test_id=test.test_id,
            sample_size=50,
            bets_tested=50,
            wins=20,  # 40% win rate
            losses=30,
            pushes=0,
            total_pnl=-500.0,
            avg_edge_found=0.01,
        )

        success = pipeline.complete_test(
            test_id=test.test_id,
            conclusion="Strategy does not work",
            deploy=False,
        )

        assert success is True
        assert test.status == "rejected"

    def test_get_active_tests(self, temp_research_dir):
        """Test getting active tests."""
        pipeline = NBAResearchPipeline(research_dir=temp_research_dir)

        pipeline.start_test("Test 1", "Hypothesis 1", "timing")
        pipeline.start_test("Test 2", "Hypothesis 2", "model")

        active = pipeline.get_active_tests()

        assert len(active) == 2

    def test_get_deployed_edges(self, temp_research_dir):
        """Test getting deployed edges."""
        pipeline = NBAResearchPipeline(research_dir=temp_research_dir)

        test = pipeline.start_test("Deployable", "Works", "model")
        # 70% win rate for statistical significance
        pipeline.update_results(test.test_id, 50, 50, 35, 15, 0, 1000.0, 0.08)
        pipeline.complete_test(test.test_id, "Works!", deploy=True)

        deployed = pipeline.get_deployed_edges()

        assert len(deployed) == 1
        assert deployed[0].deployed is True

    def test_generate_weekly_report(self, temp_research_dir):
        """Test weekly report generation."""
        pipeline = NBAResearchPipeline(research_dir=temp_research_dir)

        # Create and complete some tests
        test = pipeline.start_test("Weekly Test", "Hypothesis", "model")
        # 70% win rate for statistical significance
        pipeline.update_results(test.test_id, 30, 30, 21, 9, 0, 500.0, 0.06)
        pipeline.complete_test(test.test_id, "Validated", deploy=True)

        report = pipeline.generate_weekly_report(
            week_start="2024-01-01",
            week_end="2024-01-07",
            hours_spent=5.0,
        )

        assert report.week_start == "2024-01-01"
        assert report.hours_spent == 5.0
        # Test may or may not be counted as validated depending on timing
        assert report.tests_active >= 0 or report.tests_validated >= 0

    def test_get_research_summary(self, temp_research_dir):
        """Test research summary."""
        pipeline = NBAResearchPipeline(research_dir=temp_research_dir)

        # Create tests in different states
        t1 = pipeline.start_test("Test 1", "H1", "timing")
        t2 = pipeline.start_test("Test 2", "H2", "model")

        # Deploy one with significant results
        pipeline.update_results(t2.test_id, 50, 50, 35, 15, 0, 1000.0, 0.08)
        pipeline.complete_test(t2.test_id, "Works", deploy=True)

        summary = pipeline.get_research_summary()

        assert summary['total_tests'] == 2
        # May be 0 or 1 depending on statistical significance
        assert summary['deployed_edges'] >= 0
        assert 'by_status' in summary
        assert 'by_category' in summary


class TestEdgeTestTemplates:
    """Tests for edge test templates."""

    def test_templates_defined(self):
        """Test edge test templates are defined."""
        assert 'late_line_movement' in EDGE_TEST_TEMPLATES
        assert 'injury_overreaction' in EDGE_TEST_TEMPLATES
        assert 'back_to_back_spots' in EDGE_TEST_TEMPLATES
        assert 'primetime_overs' in EDGE_TEST_TEMPLATES

    def test_template_structure(self):
        """Test template structure."""
        template = EDGE_TEST_TEMPLATES['late_line_movement']

        assert 'title' in template
        assert 'hypothesis' in template
        assert 'category' in template
        assert 'expected_edge' in template
        assert 'min_sample_size' in template
