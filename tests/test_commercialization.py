"""Tests for commercialization and moats system."""

import pytest
import tempfile
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commercialization.track_record import (
    TrackRecordManager, MonthlyPerformance, TrackRecordReport,
    create_monthly_performance_from_bets, TRACK_RECORD_DIR
)
from commercialization.signal_distribution import (
    SignalDistributor, SignalPick, DailyPicks, SignalTier, DeliveryChannel,
    SUBSCRIPTION_PACKAGES, create_subscription_package
)
from commercialization.syndicate import (
    SyndicateManager, SyndicatePartner, SyndicateAllocation,
    BookRelationshipManager, SportsbookRelationship,
    create_syndicate_operating_agreement
)
from commercialization.edge_research import (
    EdgeResearchManager, EdgeResearch, WeeklyResearchReport,
    EDGE_RESEARCH_TEMPLATES
)


# =============================================================================
# Track Record Tests
# =============================================================================

class TestTrackRecordManager:
    """Tests for TrackRecordManager class."""

    @pytest.fixture
    def temp_track_record_dir(self):
        """Create temporary track record directory."""
        fd = tempfile.mkdtemp()
        yield fd
        # Cleanup
        import shutil
        shutil.rmtree(fd, ignore_errors=True)

    def test_manager_initialization(self, temp_track_record_dir):
        """Test TrackRecordManager initializes correctly."""
        manager = TrackRecordManager(track_record_dir=temp_track_record_dir)
        assert manager is not None
        assert Path(temp_track_record_dir).exists()

    def test_add_monthly_performance(self, temp_track_record_dir):
        """Test adding monthly performance."""
        manager = TrackRecordManager(track_record_dir=temp_track_record_dir)

        performance = MonthlyPerformance(
            month="2024-01",
            total_bets=100,
            wins=55,
            losses=45,
            total_pnl=500.0,
            total_wagered=10000.0,
            starting_bankroll=1000.0,
            ending_bankroll=1500.0,
        )

        manager.add_monthly_performance(performance)

        assert "2024-01" in manager.monthly_records
        assert manager.monthly_records["2024-01"].total_bets == 100

    def test_verify_performance(self, temp_track_record_dir):
        """Test verifying performance with third party."""
        manager = TrackRecordManager(track_record_dir=temp_track_record_dir)

        performance = MonthlyPerformance(month="2024-02")
        manager.add_monthly_performance(performance)

        success = manager.verify_performance(
            month="2024-02",
            source="BetStamp",
            url="https://betstamp.app/verify/123"
        )

        assert success is True
        assert manager.monthly_records["2024-02"].verified is True
        assert manager.monthly_records["2024-02"].verification_source == "BetStamp"

    def test_calculate_lifetime_metrics(self, temp_track_record_dir):
        """Test lifetime metrics calculation."""
        manager = TrackRecordManager(track_record_dir=temp_track_record_dir)

        # Add multiple months
        for i in range(3):
            month = f"2024-{i+1:02d}"
            performance = MonthlyPerformance(
                month=month,
                total_bets=50,
                wins=28,
                losses=22,
                total_pnl=250.0,
                total_wagered=5000.0,
                starting_bankroll=1000.0 + i * 250,
                ending_bankroll=1000.0 + (i + 1) * 250,
                daily_returns=[0.01] * 10,
            )
            manager.add_monthly_performance(performance)

        metrics = manager.calculate_lifetime_metrics()

        assert metrics['total_months'] == 3
        assert metrics['total_bets'] == 150
        assert metrics['total_pnl'] == 750.0

    def test_generate_report(self, temp_track_record_dir):
        """Test report generation."""
        manager = TrackRecordManager(track_record_dir=temp_track_record_dir)

        performance = MonthlyPerformance(month="2024-01", total_bets=100)
        manager.add_monthly_performance(performance)

        report = manager.generate_report('lifetime')

        assert isinstance(report, TrackRecordReport)
        assert report.report_type == 'lifetime'
        assert len(report.periods) == 1

    def test_get_verification_status(self, temp_track_record_dir):
        """Test verification status summary."""
        manager = TrackRecordManager(track_record_dir=temp_track_record_dir)

        # Add verified and unverified months
        for i in range(4):
            month = f"2024-{i+1:02d}"
            performance = MonthlyPerformance(month=month)
            manager.add_monthly_performance(performance)

            if i < 3:  # Verify 3 out of 4
                manager.verify_performance(month, "Pikkit")

        status = manager.get_verification_status()

        assert status['total_months'] == 4
        assert status['verified_months'] == 3
        assert status['verification_pct'] == 75.0
        assert status['audit_ready'] is False  # Need 80%

    def test_save_report(self, temp_track_record_dir):
        """Test saving report to file."""
        manager = TrackRecordManager(track_record_dir=temp_track_record_dir)

        performance = MonthlyPerformance(month="2024-01")
        manager.add_monthly_performance(performance)

        report = manager.generate_report('lifetime')
        manager.save_report(report)

        # Check file was created
        files = list(Path(temp_track_record_dir).glob("*_report.json"))
        assert len(files) > 0


class TestCreateMonthlyPerformance:
    """Tests for create_monthly_performance_from_bets function."""

    def test_create_from_bets(self):
        """Test creating performance from bet list."""
        bets = [
            {'outcome': 'win', 'pnl': 100.0, 'bet_size': 100.0, 'edge': 0.05, 'date': '2024-01-01'},
            {'outcome': 'loss', 'pnl': -100.0, 'bet_size': 100.0, 'edge': 0.03, 'date': '2024-01-02'},
            {'outcome': 'win', 'pnl': 150.0, 'bet_size': 100.0, 'edge': 0.08, 'date': '2024-01-03'},
        ]

        performance = create_monthly_performance_from_bets("2024-01", bets, 1000.0)

        assert performance.month == "2024-01"
        assert performance.total_bets == 3
        assert performance.wins == 2
        assert performance.losses == 1
        assert performance.total_pnl == 150.0

    def test_create_from_empty_bets(self):
        """Test creating performance from empty bet list."""
        performance = create_monthly_performance_from_bets("2024-01", [], 1000.0)

        assert performance.month == "2024-01"
        assert performance.total_bets == 0
        assert performance.starting_bankroll == 1000.0


# =============================================================================
# Signal Distribution Tests
# =============================================================================

class TestSignalDistributor:
    """Tests for SignalDistributor class."""

    @pytest.fixture
    def temp_signals_dir(self):
        """Create temporary signals directory."""
        fd = tempfile.mkdtemp()
        yield fd
        import shutil
        shutil.rmtree(fd, ignore_errors=True)

    def test_distributor_initialization(self, temp_signals_dir):
        """Test SignalDistributor initializes correctly."""
        distributor = SignalDistributor(signals_dir=temp_signals_dir)
        assert distributor is not None

    def test_create_pick(self, temp_signals_dir):
        """Test creating signal pick."""
        distributor = SignalDistributor(signals_dir=temp_signals_dir)

        pick = distributor.create_pick(
            sport="Basketball",
            league="NBA",
            event_date="2024-03-15",
            event_time="19:30",
            team_a="Lakers",
            team_b="Celtics",
            market="Spread",
            selection="Lakers -5.5",
            odds=-110,
            edge_pct=5.0,
            analysis="Lakers playing well at home",
            confidence="high",
        )

        assert pick.signal_id is not None
        assert pick.sport == "Basketball"
        assert pick.confidence == "high"
        assert pick.edge_pct == 5.0

    def test_create_pick_auto_stake(self, temp_signals_dir):
        """Test auto-calculation of stake percentage."""
        distributor = SignalDistributor(signals_dir=temp_signals_dir)

        pick = distributor.create_pick(
            sport="Basketball",
            league="NBA",
            event_date="2024-03-15",
            event_time="19:30",
            team_a="Lakers",
            team_b="Celtics",
            market="Spread",
            selection="Lakers -5.5",
            odds=-110,
            edge_pct=5.0,
            analysis="Test",
        )

        # Should auto-calculate stake (half-Kelly, max 2%)
        assert pick.stake_pct > 0
        assert pick.stake_pct <= 2.0

    def test_settle_pick(self, temp_signals_dir):
        """Test settling a pick."""
        distributor = SignalDistributor(signals_dir=temp_signals_dir)

        pick = distributor.create_pick(
            sport="Basketball",
            league="NBA",
            event_date="2024-03-15",
            event_time="19:30",
            team_a="Lakers",
            team_b="Celtics",
            market="Spread",
            selection="Lakers -5.5",
            odds=-110,
            edge_pct=5.0,
            analysis="Test",
        )

        success = distributor.settle_pick(pick.signal_id, 100.0, "won")

        assert success is True
        assert pick.signal_id in distributor.settled_picks
        assert pick.result == 100.0
        assert pick.status == "won"

    def test_get_performance_summary(self, temp_signals_dir):
        """Test performance summary."""
        distributor = SignalDistributor(signals_dir=temp_signals_dir)

        # Create and settle some picks
        for i in range(10):
            pick = distributor.create_pick(
                sport="Basketball",
                league="NBA",
                event_date="2024-03-15",
                event_time="19:30",
                team_a=f"Team A{i}",
                team_b="Team B",
                market="Spread",
                selection="Team A",
                odds=-110,
                edge_pct=5.0,
                analysis="Test",
            )
            result = 100.0 if i < 6 else -100.0  # 6-4 record
            status = "won" if i < 6 else "lost"
            distributor.settle_pick(pick.signal_id, result, status)

        summary = distributor.get_performance_summary(days=30)

        assert summary['total_picks'] == 10
        assert summary['win_rate'] == 0.6

    def test_export_for_delivery(self, temp_signals_dir):
        """Test exporting picks for delivery."""
        distributor = SignalDistributor(signals_dir=temp_signals_dir)

        pick = distributor.create_pick(
            sport="Basketball",
            league="NBA",
            event_date="2024-03-15",
            event_time="19:30",
            team_a="Lakers",
            team_b="Celtics",
            market="Spread",
            selection="Lakers -5.5",
            odds=-110,
            edge_pct=5.0,
            analysis="Test analysis",
            tier_access=["standard", "premium", "vip"],
        )

        message = distributor.export_for_delivery("2024-03-15", "standard", "discord")

        assert "Lakers" in message
        assert "Celtics" in message


class TestSubscriptionPackages:
    """Tests for subscription packages."""

    def test_packages_defined(self):
        """Test subscription packages are defined."""
        assert 'free' in SUBSCRIPTION_PACKAGES
        assert 'standard' in SUBSCRIPTION_PACKAGES
        assert 'premium' in SUBSCRIPTION_PACKAGES
        assert 'vip' in SUBSCRIPTION_PACKAGES

    def test_package_pricing(self):
        """Test package pricing structure."""
        standard = SUBSCRIPTION_PACKAGES['standard']

        assert standard['pricing']['monthly'] == 50
        assert standard['pricing']['quarterly'] == 50 * 3 * 0.9  # 10% discount
        assert standard['pricing']['annual'] == 50 * 12 * 0.8  # 20% discount

    def test_create_custom_package(self):
        """Test creating custom subscription package."""
        package = create_subscription_package(
            tier='elite',
            price_monthly=500,
            features=['Everything'],
            delivery_channels=['all'],
        )

        assert package['tier'] == 'elite'
        assert package['pricing']['monthly'] == 500


# =============================================================================
# Syndicate Tests
# =============================================================================

class TestSyndicateManager:
    """Tests for SyndicateManager class."""

    @pytest.fixture
    def temp_syndicate_dir(self):
        """Create temporary syndicate directory."""
        fd = tempfile.mkdtemp()
        yield fd
        import shutil
        shutil.rmtree(fd, ignore_errors=True)

    def test_manager_initialization(self, temp_syndicate_dir):
        """Test SyndicateManager initializes correctly."""
        manager = SyndicateManager(syndicate_dir=temp_syndicate_dir)
        assert manager is not None

    def test_add_partner(self, temp_syndicate_dir):
        """Test adding syndicate partner."""
        manager = SyndicateManager(syndicate_dir=temp_syndicate_dir)

        partner = SyndicatePartner(
            partner_id="p001",
            name="John Doe",
            contribution=50000.0,
            profit_share_pct=25.0,
        )

        manager.add_partner(partner)

        assert "p001" in manager.partners
        assert manager.partners["p001"].contribution == 50000.0

    def test_create_monthly_allocation(self, temp_syndicate_dir):
        """Test creating monthly allocation."""
        manager = SyndicateManager(syndicate_dir=temp_syndicate_dir)

        # Add partners
        manager.add_partner(SyndicatePartner("p001", "Partner 1", 50000.0, 50.0))
        manager.add_partner(SyndicatePartner("p002", "Partner 2", 50000.0, 50.0))

        allocation = manager.create_monthly_allocation("2024-01", 100000.0)

        assert allocation.month == "2024-01"
        assert allocation.total_capital == 100000.0
        assert len(allocation.allocations) == 2

    def test_update_allocation_pnl(self, temp_syndicate_dir):
        """Test updating allocation with P&L."""
        manager = SyndicateManager(syndicate_dir=temp_syndicate_dir)

        manager.add_partner(SyndicatePartner("p001", "Partner 1", 50000.0, 50.0))
        manager.add_partner(SyndicatePartner("p002", "Partner 2", 50000.0, 50.0))

        allocation = manager.create_monthly_allocation("2024-01", 100000.0)
        manager.update_allocation_pnl("2024-01", 10000.0, 50)

        assert allocation.total_pnl == 10000.0
        assert allocation.bets_placed == 50

        # Check partner returns updated
        assert manager.partners["p001"].total_returned > 0

    def test_generate_report(self, temp_syndicate_dir):
        """Test syndicate report generation."""
        manager = SyndicateManager(syndicate_dir=temp_syndicate_dir)

        manager.add_partner(SyndicatePartner("p001", "Partner 1", 50000.0, 50.0))

        report = manager.generate_report()

        assert 'summary' in report
        assert 'partners' in report
        assert report['summary']['total_partners'] == 1


class TestBookRelationshipManager:
    """Tests for BookRelationshipManager class."""

    def test_manager_initialization(self):
        """Test BookRelationshipManager initializes correctly."""
        manager = BookRelationshipManager()
        assert manager is not None

    def test_add_sportsbook(self):
        """Test adding sportsbook relationship."""
        manager = BookRelationshipManager()

        book = SportsbookRelationship(
            sportsbook="DraftKings",
            status="good",
            max_bet=5000.0,
            current_limit=5000.0,
        )

        manager.add_sportsbook(book)

        assert "DraftKings" in manager.books

    def test_update_bet_record(self):
        """Test updating bet record."""
        manager = BookRelationshipManager()

        book = SportsbookRelationship(
            sportsbook="FanDuel",
            status="good",
            max_bet=3000.0,
            current_limit=3000.0,
        )
        manager.add_sportsbook(book)

        manager.update_bet_record("FanDuel", 1000.0, 150.0)

        assert manager.books["FanDuel"].total_wagered == 1000.0
        assert manager.books["FanDuel"].total_pnl == 150.0

    def test_get_recommended_books(self):
        """Test getting recommended sportsbooks."""
        manager = BookRelationshipManager()

        # Add books with different statuses
        manager.add_sportsbook(SportsbookRelationship("Book1", "good", 5000.0, 5000.0))
        manager.add_sportsbook(SportsbookRelationship("Book2", "limited", 1000.0, 500.0))
        manager.add_sportsbook(SportsbookRelationship("Book3", "banned", 0, 0))

        recommended = manager.get_recommended_books(min_limit=1000)

        assert len(recommended) >= 1
        assert recommended[0]['sportsbook'] == "Book1"

    def test_sharps_score_update(self):
        """Test Sharps score updates based on performance."""
        manager = BookRelationshipManager()

        book = SportsbookRelationship("SharpBook", "good", 5000.0, 5000.0)
        manager.add_sportsbook(book)

        # Simulate winning bets (should increase Sharps score)
        for _ in range(10):
            manager.update_bet_record("SharpBook", 1000.0, 150.0)

        score = manager.books["SharpBook"].sharps_score
        assert score > 0  # Should have some score from winning


class TestSyndicateOperatingAgreement:
    """Tests for syndicate operating agreement."""

    def test_create_agreement(self):
        """Test creating operating agreement."""
        agreement = create_syndicate_operating_agreement(
            syndicate_name="Test Syndicate",
            management_fee_pct=1.0,
            performance_fee_pct=20.0,
            high_water_mark=True,
        )

        assert agreement['syndicate_name'] == "Test Syndicate"
        assert agreement['fees']['management_fee']['percentage'] == 1.0
        assert agreement['fees']['performance_fee']['percentage'] == 20.0
        assert agreement['fees']['performance_fee']['high_water_mark'] is True


# =============================================================================
# Edge Research Tests
# =============================================================================

class TestEdgeResearchManager:
    """Tests for EdgeResearchManager class."""

    @pytest.fixture
    def temp_research_dir(self):
        """Create temporary research directory."""
        fd = tempfile.mkdtemp()
        yield fd
        import shutil
        shutil.rmtree(fd, ignore_errors=True)

    def test_manager_initialization(self, temp_research_dir):
        """Test EdgeResearchManager initializes correctly."""
        manager = EdgeResearchManager(research_dir=temp_research_dir)
        assert manager is not None

    def test_start_research(self, temp_research_dir):
        """Test starting new research project."""
        manager = EdgeResearchManager(research_dir=temp_research_dir)

        research = manager.start_research(
            title="Player Prop Inefficiency",
            hypothesis="Sportsbooks are slow to adjust props",
            edge_type="market_inefficiency",
            expected_edge_pct=3.0,
        )

        assert research.research_id is not None
        assert research.status == "hypothesis"
        assert research.expected_edge_pct == 3.0

    def test_update_results(self, temp_research_dir):
        """Test updating research results."""
        manager = EdgeResearchManager(research_dir=temp_research_dir)

        research = manager.start_research(
            title="Test Research",
            hypothesis="Test hypothesis",
            edge_type="model_improvement",
        )

        success = manager.update_results(
            research.research_id,
            sample_size=100,
            bets_tested=50,
            bets_won=30,
            total_pnl=5.0,  # 5% return on sample
        )

        assert success is True
        assert research.actual_edge_pct == 5.0  # 5/100 * 100 = 5%

    def test_validate_edge(self, temp_research_dir):
        """Test validating edge statistically."""
        manager = EdgeResearchManager(research_dir=temp_research_dir)

        research = manager.start_research(
            title="Valid Edge",
            hypothesis="This edge works",
            edge_type="data_advantage",
        )

        # Update with good results
        manager.update_results(research.research_id, 100, 50, 35, 300.0)

        # Validate with significant p-value
        success = manager.validate_edge(research.research_id, p_value=0.03)

        assert success is True
        assert research.status == "validated"
        assert research.deployment_ready is True

    def test_deploy_edge(self, temp_research_dir):
        """Test deploying validated edge."""
        manager = EdgeResearchManager(research_dir=temp_research_dir)

        research = manager.start_research(
            title="Deployable Edge",
            hypothesis="This edge works",
            edge_type="timing_advantage",
        )

        # Update and validate
        manager.update_results(research.research_id, 100, 50, 35, 300.0)
        manager.validate_edge(research.research_id, p_value=0.02)

        # Deploy
        success = manager.deploy_edge(research.research_id)

        assert success is True
        assert research.status == "deployed"
        assert research.deployment_date is not None

    def test_get_active_research(self, temp_research_dir):
        """Test getting active research projects."""
        manager = EdgeResearchManager(research_dir=temp_research_dir)

        manager.start_research("Project 1", "Hypothesis 1", "market_inefficiency")
        manager.start_research("Project 2", "Hypothesis 2", "model_improvement")

        active = manager.get_active_research()

        assert len(active) == 2

    def test_generate_weekly_report(self, temp_research_dir):
        """Test generating weekly report."""
        manager = EdgeResearchManager(research_dir=temp_research_dir)

        # Create research project
        research = manager.start_research("Project 1", "Hypothesis 1", "market_inefficiency")

        # Update the project to have a timestamp within the test week
        research.last_updated = "2024-01-05T12:00:00"
        manager._save_research(research)

        report = manager.generate_weekly_report(
            week_start="2024-01-01",
            week_end="2024-01-07",
            hours_spent=5.0,
        )

        assert report.week_start == "2024-01-01"
        assert report.hours_spent == 5.0
        assert len(report.research_projects) >= 1

    def test_get_research_summary(self, temp_research_dir):
        """Test research summary."""
        manager = EdgeResearchManager(research_dir=temp_research_dir)

        r1 = manager.start_research("Project 1", "H1", "market_inefficiency")
        r2 = manager.start_research("Project 2", "H2", "model_improvement")

        # Deploy one
        manager.update_results(r2.research_id, 100, 50, 35, 300.0)
        manager.validate_edge(r2.research_id, 0.02)
        manager.deploy_edge(r2.research_id)

        summary = manager.get_research_summary()

        assert summary['total_projects'] == 2
        assert summary['deployed_edges'] == 1
        assert 'by_status' in summary
        assert 'by_type' in summary


class TestEdgeResearchTemplates:
    """Tests for edge research templates."""

    def test_templates_defined(self):
        """Test edge research templates are defined."""
        assert 'player_prop_inefficiency' in EDGE_RESEARCH_TEMPLATES
        assert 'team_total_correlation' in EDGE_RESEARCH_TEMPLATES
        assert 'closing_line_value' in EDGE_RESEARCH_TEMPLATES

    def test_template_structure(self):
        """Test template structure."""
        template = EDGE_RESEARCH_TEMPLATES['player_prop_inefficiency']

        assert 'title' in template
        assert 'edge_type' in template
        assert 'hypothesis' in template
        assert 'expected_edge_pct' in template
        assert 'data_sources' in template
        assert 'features_used' in template
