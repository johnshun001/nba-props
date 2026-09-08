"""Tests for SQLite bet tracking."""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from schemas.validation import Bet
from storage.bet_tracker import (
    init_db, log_bet, update_bet_outcome, get_bet, get_all_bets,
    get_unsettled_bets, get_settled_bets, get_bets_by_player,
    get_bets_by_prop_type, get_betting_summary, get_bets_by_date_range,
    delete_bet, clear_all_bets, get_db_connection
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
def sample_bet():
    """Create a sample bet for testing."""
    return Bet(
        player_id="lebron_james",
        prop_type="points",
        line=25.5,
        bet_size=10.0,
        edge=0.08,
        bet_side="over",
        sportsbook="draftkings",
        odds=-110,
        timestamp=datetime.utcnow()
    )


class TestInitDB:
    """Tests for database initialization."""

    def test_init_db_creates_table(self, temp_db):
        """init_db should create the bets table."""
        with get_db_connection(temp_db) as conn:
            result = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='bets'
            """).fetchone()

            assert result is not None
            assert result[0] == 'bets'

    def test_init_db_creates_indexes(self, temp_db):
        """init_db should create required indexes."""
        with get_db_connection(temp_db) as conn:
            indexes = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND tbl_name='bets'
            """).fetchall()

            index_names = [idx[0] for idx in indexes]
            assert 'idx_bets_player' in index_names
            assert 'idx_bets_prop_type' in index_names
            assert 'idx_bets_timestamp' in index_names


class TestLogBet:
    """Tests for logging bets."""

    def test_log_bet_success(self, temp_db, sample_bet):
        """log_bet should successfully insert a bet."""
        bet_id = log_bet(sample_bet, temp_db)

        assert bet_id > 0

        bet = get_bet(bet_id, temp_db)
        assert bet is not None
        assert bet['player_id'] == sample_bet.player_id
        assert bet['prop_type'] == sample_bet.prop_type
        assert bet['line'] == sample_bet.line
        assert bet['bet_size'] == sample_bet.bet_size

    def test_log_bet_multiple_bets(self, temp_db, sample_bet):
        """Should be able to log multiple bets."""
        bet_id1 = log_bet(sample_bet, temp_db)
        bet_id2 = log_bet(sample_bet, temp_db)

        assert bet_id1 != bet_id2

        all_bets = get_all_bets(temp_db)
        assert len(all_bets) == 2


class TestUpdateBetOutcome:
    """Tests for updating bet outcomes."""

    def test_update_bet_outcome_win(self, temp_db, sample_bet):
        """update_bet_outcome should update a winning bet."""
        bet_id = log_bet(sample_bet, temp_db)

        pnl = 9.09  # Win at -110
        success = update_bet_outcome(bet_id, "win", pnl, temp_db)

        assert success is True

        bet = get_bet(bet_id, temp_db)
        assert bet['outcome'] == 'win'
        assert abs(bet['pnl'] - pnl) < 0.01
        assert bet['settled_at'] is not None

    def test_update_bet_outcome_loss(self, temp_db, sample_bet):
        """update_bet_outcome should update a losing bet."""
        bet_id = log_bet(sample_bet, temp_db)

        pnl = -10.0  # Loss
        success = update_bet_outcome(bet_id, "loss", pnl, temp_db)

        assert success is True

        bet = get_bet(bet_id, temp_db)
        assert bet['outcome'] == 'loss'
        assert bet['pnl'] == pnl

    def test_update_bet_outcome_push(self, temp_db, sample_bet):
        """update_bet_outcome should update a push."""
        bet_id = log_bet(sample_bet, temp_db)

        pnl = 0.0  # Push
        success = update_bet_outcome(bet_id, "push", pnl, temp_db)

        assert success is True

        bet = get_bet(bet_id, temp_db)
        assert bet['outcome'] == 'push'
        assert bet['pnl'] == 0.0

    def test_update_nonexistent_bet(self, temp_db):
        """update_bet_outcome should return False for nonexistent bet."""
        success = update_bet_outcome(99999, "win", 10.0, temp_db)
        assert success is False


class TestGetBets:
    """Tests for retrieving bets."""

    def test_get_all_bets(self, temp_db, sample_bet):
        """get_all_bets should return all bets."""
        log_bet(sample_bet, temp_db)
        log_bet(sample_bet, temp_db)
        log_bet(sample_bet, temp_db)

        all_bets = get_all_bets(temp_db)
        assert len(all_bets) == 3

    def test_get_unsettled_bets(self, temp_db, sample_bet):
        """get_unsettled_bets should return only unsettled bets."""
        bet_id1 = log_bet(sample_bet, temp_db)
        bet_id2 = log_bet(sample_bet, temp_db)

        update_bet_outcome(bet_id1, "win", 9.09, temp_db)

        unsettled = get_unsettled_bets(temp_db)
        assert len(unsettled) == 1
        assert unsettled[0]['id'] == bet_id2

    def test_get_settled_bets(self, temp_db, sample_bet):
        """get_settled_bets should return only settled bets."""
        bet_id1 = log_bet(sample_bet, temp_db)
        bet_id2 = log_bet(sample_bet, temp_db)

        update_bet_outcome(bet_id1, "win", 9.09, temp_db)

        settled = get_settled_bets(temp_db)
        assert len(settled) == 1
        assert settled[0]['id'] == bet_id1

    def test_get_bets_by_player(self, temp_db, sample_bet):
        """get_bets_by_player should filter by player."""
        log_bet(sample_bet, temp_db)

        # Create bet for different player
        other_bet = Bet(
            player_id="curry_lebron",
            prop_type="points",
            line=28.5,
            bet_size=10.0,
            edge=0.06,
            bet_side="over",
            sportsbook="fanduel",
            odds=-110
        )
        log_bet(other_bet, temp_db)

        lebron_bets = get_bets_by_player("lebron_james", temp_db)
        assert len(lebron_bets) == 1

    def test_get_bets_by_prop_type(self, temp_db, sample_bet):
        """get_bets_by_prop_type should filter by prop type."""
        log_bet(sample_bet, temp_db)

        # Create bet for different prop type
        reb_bet = Bet(
            player_id="lebron_james",
            prop_type="rebounds",
            line=8.5,
            bet_size=10.0,
            edge=0.05,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110
        )
        log_bet(reb_bet, temp_db)

        points_bets = get_bets_by_prop_type("points", temp_db)
        assert len(points_bets) == 1


class TestBettingSummary:
    """Tests for betting summary statistics."""

    def test_get_betting_summary_empty(self, temp_db):
        """get_betting_summary should handle empty database."""
        summary = get_betting_summary(temp_db)

        assert summary['total_bets'] == 0
        assert summary['wins'] == 0
        assert summary['losses'] == 0
        assert summary['total_pnl'] == 0.0

    def test_get_betting_summary_with_bets(self, temp_db, sample_bet):
        """get_betting_summary should calculate correct statistics."""
        # Log 3 bets: 2 wins, 1 loss
        bet_id1 = log_bet(sample_bet, temp_db)
        bet_id2 = log_bet(sample_bet, temp_db)
        bet_id3 = log_bet(sample_bet, temp_db)

        update_bet_outcome(bet_id1, "win", 9.09, temp_db)
        update_bet_outcome(bet_id2, "win", 9.09, temp_db)
        update_bet_outcome(bet_id3, "loss", -10.0, temp_db)

        summary = get_betting_summary(temp_db)

        assert summary['total_bets'] == 3
        assert summary['wins'] == 2
        assert summary['losses'] == 1
        assert summary['win_rate'] == 2/3  # 2 wins / 3 total decisive bets
        assert abs(summary['total_pnl'] - 8.18) < 0.01

    def test_get_betting_summary_roi(self, temp_db, sample_bet):
        """get_betting_summary should calculate ROI."""
        bet_id = log_bet(sample_bet, temp_db)
        update_bet_outcome(bet_id, "win", 9.09, temp_db)

        summary = get_betting_summary(temp_db)

        # ROI = (9.09 / 10.0) * 100 = 90.9%
        assert abs(summary['roi'] - 90.9) < 0.1


class TestDateRangeQueries:
    """Tests for date range queries."""

    def test_get_bets_by_date_range(self, temp_db, sample_bet):
        """get_bets_by_date_range should filter by date."""
        # Bet in the past
        past_bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110,
            timestamp=datetime.utcnow() - timedelta(days=7)
        )

        log_bet(past_bet, temp_db)
        log_bet(sample_bet, temp_db)  # Current bet

        # Query last 3 days
        start = datetime.utcnow() - timedelta(days=3)
        end = datetime.utcnow()

        recent_bets = get_bets_by_date_range(start, end, temp_db)
        assert len(recent_bets) == 1

    def test_get_bets_by_date_range_all(self, temp_db, sample_bet):
        """get_bets_by_date_range should return all if range is wide enough."""
        past_bet = Bet(
            player_id="lebron_james",
            prop_type="points",
            line=25.5,
            bet_size=10.0,
            edge=0.08,
            bet_side="over",
            sportsbook="draftkings",
            odds=-110,
            timestamp=datetime.utcnow() - timedelta(days=7)
        )

        log_bet(past_bet, temp_db)
        log_bet(sample_bet, temp_db)

        # Query wide range
        start = datetime.utcnow() - timedelta(days=30)
        end = datetime.utcnow()

        all_bets = get_bets_by_date_range(start, end, temp_db)
        assert len(all_bets) == 2


class TestDeleteBet:
    """Tests for deleting bets."""

    def test_delete_bet_success(self, temp_db, sample_bet):
        """delete_bet should remove a bet."""
        bet_id = log_bet(sample_bet, temp_db)

        success = delete_bet(bet_id, temp_db)
        assert success is True

        bet = get_bet(bet_id, temp_db)
        assert bet is None

    def test_delete_nonexistent_bet(self, temp_db):
        """delete_bet should return False for nonexistent bet."""
        success = delete_bet(99999, temp_db)
        assert success is False


class TestClearAllBets:
    """Tests for clearing all bets."""

    def test_clear_all_bets(self, temp_db, sample_bet):
        """clear_all_bets should remove all bets."""
        log_bet(sample_bet, temp_db)
        log_bet(sample_bet, temp_db)
        log_bet(sample_bet, temp_db)

        clear_all_bets(temp_db)

        all_bets = get_all_bets(temp_db)
        assert len(all_bets) == 0


class TestIntegration:
    """Integration tests for bet tracking workflow."""

    def test_full_bet_lifecycle(self, temp_db, sample_bet):
        """Test complete bet lifecycle: log -> settle -> query."""
        # Log bet
        bet_id = log_bet(sample_bet, temp_db)

        # Verify it's unsettled
        unsettled = get_unsettled_bets(temp_db)
        assert len(unsettled) == 1
        assert unsettled[0]['id'] == bet_id

        # Settle the bet
        update_bet_outcome(bet_id, "win", 9.09, temp_db)

        # Verify it's settled
        settled = get_settled_bets(temp_db)
        assert len(settled) == 1
        assert settled[0]['outcome'] == 'win'

        # Verify summary is updated
        summary = get_betting_summary(temp_db)
        assert summary['wins'] == 1
        assert summary['total_pnl'] > 0

    def test_multiple_players_and_props(self, temp_db):
        """Test tracking bets across multiple players and prop types."""
        bets_data = [
            {"player_id": "lebron_james", "prop_type": "points", "line": 25.5},
            {"player_id": "lebron_james", "prop_type": "rebounds", "line": 8.5},
            {"player_id": "curry_stephen", "prop_type": "points", "line": 28.5},
            {"player_id": "curry_stephen", "prop_type": "threes", "line": 4.5},
            {"player_id": "jokic_nikola", "prop_type": "assists", "line": 9.5},
        ]

        for data in bets_data:
            bet = Bet(
                player_id=data["player_id"],
                prop_type=data["prop_type"],
                line=data["line"],
                bet_size=10.0,
                edge=0.05,
                bet_side="over",
                sportsbook="draftkings",
                odds=-110
            )
            log_bet(bet, temp_db)

        # Verify total count
        all_bets = get_all_bets(temp_db)
        assert len(all_bets) == 5

        # Verify player filtering
        lebron_bets = get_bets_by_player("lebron_james", temp_db)
        assert len(lebron_bets) == 2

        # Verify prop type filtering
        points_bets = get_bets_by_prop_type("points", temp_db)
        assert len(points_bets) == 2
