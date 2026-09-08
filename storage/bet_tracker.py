"""SQLite bet tracking for NBA prop betting system.

Logs every bet placed to a SQLite database for tracking and analysis.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from schemas.validation import Bet, BetResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "bets.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL,
    prop_type TEXT NOT NULL,
    line REAL NOT NULL,
    bet_size REAL NOT NULL,
    bet_side TEXT NOT NULL,
    sportsbook TEXT NOT NULL,
    odds INTEGER NOT NULL,
    edge REAL NOT NULL,
    outcome TEXT,
    pnl REAL,
    timestamp DATETIME NOT NULL,
    settled_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_bets_player ON bets(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_bets_prop_type ON bets(prop_type);",
    "CREATE INDEX IF NOT EXISTS idx_bets_timestamp ON bets(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_bets_outcome ON bets(outcome);",
]


@contextmanager
def get_db_connection(db_path: str = DB_PATH):
    """Context manager for database connections."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH):
    """Initialize the database with required tables and indexes."""
    with get_db_connection(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        for index_sql in CREATE_INDEX_SQL:
            conn.execute(index_sql)
        conn.commit()


def log_bet(bet: Bet, db_path: str = DB_PATH) -> int:
    """
    Log a bet to the database.

    Args:
        bet: Bet object to log
        db_path: Path to SQLite database

    Returns:
        ID of the inserted bet
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO bets (
                player_id, prop_type, line, bet_size, bet_side,
                sportsbook, odds, edge, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            bet.player_id,
            bet.prop_type,
            bet.line,
            bet.bet_size,
            bet.bet_side,
            bet.sportsbook,
            bet.odds,
            bet.edge,
            bet.timestamp.isoformat()
        ])
        conn.commit()
        return cursor.lastrowid


def update_bet_outcome(bet_id: int, outcome: str, pnl: float,
                       db_path: str = DB_PATH) -> bool:
    """
    Update a bet with its outcome and P&L.

    Args:
        bet_id: ID of the bet to update
        outcome: Bet outcome ('win', 'loss', 'push')
        pnl: Profit and loss in dollars
        db_path: Path to SQLite database

    Returns:
        True if update was successful
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("""
            UPDATE bets
            SET outcome = ?, pnl = ?, settled_at = ?
            WHERE id = ?
        """, [outcome, pnl, datetime.utcnow().isoformat(), bet_id])
        conn.commit()
        return cursor.rowcount > 0


def get_bet(bet_id: int, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Get a single bet by ID.

    Args:
        bet_id: ID of the bet
        db_path: Path to SQLite database

    Returns:
        Bet as dictionary or None if not found
    """
    with get_db_connection(db_path) as conn:
        row = conn.execute("""
            SELECT * FROM bets WHERE id = ?
        """, [bet_id]).fetchone()

        if row is None:
            return None

        return dict(row)


def get_all_bets(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Get all bets from the database.

    Args:
        db_path: Path to SQLite database

    Returns:
        List of bets as dictionaries
    """
    with get_db_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM bets ORDER BY timestamp DESC
        """).fetchall()

        return [dict(row) for row in rows]


def get_unsettled_bets(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Get all unsettled bets (no outcome yet).

    Args:
        db_path: Path to SQLite database

    Returns:
        List of unsettled bets as dictionaries
    """
    with get_db_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM bets
            WHERE outcome IS NULL
            ORDER BY timestamp DESC
        """).fetchall()

        return [dict(row) for row in rows]


def get_settled_bets(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Get all settled bets (with outcome).

    Args:
        db_path: Path to SQLite database

    Returns:
        List of settled bets as dictionaries
    """
    with get_db_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM bets
            WHERE outcome IS NOT NULL
            ORDER BY timestamp DESC
        """).fetchall()

        return [dict(row) for row in rows]


def get_bets_by_player(player_id: str, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Get all bets for a specific player.

    Args:
        player_id: Player ID to filter by
        db_path: Path to SQLite database

    Returns:
        List of bets as dictionaries
    """
    with get_db_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM bets
            WHERE player_id = ?
            ORDER BY timestamp DESC
        """, [player_id]).fetchall()

        return [dict(row) for row in rows]


def get_bets_by_prop_type(prop_type: str, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Get all bets for a specific prop type.

    Args:
        prop_type: Prop type to filter by
        db_path: Path to SQLite database

    Returns:
        List of bets as dictionaries
    """
    with get_db_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM bets
            WHERE prop_type = ?
            ORDER BY timestamp DESC
        """, [prop_type]).fetchall()

        return [dict(row) for row in rows]


def get_betting_summary(db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Get summary statistics for all bets.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dictionary with summary statistics
    """
    with get_db_connection(db_path) as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_bets,
                COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
                COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losses,
                COUNT(CASE WHEN outcome = 'push' THEN 1 END) as pushes,
                COALESCE(SUM(CASE WHEN outcome IS NOT NULL THEN bet_size ELSE 0 END), 0) as total_wagered,
                COALESCE(SUM(CASE WHEN outcome IS NOT NULL THEN pnl ELSE 0 END), 0) as total_pnl,
                COALESCE(AVG(CASE WHEN outcome IS NOT NULL THEN edge ELSE 0 END), 0) as avg_edge
            FROM bets
        """).fetchone()

        if row is None:
            return {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "total_wagered": 0.0,
                "total_pnl": 0.0,
                "avg_edge": 0.0,
                "win_rate": 0.0,
                "roi": 0.0
            }

        summary = dict(row)

        # Calculate derived metrics
        total_decisive = summary['wins'] + summary['losses']
        summary['win_rate'] = summary['wins'] / total_decisive if total_decisive > 0 else 0.0
        summary['roi'] = (summary['total_pnl'] / summary['total_wagered'] * 100) if summary['total_wagered'] and summary['total_wagered'] > 0 else 0.0

        return summary


def get_bets_by_date_range(start_date: datetime, end_date: datetime,
                           db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Get bets within a date range.

    Args:
        start_date: Start of date range
        end_date: End of date range
        db_path: Path to SQLite database

    Returns:
        List of bets as dictionaries
    """
    with get_db_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM bets
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        """, [start_date.isoformat(), end_date.isoformat()]).fetchall()

        return [dict(row) for row in rows]


def delete_bet(bet_id: int, db_path: str = DB_PATH) -> bool:
    """
    Delete a bet from the database.

    Args:
        bet_id: ID of the bet to delete
        db_path: Path to SQLite database

    Returns:
        True if deletion was successful
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("""
            DELETE FROM bets WHERE id = ?
        """, [bet_id])
        conn.commit()
        return cursor.rowcount > 0


def clear_all_bets(db_path: str = DB_PATH):
    """Clear all bets from the database (for testing)."""
    with get_db_connection(db_path) as conn:
        conn.execute("DELETE FROM bets")
        conn.commit()
