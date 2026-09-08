"""Revenue attribution system for NBA prop betting.

Tracks value from each component:
- line_shopping_value: Value gained from selecting best line
- kelly_edge: Value from Kelly sizing optimization
- model_edge: Value from model prediction accuracy
- matchup_edge: Value from matchup-specific adjustments

Aggregates after 100 bets: total by category
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass

from schemas.validation import RevenueAttribution


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "attribution.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS revenue_attribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_id TEXT NOT NULL,
    line_shopping_value REAL NOT NULL,
    kelly_edge REAL NOT NULL,
    model_edge REAL NOT NULL,
    matchup_edge REAL NOT NULL,
    total_edge REAL NOT NULL,
    timestamp DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_attribution_bet_id ON revenue_attribution(bet_id);",
    "CREATE INDEX IF NOT EXISTS idx_attribution_timestamp ON revenue_attribution(timestamp);",
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
    """Initialize the attribution database."""
    with get_db_connection(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        for index_sql in CREATE_INDEX_SQL:
            conn.execute(index_sql)
        conn.commit()


def log_attribution(attribution: RevenueAttribution, db_path: str = DB_PATH) -> int:
    """
    Log revenue attribution for a bet.

    Args:
        attribution: RevenueAttribution object
        db_path: Path to SQLite database

    Returns:
        ID of the inserted record
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO revenue_attribution (
                bet_id, line_shopping_value, kelly_edge, model_edge,
                matchup_edge, total_edge, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            attribution.bet_id,
            attribution.line_shopping_value,
            attribution.kelly_edge,
            attribution.model_edge,
            attribution.matchup_edge,
            attribution.total_edge,
            attribution.timestamp.isoformat()
        ])
        conn.commit()
        return cursor.lastrowid


def get_attribution_by_bet(bet_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Get attribution record for a specific bet."""
    with get_db_connection(db_path) as conn:
        row = conn.execute("""
            SELECT * FROM revenue_attribution WHERE bet_id = ?
        """, [bet_id]).fetchone()

        if row is None:
            return None
        return dict(row)


def get_all_attributions(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Get all attribution records."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM revenue_attribution ORDER BY timestamp DESC
        """).fetchall()
        return [dict(row) for row in rows]


def get_aggregate_attribution(min_bets: int = 100, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Get aggregate attribution statistics.

    Only returns results if at least min_bets records exist.

    Args:
        min_bets: Minimum number of bets required
        db_path: Path to SQLite database

    Returns:
        Dictionary with aggregate statistics or None if insufficient data
    """
    with get_db_connection(db_path) as conn:
        # Check if we have enough bets
        count = conn.execute("""
            SELECT COUNT(*) as cnt FROM revenue_attribution
        """).fetchone()['cnt']

        if count < min_bets:
            return None

        row = conn.execute("""
            SELECT
                COUNT(*) as total_bets,
                SUM(line_shopping_value) as total_line_shopping,
                SUM(kelly_edge) as total_kelly_edge,
                SUM(model_edge) as total_model_edge,
                SUM(matchup_edge) as total_matchup_edge,
                SUM(total_edge) as total_edge,
                AVG(line_shopping_value) as avg_line_shopping,
                AVG(kelly_edge) as avg_kelly_edge,
                AVG(model_edge) as avg_model_edge,
                AVG(matchup_edge) as avg_matchup_edge,
                AVG(total_edge) as avg_total_edge
            FROM revenue_attribution
        """).fetchone()

        result = dict(row)

        # Calculate percentage contribution of each component
        # Note: line_shopping_value is in points, needs to be converted to edge units (/10)
        total = result['total_edge']
        if total != 0:
            line_shopping_edge = result['total_line_shopping'] / 10
            result['line_shopping_pct'] = line_shopping_edge / total * 100
            result['kelly_edge_pct'] = result['total_kelly_edge'] / total * 100
            result['model_edge_pct'] = result['total_model_edge'] / total * 100
            result['matchup_edge_pct'] = result['total_matchup_edge'] / total * 100
        else:
            result['line_shopping_pct'] = 0.0
            result['kelly_edge_pct'] = 0.0
            result['model_edge_pct'] = 0.0
            result['matchup_edge_pct'] = 0.0

        result['total_bets'] = count

        return result


def get_attribution_by_component(db_path: str = DB_PATH) -> Dict[str, float]:
    """
    Get total attribution broken down by component.

    Returns:
        Dictionary with component totals
    """
    with get_db_connection(db_path) as conn:
        row = conn.execute("""
            SELECT
                SUM(line_shopping_value) as line_shopping,
                SUM(kelly_edge) as kelly,
                SUM(model_edge) as model,
                SUM(matchup_edge) as matchup
            FROM revenue_attribution
        """).fetchone()

        return {
            'line_shopping_value': row['line_shopping'] or 0.0,
            'kelly_edge': row['kelly'] or 0.0,
            'model_edge': row['model'] or 0.0,
            'matchup_edge': row['matchup'] or 0.0,
        }


def get_attribution_trend(n_bets: int = 50, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Get attribution trend for the last n bets.

    Args:
        n_bets: Number of recent bets to analyze
        db_path: Path to SQLite database

    Returns:
        List of attribution records
    """
    with get_db_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM revenue_attribution
            ORDER BY timestamp DESC
            LIMIT ?
        """, [n_bets]).fetchall()

        return [dict(row) for row in rows]


def get_component_ranking(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Get ranking of components by total contribution.

    Returns:
        List of components sorted by total contribution
    """
    components = get_attribution_by_component(db_path)

    ranking = [
        {'component': 'line_shopping_value', 'total': components['line_shopping_value']},
        {'component': 'kelly_edge', 'total': components['kelly_edge']},
        {'component': 'model_edge', 'total': components['model_edge']},
        {'component': 'matchup_edge', 'total': components['matchup_edge']},
    ]

    return sorted(ranking, key=lambda x: abs(x['total']), reverse=True)


def clear_all_attributions(db_path: str = DB_PATH):
    """Clear all attribution records (for testing)."""
    with get_db_connection(db_path) as conn:
        conn.execute("DELETE FROM revenue_attribution")
        conn.commit()


@dataclass
class AttributionCalculator:
    """
    Calculator for computing revenue attribution components.

    Usage:
        calc = AttributionCalculator()
        attribution = calc.calculate(
            selected_line=25.5,
            market_lines=[{'line': 24.5}, {'line': 25.5}, {'line': 24.0}],
            kelly_bet=10.0,
            flat_bet=5.0,
            model_projection=27.0,
            market_projection=25.0,
            matchup_adjustment=0.5
        )
    """

    def calculate(
        self,
        bet_id: str,
        selected_line: float,
        market_lines: List[Dict],
        kelly_bet: float,
        flat_bet: float,
        model_projection: float,
        market_projection: float,
        matchup_adjustment: float,
        odds: float = -110
    ) -> RevenueAttribution:
        """
        Calculate revenue attribution for a bet.

        Args:
            bet_id: Unique bet identifier
            selected_line: Line selected after line shopping
            market_lines: All available lines from market
            kelly_bet: Bet size from Kelly sizing
            flat_bet: Bet size from flat betting (baseline)
            model_projection: Model's projected value
            market_projection: Market's projected value (line)
            matchup_adjustment: Adjustment for matchup specifics
            odds: American odds

        Returns:
            RevenueAttribution object
        """
        # Line shopping value: difference between selected and average market line
        avg_line = sum(l['line'] for l in market_lines) / len(market_lines)
        line_shopping_value = selected_line - avg_line

        # Kelly edge: additional edge from optimal sizing
        # (kelly_bet - flat_bet) / flat_bet represents the confidence multiplier
        kelly_edge = (kelly_bet - flat_bet) / flat_bet if flat_bet > 0 else 0.0
        kelly_edge = kelly_edge * 0.1  # Scale to reasonable edge values

        # Model edge: difference between model and market
        model_edge = (model_projection - market_projection) / market_projection

        # Matchup edge: additional edge from matchup analysis
        matchup_edge = matchup_adjustment / market_projection

        # Total edge (line shopping converted to edge units)
        total_edge = line_shopping_value / 10 + kelly_edge + model_edge + matchup_edge

        return RevenueAttribution(
            bet_id=bet_id,
            line_shopping_value=round(line_shopping_value, 4),
            kelly_edge=round(kelly_edge, 4),
            model_edge=round(model_edge, 4),
            matchup_edge=round(matchup_edge, 4),
            total_edge=round(total_edge, 4)
        )
