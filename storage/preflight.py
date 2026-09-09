"""Actionable preconditions for pipeline entry points.

Running a stage before its inputs exist used to surface a raw DuckDB error --
``Catalog Error: Table with name player_game_features does not exist! Did you
mean "pg_am"?`` -- which tells someone nothing about what to run next. These
helpers turn that into a ``NOT_READY`` message naming the missing table and the
command that produces it.

``MissingPipelineData`` subclasses ``ValueError`` so entry points that already
report ``NOT_READY`` for empty inputs keep working unchanged.
"""

from __future__ import annotations


REMEDIES = {
    "raw_api_responses": "python -m scrapers.nba_scraper",
    "player_game_features": "python -m scrapers.nba_scraper, then python -m storage.materialize",
    "player_results": "python -m storage.materialize",
    "player_lookup": "python -m storage.player_lookup",
    "game_schedule": "python -m scrapers.schedule_scraper  (run it before tipoff)",
    "prop_lines": "python -m scrapers.odds_scraper, then python -m storage.materialize  (needs ODDS_API_KEY)",
    "event_game_map": "python -m storage.materialize",
    "pregame_feature_store": "python -m features.pregame_features",
    "shadow_predictions": "python -m execution.shadow_tracker run",
}


class MissingPipelineData(ValueError):
    """Raised when a stage runs before the data it depends on exists."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        lines = [
            "this step needs data that has not been collected yet.",
            "",
            "Missing table(s):",
        ]
        for table in self.missing:
            lines.append(f"  - {table}")
            remedy = REMEDIES.get(table)
            if remedy:
                lines.append(f"      run: {remedy}")
        lines += ["", "See 02_Run/ for the full command order."]
        super().__init__("\n".join(lines))


def existing_tables(con) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }


def require_tables(con, tables) -> None:
    """Raise :class:`MissingPipelineData` naming every absent table."""
    present = existing_tables(con)
    missing = [table for table in tables if table not in present]
    if missing:
        raise MissingPipelineData(missing)


__all__ = ["MissingPipelineData", "REMEDIES", "existing_tables", "require_tables"]
