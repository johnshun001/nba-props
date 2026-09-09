"""Entry points must explain missing data instead of leaking a DuckDB error.

Before this, running a stage too early produced::

    _duckdb.CatalogException: Catalog Error: Table with name
    player_game_features does not exist! Did you mean "pg_am"?

which tells someone cloning the repo nothing about what to run.
"""

from __future__ import annotations

import duckdb
import pytest

from storage.preflight import MissingPipelineData, require_tables


def test_missing_table_names_the_command_that_creates_it():
    con = duckdb.connect(":memory:")
    with pytest.raises(MissingPipelineData) as excinfo:
        require_tables(con, ["player_game_features"])
    message = str(excinfo.value)
    assert "player_game_features" in message
    assert "scrapers.nba_scraper" in message
    assert "storage.materialize" in message


def test_reports_every_missing_table_at_once():
    con = duckdb.connect(":memory:")
    with pytest.raises(MissingPipelineData) as excinfo:
        require_tables(con, ["game_schedule", "prop_lines"])
    assert excinfo.value.missing == ["game_schedule", "prop_lines"]


def test_present_tables_pass():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE player_game_features (player_id INTEGER)")
    require_tables(con, ["player_game_features"])


def test_is_a_valueerror_so_existing_handlers_still_catch_it():
    """analysis.replay already reports NOT_READY for ValueError."""
    assert issubclass(MissingPipelineData, ValueError)


def test_train_entry_point_exits_cleanly(tmp_path):
    from models.train import main

    database = tmp_path / "empty.db"
    duckdb.connect(str(database)).close()
    assert main(["--db", str(database)]) == 2


def test_replay_entry_point_exits_cleanly(tmp_path):
    from analysis.replay import main

    database = tmp_path / "empty.db"
    duckdb.connect(str(database)).close()
    assert main(["--db", str(database)]) == 2
