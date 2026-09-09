"""Normalized storage contracts for data required by pregame forecasts."""

from __future__ import annotations


PIPELINE_DDLS = (
    """CREATE TABLE IF NOT EXISTS event_game_map (
        event_id VARCHAR PRIMARY KEY, game_id VARCHAR NOT NULL,
        asof_time TIMESTAMP NOT NULL, source VARCHAR NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS team_game_logs (
        team_id INTEGER NOT NULL, game_id VARCHAR NOT NULL, game_date TIMESTAMP NOT NULL,
        opponent_id INTEGER, is_home BOOLEAN, pace FLOAT, def_rating FLOAT,
        asof_time TIMESTAMP NOT NULL, source_raw_id VARCHAR, data_version VARCHAR NOT NULL,
        PRIMARY KEY (team_id, game_id, asof_time)
    )""",
    """CREATE TABLE IF NOT EXISTS starting_lineups (
        event_id VARCHAR NOT NULL, game_id VARCHAR, player_id INTEGER NOT NULL,
        starter_probability FLOAT, bench_probability FLOAT, asof_time TIMESTAMP NOT NULL,
        source_raw_id VARCHAR, data_version VARCHAR NOT NULL,
        PRIMARY KEY (event_id, player_id, asof_time)
    )""",
    """CREATE TABLE IF NOT EXISTS player_availability (
        event_id VARCHAR NOT NULL, game_id VARCHAR, player_id INTEGER NOT NULL,
        status VARCHAR, play_probability FLOAT, teammate_absence_count INTEGER,
        teammate_absent_usage FLOAT, usage_change_indicator FLOAT,
        asof_time TIMESTAMP NOT NULL, source_raw_id VARCHAR, data_version VARCHAR NOT NULL,
        PRIMARY KEY (event_id, player_id, asof_time)
    )""",
    """CREATE TABLE IF NOT EXISTS opponent_position_allowed (
        opponent_id INTEGER NOT NULL, position VARCHAR NOT NULL,
        opponent_allowed_position FLOAT NOT NULL, asof_time TIMESTAMP NOT NULL,
        source_raw_id VARCHAR, data_version VARCHAR NOT NULL,
        PRIMARY KEY (opponent_id, position, asof_time)
    )""",
    """CREATE TABLE IF NOT EXISTS hmm_pregame_predictions (
        player_id INTEGER NOT NULL, game_id VARCHAR NOT NULL, prediction_time TIMESTAMP NOT NULL,
        hmm_expected_minutes FLOAT, hmm_p_dnp FLOAT, hmm_p_limited FLOAT,
        hmm_p_rotation FLOAT, hmm_p_featured FLOAT, hmm_minutes_uncertainty FLOAT,
        model_version VARCHAR NOT NULL,
        PRIMARY KEY (player_id, game_id, prediction_time, model_version)
    )""",
    """CREATE TABLE IF NOT EXISTS player_results (
        player_id INTEGER NOT NULL, game_id VARCHAR NOT NULL, game_date TIMESTAMP NOT NULL,
        pts FLOAT, reb FLOAT, ast FLOAT, minutes FLOAT, source_raw_id VARCHAR,
        data_version VARCHAR NOT NULL, PRIMARY KEY (player_id, game_id)
    )""",
    """CREATE TABLE IF NOT EXISTS pregame_feature_store (
        player_id INTEGER NOT NULL, game_id VARCHAR NOT NULL, prediction_time TIMESTAMP NOT NULL,
        feature_version VARCHAR NOT NULL, data_version VARCHAR NOT NULL,
        features_json TEXT NOT NULL, PRIMARY KEY (player_id, game_id, prediction_time, feature_version)
    )""",
)


def ensure_pipeline_schemas(con) -> None:
    for ddl in PIPELINE_DDLS:
        con.execute(ddl)


__all__ = ["ensure_pipeline_schemas"]
