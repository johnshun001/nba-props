"""Leakage-free, one-row-per-player/game pregame feature construction.

All statistics derived from player or team results are explicitly shifted by
one game before rolling/expanding calculations.  Context tables are joined
as-of their recorded ``prediction_time``/``asof_time`` so a row cannot see an
update published after its forecast timestamp.
"""

from __future__ import annotations

from typing import Iterable
import json

import numpy as np
import pandas as pd

from config import load_config
from storage.dates import parse_game_dates
from storage.pipeline_schema import ensure_pipeline_schemas


STATS = ("pts", "reb", "ast")
HMM_COLUMNS = (
    "hmm_expected_minutes",
    "hmm_p_dnp",
    "hmm_p_limited",
    "hmm_p_rotation",
    "hmm_p_featured",
    "hmm_minutes_uncertainty",
)
SNAPSHOT_CONTEXT_COLUMNS = (
    "starter_probability",
    "bench_probability",
    "teammate_absence_count",
    "teammate_absent_usage",
    "usage_change_indicator",
    "opponent_allowed_position",
)

FEATURE_COLUMNS = [
    *(f"{stat}_avg_{window}" for stat in STATS for window in (5, 10, 20)),
    *(f"{stat}_ewm" for stat in STATS),
    *(f"{stat}_per_minute_10" for stat in STATS),
    *(f"{stat}_season_avg" for stat in STATS),
    *(f"{stat}_career_avg" for stat in STATS),
    *HMM_COLUMNS,
    "starter_probability",
    "bench_probability",
    "is_home",
    "rest_days",
    "is_back_to_back",
    "opponent_pace",
    "opponent_def_rating",
    "opponent_allowed_position",
    "teammate_absence_count",
    "teammate_absent_usage",
    "usage_change_indicator",
    *(f"{stat}_volatility_10" for stat in STATS),
    *(f"{stat}_trend_10" for stat in STATS),
]


def _require(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _prior_rolling(series: pd.Series, window: int, function: str = "mean") -> pd.Series:
    prior = series.shift(1)
    rolling = prior.rolling(window, min_periods=1)
    if function == "std":
        return rolling.std(ddof=0)
    return rolling.mean()


def _prior_trend(series: pd.Series, window: int = 10) -> pd.Series:
    def slope(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 2:
            return 0.0
        x = np.arange(values.size, dtype=float)[mask]
        return float(np.polyfit(x, values[mask], 1)[0])

    return series.shift(1).rolling(window, min_periods=2).apply(slope, raw=True)


def _asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    by: list[str],
    columns: list[str],
    right_time: str = "asof_time",
) -> pd.DataFrame:
    if right is None or right.empty:
        for column in columns:
            if column not in left:
                left[column] = np.nan
        return left
    _require(right, [*by, right_time, *columns], "context table")
    right_frame = right[[*by, right_time, *columns]].copy()
    right_frame[right_time] = pd.to_datetime(
        right_frame[right_time], utc=True, errors="coerce"
    ).astype("datetime64[ns, UTC]")
    right_frame = right_frame.dropna(subset=[right_time]).rename(columns={right_time: "context_time"})
    left_frame = left.copy()
    left_frame["prediction_time"] = pd.to_datetime(
        left_frame["prediction_time"], utc=True, errors="coerce"
    ).astype("datetime64[ns, UTC]")
    # DuckDB/pandas commonly disagree on int32 vs int64 identifier widths.
    original_keys = []
    for position, key in enumerate(by):
        original = f"__original_key_{position}"
        left_frame[original] = left_frame[key]
        original_keys.append((key, original))
        left_frame[key] = left_frame[key].astype("string")
        right_frame[key] = right_frame[key].astype("string")
    # pandas requires global ordering by the merge key, even when ``by`` is used.
    left_frame["__order"] = np.arange(len(left_frame))
    left_frame = left_frame.sort_values(["prediction_time", *by])
    right_frame = right_frame.sort_values(["context_time", *by])
    merged = pd.merge_asof(
        left_frame,
        right_frame,
        left_on="prediction_time",
        right_on="context_time",
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("__order").drop(columns=["__order", "context_time"])
    for key, original in original_keys:
        merged[key] = merged.pop(original)
    return merged


def build_team_context(team_games: pd.DataFrame) -> pd.DataFrame:
    """Create pregame opponent pace/defense values from prior team games."""
    _require(team_games, ["team_id", "game_date", "pace", "def_rating"], "team_games")
    teams = team_games.copy()
    teams["game_date"] = parse_game_dates(teams["game_date"])
    teams = teams.sort_values(["team_id", "game_date"])
    grouped = teams.groupby("team_id", sort=False, group_keys=False)
    teams["opponent_pace"] = grouped["pace"].transform(
        lambda s: s.shift(1).ewm(span=10, adjust=False, min_periods=1).mean()
    )
    teams["opponent_def_rating"] = grouped["def_rating"].transform(
        lambda s: s.shift(1).ewm(span=10, adjust=False, min_periods=1).mean()
    )
    return teams.rename(columns={"team_id": "opponent_id", "game_date": "asof_time"})[
        ["opponent_id", "asof_time", "opponent_pace", "opponent_def_rating"]
    ]


def build_pregame_features(
    player_games: pd.DataFrame,
    *,
    team_games: pd.DataFrame | None = None,
    hmm_predictions: pd.DataFrame | None = None,
    lineup_predictions: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
    opponent_allowed: pd.DataFrame | None = None,
    feature_version: str | None = None,
) -> pd.DataFrame:
    """Return a stable pregame feature table with one row per player/game.

    ``player_games`` may contain completed-game targets, but none of the feature
    columns read the current row's result. ``prediction_time`` defaults to the
    known tip/game timestamp and must be supplied for live snapshots.
    """
    _require(player_games, ["player_id", "game_id", "game_date", *STATS, "minutes"], "player_games")
    config = load_config()
    version = feature_version or config["versions"]["feature"]
    frame = player_games.copy()
    frame["game_date"] = parse_game_dates(frame["game_date"])
    if "prediction_time" not in frame:
        frame["prediction_time"] = frame.get("tip_time_utc", frame["game_date"])
    frame["prediction_time"] = pd.to_datetime(frame["prediction_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["player_id", "game_id", "game_date", "prediction_time"])
    frame = frame.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    if frame.duplicated(["player_id", "game_id", "prediction_time"]).any():
        raise ValueError("player_games contains duplicate player/game/prediction_time rows")

    for column in [*STATS, "minutes"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby("player_id", sort=False, group_keys=False)
    windows = tuple(int(v) for v in config.get("rolling_windows", [5, 10, 20]))
    for stat in STATS:
        for window in windows:
            frame[f"{stat}_avg_{window}"] = grouped[stat].transform(
                lambda s, w=window: _prior_rolling(s, w)
            )
        frame[f"{stat}_ewm"] = grouped[stat].transform(
            lambda s: s.shift(1).ewm(span=10, adjust=False, min_periods=1).mean()
        )
        numerator = grouped[stat].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
        denominator = grouped["minutes"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).sum()
        )
        frame[f"{stat}_per_minute_10"] = numerator / denominator.replace(0.0, np.nan)
        frame[f"{stat}_career_avg"] = grouped[stat].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        season_key = frame["season"] if "season" in frame else frame["game_date"].dt.year
        frame[f"{stat}_season_avg"] = frame.groupby(
            [frame["player_id"], season_key], sort=False
        )[stat].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        frame[f"{stat}_volatility_10"] = grouped[stat].transform(
            lambda s: _prior_rolling(s, 10, "std")
        )
        frame[f"{stat}_trend_10"] = grouped[stat].transform(_prior_trend)

    previous_game = grouped["game_date"].shift(1)
    frame["rest_days"] = (frame["game_date"] - previous_game).dt.total_seconds().div(86400).clip(0, 14)
    frame["is_back_to_back"] = (frame["rest_days"] <= 1.0).astype(float)
    if "is_home" not in frame:
        matchup = frame.get("matchup", pd.Series("", index=frame.index)).astype(str)
        frame["is_home"] = (~matchup.str.contains("@", regex=False)).astype(float)

    team_context = build_team_context(team_games) if team_games is not None and not team_games.empty else None
    if "opponent_id" not in frame:
        frame["opponent_id"] = np.nan
    frame = _asof_join(frame, team_context, ["opponent_id"], ["opponent_pace", "opponent_def_rating"])
    frame = _asof_join(frame, hmm_predictions, ["player_id"], list(HMM_COLUMNS))
    frame = _asof_join(
        frame,
        lineup_predictions,
        ["player_id", "game_id"],
        ["starter_probability", "bench_probability"],
    )
    frame = _asof_join(
        frame,
        availability,
        ["player_id", "game_id"],
        ["teammate_absence_count", "teammate_absent_usage", "usage_change_indicator"],
    )
    if opponent_allowed is not None and not opponent_allowed.empty and "position" in frame:
        frame = _asof_join(
            frame,
            opponent_allowed,
            ["opponent_id", "position"],
            ["opponent_allowed_position"],
        )
    elif "opponent_allowed_position" not in frame:
        frame["opponent_allowed_position"] = np.nan

    defaults = {
        "hmm_p_dnp": 0.0,
        "hmm_p_limited": 0.0,
        "hmm_p_rotation": 0.5,
        "hmm_p_featured": 0.5,
        "starter_probability": 0.5,
        "bench_probability": 0.5,
        "teammate_absence_count": 0.0,
        "teammate_absent_usage": 0.0,
        "usage_change_indicator": 0.0,
    }
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = defaults.get(column, np.nan)
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if column in defaults:
            frame[column] = frame[column].fillna(defaults[column])

    frame["feature_version"] = version
    identity = ["player_id", "game_id", "prediction_time", "feature_version"]
    context_keys = [column for column in ("team_id", "opponent_id", "position") if column in frame]
    return frame[[
        *identity, "game_date", *(["season"] if "season" in frame else []),
        *context_keys, *STATS, "minutes", *FEATURE_COLUMNS,
    ]]


def apply_snapshot_context(
    frame: pd.DataFrame,
    *,
    lineup_predictions: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
    opponent_allowed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-evaluate time-varying context at each row's prediction timestamp.

    Player rolling statistics and walk-forward HMM outputs are invariant within
    a game's pre-tip window. Lineup, injury, and position-defense snapshots are
    not, so historical replay uses this helper after attaching the exact odds
    snapshot time.
    """
    _require(frame, ["player_id", "game_id", "prediction_time"], "snapshot feature frame")
    output = frame.drop(columns=[c for c in SNAPSHOT_CONTEXT_COLUMNS if c in frame]).copy()
    output = _asof_join(
        output, lineup_predictions, ["player_id", "game_id"],
        ["starter_probability", "bench_probability"],
    )
    output = _asof_join(
        output, availability, ["player_id", "game_id"],
        ["teammate_absence_count", "teammate_absent_usage", "usage_change_indicator"],
    )
    if opponent_allowed is not None and not opponent_allowed.empty and {"opponent_id", "position"}.issubset(output):
        output = _asof_join(
            output, opponent_allowed, ["opponent_id", "position"],
            ["opponent_allowed_position"],
        )
    else:
        output["opponent_allowed_position"] = np.nan

    defaults = {
        "starter_probability": 0.5,
        "bench_probability": 0.5,
        "teammate_absence_count": 0.0,
        "teammate_absent_usage": 0.0,
        "usage_change_indicator": 0.0,
    }
    for column in SNAPSHOT_CONTEXT_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
        if column in defaults:
            output[column] = output[column].fillna(defaults[column])
    return output


def persist_feature_store(con, frame: pd.DataFrame) -> int:
    """Persist versioned pregame features under the four-column identity key."""
    _require(frame, ["player_id", "game_id", "prediction_time", "feature_version", *FEATURE_COLUMNS], "feature frame")
    ensure_pipeline_schemas(con)
    data_version = load_config()["versions"]["data"]
    count = 0
    for row in frame.itertuples(index=False):
        payload = {
            column: (None if pd.isna(getattr(row, column)) else float(getattr(row, column)))
            for column in FEATURE_COLUMNS
        }
        con.execute("""
            INSERT OR REPLACE INTO pregame_feature_store (
                player_id, game_id, prediction_time, feature_version,
                data_version, features_json
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, [
            int(row.player_id), str(row.game_id), row.prediction_time,
            str(row.feature_version), data_version,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        ])
        count += 1
    return count


def main() -> int:
    import duckdb
    from config import project_path
    from models.train import load_training_frame

    with duckdb.connect(str(project_path("database"))) as con:
        frame = load_training_frame(con)
        count = persist_feature_store(con, frame)
    print(f"Stored {count} pregame feature rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
