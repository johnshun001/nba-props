"""Tests for teammate/minutes shock classification."""

import pandas as pd
import pytest

from models.teammate_shock import classify_shock_games


def test_classify_shock_games_uses_prior_games_only():
    games = pd.DataFrame({"minutes": [10, 10, 10, 10, 10, 30]})

    result = classify_shock_games(games, z_threshold=1.0, window=5)

    assert result.loc[:4, "is_shock"].eq(False).all()
    assert bool(result.loc[5, "is_shock"]) is True
    assert result.loc[5, "rolling_mean_min"] == pytest.approx(10.0)


def test_classify_shock_games_handles_empty_input():
    result = classify_shock_games(pd.DataFrame(columns=["minutes"]))

    assert result.empty
    assert {"is_shock", "rolling_mean_min", "rolling_std_min"} <= set(result.columns)


def test_classify_shock_games_rejects_invalid_window():
    with pytest.raises(ValueError, match="window must be at least 1"):
        classify_shock_games(pd.DataFrame({"minutes": [10]}), window=0)
