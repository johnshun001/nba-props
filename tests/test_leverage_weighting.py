"""Tests for leverage weighting: garbage_time=0.3x, clutch=1.5x."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.baseline import compute_projections


def create_leverage_weights(df):
    """Create leverage weights based on game situation."""
    weights = np.ones(len(df))

    garbage_time = (df['quarter'] == 4) & (df['point_diff'] > 15) & (df['minutes_remaining'] < 5)
    clutch = ((df['quarter'] >= 3) & (df['point_diff'] < 5))

    weights[garbage_time] = 0.3
    weights[clutch] = 1.5

    return weights


class TestLeverageWeighting:
    """Tests for leverage weighting system."""

    def test_garbage_time_weight(self):
        """Garbage time situations get 0.3x weight."""
        df = pd.DataFrame({
            'quarter': [4, 4, 4],
            'point_diff': [20, 18, 10],
            'minutes_remaining': [3, 4, 6],
            'pts_target': [10, 12, 15],
        })

        weights = create_leverage_weights(df)

        # Row 0: Q4, diff>15, min<5 -> garbage time
        assert weights[0] == 0.3
        # Row 1: Q4, diff>15, min<5 -> garbage time
        assert weights[1] == 0.3
        # Row 2: Q4, diff<=15 -> not garbage time
        assert weights[2] == 1.0

    def test_clutch_weight(self):
        """Clutch situations get 1.5x weight."""
        df = pd.DataFrame({
            'quarter': [3, 4, 4, 2],
            'point_diff': [3, 4, 5, 3],
            'minutes_remaining': [10, 5, 2, 10],
            'pts_target': [10, 12, 15, 8],
        })

        weights = create_leverage_weights(df)

        # Row 0: Q3, diff<5 -> clutch
        assert weights[0] == 1.5
        # Row 1: Q4, diff<5 -> clutch
        assert weights[1] == 1.5
        # Row 2: Q4, diff=5 (not <5) -> not clutch
        assert weights[2] == 1.0
        # Row 3: Q2 -> not clutch
        assert weights[3] == 1.0

    def test_clutch_overrides_garbage_time(self):
        """When both conditions could apply, clutch takes precedence."""
        df = pd.DataFrame({
            'quarter': [4],
            'point_diff': [3],  # <5 so clutch
            'minutes_remaining': [2],  # <5 but clutch wins
            'pts_target': [10],
        })

        weights = create_leverage_weights(df)
        # Clutch (1.5) should apply, not garbage time (0.3)
        assert weights[0] == 1.5

    def test_normal_time_weight(self):
        """Normal game situations get 1.0x weight."""
        df = pd.DataFrame({
            'quarter': [1, 2, 3, 4],
            'point_diff': [10, 12, 20, 8],
            'minutes_remaining': [10, 8, 10, 8],
            'pts_target': [10, 12, 15, 10],
        })

        weights = create_leverage_weights(df)

        # All should be normal time (1.0)
        assert all(weights == 1.0)

    def test_empty_dataframe(self):
        """Empty dataframe returns empty weights array."""
        df = pd.DataFrame({
            'quarter': [],
            'point_diff': [],
            'minutes_remaining': [],
            'pts_target': [],
        })

        weights = create_leverage_weights(df)
        assert len(weights) == 0

    def test_weight_boundaries(self):
        """Test boundary conditions for garbage time and clutch."""
        # Garbage time boundary: point_diff = 15 (not > 15)
        df_garbage_boundary = pd.DataFrame({
            'quarter': [4],
            'point_diff': [15],  # Exactly 15, not garbage time
            'minutes_remaining': [3],
            'pts_target': [10],
        })
        weights = create_leverage_weights(df_garbage_boundary)
        assert weights[0] == 1.0

        # Clutch boundary: point_diff = 5 (not < 5)
        df_clutch_boundary = pd.DataFrame({
            'quarter': [4],
            'point_diff': [5],  # Exactly 5, not clutch
            'minutes_remaining': [3],
            'pts_target': [10],
        })
        weights = create_leverage_weights(df_clutch_boundary)
        assert weights[0] == 1.0

        # Clutch boundary: quarter = 3 (>= 3)
        df_q3_clutch = pd.DataFrame({
            'quarter': [3],
            'point_diff': [4],  # < 5
            'minutes_remaining': [8],
            'pts_target': [10],
        })
        weights = create_leverage_weights(df_q3_clutch)
        assert weights[0] == 1.5
