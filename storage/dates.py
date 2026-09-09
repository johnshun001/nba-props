"""Robust parsing for stored NBA date columns.

``player_game_features.game_date`` is not written in a single format. Rows
materialized before ``storage/materialize.py`` normalized to ISO retain the
``nba_api`` presentation format (``Nov 17, 2023``), while newer rows are
``2025-11-18``. When both live in one column, ``pandas`` infers a single
format for the whole Series and silently coerces everything that does not
match to ``NaT`` -- which drops those rows from training and feature builds
with no error.

Parsing each value independently costs a little speed and removes an entire
class of silent data loss.
"""

from __future__ import annotations

import pandas as pd


def parse_game_dates(values, *, utc: bool = True) -> pd.Series:
    """Parse a possibly mixed-format date column without dropping rows."""
    return pd.to_datetime(values, format="mixed", utc=utc, errors="coerce")


__all__ = ["parse_game_dates"]
