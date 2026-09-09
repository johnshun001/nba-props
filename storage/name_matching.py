"""Single source of truth for player-name matching.

Sportsbook feeds publish ASCII names (``Luka Doncic``) while ``nba_api``
publishes diacritics (``Luka Dončić``).  Both sides of every join must
normalize identically or rows disappear silently with no error, so the SQL
and pandas implementations below are kept deliberately in lockstep and are
covered by a test that compares them on the same inputs.

Normalization is intentionally conservative: fold accents, drop the
punctuation that varies between feeds (``R.J.``/``RJ``, ``De'Aaron``/``DeAaron``),
treat hyphens as spaces (``Karl-Anthony``/``Karl Anthony``), collapse
whitespace, and lowercase.  Nothing here drops name components, because that
would risk collapsing two distinct players onto one key.
"""

from __future__ import annotations

import unicodedata

import pandas as pd


DROPPED_CHARACTERS = (".", "'", "\u2019", "\u02bc")


def normalize_name(value: object) -> str:
    """Fold accents, drop varying punctuation, collapse whitespace, lowercase."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    for character in DROPPED_CHARACTERS:
        stripped = stripped.replace(character, "")
    return " ".join(stripped.replace("-", " ").split()).lower()


def normalize_name_series(series: pd.Series) -> pd.Series:
    """Vectorized :func:`normalize_name` for join keys built in pandas."""
    return series.map(normalize_name)


def normalized_name_sql(expression: str) -> str:
    """DuckDB expression producing the same key as :func:`normalize_name`."""
    accented = f"strip_accents(CAST({expression} AS VARCHAR))"
    unpunctuated = f"regexp_replace({accented}, '[.''\u2019\u02bc]', '', 'g')"
    spaced = f"replace({unpunctuated}, '-', ' ')"
    return f"lower(trim(regexp_replace({spaced}, '\\s+', ' ', 'g')))"


__all__ = ["DROPPED_CHARACTERS", "normalize_name", "normalize_name_series", "normalized_name_sql"]
