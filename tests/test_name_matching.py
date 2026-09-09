"""Player-name normalization must behave identically in SQL and pandas.

The bug these cover: ``player_lookup`` stores ``Luka Dončić`` while sportsbook
feeds publish ``Luka Doncic``. An exact ``lower(trim(...))`` join drops those
rows silently -- no error, no warning, just missing data downstream.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from storage.name_matching import (
    normalize_name,
    normalize_name_series,
    normalized_name_sql,
)


EQUIVALENT_NAMES = [
    ("Luka Dončić", "Luka Doncic"),
    ("Nikola Jokić", "Nikola Jokic"),
    ("Kristaps Porziņģis", "Kristaps Porzingis"),
    ("Alperen Şengün", "Alperen Sengun"),
    ("R.J. Barrett", "RJ Barrett"),
    ("De'Aaron Fox", "DeAaron Fox"),
    ("De’Aaron Fox", "DeAaron Fox"),
    ("Karl-Anthony Towns", "Karl Anthony Towns"),
    ("  Jaylen   Brown  ", "Jaylen Brown"),
    ("LEBRON JAMES", "LeBron James"),
]

DISTINCT_NAMES = [
    ("Jalen Green", "Jalen Brown"),
    ("Jaylen Brown", "Jalen Brown"),
    ("Jalen Williams", "Jaylin Williams"),
]


@pytest.mark.parametrize("left,right", EQUIVALENT_NAMES)
def test_equivalent_names_normalize_together(left, right):
    assert normalize_name(left) == normalize_name(right)


@pytest.mark.parametrize("left,right", DISTINCT_NAMES)
def test_distinct_players_stay_distinct(left, right):
    assert normalize_name(left) != normalize_name(right)


@pytest.mark.parametrize("left,right", EQUIVALENT_NAMES + DISTINCT_NAMES)
def test_sql_and_pandas_agree(left, right):
    """If these ever diverge, joins silently lose rows again."""
    con = duckdb.connect(":memory:")
    expression = normalized_name_sql("?")
    for value in (left, right):
        sql_result = con.execute(f"SELECT {expression}", [value]).fetchone()[0]
        assert sql_result == normalize_name(value), value


def test_normalize_handles_missing_values():
    assert normalize_name(None) == ""
    assert normalize_name(float("nan")) == ""
    series = normalize_name_series(pd.Series(["Luka Dončić", None]))
    assert list(series) == ["luka doncic", ""]


def test_join_recovers_accented_rows():
    """End-to-end: the join that was dropping rows now keeps them."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE lookup (player_id INTEGER, full_name VARCHAR)")
    con.execute("INSERT INTO lookup VALUES (1, 'Luka Dončić'), (2, 'Jaylen Brown')")
    con.execute("CREATE TABLE lines (player_name VARCHAR)")
    con.execute("INSERT INTO lines VALUES ('Luka Doncic'), ('Jaylen Brown')")

    naive = con.execute("""
        SELECT COUNT(*) FROM lines l
        JOIN lookup k ON lower(trim(k.full_name)) = lower(trim(l.player_name))
    """).fetchone()[0]
    normalized = con.execute(f"""
        SELECT COUNT(*) FROM lines l
        JOIN lookup k
          ON {normalized_name_sql("k.full_name")} = {normalized_name_sql("l.player_name")}
    """).fetchone()[0]

    assert naive == 1, "precondition: the naive join drops the accented player"
    assert normalized == 2
