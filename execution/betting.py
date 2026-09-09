"""Deterministic two-sided selection, line shopping, Kelly, and exposure caps."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from config import load_config


def american_to_decimal(odds: int | float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def implied_probability(odds: int | float) -> float:
    return 1.0 / american_to_decimal(odds)


def pair_prop_prices(rows: pd.DataFrame) -> pd.DataFrame:
    """Pair Over/Under rows without crossing event, book, line, or snapshot."""
    if rows.empty:
        return rows.copy()
    if {"over_price", "under_price"}.issubset(rows.columns):
        return rows.dropna(subset=["over_price", "under_price"]).copy()
    required = {"event_id", "player_name", "market", "line", "bookmaker", "asof_time", "side", "price"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Odds rows are missing: {missing}")
    keys = ["event_id", "player_name", "market", "line", "bookmaker", "asof_time"]
    normalized = rows.copy()
    normalized["side"] = normalized["side"].str.lower()
    if normalized.duplicated([*keys, "side"]).any():
        raise ValueError("Duplicate price for an event/player/market/line/book/snapshot/side")
    pivot = normalized.pivot(index=keys, columns="side", values="price").reset_index()
    return pivot.rename(columns={"over": "over_price", "under": "under_price"}).dropna(
        subset=["over_price", "under_price"]
    )


def no_vig_probabilities(over_odds: int, under_odds: int) -> tuple[float, float]:
    over = implied_probability(over_odds)
    under = implied_probability(under_odds)
    total = over + under
    return over / total, under / total


def expected_value(probability: float, american_odds: int) -> float:
    decimal = american_to_decimal(american_odds)
    return float(probability) * (decimal - 1.0) - (1.0 - float(probability))


def fractional_kelly(probability: float, american_odds: int, *, fraction: float | None = None, cap: float | None = None) -> float:
    thresholds = load_config()["thresholds"]
    fraction = float(thresholds["kelly_fraction"] if fraction is None else fraction)
    cap = float(thresholds["max_player_exposure"] if cap is None else cap)
    b = american_to_decimal(american_odds) - 1.0
    raw = (b * float(probability) - (1.0 - float(probability))) / b
    return float(np.clip(raw * fraction, 0.0, cap))


def select_opportunities(predictions: pd.DataFrame, minimum_edge: float | None = None) -> pd.DataFrame:
    """Evaluate both sides and shop by expected value with deterministic ties."""
    if predictions.empty:
        return predictions.copy()
    minimum_edge = float(load_config()["thresholds"]["minimum_edge"] if minimum_edge is None else minimum_edge)
    candidates = []
    identity = ["event_id", "player_id", "stat"]
    for _, row in predictions.iterrows():
        book_over, book_under = no_vig_probabilities(int(row["over_price"]), int(row["under_price"]))
        for side in ("OVER", "UNDER"):
            probability = float(row["p_over"] if side == "OVER" else row["p_under"])
            price = int(row["over_price"] if side == "OVER" else row["under_price"])
            book_probability = book_over if side == "OVER" else book_under
            edge = probability - book_probability
            candidates.append({
                **row.to_dict(), "side": side, "price": price,
                "model_probability": probability, "book_probability": book_probability,
                "edge": edge, "expected_value": expected_value(probability, price),
                "kelly_fraction": fractional_kelly(probability, price),
            })
    candidates = pd.DataFrame(candidates)
    candidates = candidates[(candidates["edge"] >= minimum_edge) & (candidates["expected_value"] > 0)].copy()
    if candidates.empty:
        return candidates
    candidates["line_preference"] = np.where(
        candidates["side"] == "OVER", -candidates["sportsbook_line"], candidates["sportsbook_line"]
    )
    # Highest EV wins.  When EV is exactly tied, prefer the easier threshold
    # (lower for OVER, higher for UNDER), then the larger American price and a
    # stable bookmaker key.  ``head`` is intentional: using ``tail`` after a
    # descending sort silently selected the worst offer in each market.
    candidates = candidates.sort_values(
        [*identity, "side", "expected_value", "line_preference", "price", "bookmaker"],
        ascending=[True, True, True, True, False, False, False, True],
        kind="mergesort",
    )
    return (
        candidates.groupby([*identity, "side"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def apply_exposure_caps(opportunities: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    """Allocate recommendations while capping player, game, and correlated props."""
    if opportunities.empty:
        return opportunities.assign(stake=pd.Series(dtype=float), exposure_status=pd.Series(dtype=str))
    thresholds = load_config()["thresholds"]
    player_cap = bankroll * float(thresholds["max_player_exposure"])
    game_cap = bankroll * float(thresholds["max_game_exposure"])
    correlated_cap = bankroll * float(thresholds["max_correlated_exposure"])
    player_used, game_used, correlation_used = defaultdict(float), defaultdict(float), defaultdict(float)
    rows = []
    ordered = opportunities.sort_values(["expected_value", "event_id", "player_id", "stat"], ascending=[False, True, True, True])
    for _, row in ordered.iterrows():
        player_key = (row["event_id"], row["player_id"])
        game_key = row["event_id"]
        correlation_key = (row["event_id"], row["player_id"], row["side"])
        requested = bankroll * float(row["kelly_fraction"])
        available = min(player_cap - player_used[player_key], game_cap - game_used[game_key], correlated_cap - correlation_used[correlation_key])
        stake = max(0.0, min(requested, available))
        player_used[player_key] += stake
        game_used[game_key] += stake
        correlation_used[correlation_key] += stake
        rows.append({**row.to_dict(), "stake": round(stake, 2), "exposure_status": "APPROVED" if stake > 0 else "EXPOSURE_CAPPED"})
    return pd.DataFrame(rows)


__all__ = ["american_to_decimal", "apply_exposure_caps", "fractional_kelly", "no_vig_probabilities", "pair_prop_prices", "select_opportunities"]
