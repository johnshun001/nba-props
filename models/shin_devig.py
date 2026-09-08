# Shin (1993) vig removal for two-outcome markets (over/under props)
# Reference: Shin, H.S. (1993) — "Measuring the Incidence of Insider Trading"
#
# The Shin model assumes a fraction z of bettors are insiders with perfect info.
# It solves for z numerically, then returns true probabilities stripped of vig.
#
# For a two-outcome market with implied probs p1_raw, p2_raw (sum > 1 due to vig):
#   Shin solves: p1_true = sqrt(z^2 + 4(1-z) * (p1_raw / sum_raw)^2 * sum_raw) - z
#                          -------------------------------------------------------
#                                         2(1-z)
# where sum_raw = p1_raw + p2_raw (the overround)

import numpy as np
from scipy.optimize import brentq


def american_to_implied(american_odds: float) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def implied_to_american(prob: float) -> float:
    """Convert true probability back to American odds."""
    if prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return round(-prob / (1 - prob) * 100, 1)
    else:
        return round((1 - prob) / prob * 100, 1)


def solve_shin_z(p1_raw: float, p2_raw: float) -> float:
    """
    Solve for Shin's insider fraction z numerically using Brent's method.
    z is bounded (0, 1) — in practice z is small (0.01 to 0.10 for sports).
    """
    overround = p1_raw + p2_raw  # > 1.0 due to vig

    def equation(z):
        # Shin condition: sum of true probs must equal 1
        # true_pi = (sqrt(z^2 + 4*(1-z)*pi_raw*overround^-1) - z) / (2*(1-z))
        # sum(true_pi) - 1 = 0
        total = 0.0
        for p_raw in [p1_raw, p2_raw]:
            qi = p_raw / overround  # normalize
            inside = z ** 2 + 4 * (1 - z) * qi
            if inside < 0:
                return float("inf")
            true_p = (np.sqrt(inside) - z) / (2 * (1 - z))
            total += true_p
        return total - 1.0

    try:
        z = brentq(equation, 1e-6, 0.5, xtol=1e-8)
        return z
    except ValueError:
        # fallback to proportional normalization if solver fails
        return None


def shin_devig(over_american: float, under_american: float) -> dict:
    """
    Apply Shin de-vig to a two-outcome prop market.

    Args:
        over_american:  American odds for the over
        under_american: American odds for the under

    Returns dict with:
        p_over_true:   True probability of over
        p_under_true:  True probability of under
        overround:     Raw overround (vig measure)
        shin_z:        Estimated insider fraction
        method:        'shin' or 'proportional' (fallback)
    """
    p_over_raw = american_to_implied(over_american)
    p_under_raw = american_to_implied(under_american)
    overround = p_over_raw + p_under_raw

    z = solve_shin_z(p_over_raw, p_under_raw)

    if z is not None:
        # Apply Shin formula
        def shin_true(p_raw, z, overround):
            qi = p_raw / overround
            return (np.sqrt(z ** 2 + 4 * (1 - z) * qi) - z) / (2 * (1 - z))

        p_over_true = shin_true(p_over_raw, z, overround)
        p_under_true = shin_true(p_under_raw, z, overround)
        method = "shin"
    else:
        # Proportional normalization fallback
        p_over_true = p_over_raw / overround
        p_under_true = p_under_raw / overround
        z = None
        method = "proportional"

    return {
        "p_over_true": round(p_over_true, 6),
        "p_under_true": round(p_under_true, 6),
        "overround": round(overround, 6),
        "shin_z": round(z, 6) if z is not None else None,
        "method": method,
    }


def kelly_fraction(p_true: float, american_odds: float, max_fraction: float = 0.02) -> float:
    """
    Full Kelly fraction capped at max_fraction (default 2%).

    Formula: f* = (b*p - q) / b
    where:
        b = decimal odds - 1
        p = true probability of winning
        q = 1 - p
    """
    if american_odds > 0:
        b = american_odds / 100.0
    else:
        b = 100.0 / abs(american_odds)

    p = p_true
    q = 1.0 - p
    f = (b * p - q) / b

    if f <= 0:
        return 0.0  # no edge, no bet

    return round(min(f, max_fraction), 6)


def devig_prop_line(over_american: float, under_american: float,
                    projected_prob_over: float) -> dict:
    """
    Full pipeline: devig a prop line and compute Kelly fraction for the over.

    Args:
        over_american:        American odds for the over
        under_american:       American odds for the under
        projected_prob_over:  Your model's estimated P(over)

    Returns full breakdown including edge and Kelly fraction.
    """
    devig = shin_devig(over_american, under_american)
    book_p_over = devig["p_over_true"]
    edge = projected_prob_over - book_p_over
    kelly = kelly_fraction(projected_prob_over, over_american) if edge > 0 else 0.0

    return {
        **devig,
        "projected_p_over": round(projected_prob_over, 6),
        "book_p_over": book_p_over,
        "edge": round(edge, 6),
        "kelly_fraction": kelly,
        "bet_signal": "OVER" if edge > 0.02 else "PASS",
    }


if __name__ == "__main__":
    # Sanity check — standard -110 / -110 line (50/50 market with vig)
    print("=== Shin De-Vig Sanity Check ===\n")

    result = shin_devig(-110, -110)
    print(f"Standard -110/-110 line:")
    print(f"  Overround:  {result['overround']}  (expect ~1.047)")
    print(f"  P(over):    {result['p_over_true']}  (expect ~0.50)")
    print(f"  P(under):   {result['p_under_true']}  (expect ~0.50)")
    print(f"  Shin z:     {result['shin_z']}")
    print(f"  Method:     {result['method']}\n")

    # Skewed line — book favors under
    result2 = shin_devig(-130, +110)
    print(f"Skewed -130 / +110 line:")
    print(f"  Overround:  {result2['overround']}")
    print(f"  P(over):    {result2['p_over_true']}")
    print(f"  P(under):   {result2['p_under_true']}")
    print(f"  Shin z:     {result2['shin_z']}\n")

    # Full pipeline — model says 58% on over, line is -110/-110
    result3 = devig_prop_line(-110, -110, projected_prob_over=0.58)
    print(f"Model says 58% on over, line is -110/-110:")
    print(f"  Book true P(over): {result3['book_p_over']}")
    print(f"  Edge:              {result3['edge']}")
    print(f"  Kelly fraction:    {result3['kelly_fraction']}")
    print(f"  Signal:            {result3['bet_signal']}")
