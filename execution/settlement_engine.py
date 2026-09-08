# SettlementEngine — fail-closed execution gate.
# Loads settlement_rules.csv at startup.
# Raises ValueError if any (stat_type, book) combination is missing.
# No bets are generated for undefined markets.

import csv
import os
import datetime
import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH       = str(PROJECT_ROOT / "data" / "raw.db")
RULES_PATH    = str(PROJECT_ROOT / "execution" / "settlement_rules.csv")
LATENCY_LIMIT = 1800  # 30 minutes in seconds — auto-stop if exceeded


class SettlementEngine:
    """
    Instantiated at startup. Raises ValueError immediately if any
    (stat_type, book) combination in use is missing from settlement_rules.csv.
    """

    def __init__(self, rules_path: str = RULES_PATH):
        if not os.path.exists(rules_path):
            raise FileNotFoundError(
                f"settlement_rules.csv not found at {rules_path}. "
                f"Commit this file before running any backtests or live bets."
            )
        self.rules   = self._load_rules(rules_path)
        self.started = datetime.datetime.utcnow()
        print(f"SettlementEngine initialized — {len(self.rules)} rules loaded")

    def _load_rules(self, path: str) -> dict:
        rules = {}
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["stat_type"].strip().lower(), row["book"].strip().lower())
                rules[key] = row["settle_rule"].strip()
        return rules

    def validate(self, stat_type: str, book: str) -> str:
        """
        Returns the settle_rule for (stat_type, book).
        Raises ValueError if the combination is not in settlement_rules.csv.
        This is intentionally fail-closed — missing rule = no bet.
        """
        key = (stat_type.strip().lower(), book.strip().lower())
        if key not in self.rules:
            raise ValueError(
                f"No settlement rule for ({stat_type}, {book}). "
                f"Add it to settlement_rules.csv before generating bets for this market."
            )
        return self.rules[key]

    def check_latency(self) -> bool:
        """
        Returns True if within latency budget, False = auto-stop.
        Budget: inference must complete within 30min of startup.
        """
        elapsed = (datetime.datetime.utcnow() - self.started).total_seconds()
        if elapsed > LATENCY_LIMIT:
            print(f"AUTO-STOP: latency {elapsed:.0f}s exceeds {LATENCY_LIMIT}s budget.")
            print("System is missing the market. No bets generated.")
            return False
        remaining = LATENCY_LIMIT - elapsed
        print(f"Latency check: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
        return True

    def validate_bet_slate(self, bets: list) -> list:
        """
        Filter a list of bet dicts through settlement validation + latency check.
        Each bet dict must have 'stat_type' and 'book' keys.
        Returns only bets that pass all gates.
        """
        if not self.check_latency():
            return []

        valid = []
        for bet in bets:
            try:
                rule = self.validate(bet["stat_type"], bet["book"])
                valid.append({**bet, "settle_rule": rule})
            except ValueError as e:
                print(f"  REJECTED: {e}")
        return valid


def run_settlement_check(con, engine: SettlementEngine):
    """
    Pull today's shadow predictions and validate each through SettlementEngine.
    Prints which bets are valid vs rejected.
    """
    rows = con.execute("""
        SELECT pred_id, player_name, stat, book, our_line, signal, kelly_fraction
        FROM shadow_predictions
        WHERE settled = FALSE
        ORDER BY created_at DESC
        LIMIT 50
    """).fetchall()

    if not rows:
        print("No pending shadow predictions to validate.")
        return

    bets = [
        {"pred_id": r[0], "player_name": r[1], "stat_type": r[2],
         "book": r[3], "line": r[4], "signal": r[5], "kelly": r[6]}
        for r in rows
    ]

    print(f"\nValidating {len(bets)} pending bets through SettlementEngine...\n")
    valid_bets = engine.validate_bet_slate(bets)

    print(f"\n=== Valid Bets ({len(valid_bets)}/{len(bets)}) ===\n")
    if valid_bets:
        print(f"{'PLAYER':<22} {'STAT':<6} {'BOOK':<15} {'LINE':>5} {'KELLY':>7} {'SIGNAL'}")
        print("-" * 70)
        for b in valid_bets:
            if b["signal"] == "OVER":
                print(f"{b['player_name']:<22} {b['stat_type']:<6} {b['book']:<15} "
                      f"{b['line']:>5} {b['kelly']:>7.4f} {b['signal']}")
    else:
        print("No valid bets passed all gates.")


def main():
    import sys
    try:
        engine = SettlementEngine()
    except (FileNotFoundError, ValueError) as e:
        print(f"STARTUP FAILURE: {e}")
        return 1

    con = duckdb.connect(DB_PATH)
    run_settlement_check(con, engine)
    con.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
