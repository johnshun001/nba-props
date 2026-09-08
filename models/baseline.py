import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")
LOOKBACK_GAMES = 10

MARKET_TO_STAT = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
}

def connect_db():
    return duckdb.connect(DB_PATH)

def load_game_features(con):
    return con.execute("""
        SELECT player_id, game_date, minutes, pts, reb, ast
        FROM player_game_features
        WHERE minutes > 0
        ORDER BY player_id, game_date ASC
    """).df()

def load_prop_lines(con):
    return con.execute("""
        SELECT DISTINCT player_name, market, line, bookmaker
        FROM prop_lines
        WHERE market IN ('player_points', 'player_rebounds', 'player_assists')
    """).df()

def load_lookup(con):
    return con.execute("""
        SELECT player_id, full_name, team FROM player_lookup
    """).df()

def compute_projections(df):
    results = []
    for player_id, group in df.groupby("player_id"):
        group = group.sort_values("game_date").tail(LOOKBACK_GAMES)
        if len(group) < 3:
            continue
        avg_min = group["minutes"].mean()
        for stat in ["pts", "reb", "ast"]:
            per_min = (group[stat] / group["minutes"].replace(0, np.nan)).mean()
            proj = avg_min * per_min if pd.notna(per_min) else None
            std = group[stat].std()
            results.append({
                "player_id": player_id,
                "stat": stat,
                "projected": round(proj, 2) if proj is not None else None,
                "std": round(std, 2) if pd.notna(std) else None,
                "avg_minutes": round(avg_min, 2),
                "n_games": len(group),
            })
    return pd.DataFrame(results)

def normalize_name(name):
    return name.strip().lower().replace(".", "").replace("'", "")

def match_lines_to_projections(proj_df, prop_df, lookup_df):
    # build name → player_id map with normalized keys
    name_map = {normalize_name(row["full_name"]): row["player_id"]
                for _, row in lookup_df.iterrows()}

    stat_map = MARKET_TO_STAT
    matched = []

    for _, line_row in prop_df.iterrows():
        norm = normalize_name(line_row["player_name"])
        player_id = name_map.get(norm)
        if player_id is None:
            continue

        stat = stat_map.get(line_row["market"])
        if stat is None:
            continue

        proj_row = proj_df[
            (proj_df["player_id"] == player_id) &
            (proj_df["stat"] == stat)
        ]
        if proj_row.empty:
            continue

        proj = proj_row.iloc[0]["projected"]
        std = proj_row.iloc[0]["std"]
        avg_min = proj_row.iloc[0]["avg_minutes"]
        n = proj_row.iloc[0]["n_games"]
        line = line_row["line"]

        edge = round(proj - line, 2) if proj is not None else None
        edge_pct = round((proj - line) / line * 100, 1) if proj and line else None
        signal = "OVER" if edge and edge > 0 else "UNDER" if edge and edge < 0 else "NONE"

        matched.append({
            "player": line_row["player_name"],
            "market": line_row["market"].replace("player_", ""),
            "book": line_row["bookmaker"],
            "line": line,
            "projected": proj,
            "std": std,
            "avg_min": avg_min,
            "n_games": n,
            "edge": edge,
            "edge_pct": f"{edge_pct}%" if edge_pct is not None else "N/A",
            "signal": signal,
        })

    return pd.DataFrame(matched)

def print_results(matched_df):
    if matched_df.empty:
        print("No matches found between prop lines and tracked players.")
        print("The players in today's prop lines may not overlap with the 10 scraped players.")
        return

    print(f"{'PLAYER':<25} {'MARKET':<10} {'BOOK':<15} {'LINE':>6} {'PROJ':>6} {'EDGE':>6} {'EDGE%':>7} {'SIGNAL'}")
    print("-" * 90)
    for _, r in matched_df.sort_values("edge_pct", ascending=False).iterrows():
        print(f"{r['player']:<25} {r['market']:<10} {r['book']:<15} "
              f"{r['line']:>6} {str(r['projected']):>6} {str(r['edge']):>6} "
              f"{str(r['edge_pct']):>7}  {r['signal']}")

def main():
    con = connect_db()
    game_df = load_game_features(con)
    prop_df = load_prop_lines(con)
    lookup_df = load_lookup(con)
    con.close()

    if game_df.empty:
        print("No game features. Run materialize.py first.")
        return

    proj_df = compute_projections(game_df)
    matched_df = match_lines_to_projections(proj_df, prop_df, lookup_df)

    print(f"Tracked players: {game_df['player_id'].nunique()}")
    print(f"Prop lines loaded: {len(prop_df)}")
    print(f"Matches found: {len(matched_df)}\n")
    print_results(matched_df)

if __name__ == "__main__":
    main()
