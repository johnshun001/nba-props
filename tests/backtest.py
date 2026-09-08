import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")

SEASONS: Dict[str, Tuple[str, str]] = {
    "2022-23": ("2022-10-01", "2023-06-30"),
    "2023-24": ("2023-10-01", "2024-06-30"),
    "2024-25": ("2024-10-01", "2025-06-30"),
}
TRAIN_SEASON = "2022-23"
EVAL_SEASON  = "2023-24"
STATS        = ["pts", "reb", "ast"]
MIN_TRAIN_GAMES = 20


def load_games(con, player_id: int, stat: str, season_str: str) -> np.ndarray:
    if season_str not in SEASONS:
        raise ValueError(f"Unknown season_str: {season_str}")
    start, end = SEASONS[season_str]
    q = (
        "SELECT game_date, " + stat + " AS v, minutes "
        "FROM player_game_features "
        "WHERE player_id = ? AND game_date >= ? AND game_date <= ? "
        "AND " + stat + " IS NOT NULL AND minutes IS NOT NULL AND minutes >= 1 "
        "ORDER BY game_date ASC"
    )
    rows = con.execute(q, [int(player_id), start, end]).fetchall()
    if not rows:
        return np.array([], dtype=float)
    vals: List[float] = []
    for r in rows:
        v, m = r[1], r[2]
        if v is None or m is None:
            continue
        if float(m) < 1.0:
            continue
        vals.append(float(v))
    return np.asarray(vals, dtype=float)


def rolling_baseline(vals: np.ndarray, window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vals, dtype=float).reshape(-1)
    if v.size <= window:
        return np.array([], dtype=float), np.array([], dtype=float)
    preds: List[float] = []
    actuals: List[float] = []
    for i in range(window, v.size):
        preds.append(float(np.mean(v[i - window:i])))
        actuals.append(float(v[i]))
    return np.asarray(preds, dtype=float), np.asarray(actuals, dtype=float)


def nonconf_scores_from_train(train_vals: np.ndarray, window: int = 10) -> np.ndarray:
    preds, actuals = rolling_baseline(train_vals, window=window)
    if preds.size == 0:
        return np.array([], dtype=float)
    return np.abs(preds - actuals)


def eval_player_stat(con, player_id: int, player_name: str, stat: str) -> Optional[Dict[str, Any]]:
    train_vals = load_games(con, player_id, stat, TRAIN_SEASON)
    eval_vals  = load_games(con, player_id, stat, EVAL_SEASON)

    if train_vals.size < MIN_TRAIN_GAMES or eval_vals.size < 5:
        return None

    nonconf_scores = nonconf_scores_from_train(train_vals, window=10)
    if nonconf_scores.size == 0:
        return None

    threshold = float(np.quantile(nonconf_scores, 0.90))

    if eval_vals.size < 10:
        return None

    # Use last 10 train games as context prefix so eval rolling baseline is clean
    context   = np.concatenate([train_vals[-10:], eval_vals]).astype(float)
    start_idx = 10 + 10   # 10 context prefix + 10 warmup within eval
    end_idx   = context.size
    n_valid   = end_idx - start_idx

    if n_valid < 5:
        return None

    hits: List[float]     = []
    abs_errs: List[float] = []
    cover: List[float]    = []

    for j in range(start_idx, end_idx):
        predicted = float(np.mean(context[j - 10:j]))
        actual    = float(context[j])
        hits.append(1.0 if actual > predicted else 0.0)
        abs_errs.append(abs(actual - predicted))
        cover.append(1.0 if (predicted - threshold <= actual <= predicted + threshold) else 0.0)

    hit_rate      = float(np.mean(hits))      if hits      else float("nan")
    mae           = float(np.mean(abs_errs))  if abs_errs  else float("nan")
    conf_coverage = float(np.mean(cover))     if cover     else float("nan")
    robust = bool((hit_rate > 0.52) and (conf_coverage >= 0.70))

    return {
        "player_id":     int(player_id),
        "player_name":   str(player_name),
        "stat":          str(stat),
        "n_train":       int(train_vals.size),
        "n_eval":        int(eval_vals.size),
        "hit_rate":      hit_rate,
        "mae":           mae,
        "conf_coverage": conf_coverage,
        "robust":        robust,
    }


def run_backtest(con) -> pd.DataFrame:
    players = con.execute(
        "SELECT player_id, full_name FROM player_lookup "
        "WHERE player_id IS NOT NULL ORDER BY player_id"
    ).fetchall()

    rows: List[Dict[str, Any]] = []
    for pid, name in players:
        player_id   = int(pid)
        player_name = str(name) if name is not None else str(player_id)
        for stat in STATS:
            try:
                res = eval_player_stat(con, player_id, player_name, stat)
            except Exception as e:
                print(f"  eval failed player_id={player_id} stat={stat}: {e}")
                res = None
            if res is not None:
                rows.append(res)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["robust", "hit_rate", "conf_coverage", "player_name", "stat"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
    return df


def print_report(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("No results -- not enough historical data yet.")
        print("(Need 2022-23 and 2023-24 game logs ingested.)")
        return

    cols = ["player_name", "stat", "n_train", "n_eval", "hit_rate", "mae", "conf_coverage", "robust"]
    disp = df[cols].copy()
    disp["hit_rate"]      = disp["hit_rate"].map(lambda x: f"{float(x):.3f}")
    disp["mae"]           = disp["mae"].map(lambda x: f"{float(x):.3f}")
    disp["conf_coverage"] = disp["conf_coverage"].map(lambda x: f"{float(x):.3f}")
    disp["robust"]        = disp["robust"].map(lambda x: "Y" if bool(x) else "N")

    header = ["player", "stat", "n_tr", "n_ev", "hit", "mae", "cov90", "rob"]
    widths = [22, 5, 6, 6, 6, 7, 7, 4]
    fmt_header = " ".join(h.ljust(w) for h, w in zip(header, widths))
    print(fmt_header)
    print("-" * len(fmt_header))

    for _, r in disp.iterrows():
        row = [
            str(r["player_name"])[:widths[0]].ljust(widths[0]),
            str(r["stat"]).ljust(widths[1]),
            str(r["n_train"]).rjust(widths[2]),
            str(r["n_eval"]).rjust(widths[3]),
            str(r["hit_rate"]).rjust(widths[4]),
            str(r["mae"]).rjust(widths[5]),
            str(r["conf_coverage"]).rjust(widths[6]),
            str(r["robust"]).rjust(widths[7]),
        ]
        print(" ".join(row))

    overall_hit  = float(df["hit_rate"].mean())
    overall_mae  = float(df["mae"].mean())
    robust_count = int(df["robust"].sum())

    print(f"\nSUMMARY: mean_hit_rate={overall_hit:.3f}  mean_mae={overall_mae:.3f}  robust_count={robust_count}")

    robust_players = df[df["robust"]].copy()
    print("\nROBUST PLAYERS:")
    if robust_players.empty:
        print("  (none yet -- need more historical seasons)")
    else:
        for name in sorted(set(robust_players["player_name"].tolist())):
            print(f"  {name}")


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df = run_backtest(con)
        print_report(df)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
