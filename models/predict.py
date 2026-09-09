"""Generate calibrated distribution forecasts from saved ensemble artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import project_path
from models.ensemble import PooledPropEnsemble, SUPPORTED_STATS


def predict_frame(frame: pd.DataFrame, artifact_dir: str | Path | None = None) -> pd.DataFrame:
    directory = Path(artifact_dir) if artifact_dir else project_path("artifacts")
    outputs: list[pd.DataFrame] = []
    for stat, group in frame.groupby("stat", sort=False):
        if stat not in SUPPORTED_STATS:
            continue
        model = PooledPropEnsemble.load(directory / f"{stat}.pkl")
        if "feature_version" in group:
            versions = set(group["feature_version"].dropna().astype(str))
            if versions and versions != {model.artifact.feature_version}:
                raise ValueError(
                    f"Feature version mismatch for {stat}: expected {model.artifact.feature_version}, got {sorted(versions)}"
                )
        predicted = model.predict_distribution(group, group["sportsbook_line"])
        identity = [c for c in ["player_id", "game_id", "event_id", "prediction_time", "bookmaker", "stat"] if c in group]
        outputs.append(pd.concat([group[identity].reset_index(drop=True), predicted.reset_index(drop=True)], axis=1))
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Parquet or CSV pregame feature rows")
    parser.add_argument("--artifact-dir", type=Path, default=project_path("artifacts"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    frame = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    output = predict_frame(frame, args.artifact_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".parquet":
            output.to_parquet(args.output, index=False)
        else:
            output.to_csv(args.output, index=False)
    else:
        print(json.dumps(output.to_dict("records"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
