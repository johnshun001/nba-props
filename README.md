# NBA Props Research Toolkit

[![Tests](https://github.com/johnshun001/nba-props/actions/workflows/tests.yml/badge.svg)](https://github.com/johnshun001/nba-props/actions/workflows/tests.yml)

A Python toolkit for collecting NBA player-prop data, training probability
models, backtesting predictions, and checking execution quality.

## Project guide

The top-level numbered folders are the easiest path through the project:

| Step | Folder | Use it for |
| --- | --- | --- |
| 1 | [`01_Setup/`](01_Setup/) | Install dependencies, configure the environment, and verify the repo. |
| 2 | [`02_Run/`](02_Run/) | Collect data, train models, and run checks. |
| 3 | [`03_Results/`](03_Results/) | Understand where generated outputs are saved and what to share. |

## Current status

Last verified 2026-09-08 against a local database holding 10 tracked players,
2,186 player-game rows, and 5,919 sportsbook lines.

### Working end to end

- **Ingestion and materialization.** `raw_api_responses` keeps immutable source
  JSON. `storage/materialize.py` parses it into `player_game_features`,
  `player_results`, and `prop_lines` with deterministic reproducibility hashes.
- **Event mapping.** `storage/event_mapping.py` matches sportsbook events to NBA
  game IDs on normalized team names plus a two-hour tip window, and rejects
  ambiguous matches instead of guessing.
- **Leakage-free features.** Every rolling statistic in
  `features/pregame_features.py` is shifted by one game before the window is
  applied, and context tables join through a backward `merge_asof` on
  `prediction_time`. Same-game actual minutes are never a model input.
- **Pooled ensemble training.** `models/train.py` fits LightGBM quantiles,
  XGBoost, a quantile forest, and a PyTorch joint-quantile member over ordered,
  disjoint train, validation, calibration, and test periods. Held-out test MAE
  is 7.28 points, 2.15 rebounds, 1.82 assists.
- **Selection and staking.** `execution/betting.py` converts prices to no-vig
  probabilities, evaluates both sides, shops by expected value with fully
  deterministic tiebreaks, and applies fractional Kelly under player, game, and
  correlated exposure caps.
- **Execution controls.** `health_check`, `close_spec`, `settlement_engine`, and
  `drift_monitor` each run and report correctly.
- **Tests.** `python -m pytest -q` reports `237 passed`.

### Known gaps

These are data-coverage problems rather than model problems. Every gate below
fails closed, so the pipeline reports `NOT_READY` instead of emitting a bet it
cannot support.

- **Player-name joins do not fold diacritics.** `player_lookup` stores
  `Luka Dončić` while the odds feed returns `Luka Doncic`, so the
  `lower(trim(...))` joins in `analysis/replay.py`, `models/train.py`, and
  `execution/shadow_tracker.py` silently drop those rows.
- **Tip times are lost once a game starts.** `scrapers/schedule_scraper.py`
  reads `GAME_STATUS_TEXT`, which becomes `Final` or `1st Qtr` after tipoff and
  no longer parses to a timestamp. `map_events_to_games` requires a non-null
  `tip_time_utc`, so a game collected late can never be mapped. Collect the
  schedule before tipoff.
- **Calibration is not fit yet.** Too few rows carry a sportsbook line inside the
  calibration window, so `calibration_status` stays `NOT_READY` and reported
  probabilities are uncalibrated. `analysis/replay.py` deliberately drops any
  candidate that is not `READY`, so replay currently produces no bets.
- **Coverage is narrow.** `scrapers/nba_scraper.py` tracks a hardcoded list of
  ten `PLAYER_IDS`.

### Not on the main path

`nba_analytics/`, `commercialization/`, and several modules under `models/`
(`calibration`, `conformal`, `plc_priors`, `shin_devig`, `lineup_uncertainty`)
are tested but are not imported by any pipeline entry point. Treat them as a
library rather than as part of the running system.

## 1. Setup

You need Git and Python 3.11.

```bash
git clone https://github.com/johnshun001/nba-props.git
cd nba-props
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
# Optional PyTorch quantile member (used automatically when installed)
python -m pip install -r requirements-torch.txt
python setup_env.py
```

On Windows PowerShell, replace the activation command with:

```powershell
.venv\Scripts\Activate.ps1
```

The final setup command creates a fresh local database at `data/raw.db`.

For a plain-English map of what each file and folder is for, see
[`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md).

### Optional: sportsbook odds

NBA statistics use the public `nba_api` package. Sportsbook odds require a key
from [The Odds API](https://the-odds-api.com/).

```bash
export ODDS_API_KEY="your-key-here"
```

Each user must supply their own key. Never commit it to Git.

## 2. Run

Always run commands from the repository root with the virtual environment
activated.

### Check that everything works

```bash
python -m pytest -q
```

Expected result: `237 passed`.

### Configure the pipeline

`config/pipeline.json` is the single source for seasons, statistics, rolling
windows, thresholds, feature/data/model versions, exposure caps, and
project-relative artifact paths.

### Collect and prepare data

```bash
export NBA_SEASON="2024-25"
python -m scrapers.nba_scraper
python -m storage.player_lookup
python -m scrapers.schedule_scraper
python -m scrapers.context_scraper --season 2024-25
python -m storage.materialize
python -m features.pregame_features
```

Sportsbook odds are optional and require `ODDS_API_KEY`:

```bash
python -m scrapers.odds_scraper
python -m storage.materialize
```

Historical sportsbook snapshots use an explicit timestamp:

```bash
python -m scrapers.odds_scraper --historical-date 2025-01-15T23:00:00Z
python -m storage.materialize
```

Timestamped lineup, injury/availability, and opponent-by-position feeds can be
loaded from provider-neutral JSON:

```bash
python -m scrapers.context_scraper \
  --lineups-json data/import/lineups.json \
  --availability-json data/import/availability.json \
  --position-defense-json data/import/position-defense.json
```

To collect the additional historical season configured in the project:

```bash
python -m scripts.ingest_historical
```

### Train the models

Run these after data collection:

```bash
python -m features.pregame_features
python -m models.train
```

The pooled points, rebounds, and assists models use LightGBM quantiles,
XGBoost regression, the quantile-forest benchmark, and—when installed—a small
PyTorch joint-quantile model. Training uses ordered, disjoint train,
validation, calibration, and untouched test periods. Same-game actual minutes
are never model inputs; expected minutes, DNP/role probabilities, and minutes
uncertainty come directly from the pregame HMM path.

### Generate predictions

Pass CSV or Parquet pregame rows containing the stored feature columns plus
`stat` and `sportsbook_line`:

```bash
python -m models.predict data/prediction_slate.parquet \
  --output data/predictions.parquet
```

Output includes q10/q25/median/q75/q90, calibrated over/under probabilities,
and conformal intervals fit on held-out residuals.

### Replay historical sportsbook decisions

```bash
python -m analysis.replay
```

Replay requires paired snapshots mapped to the correct NBA event and settles
against actual player results. It reports MAE, RMSE, direction accuracy, Brier
score, log loss, calibration error, two-sided win rate, ROI after vig, CLV,
maximum drawdown, and breakdowns by player/stat/edge/book. It never substitutes
a simulated rolling-average line. Replay prints `NOT_READY` and exits when those
preconditions are unmet; see [Known gaps](#known-gaps).

### Run the reports and checks

```bash
python -m tests.backtest
python -m execution.health_check
python -m execution.shadow_tracker run
python -m execution.close_spec
python -m execution.close_spec settle
python -m execution.settlement_engine
python -m execution.drift_monitor
```

Some execution checks need schedule, odds, and shadow-prediction records in the
local database. If that data does not exist yet, the command will report that
there is nothing to evaluate.

Shadow mode shops both OVER and UNDER opportunities, uses calibrated
probabilities and fractional Kelly, and caps player/game/correlated exposure.
Fewer than 100 properly settled forecasts—or missing CLV/calibration evidence—
returns `NOT_READY`. Promotion requires both the CLV and calibration gates.

## 3. Results

Commands print summaries in the terminal and save generated files locally:

| Result | Location |
| --- | --- |
| Collected and materialized data | `data/raw.db` |
| Walk-forward predictions | `data/replay_results.csv` |
| Minutes models | `models/hmm_store/` |
| Quantile-forest models | `models/qrf_store/` |
| Teammate-shock models | `models/shock_store/` |
| Calibration models | `models/calibration_store/` |
| Pooled ensemble models + metadata | `models/ensemble_store/` |
| Generated reports | `output/` and the relevant analytics folders |
| Automated test history | [GitHub Actions](https://github.com/johnshun001/nba-props/actions) |

These results are excluded from Git because they are generated, can be large,
and may contain private operational data. To give a collaborator your exact
current results, send the required database or model directories separately
through a private file-sharing channel.

Do not open pickle model files from untrusted sources.

## Command summary

| Goal | Command |
| --- | --- |
| Initialize the database | `python setup_env.py` |
| Verify the installation | `python -m pytest -q` |
| Collect player game logs | `python -m scrapers.nba_scraper` |
| Collect sportsbook odds | `python -m scrapers.odds_scraper` |
| Build raw result/odds tables | `python -m storage.materialize` |
| Build versioned pregame features | `python -m features.pregame_features` |
| Train minutes models | `python -m models.hmm_minutes` |
| Train pooled prop models | `python -m models.train` |
| Predict a slate | `python -m models.predict data/prediction_slate.parquet` |
| Generate replay results | `python -m analysis.replay` |
| Run shadow mode | `python -m execution.shadow_tracker run` |
| Run historical backtest | `python -m tests.backtest` |

## Project folders

| Folder | Purpose |
| --- | --- |
| `scrapers/` | NBA schedule, game-log, and sportsbook collection |
| `storage/` | Database setup, materialization, and bet tracking |
| `models/` | Statistical and machine-learning models |
| `features/` | Leakage-free pregame features and versioned feature storage |
| `config/` | Central pipeline configuration and versions |
| `analysis/` | Backtests, experiments, Monte Carlo, and attribution |
| `execution/` | Health, close, settlement, shadow, and drift controls |
| `nba_analytics/` | Market-efficiency and performance research |
| `commercialization/` | Track-record, signal, and research utilities |
| `schemas/` | Pydantic validation models |
| `tests/` | Automated tests and offline replay tools |

## Notes

- Upstream NBA endpoints can throttle or change. If collection fails, wait
  briefly, rerun it, and inspect the console output.
- `execution/settlement_rules.csv` is version-controlled because the settlement
  engine fails closed when a required rule is missing.
- This is research software, not financial advice. Model output is uncertain;
  comply with applicable laws and sportsbook rules.
