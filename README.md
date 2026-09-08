# NBA Props Research Toolkit

[![Tests](https://github.com/johnshun001/nba-props/actions/workflows/tests.yml/badge.svg)](https://github.com/johnshun001/nba-props/actions/workflows/tests.yml)

A Python toolkit for collecting NBA player-prop data, training probability
models, backtesting predictions, and checking execution quality.

## 1. Setup

You need Git and Python 3.11.

```bash
git clone https://github.com/johnshun001/nba-props.git
cd nba-props
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python setup_env.py
```

On Windows PowerShell, replace the activation command with:

```powershell
.venv\Scripts\Activate.ps1
```

The final setup command creates a fresh local database at `data/raw.db`.

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

Expected result: `224 passed`.

### Collect and prepare data

```bash
export NBA_SEASON="2024-25"
python -m scrapers.nba_scraper
python -m storage.player_lookup
python -m scrapers.schedule_scraper
python -m storage.materialize
```

Sportsbook odds are optional and require `ODDS_API_KEY`:

```bash
python -m scrapers.odds_scraper
python -m storage.materialize
```

To collect the additional historical season configured in the project:

```bash
python -m scripts.ingest_historical
```

### Train the models

Run these after data collection:

```bash
python -m models.hmm_minutes
python -m models.qrf_model
python -m models.teammate_shock
python -m tests.replay
python -m models.calibration
```

### Run the reports and checks

```bash
python -m tests.backtest
python -m execution.health_check
python -m execution.shadow_tracker
python -m execution.close_spec
python -m execution.settlement_engine
python -m execution.drift_monitor
```

Some execution checks need schedule, odds, and shadow-prediction records in the
local database. If that data does not exist yet, the command will report that
there is nothing to evaluate.

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
| Build feature tables | `python -m storage.materialize` |
| Train minutes models | `python -m models.hmm_minutes` |
| Train prop models | `python -m models.qrf_model` |
| Generate replay results | `python -m tests.replay` |
| Run historical backtest | `python -m tests.backtest` |

## Project folders

| Folder | Purpose |
| --- | --- |
| `scrapers/` | NBA schedule, game-log, and sportsbook collection |
| `storage/` | Database setup, materialization, and bet tracking |
| `models/` | Statistical and machine-learning models |
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
