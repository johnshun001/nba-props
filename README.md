# NBA Props Research Toolkit

A Python research pipeline for collecting NBA player-game and sportsbook data,
building probabilistic player-prop models, replaying historical decisions, and
monitoring execution quality.

The repository includes:

- NBA and sportsbook scrapers backed by DuckDB
- baseline, HMM minutes, quantile-forest, calibration, and uncertainty models
- walk-forward replay, backtesting, Monte Carlo, cohort, and attribution tools
- settlement, closing-line, health, drift, and shadow-tracking controls
- 221 automated tests covering the reusable analytics components

> This is research software, not financial advice. Model output is uncertain;
> comply with all laws and sportsbook rules that apply to you.

## Quick start

Requirements: Git and Python 3.11.

```bash
git clone https://github.com/johnshun001/nba-props.git
cd nba-props
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python setup_env.py
python -m pytest -q
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

`setup_env.py` creates `data/raw.db` and the raw-ingestion table. Databases,
API keys, generated reports, and trained pickle files are intentionally local
and excluded from Git.

## Optional odds API configuration

NBA statistics use the public `nba_api` package. Live sportsbook collection uses
[The Odds API](https://the-odds-api.com/) and requires each user to provide their
own key:

```bash
cp .env.example .env
export ODDS_API_KEY="your-key-here"
```

The scripts read environment variables from the shell; they do not load `.env`
automatically. Never commit a real API key.

## Build the local dataset

Run commands from the repository root. The default example season is `2024-25`;
override it when needed with `NBA_SEASON`.

```bash
export NBA_SEASON="2024-25"
python -m scrapers.nba_scraper
python -m storage.player_lookup
python -m scrapers.schedule_scraper
python -m scrapers.odds_scraper       # requires ODDS_API_KEY
python -m storage.materialize
```

For the additional historical season configured in the script:

```bash
python -m scripts.ingest_historical
```

The upstream NBA endpoints can throttle or change. Re-run failed collection
commands after a short pause and inspect their console output before modeling.

## Train and evaluate models

After `player_game_features` has been materialized:

```bash
python -m models.hmm_minutes
python -m models.qrf_model
python -m models.teammate_shock
python -m tests.replay
python -m models.calibration
python -m tests.backtest
```

Generated model artifacts are written under `models/*_store/`; replay output is
written to `data/replay_results.csv`. These files are reproducible runtime
artifacts and are ignored by Git.

## Execution checks

The execution modules use the same local DuckDB database:

```bash
python -m execution.health_check
python -m execution.close_spec
python -m execution.shadow_tracker
python -m execution.settlement_engine
python -m execution.drift_monitor
```

`execution/settlement_rules.csv` is version-controlled because the settlement
engine fails closed when a required rule is absent.

## Project layout

```text
analysis/           Backtests, experiments, Monte Carlo, and attribution
commercialization/  Track-record, research, signal, and syndicate utilities
execution/          Health, close, settlement, shadow, and drift controls
models/             Statistical and machine-learning models
nba_analytics/      Market-efficiency and performance research
schemas/            Pydantic validation models
scrapers/           NBA schedule, game-log, and sportsbook collectors
scripts/            End-to-end ingestion helpers
storage/            Database materialization and bet tracking
tests/              Automated tests and offline replay tools
```

## Sharing data or pretrained models

The GitHub repository contains source code only. If a collaborator needs the
exact current state, send `data/raw.db` and the relevant `models/*_store/`
directories through a private file-sharing channel. DuckDB files may contain
collected operational history, and pickle files should only be opened when they
come from a trusted sender.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q .
python -m pytest -q
```

GitHub Actions runs the same compile and test checks on every push and pull
request.
