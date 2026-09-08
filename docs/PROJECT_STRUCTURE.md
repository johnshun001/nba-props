# Project Structure

Use this as the quick map for what each file or folder is for. The project is
organized around three jobs: set up the environment, run the pipeline, and read
the results.

## 1. Setup

These files help someone clone the repo, install dependencies, configure local
settings, and verify that the project is ready to run.

| Path | What it does |
| --- | --- |
| `01_Setup/` | Sequential guide for installing and verifying the project. |
| `README.md` | Main starting point with setup, run, and results commands. |
| `.env.example` | Shows the optional environment variables without exposing secrets. |
| `requirements.txt` | Runtime Python dependencies. |
| `requirements-dev.txt` | Test and development dependencies. |
| `pyproject.toml` | Build metadata and packaging configuration. |
| `pytest.ini` | Pytest configuration. |
| `setup_env.py` | Creates local directories and initializes `data/raw.db`. |
| `.github/workflows/tests.yml` | Runs the automated test suite on GitHub. |
| `.gitignore` | Keeps generated data, model artifacts, secrets, and caches out of Git. |

Start here when a new person is getting the repo running:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python setup_env.py
python -m pytest -q
```

## 2. Run

These folders contain the code that collects data, builds features, trains
models, backtests predictions, and checks operational quality.

| Path | What it does |
| --- | --- |
| `02_Run/` | Sequential guide for running the data, model, and check pipelines. |
| `scrapers/` | Pulls NBA game logs, schedule data, and optional sportsbook odds. |
| `scripts/` | One-off or convenience ingestion commands. |
| `storage/` | Creates lookup tables, materializes features, and tracks bets. |
| `features/` | Builds model-ready feature sets. |
| `models/` | Trains, calibrates, and serves prediction models. |
| `analysis/` | Runs backtests, simulations, cohorts, and attribution analysis. |
| `execution/` | Checks health, settlement, shadow tracking, close quality, and drift. |
| `nba_analytics/` | Research tools for market efficiency and performance attribution. |
| `commercialization/` | Utilities for track records, distribution, and signal packaging. |
| `schemas/` | Shared validation models. |
| `config/` | Local pipeline configuration. |
| `tests/` | Automated tests plus replay and backtest entry points. |

The usual run order is:

```bash
python -m scrapers.nba_scraper
python -m storage.player_lookup
python -m scrapers.schedule_scraper
python -m storage.materialize
python -m models.hmm_minutes
python -m models.qrf_model
python -m models.teammate_shock
python -m tests.replay
python -m tests.backtest
```

## 3. Results

These paths are created locally while the project runs. They are intentionally
not committed because they can be large, generated, or private.

| Path | What it contains |
| --- | --- |
| `03_Results/` | Sequential guide for finding and sharing generated results. |
| `data/raw.db` | Local SQLite database created by setup and populated by the pipeline. |
| `data/replay_results.csv` | Walk-forward prediction output. |
| `models/hmm_store/` | Saved minutes-model artifacts. |
| `models/qrf_store/` | Saved quantile-forest artifacts. |
| `models/shock_store/` | Saved teammate-shock artifacts. |
| `models/calibration_store/` | Saved calibration artifacts. |
| `output/` | Generated reports and analysis outputs. |
| `logs/` | Runtime logs. |
| `signals/` | Generated signal outputs, when produced. |
| `track_record/` | Generated performance-history outputs, when produced. |
| `syndicate/*.json` | Generated syndicate payloads. |

When sharing the repo, send the GitHub link for the code. Send generated
databases, model directories, or reports separately only when the other person
needs the exact local results rather than a fresh run.
