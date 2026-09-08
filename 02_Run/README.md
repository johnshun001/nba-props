# 02 Run

Use this section when the environment is ready and you want to collect data,
train models, or run checks.

## Data Pipeline

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

## Model Pipeline

```bash
python -m models.hmm_minutes
python -m models.qrf_model
python -m models.teammate_shock
python -m tests.replay
python -m models.calibration
```

## Checks And Reports

```bash
python -m tests.backtest
python -m execution.health_check
python -m execution.shadow_tracker
python -m execution.close_spec
python -m execution.settlement_engine
python -m execution.drift_monitor
```

## Run Folders

| Folder | Purpose |
| --- | --- |
| `scrapers/` | Collect NBA, schedule, and optional odds data. |
| `storage/` | Build lookup tables, materialize data, and track bets. |
| `features/` | Build model-ready inputs. |
| `models/` | Train, calibrate, and run prediction models. |
| `analysis/` | Run experiments, backtests, and attribution. |
| `execution/` | Run operational checks and settlement workflows. |
| `tests/` | Verify the project and run replay or backtest entry points. |
