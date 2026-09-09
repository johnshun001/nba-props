# 02 Run

Use this section when the environment is ready and you want to collect data,
train models, or run checks. Run everything from the repository root with the
virtual environment activated.

## 1. Collect

> [!WARNING]
> Scrape the schedule **before tipoff**. `schedule_scraper` reads
> `GAME_STATUS_TEXT`, which becomes `Final` or `1st Qtr` once a game starts and
> no longer parses to a timestamp. Event mapping requires a tip time, so a game
> collected late can never be mapped to its sportsbook event.

```bash
export NBA_SEASON="2024-25"
python -m scrapers.nba_scraper        # player game logs
python -m storage.player_lookup       # player id to name table
python -m scrapers.schedule_scraper   # schedule and tip times
```

Sportsbook odds are optional and require `ODDS_API_KEY`:

```bash
python -m scrapers.odds_scraper
```

Historical snapshots take an explicit timestamp:

```bash
python -m scrapers.odds_scraper --historical-date 2025-01-15T23:00:00Z
```

Timestamped lineup, availability, and opponent-by-position feeds load from
provider-neutral JSON:

```bash
python -m scrapers.context_scraper \
  --lineups-json data/import/lineups.json \
  --availability-json data/import/availability.json \
  --position-defense-json data/import/position-defense.json
```

To collect the additional historical season configured in `config/pipeline.json`:

```bash
python -m scripts.ingest_historical
```

## 2. Materialize and build features

```bash
python -m storage.materialize         # raw JSON to typed tables + event mapping
python -m features.pregame_features   # versioned, leakage-free feature store
```

## 3. Train

```bash
python -m models.train
```

Fits one pooled ensemble per stat (points, rebounds, assists) over ordered,
disjoint train, validation, calibration, and test periods. Writes artifacts and
a metadata JSON to `models/ensemble_store/`. Also saves one minutes HMM per
player with enough history.

## 4. Predict

Pass CSV or Parquet rows containing the stored feature columns plus `stat` and
`sportsbook_line`:

```bash
python -m models.predict data/slate.parquet --output data/predictions.parquet
```

Output includes `q10`/`q25`/`median`/`q75`/`q90`, over/under probabilities,
conformal intervals, and a `calibration_status` flag.

## 5. Evaluate

```bash
python -m analysis.replay             # walk-forward replay against real lines
python -m tests.backtest              # historical backtest
python -m execution.health_check      # data coverage and no-trade gate
python -m execution.shadow_tracker    # paper trading; never places wagers
python -m execution.close_spec        # capture closing lines
python -m execution.close_spec settle
python -m execution.settlement_engine
python -m execution.drift_monitor     # CLV, calibration, promotion gates
```

Several checks need schedule, odds, and shadow records already in the database.
When that data is missing they report `NOT_READY` rather than guessing — that is
expected behavior, not a failure.

## Configuration

`config/pipeline.json` is the single source for seasons, statistics, rolling
windows, split fractions, thresholds, exposure caps, versions, and artifact
paths. Change it there rather than editing modules.

## Run folders

| Folder | Purpose |
| --- | --- |
| `scrapers/` | Collect NBA logs, schedule, odds, and context feeds |
| `storage/` | Schemas, materialization, event mapping, bet tracking |
| `features/` | Build leakage-free model inputs |
| `models/` | Train, predict, and store ensemble and minutes models |
| `analysis/` | Replay, backtests, and attribution |
| `execution/` | Selection, staking, and operational checks |
| `tests/` | Verify the project and run replay or backtest entry points |
