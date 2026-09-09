# 03 Results

Use this section when you want to know where outputs go after the project runs.

## Output locations

| Path | Contents |
| --- | --- |
| `data/raw.db` | Local **DuckDB** database holding raw responses, features, lines, and results |
| `data/replay_results.csv` | Walk-forward replay output |
| `models/ensemble_store/` | Pooled ensemble artifacts plus a metadata JSON per stat |
| `models/hmm_store/` | Saved minutes-model artifacts |
| `models/qrf_store/` | Saved quantile-forest artifacts |
| `models/shock_store/` | Saved teammate-shock artifacts |
| `models/calibration_store/` | Saved calibration artifacts |
| `output/` | Generated reports and analysis files |
| `logs/` | Runtime logs |

## Reading the metadata

Each trained stat writes `models/ensemble_store/<stat>.json` next to its
artifact. Check these fields first:

| Field | Meaning |
| --- | --- |
| `split_boundaries` | Exact timestamps separating train, validation, calibration, and test |
| `test_metrics` | MAE and RMSE on the untouched test period |
| `calibration_status` | `READY` only when the isotonic calibrator actually fit |
| `conformal_rows` | Number of held-out residuals backing the prediction intervals |

If `calibration_status` is `NOT_READY`, reported probabilities are uncalibrated
and downstream betting stages will deliberately emit nothing.

## Inspecting the database

```bash
python -c "import duckdb; con=duckdb.connect('data/raw.db', read_only=True); \
print(con.execute('select table_name from information_schema.tables').fetchall())"
```

## Sharing results

The repository holds source code and instructions. Generated databases, trained
model stores, reports, and logs stay local by default because they are large,
rebuildable, and may contain private operational data.

Send generated files separately only when a collaborator needs your exact local
run instead of rebuilding from the code.
