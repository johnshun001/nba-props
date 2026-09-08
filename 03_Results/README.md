# 03 Results

Use this section when you want to know where outputs go after the project runs.

## Output Locations

| Path | Contents |
| --- | --- |
| `data/raw.db` | Local SQLite database. |
| `data/replay_results.csv` | Walk-forward prediction output. |
| `models/hmm_store/` | Saved minutes-model artifacts. |
| `models/qrf_store/` | Saved quantile-forest artifacts. |
| `models/shock_store/` | Saved teammate-shock artifacts. |
| `models/calibration_store/` | Saved calibration artifacts. |
| `output/` | Generated reports and analysis files. |
| `logs/` | Runtime logs. |
| `signals/` | Generated signal files, when produced. |
| `track_record/` | Generated performance-history files, when produced. |
| `syndicate/*.json` | Generated syndicate payloads. |

## Sharing Results

The GitHub repo is for source code and instructions. Generated databases,
trained model stores, reports, logs, and signal files stay local by default.

Send generated result files separately only when a collaborator needs your exact
local run instead of rebuilding the results from the code.
