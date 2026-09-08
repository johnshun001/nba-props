# 01 Setup

Use this section when someone is opening the project for the first time.

## Start Here

Run these commands from the repository root:

```bash
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

## Setup Files

| Path | Purpose |
| --- | --- |
| `README.md` | Main guide for setup, run, and results. |
| `.env.example` | Template for optional secrets such as `ODDS_API_KEY`. |
| `requirements.txt` | Runtime dependencies. |
| `requirements-dev.txt` | Runtime plus test dependencies. |
| `pyproject.toml` | Build and package metadata. |
| `pytest.ini` | Test configuration. |
| `setup_env.py` | Creates local folders and initializes `data/raw.db`. |
| `.github/workflows/tests.yml` | GitHub Actions test workflow. |

The setup step creates local generated files. Those are ignored by Git and can
be rebuilt by each person who clones the project.
