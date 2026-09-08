import sys
import subprocess
import importlib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = str(PROJECT_ROOT / "data" / "raw.db")

def check_python_version():
    v = sys.version_info
    print(f"Python version: {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) < (3, 10):
        print("WARNING: Python < 3.10 detected. Upgrade recommended.")

def is_importable(package):
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False

def pip_install(package):
    print(f"Installing {package}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR installing {package}: {result.stderr}")
    else:
        print(f"OK: {package} installed")

def ensure_packages():
    packages = ["duckdb", "nba_api", "pandas", "requests", "numpy"]
    for pkg in packages:
        if is_importable(pkg):
            print(f"OK: {pkg} already available")
        else:
            pip_install(pkg)

def ensure_parent_dir():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    print(f"OK: data directory ready at {Path(DB_PATH).parent}")

def connect_duckdb():
    import duckdb
    con = duckdb.connect(DB_PATH)
    print(f"OK: connected to {DB_PATH}")
    return con

def create_schema(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_api_responses (
            raw_id                  VARCHAR PRIMARY KEY,
            source                  VARCHAR NOT NULL,
            endpoint                VARCHAR NOT NULL,
            params_hash             VARCHAR NOT NULL,
            ingestion_timestamp_utc TIMESTAMP NOT NULL,
            scrape_window_id        VARCHAR,
            raw_payload_json        TEXT NOT NULL,
            http_status             INTEGER,
            parse_status            VARCHAR
        )
    """)
    print("OK: raw_api_responses table ready")

def make_raw_id(source, endpoint, params_hash, ingestion_timestamp_utc):
    raw = f"{source}{endpoint}{params_hash}{str(ingestion_timestamp_utc)}"
    return hashlib.sha256(raw.encode()).hexdigest()

def insert_dummy_row(con):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    source = "test"
    endpoint = "dummy_endpoint"
    params_hash = hashlib.sha256(b"{}").hexdigest()
    raw_id = make_raw_id(source, endpoint, params_hash, now)
    payload = json.dumps({"test": True})

    con.execute("""
        INSERT OR IGNORE INTO raw_api_responses
            (raw_id, source, endpoint, params_hash, ingestion_timestamp_utc,
             scrape_window_id, raw_payload_json, http_status, parse_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [raw_id, source, endpoint, params_hash, now, None, payload, 200, "ok"])
    print(f"OK: dummy row inserted with raw_id={raw_id[:16]}...")
    return raw_id

def select_row_by_raw_id(con, raw_id):
    result = con.execute(
        "SELECT * FROM raw_api_responses WHERE raw_id = ?", [raw_id]
    ).fetchone()
    return result

def sanity_check(con):
    raw_id = insert_dummy_row(con)
    row = select_row_by_raw_id(con, raw_id)
    if row:
        print(f"OK: row confirmed in DB -> source={row[1]}, endpoint={row[2]}, status={row[8]}")
        print("Layer 0 initialized. raw.db ready.")
    else:
        print("ERROR: row not found after insert — something is wrong")

def main():
    check_python_version()
    ensure_packages()
    ensure_parent_dir()
    con = connect_duckdb()
    create_schema(con)
    sanity_check(con)
    con.close()

if __name__ == "__main__":
    main()
