"""Central, project-relative configuration for the NBA props pipeline."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.json"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def project_path(key: str) -> Path:
    """Resolve a configured path beneath the repository root."""
    value = load_config()["paths"][key]
    path = (PROJECT_ROOT / value).resolve()
    if PROJECT_ROOT not in path.parents and path != PROJECT_ROOT:
        raise ValueError(f"Configured path escapes project root: {value}")
    return path


__all__ = ["CONFIG_PATH", "PROJECT_ROOT", "load_config", "project_path"]
