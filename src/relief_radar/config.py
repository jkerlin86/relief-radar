"""Configuration loading for the Relief Radar pipeline."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load the pipeline config (defaults to configs/default.json)."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.json"
    with open(cfg_path, encoding="utf-8") as fh:
        return json.load(fh)


def project_path(*parts: str) -> Path:
    """Absolute path inside the project tree."""
    return PROJECT_ROOT.joinpath(*parts)
