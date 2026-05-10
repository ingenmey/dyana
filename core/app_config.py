from __future__ import annotations

import json
from pathlib import Path


DEFAULT_APP_CONFIG = {
    "EXCLUDED_ELEMENTS": [
        "Li",
        "Na",
        "Mg",
        "K",
        "Ca",
        "Rb",
        "Sr",
        "Cs",
        "Ba",
        "Cu",
        "Ag",
        "Au",
        "Pt",
        "Zn",
        "Fe",
        "Co",
    ],
    "NEIGHBOR_SEARCH_SCALE": 1.164,
    "BOND_DISTANCE_SCALE": 1.4,
    "OUTPUT_FORCE_DEFAULT": False,
}


def default_config_path() -> Path:
    """Return the conventional repo-root config path used during source-tree runs."""
    return Path(__file__).resolve().parents[1] / "config.json"


def load_app_config(config_path: str | Path | None = None) -> dict:
    """Load optional JSON config overrides on top of built-in application defaults."""
    config = dict(DEFAULT_APP_CONFIG)
    path = Path(config_path) if config_path is not None else default_config_path()

    try:
        with open(path, "r", encoding="utf-8") as fin:
            raw = json.load(fin)
    except (OSError, json.JSONDecodeError):
        return config

    if not isinstance(raw, dict):
        return config

    config.update(raw)
    return config
