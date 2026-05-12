from __future__ import annotations

import json
from pathlib import Path

from .atomic_properties import elem_number


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
    "BOND_DISTANCE_OVERRIDES": {},
    "OUTPUT_FORCE_DEFAULT": False,
}


def default_config_path() -> Path:
    """Return the conventional repo-root config path used during source-tree runs."""
    return Path(__file__).resolve().parents[1] / "config.json"


def normalize_bond_distance_overrides(raw_overrides) -> dict[str, float]:
    """Normalize optional pair-specific absolute bond cutoffs from config JSON."""
    if not isinstance(raw_overrides, dict):
        return {}

    overrides: dict[str, float] = {}
    for raw_key, raw_value in raw_overrides.items():
        if not isinstance(raw_key, str):
            continue
        parts = raw_key.split("-")
        if len(parts) != 2:
            continue

        symbols = []
        for part in parts:
            symbol = part.strip().capitalize()
            if not symbol or symbol not in elem_number:
                symbols = []
                break
            symbols.append(symbol)
        if len(symbols) != 2:
            continue

        try:
            cutoff = float(raw_value)
        except (TypeError, ValueError):
            continue

        if cutoff <= 0.0:
            continue

        first, second = sorted(symbols)
        overrides[f"{first}-{second}"] = cutoff

    return overrides


def load_app_config(config_path: str | Path | None = None) -> dict:
    """Load optional JSON config overrides on top of built-in application defaults."""
    config = dict(DEFAULT_APP_CONFIG)
    config["BOND_DISTANCE_OVERRIDES"] = dict(DEFAULT_APP_CONFIG["BOND_DISTANCE_OVERRIDES"])
    path = Path(config_path) if config_path is not None else default_config_path()

    try:
        with open(path, "r", encoding="utf-8") as fin:
            raw = json.load(fin)
    except (OSError, json.JSONDecodeError):
        return config

    if not isinstance(raw, dict):
        return config

    config.update(raw)
    config["BOND_DISTANCE_OVERRIDES"] = normalize_bond_distance_overrides(
        raw.get("BOND_DISTANCE_OVERRIDES", {})
    )
    return config
