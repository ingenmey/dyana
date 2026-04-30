"""Metadata helpers for reproducible analysis outputs."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np


def build_run_metadata(analysis: str, parameters=None, trajectory=None, extra=None) -> dict:
    """Build lightweight run metadata for an analysis output."""
    data = {
        "analysis": analysis,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
        },
    }

    for package in ("scipy", "networkx", "matplotlib"):
        try:
            data["dependencies"][package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            data["dependencies"][package] = None

    git_commit = _git_commit()
    if git_commit:
        data["git_commit"] = git_commit

    if parameters is not None:
        data["parameters"] = _jsonable(parameters)
    if trajectory is not None:
        data["trajectory"] = _jsonable(trajectory)
    if extra:
        data.update(_jsonable(extra))

    return data


def write_metadata(metadata: dict, filename: str | Path):
    """Write metadata as stable, indented JSON."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(_jsonable(metadata), fout, indent=2, sort_keys=True)
        fout.write("\n")


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(value.__dict__)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

