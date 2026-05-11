"""Header formatting for supported Dyana interactive runs."""

from __future__ import annotations

import configparser
import importlib.metadata
from datetime import datetime
from pathlib import Path


def _resolve_absolute_path(path: str | Path) -> str:
    """Return an absolute normalized path string for console display/use."""
    return str(Path(path).expanduser().resolve(strict=False))


def resolve_dyana_version() -> str:
    """Return the installed Dyana version or fall back to setup metadata."""
    try:
        return importlib.metadata.version("dyana")
    except importlib.metadata.PackageNotFoundError:
        setup_cfg = Path(__file__).resolve().parents[1] / "setup.cfg"
        if setup_cfg.exists():
            parser = configparser.ConfigParser()
            parser.read(setup_cfg, encoding="utf-8")
            return parser.get("metadata", "version", fallback="unknown")
        return "unknown"


def format_started_at() -> str:
    """Return a local timestamp for the start of an interactive run."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def build_run_header(
    traj_file: str | Path,
    *,
    traj_format: str | None,
    output_dir: str | Path,
    console_log_path: str | Path,
    input_log_path: str | Path | None = None,
    prepared_setup: str | Path | None = None,
) -> tuple[str, list[str]]:
    """Build the supported run-header title and lines."""
    header_title = f"Dyana {resolve_dyana_version()}"
    trajectory_path = _resolve_absolute_path(traj_file)
    header_lines = [
        f"Started: {format_started_at()}",
        f"Trajectory: {trajectory_path}" if traj_format is None else f"Trajectory: {trajectory_path} ({traj_format})",
        f"Output dir: {_resolve_absolute_path(output_dir)}",
        f"Console log: {_resolve_absolute_path(console_log_path)}",
    ]
    if input_log_path is not None:
        header_lines.append(f"Input log: {_resolve_absolute_path(input_log_path)}")
    if prepared_setup is not None:
        header_lines.append(f"Prepared setup: {_resolve_absolute_path(prepared_setup)}")
    return header_title, header_lines
