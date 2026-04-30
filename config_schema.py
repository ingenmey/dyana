"""Typed configuration scaffolding for non-interactive Dyana runs.

This module is intentionally lightweight in phase 1. The existing interactive
CLI remains the default path, while these dataclasses provide a stable shape
for future JSON/YAML-driven runs and for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameLoopConfig:
    start_frame: int = 1
    nframes: int = -1
    frame_stride: int = 1
    update_compounds: bool = False

    def __post_init__(self):
        if self.start_frame < 1:
            raise ValueError("start_frame must be >= 1.")
        if self.nframes != -1 and self.nframes < 1:
            raise ValueError("nframes must be >= 1 or -1.")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1.")


@dataclass(frozen=True)
class TopologyConfig:
    excluded_elements: set[str] = field(default_factory=set)
    neighbor_search_scale: float = 1.164
    bond_distance_scale: float = 1.4

    def __post_init__(self):
        if self.neighbor_search_scale <= 0:
            raise ValueError("neighbor_search_scale must be positive.")
        if self.bond_distance_scale <= 0:
            raise ValueError("bond_distance_scale must be positive.")


@dataclass(frozen=True)
class AnalysisRunConfig:
    analysis: str
    trajectory_file: str
    trajectory_format: str | None = None
    output_dir: str = "."
    frame_loop: FrameLoopConfig = field(default_factory=FrameLoopConfig)
    parameters: dict[str, Any] = field(default_factory=dict)


def load_analysis_run_config(path: str | Path) -> AnalysisRunConfig:
    """Load a JSON analysis-run config into typed dataclasses."""
    with open(path, "r", encoding="utf-8") as fin:
        raw = json.load(fin)

    frame_loop = FrameLoopConfig(**raw.get("frame_loop", {}))
    return AnalysisRunConfig(
        analysis=raw["analysis"],
        trajectory_file=raw["trajectory_file"],
        trajectory_format=raw.get("trajectory_format"),
        output_dir=raw.get("output_dir", "."),
        frame_loop=frame_loop,
        parameters=raw.get("parameters", {}),
    )
