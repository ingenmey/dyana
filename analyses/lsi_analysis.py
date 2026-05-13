from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from analyses.common.base_analysis import BaseAnalysis
from analyses.common.histogram import HistogramND
from framework.analysis_params import AtomLabelsParam, BoolParam, CompoundParam, FloatParam, IntParam
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, write_histogram_1d, write_table


@dataclass(frozen=True)
class LSIConfig:
    """Configuration for Local Structure Index analysis."""

    compound_index: int
    site_labels: list[str]
    cutoff: float = 3.7
    bin_count_local: int = 100
    histogram_min: float = 0.0
    histogram_max: float = 0.4
    output_frame_mean: bool = True
    optional_threshold: float | None = None

    def __post_init__(self):
        if self.compound_index < 0:
            raise ValueError("compound_index must be >= 0.")
        if not self.site_labels:
            raise ValueError("site_labels must not be empty.")
        if self.cutoff <= 0:
            raise ValueError("cutoff must be > 0.")
        if self.bin_count_local < 1:
            raise ValueError("bin_count_local must be >= 1.")
        if self.histogram_min < 0:
            raise ValueError("histogram_min must be >= 0.")
        if self.histogram_max <= self.histogram_min:
            raise ValueError("histogram_max must be greater than histogram_min.")
        if self.optional_threshold is not None and self.optional_threshold < 0:
            raise ValueError("optional_threshold must be >= 0 or None.")


class LSIAnalysis(BaseAnalysis):
    """Local Structure Index analysis on one site atom per molecule."""

    CONFIG_CLASS = LSIConfig
    CONFIG_SCHEMA = [
        CompoundParam(
            name="compound_index",
            role="reference",
            prompt="Choose the site compound (number): ",
        ),
        AtomLabelsParam(
            name="site_labels",
            role="reference",
            compound="compound_index",
            prompt="Which atom(s) define the molecular site? (comma-separated) ",
        ),
        FloatParam(
            name="cutoff",
            prompt="Neighbour cutoff distance Angstrom: ",
            default=3.7,
            minval=0.0,
        ),
        IntParam(
            name="bin_count_local",
            prompt="Enter the number of bins for local LSI distribution: ",
            default=100,
            minval=1,
        ),
        FloatParam(
            name="histogram_min",
            prompt="Enter the minimum LSI value for the histogram (Angstrom^2): ",
            default=0.0,
            minval=0.0,
        ),
        FloatParam(
            name="histogram_max",
            prompt="Enter the maximum LSI value for the histogram (Angstrom^2): ",
            default=0.4,
            minval=0.0,
        ),
        BoolParam(
            name="output_frame_mean",
            prompt="Write per-frame mean/std/count output?",
            default=True,
        ),
        FloatParam(
            name="optional_threshold",
            prompt="Optional LSI threshold for per-frame fractions (Angstrom^2): ",
            default=None,
            display_default="None",
            minval=0.0,
            allow_none=True,
        ),
    ]

    def configure(self, config: LSIConfig):
        self.bind_config(config)
        (self.compound_type, self.compound_key), = self.resolve_compound_types([self.compound_index])
        self.site_selection = self.traj.topology_frame.resolve_selection(self.compound_type, self.site_labels)
        if len(self.site_selection.local_indices) != 1:
            raise ValueError("LSI requires exactly one selected site atom per molecule.")

        histogram_edges = np.linspace(
            self.histogram_min,
            self.histogram_max,
            self.bin_count_local + 1,
        )
        self.local_hist = HistogramND([histogram_edges], mode="linear")
        self.frame_rows: list[tuple[float, ...]] = []

        self.rebuild_runtime_state()
        if self.site_indices.size < 3:
            raise ValueError("LSI requires at least three molecular sites in the initial frame.")
        self.mark_configured()

    def rebuild_runtime_state(self):
        self.site_indices = self.traj.topology_frame.get_atom_ids_for_local_indices(
            self.compound_type,
            self.site_selection.local_indices,
        )

    def post_compound_update(self):
        topology_frame = self.traj.topology_frame
        if not topology_frame.has_compound_type_key(self.compound_key):
            return False

        self.compound_type = topology_frame.get_compound_type_by_key(self.compound_key)
        self.rebuild_runtime_state()
        return self.site_indices.size >= 3

    def process_frame(self):
        if self.site_indices.size < 3:
            return

        site_coords = np.mod(self.traj.coords[self.site_indices], self.traj.box_size)
        box = self.traj.box_size
        tree = cKDTree(site_coords, boxsize=box)
        local_lsi_values = []

        for site_position, site_coord in enumerate(site_coords):
            inside_positions = [
                int(position)
                for position in tree.query_ball_point(site_coord, self.cutoff)
                if int(position) != site_position
            ]
            n_inside = len(inside_positions)
            if n_inside < 1:
                continue

            # Standard LSI uses all neighbors with r < cutoff plus the nearest
            # neighbor with r >= cutoff.
            k = n_inside + 2
            inside = np.array([], dtype=float)
            outside = np.array([], dtype=float)

            while True:
                distances, neighbor_positions = tree.query(site_coord, k=k)
                distances = np.atleast_1d(distances)
                neighbor_positions = np.atleast_1d(neighbor_positions)

                mask = neighbor_positions != site_position
                distances = distances[mask]

                finite = np.isfinite(distances) & (distances > 1e-12)
                distances = distances[finite]

                inside = distances[distances < self.cutoff]
                outside = distances[distances >= self.cutoff]

                if inside.size >= 1 and outside.size >= 1:
                    break

                if distances.size >= len(site_coords) - 1:
                    inside = np.array([], dtype=float)
                    outside = np.array([], dtype=float)
                    break

                k = min(k + 4, len(site_coords))

            if inside.size < 1 or outside.size < 1:
                continue

            distances_lsi = np.concatenate([inside, outside[:1]])
            shell_gaps = distances_lsi[1:] - distances_lsi[:-1]
            if shell_gaps.size == 0:
                continue

            mean_gap = float(np.mean(shell_gaps))
            lsi_value = float(np.mean((shell_gaps - mean_gap) ** 2))
            local_lsi_values.append(lsi_value)

        if not local_lsi_values:
            return

        values = np.asarray(local_lsi_values, dtype=float)
        self.local_hist.add(values)

        if self.output_frame_mean:
            row = [
                self.frame_idx + 1,
                float(np.mean(values)),
                float(np.std(values)),
                int(values.size),
            ]
            if self.optional_threshold is not None:
                row.extend(
                    [
                        float(np.mean(values <= self.optional_threshold)),
                        float(np.mean(values > self.optional_threshold)),
                    ]
                )
            self.frame_rows.append(tuple(row))

    def postprocess(self):
        filename_part = format_selection(self.site_labels, self.compound_type.formula)
        has_data = self.local_hist.counts.sum() > 0

        if not has_data:
            console.warn("No local LSI values were accumulated.")
            return

        self.local_hist.normalize(field="count", method="total", total=100)
        local_filename = build_output_filename("lsi_local", [filename_part])
        write_histogram_1d(local_filename, self.local_hist, headers=["LSI/Angstrom^2", "P(LSI)"])
        console.success(f"Saved local LSI distribution to {local_filename}")

        if self.output_frame_mean:
            headers = ["frame", "mean_LSI", "std_LSI", "count"]
            if self.optional_threshold is not None:
                headers.extend(["fraction_below_threshold", "fraction_above_threshold"])

            global_filename = build_output_filename("lsi_global", [filename_part])
            write_table(global_filename, headers=headers, data=self.frame_rows)
            console.success(f"Saved per-frame LSI statistics to {global_filename}")
