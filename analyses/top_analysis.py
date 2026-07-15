from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analyses.common.base_analysis import BaseAnalysis
from analyses.common.histogram import HistogramND
from analyses.common.pair_selectors import PairSelector, PairSelectorSpec, pair_selector_schema
from framework.analysis_params import CompoundParam, IntParam
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, format_selection_group, write_histogram_1d


@dataclass(frozen=True)
class TetrahedralOrderConfig:
    """Configuration for tetrahedral orientational/translational order analysis."""

    ref_compound_index: int
    selector: PairSelectorSpec
    bin_count_q: int = 100
    bin_count_s: int = 10000

    def __post_init__(self):
        if self.ref_compound_index < 0:
            raise ValueError("ref_compound_index must be >= 0.")
        if self.bin_count_q < 1:
            raise ValueError("bin_count_q must be >= 1.")
        if self.bin_count_s < 1:
            raise ValueError("bin_count_s must be >= 1.")


class TetrahedralOrderAnalysis(BaseAnalysis):
    """Tetrahedral orientational/translational order analysis."""

    CONFIG_CLASS = TetrahedralOrderConfig
    CONFIG_SCHEMA = [
        CompoundParam(name="ref_compound_index", role="reference"),
        pair_selector_schema(
            name="selector",
            label="the tetrahedral neighbour definition",
            ref_compound_field="ref_compound_index",
            default_use_distance=False,
            default_use_rank=True,
            default_min_rank=1,
            default_max_rank=4,
        ),
        IntParam(
            name="bin_count_q",
            prompt="Enter the number of bins for angular tetrahedral order distribution q: ",
            default=100,
            minval=1,
        ),
        IntParam(
            name="bin_count_s",
            prompt="Enter the number of bins for radial tetrahedral order distribution S: ",
            default=10000,
            minval=1,
        ),
    ]

    def configure(self, config: TetrahedralOrderConfig):
        self.bind_config(config, exclude=("selector",))
        self.selector_spec = config.selector
        (self.ref_type, self.ref_key), = self.resolve_compound_types([self.ref_compound_index])
        self.tetra_selector = PairSelector(self, self.ref_type, self.ref_key, self.selector_spec)
        self.rebuild_runtime_state()

        if self.ref_indices.size == 0:
            raise ValueError("No reference atoms matched the given labels in the initial frame.")
        if self.obs_indices.size < 4:
            raise ValueError("Observed selections must provide at least four atoms in the initial frame.")

        self.hist_q = HistogramND([np.linspace(0.0, 1.0, self.bin_count_q + 1)], mode="linear")
        self.hist_s = HistogramND([np.linspace(0.0, 1.0, self.bin_count_s + 1)], mode="linear")
        self.mark_configured()

    def rebuild_runtime_state(self):
        self.tetra_selector.rebuild_runtime_state()
        self.ref_type = self.tetra_selector.ref_type
        self.ref_indices = self.tetra_selector.ref_indices
        self.obs_indices = self.tetra_selector.obs_indices

    def post_compound_update(self):
        if not self.tetra_selector.reattach_and_rebuild():
            return False

        self.ref_type = self.tetra_selector.ref_type
        self.ref_indices = self.tetra_selector.ref_indices
        self.obs_indices = self.tetra_selector.obs_indices
        return self.ref_indices.size > 0 and self.obs_indices.size >= 4

    def process_frame(self):
        if self.ref_indices.size == 0 or self.obs_indices.size < 4:
            return

        q_values = []
        s_values = []
        _, deltas_by_ref, distances_by_ref = self.tetra_selector.select_frame(include_deltas=True)

        for deltas, distances in zip(deltas_by_ref, distances_by_ref):
            if len(distances) < 4:
                continue

            four_deltas = deltas[:4]
            four_distances = distances[:4]
            unit_vectors = four_deltas / np.linalg.norm(four_deltas, axis=1)[:, None]
            cosines = []
            for first in range(3):
                for second in range(first + 1, 4):
                    cosines.append(float(np.dot(unit_vectors[first], unit_vectors[second])))

            q_value = 1.0 - (3.0 / 8.0) * sum((cosine + 1.0 / 3.0) ** 2 for cosine in cosines)
            q_values.append(q_value)

            distance_mean = float(np.mean(four_distances))
            if distance_mean > 1e-8:
                s_value = 1.0 - (1.0 / 3.0) * np.sum((four_distances - distance_mean) ** 2) / (4.0 * distance_mean**2)
                s_values.append(float(s_value))

        if q_values:
            self.hist_q.add(np.array(q_values))
        if s_values:
            self.hist_s.add(np.array(s_values))

    def postprocess(self):
        if self.hist_q.counts.sum() <= 0:
            console.warn("No tetrahedral order values were accumulated.")
            return

        filename_parts = [
            format_selection(self.selector_spec.ref_labels, self.ref_type.formula),
            format_selection_group(self.tetra_selector.observed_selection_entries),
        ]
        self.hist_q.normalize(field="count", method="total", total=100)
        q_filename = build_output_filename("top_q", filename_parts)
        write_histogram_1d(q_filename, self.hist_q, headers=["q", "P(q)"])
        console.success(f"Saved tetrahedral orientational order results to {q_filename}")

        self.hist_s.normalize(field="count", method="total", total=100)
        s_filename = build_output_filename("top_s", filename_parts)
        write_histogram_1d(s_filename, self.hist_s, headers=["S", "P(S)"])
        console.success(f"Saved tetrahedral translational order results to {s_filename}")
