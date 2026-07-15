from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:
    from scipy.special import sph_harm as _sph_harm
except ImportError:
    from scipy.special import sph_harm_y as _sph_harm_y

    def _evaluate_sph_harm(m: int, phi: np.ndarray, theta: np.ndarray):
        return _sph_harm_y(6, int(m), theta, phi)

else:
    def _evaluate_sph_harm(m: int, phi: np.ndarray, theta: np.ndarray):
        return _sph_harm(int(m), 6, phi, theta)

from analyses.common.base_analysis import BaseAnalysis
from analyses.common.histogram import HistogramND
from analyses.common.pair_selectors import PairSelector, PairSelectorSpec, pair_selector_schema
from framework.analysis_params import BoolParam, CompoundParam, IntParam
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, write_histogram_1d, write_table


@dataclass(frozen=True)
class Q6Config:
    """Configuration for Steinhardt q6/Q6 analysis."""

    compound_index: int
    selector: PairSelectorSpec
    ignore_if_fewer_within_cutoff: bool = False
    bin_count_local: int = 100

    def __post_init__(self):
        if self.compound_index < 0:
            raise ValueError("compound_index must be >= 0.")
        if self.ignore_if_fewer_within_cutoff and not (self.selector.uses_distance and self.selector.uses_rank):
            raise ValueError("ignore_if_fewer_within_cutoff requires both distance and nearest-neighbour conditions.")
        if self.bin_count_local < 1:
            raise ValueError("bin_count_local must be >= 1.")


class Q6Analysis(BaseAnalysis):
    """Steinhardt plain local q6, Lechner-Dellago qbar6, and global Q6 analysis."""

    CONFIG_CLASS = Q6Config
    CONFIG_SCHEMA = [
        CompoundParam(
            name="compound_index",
            role="reference",
            prompt="Choose the site compound (number): ",
        ),
        pair_selector_schema(
            name="selector",
            label="the q6 neighbour definition",
            ref_compound_field="compound_index",
            default_use_distance=False,
            default_use_rank=True,
            default_min_rank=1,
            default_max_rank=4,
        ),
        BoolParam(
            name="ignore_if_fewer_within_cutoff",
            prompt="When both cutoff and rank are active, ignore sites with fewer than the requested neighbours inside the cutoff?",
            default=False,
        ),
        IntParam(
            name="bin_count_local",
            prompt="Enter the number of bins for local q6 distribution: ",
            default=100,
            minval=1,
        ),
    ]

    def configure(self, config: Q6Config):
        self.bind_config(config, exclude=("selector",))
        self.selector_spec = config.selector
        (self.compound_type, self.compound_key), = self.resolve_compound_types([self.compound_index])
        self.site_selector = PairSelector(self, self.compound_type, self.compound_key, self.selector_spec)
        if len(self.site_selector.ref_selection.local_indices) != 1:
            raise ValueError("q6 requires exactly one selected site atom per molecule.")
        if not np.array_equal(np.sort(self.site_selector.ref_indices), np.sort(self.site_selector.obs_indices)):
            raise ValueError("q6 requires the observed neighbour pool to match the selected site atoms.")

        self.cutoff_selector = None
        if self.ignore_if_fewer_within_cutoff:
            self.cutoff_selector = PairSelector(
                self,
                self.compound_type,
                self.compound_key,
                PairSelectorSpec(
                    ref_labels=self.selector_spec.ref_labels,
                    observed_groups=self.selector_spec.observed_groups,
                    min_distance=self.selector_spec.min_distance,
                    max_distance=self.selector_spec.max_distance,
                ),
            )

        self.ms = np.arange(-6, 7)
        self._q6_prefactor = 4.0 * np.pi / 13.0
        local_edges = np.linspace(0.0, 1.0, self.bin_count_local + 1)
        self.q6_local_hist = HistogramND([local_edges], mode="linear")
        self.qbar6_local_hist = HistogramND([local_edges.copy()], mode="linear")
        self.global_q6_rows: list[tuple[int, float]] = []

        self.rebuild_runtime_state()
        if self.site_indices.size < 2:
            raise ValueError("q6 requires at least two molecular sites in the initial frame.")
        self.mark_configured()

    def rebuild_runtime_state(self):
        self.site_selector.rebuild_runtime_state()
        self.compound_type = self.site_selector.ref_type
        self.site_indices = self.site_selector.ref_indices
        if self.cutoff_selector is not None:
            self.cutoff_selector.rebuild_runtime_state()

    def post_compound_update(self):
        if not self.site_selector.reattach_and_rebuild():
            return False
        if self.cutoff_selector is not None and not self.cutoff_selector.reattach_and_rebuild():
            return False

        self.compound_type = self.site_selector.ref_type
        self.site_indices = self.site_selector.ref_indices
        return self.site_indices.size >= 2

    def process_frame(self):
        if self.site_indices.size < 2:
            return

        n_sites = len(self.site_indices)
        n_m = len(self.ms)
        neighbor_lists: list[list[int]] = [[] for _ in range(n_sites)]
        q6m_per_site = np.zeros((n_sites, n_m), dtype=np.complex128)
        valid_site = np.zeros(n_sites, dtype=bool)
        site_position_by_atom_id = {int(atom_id): position for position, atom_id in enumerate(self.site_indices)}

        global_q6m_sum = np.zeros(n_m, dtype=np.complex128)
        global_bond_count = 0
        cutoff_ids_by_ref = None
        if self.cutoff_selector is not None:
            cutoff_ids_by_ref, _, _ = self.cutoff_selector.select_frame()

        neighbor_atom_ids_by_ref, deltas_by_ref, distances_by_ref = self.site_selector.select_frame(include_deltas=True)

        for site_position in range(n_sites):
            if cutoff_ids_by_ref is not None and len(cutoff_ids_by_ref[site_position]) < self.selector_spec.max_rank:
                continue

            neighbor_atom_ids = neighbor_atom_ids_by_ref[site_position]
            if len(neighbor_atom_ids) == 0:
                continue
            deltas = deltas_by_ref[site_position]
            distances = distances_by_ref[site_position]

            neighbor_positions = np.array(
                [site_position_by_atom_id[int(atom_id)] for atom_id in neighbor_atom_ids],
                dtype=np.intp,
            )
            neighbor_lists[site_position] = neighbor_positions.tolist()
            valid_site[site_position] = True

            unit_vectors = deltas / distances[:, None]
            theta = np.arccos(np.clip(unit_vectors[:, 2], -1.0, 1.0))
            phi = np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0])
            y6m_values = np.array(
                [_evaluate_sph_harm(int(m), phi, theta) for m in self.ms],
                dtype=np.complex128,
            )
            q6m_per_site[site_position, :] = np.mean(y6m_values, axis=1)
            global_q6m_sum += np.sum(y6m_values, axis=1)
            global_bond_count += y6m_values.shape[1]

        local_q6_values = []
        for site_position in range(n_sites):
            if not valid_site[site_position]:
                continue
            q6_value = np.sqrt(self._q6_prefactor * np.sum(np.abs(q6m_per_site[site_position]) ** 2))
            local_q6_values.append(float(np.real(q6_value)))

        local_qbar6_values = []
        for site_position in range(n_sites):
            if not valid_site[site_position]:
                continue
            indices = [site_position] + [neighbor for neighbor in neighbor_lists[site_position] if valid_site[neighbor]]
            qbar6m = np.mean(q6m_per_site[indices, :], axis=0)
            qbar6_value = np.sqrt(self._q6_prefactor * np.sum(np.abs(qbar6m) ** 2))
            local_qbar6_values.append(float(np.real(qbar6_value)))

        if local_q6_values:
            self.q6_local_hist.add(np.array(local_q6_values, dtype=float))
        if local_qbar6_values:
            self.qbar6_local_hist.add(np.array(local_qbar6_values, dtype=float))
        if global_bond_count > 0:
            global_q6m = global_q6m_sum / global_bond_count
            global_q6 = np.sqrt(self._q6_prefactor * np.sum(np.abs(global_q6m) ** 2))
            self.global_q6_rows.append((self.frame_idx + 1, float(np.real(global_q6))))

    def postprocess(self):
        filename_part = format_selection(self.selector_spec.ref_labels, self.compound_type.formula)
        if self.q6_local_hist.counts.sum() <= 0:
            console.warn("No q6 values were accumulated.")
            return

        self.q6_local_hist.normalize(field="count", method="total", total=100)
        local_filename = build_output_filename("q6_local", [filename_part])
        write_histogram_1d(local_filename, self.q6_local_hist, headers=["q6", "P(q6)"])
        console.success(f"Saved local q6 distribution to {local_filename}")

        self.qbar6_local_hist.normalize(field="count", method="total", total=100)
        qbar6_filename = build_output_filename("qbar6_local", [filename_part])
        write_histogram_1d(qbar6_filename, self.qbar6_local_hist, headers=["qbar6", "P(qbar6)"])
        console.success(f"Saved local qbar6 distribution to {qbar6_filename}")

        global_filename = build_output_filename("q6_global", [filename_part])
        write_table(global_filename, headers=["frame", "Q6"], data=self.global_q6_rows)
        console.success(f"Saved global Q6 time series to {global_filename}")
