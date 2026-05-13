from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
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
from core.geometry import minimum_image
from framework.analysis_params import AtomLabelsParam, CompoundParam, FloatParam, IntParam
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, write_histogram_1d, write_table


@dataclass(frozen=True)
class Q6Config:
    """Configuration for Steinhardt q6/Q6 analysis."""

    compound_index: int
    site_labels: list[str]
    cutoff: float = 3.5
    bin_count_local: int = 100

    def __post_init__(self):
        if self.compound_index < 0:
            raise ValueError("compound_index must be >= 0.")
        if not self.site_labels:
            raise ValueError("site_labels must not be empty.")
        if self.cutoff <= 0:
            raise ValueError("cutoff must be > 0.")
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
        AtomLabelsParam(
            name="site_labels",
            role="reference",
            compound="compound_index",
            prompt="Which atom(s) define the molecular site? (comma-separated) ",
        ),
        FloatParam(
            name="cutoff",
            prompt="Neighbour cutoff distance Angstrom: ",
            default=3.5,
            minval=0.0,
        ),
        IntParam(
            name="bin_count_local",
            prompt="Enter the number of bins for local q6 distribution: ",
            default=100,
            minval=1,
        ),
    ]

    def configure(self, config: Q6Config):
        self.bind_config(config)
        (self.compound_type, self.compound_key), = self.resolve_compound_types([self.compound_index])
        self.site_selection = self.traj.topology_frame.resolve_selection(self.compound_type, self.site_labels)
        if len(self.site_selection.local_indices) != 1:
            raise ValueError("q6 requires exactly one selected site atom per molecule.")

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
        return self.site_indices.size >= 2

    def process_frame(self):
        if self.site_indices.size < 2:
            return

        site_coords = self.traj.coords[self.site_indices]
        box = self.traj.box_size
        tree = cKDTree(site_coords, boxsize=box)

        n_sites = len(site_coords)
        n_m = len(self.ms)
        neighbor_lists: list[list[int]] = [[] for _ in range(n_sites)]
        q6m_per_site = np.zeros((n_sites, n_m), dtype=np.complex128)
        valid_site = np.zeros(n_sites, dtype=bool)

        global_q6m_sum = np.zeros(len(self.ms), dtype=np.complex128)
        global_bond_count = 0

        # Pass 1: build neighbor lists, store per-site q6m(i), and preserve
        # the existing global Q6 accumulation behavior.
        for site_position, site_coord in enumerate(site_coords):
            neighbor_positions = [
                int(position)
                for position in tree.query_ball_point(site_coord, self.cutoff)
                if int(position) != site_position
            ]
            if not neighbor_positions:
                continue

            deltas = minimum_image(site_coords[neighbor_positions] - site_coord, box)
            distances = np.linalg.norm(deltas, axis=1)
            valid = distances > 1e-12
            if not np.any(valid):
                continue

            neighbor_positions = np.asarray(neighbor_positions, dtype=np.intp)[valid]
            deltas = deltas[valid]
            distances = distances[valid]
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

        # Pass 2: plain local Steinhardt q6(i).
        local_q6_values = []
        for site_position in range(n_sites):
            if not valid_site[site_position]:
                continue
            q6_value = np.sqrt(self._q6_prefactor * np.sum(np.abs(q6m_per_site[site_position]) ** 2))
            local_q6_values.append(float(np.real(q6_value)))

        # Pass 3: Lechner-Dellago averaged local qbar6(i). This averages the
        # complex q6m coefficients of site i and its valid neighbors before
        # forming the rotationally invariant norm.
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
        filename_part = format_selection(self.site_labels, self.compound_type.formula)
        has_data = self.q6_local_hist.counts.sum() > 0

        if not has_data:
            console.warn("No q6 values were accumulated.")
            return

        else:
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
