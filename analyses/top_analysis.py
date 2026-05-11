from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from framework.analysis_params import AtomLabelsParam, BoolParam, CompoundParam, FloatParam, ForEach, IntParam, When
from analyses.common.base_analysis import BaseAnalysis
from analyses.common.histogram import HistogramND
from core.geometry import minimum_image
from io_support.output_writer import build_output_filename, format_selection, format_selection_group, write_histogram_1d


@dataclass(frozen=True)
class TetrahedralOrderConfig:
    """Configuration for tetrahedral orientational/translational order analysis."""

    ref_compound_index: int
    ref_labels: list[str]
    obs_compound_indices: list[int]
    obs_labels_per_compound: dict[int, list[str]]
    use_cutoff: bool = False
    cutoff: float | None = None
    bin_count_q: int = 100
    bin_count_s: int = 10000

    def __post_init__(self):
        if self.ref_compound_index < 0:
            raise ValueError("ref_compound_index must be >= 0.")
        if not self.ref_labels:
            raise ValueError("ref_labels must not be empty.")
        if not self.obs_compound_indices:
            raise ValueError("obs_compound_indices must not be empty.")
        if any(idx < 0 for idx in self.obs_compound_indices):
            raise ValueError("obs_compound_indices must contain only values >= 0.")
        missing = [idx for idx in self.obs_compound_indices if idx not in self.obs_labels_per_compound]
        if missing:
            raise ValueError(f"obs_labels_per_compound is missing labels for observed compound indices: {missing}")
        for idx, labels in self.obs_labels_per_compound.items():
            if idx < 0:
                raise ValueError("obs_labels_per_compound keys must be >= 0.")
            if not labels:
                raise ValueError(f"obs_labels_per_compound[{idx}] must not be empty.")
        if self.use_cutoff:
            if self.cutoff is None:
                raise ValueError("cutoff must be provided when use_cutoff is True.")
            if self.cutoff < 0:
                raise ValueError("cutoff must be >= 0.")
        elif self.cutoff is not None and self.cutoff < 0:
            raise ValueError("cutoff must be >= 0 or None.")
        if self.bin_count_q < 1:
            raise ValueError("bin_count_q must be >= 1.")
        if self.bin_count_s < 1:
            raise ValueError("bin_count_s must be >= 1.")


class TetrahedralOrderAnalysis(BaseAnalysis):
    """Tetrahedral orientational/translational order analysis."""

    CONFIG_CLASS = TetrahedralOrderConfig
    CONFIG_SCHEMA = [
        CompoundParam(name="ref_compound_index", role="reference"),
        AtomLabelsParam(name="ref_labels", role="reference", compound="ref_compound_index"),
        CompoundParam(name="obs_compound_indices", role="observed", multi=True),
        ForEach(
            source="obs_compound_indices",
            item_name="obs_compound_index",
            steps=[
                AtomLabelsParam(
                    name="obs_labels",
                    role="observed",
                    compound="obs_compound_index",
                ),
            ],
            collect_as="obs_labels_per_compound",
            collect_mode="dict",
        ),
        BoolParam(
            name="use_cutoff",
            prompt="Use a maximum distance cutoff for neighbor search?",
            default=False,
        ),
        When(
            source="use_cutoff",
            value=True,
            steps=[
                FloatParam(
                    name="cutoff",
                    prompt="Enter the maximum cutoff distance (Angstrom): ",
                    default=5.0,
                    minval=0.0,
                ),
            ],
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
        self.bind_config(config)
        (self.ref_type, self.ref_key), = self.resolve_compound_types([self.ref_compound_index])
        resolved_obs = self.resolve_compound_types(self.obs_compound_indices)
        self.obs_types = [compound_type for compound_type, _ in resolved_obs]
        self.obs_keys = [key for _, key in resolved_obs]
        self.cutoff = self.cutoff if self.use_cutoff else None

        topology_frame = self.traj.topology_frame
        self.ref_selection = topology_frame.resolve_selection(self.ref_type, self.ref_labels)
        self.obs_labels_per_compound = {
            key: list(self.obs_labels_per_compound[idx])
            for idx, key in zip(self.obs_compound_indices, self.obs_keys)
        }
        self.obs_selections_by_key = {
            key: topology_frame.resolve_selection(compound_type, self.obs_labels_per_compound[key])
            for compound_type, key in zip(self.obs_types, self.obs_keys)
        }
        self.observed_selection_entries = [
            (self.obs_labels_per_compound[key], compound_type.formula)
            for compound_type, key in zip(self.obs_types, self.obs_keys)
        ]

        self.rebuild_runtime_state()
        if self.ref_indices.size == 0:
            raise ValueError("No reference atoms matched the given labels in the initial frame.")
        if self.obs_indices.size < 4:
            raise ValueError("Observed selections must provide at least four atoms in the initial frame.")

        self.hist_q = HistogramND([np.linspace(0.0, 1.0, self.bin_count_q + 1)], mode="linear")
        self.hist_s = HistogramND([np.linspace(0.0, 1.0, self.bin_count_s + 1)], mode="linear")
        self.mark_configured()

    def rebuild_runtime_state(self):
        topology_frame = self.traj.topology_frame
        self.ref_indices = topology_frame.get_atom_ids_for_local_indices(
            self.ref_type,
            self.ref_selection.local_indices,
        )
        obs_parts = [
            topology_frame.get_atom_ids_for_local_indices(
                compound_type,
                self.obs_selections_by_key[key].local_indices,
            )
            for compound_type, key in zip(self.obs_types, self.obs_keys)
        ]
        self.obs_indices = (
            np.concatenate([part for part in obs_parts if part.size > 0])
            if any(part.size > 0 for part in obs_parts)
            else np.empty(0, dtype=np.int32)
        )

    def post_compound_update(self):
        topology_frame = self.traj.topology_frame
        if not topology_frame.has_compound_type_key(self.ref_key):
            return False
        if any(not topology_frame.has_compound_type_key(key) for key in self.obs_keys):
            return False

        self.ref_type = topology_frame.get_compound_type_by_key(self.ref_key)
        self.obs_types = [topology_frame.get_compound_type_by_key(key) for key in self.obs_keys]
        self.observed_selection_entries = [
            (self.obs_labels_per_compound[key], compound_type.formula)
            for compound_type, key in zip(self.obs_types, self.obs_keys)
        ]
        self.rebuild_runtime_state()
        if self.ref_indices.size == 0 or self.obs_indices.size < 4:
            return False
        return True

    def process_frame(self):
        if self.ref_indices.size == 0 or self.obs_indices.size < 4:
            return

        coords = self.traj.coords
        box = self.traj.box_size
        obs_coords = coords[self.obs_indices]
        tree = cKDTree(obs_coords, boxsize=box)
        q_values = []
        s_values = []

        for ref_idx in self.ref_indices:
            ref_coord = coords[ref_idx]

            if self.cutoff is not None:
                candidate_positions = [
                    int(position)
                    for position in tree.query_ball_point(ref_coord, self.cutoff)
                    if self.obs_indices[int(position)] != ref_idx
                ]
                if len(candidate_positions) < 4:
                    continue

                candidate_deltas = minimum_image(obs_coords[candidate_positions] - ref_coord, box)
                candidate_distances = np.linalg.norm(candidate_deltas, axis=1)
                nearest = np.argsort(candidate_distances)[:4]
                four_deltas = candidate_deltas[nearest]
                four_distances = candidate_distances[nearest]
            else:
                max_neighbors = min(5, len(obs_coords))
                distances, positions = tree.query(ref_coord, k=max_neighbors)
                distances = np.atleast_1d(distances)
                positions = np.atleast_1d(positions)
                filtered = [
                    (float(distance), int(position))
                    for distance, position in zip(distances, positions)
                    if np.isfinite(distance) and self.obs_indices[int(position)] != ref_idx
                ]
                if len(filtered) < 4:
                    continue

                filtered = filtered[:4]
                selected_positions = np.array([position for _, position in filtered], dtype=np.intp)
                four_deltas = minimum_image(obs_coords[selected_positions] - ref_coord, box)
                four_distances = np.array([distance for distance, _ in filtered], dtype=float)

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
        comment_lines = [f"cutoff = {self.cutoff:.6f} Angstrom"] if self.cutoff is not None else None
        filename_parts = [
            format_selection(self.ref_labels, self.ref_type.formula),
            format_selection_group(self.observed_selection_entries),
        ]

        if self.hist_q.counts.sum() > 0:
            self.hist_q.normalize(field="count", method="total", total=100)
            q_filename = build_output_filename("top_q", filename_parts)
            write_histogram_1d(q_filename, self.hist_q, headers=["q", "P(q)"], comment_lines=comment_lines)
            print(f"Saved tetrahedral orientational order results to {q_filename}")
        else:
            print("No valid tetrahedral orientational order values were accumulated.")

        if self.hist_s.counts.sum() > 0:
            self.hist_s.normalize(field="count", method="total", total=100)
            s_filename = build_output_filename("top_s", filename_parts)
            write_histogram_1d(s_filename, self.hist_s, headers=["S", "P(S)"], comment_lines=comment_lines)
            print(f"Saved tetrahedral translational order results to {s_filename}")
        else:
            print("No valid tetrahedral translational order values were accumulated.")
