from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from framework.analysis_params import AtomLabelsParam, BoolParam, CompoundParam, FloatParam, ForEach
from analyses.common.base_analysis import BaseAnalysis
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, format_selection_group, write_table


@dataclass(frozen=True)
class NeighborCountConfig:
    """Configuration for neighbour-count probability analysis."""

    ref_compound_index: int
    ref_labels: list[str]
    obs_compound_indices: list[int]
    obs_labels_per_compound: dict[int, list[str]]
    exclude_same_molecule: bool = True
    r_cut: float = 3.5

    def __post_init__(self):
        if self.ref_compound_index < 0:
            raise ValueError("ref_compound_index must be >= 0.")
        if not self.ref_labels:
            raise ValueError("ref_labels must not be empty.")
        if not self.obs_compound_indices:
            raise ValueError("obs_compound_indices must not be empty.")
        if any(idx < 0 for idx in self.obs_compound_indices):
            raise ValueError("obs_compound_indices must contain only values >= 0.")
        if self.r_cut <= 0:
            raise ValueError("r_cut must be positive.")
        missing = [idx for idx in self.obs_compound_indices if idx not in self.obs_labels_per_compound]
        if missing:
            raise ValueError(f"obs_labels_per_compound is missing labels for observed compound indices: {missing}")
        for idx, labels in self.obs_labels_per_compound.items():
            if idx < 0:
                raise ValueError("obs_labels_per_compound keys must be >= 0.")
            if not labels:
                raise ValueError(f"obs_labels_per_compound[{idx}] must not be empty.")


class NeighborCountAnalysis(BaseAnalysis):
    """Neighbour-count probability P(n) analysis."""

    CONFIG_CLASS = NeighborCountConfig
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
            name="exclude_same_molecule",
            prompt="Exclude observed atoms that belong to the same molecule as the reference atom?",
            default=True,
        ),
        FloatParam(
            name="r_cut",
            prompt="Neighbour cutoff distance Angstrom: ",
            default=3.5,
            minval=0.1,
        ),
    ]

    def configure(self, config: NeighborCountConfig):
        self.bind_config(config)
        (self.ref_type, self.ref_key), = self.resolve_compound_types([self.ref_compound_index])
        resolved_obs = self.resolve_compound_types(self.obs_compound_indices)
        self.obs_types = [compound_type for compound_type, _ in resolved_obs]
        self.obs_keys = [key for _, key in resolved_obs]
        topology_frame = self.traj.topology_frame

        self.obs_labels_per_compound = {
            key: list(self.obs_labels_per_compound[idx])
            for idx, key in zip(self.obs_compound_indices, self.obs_keys)
        }
        self.ref_selection = topology_frame.resolve_selection(self.ref_type, self.ref_labels)
        self.obs_selections_by_key = {
            key: topology_frame.resolve_selection(compound_type, self.obs_labels_per_compound[key])
            for compound_type, key in zip(self.obs_types, self.obs_keys)
        }
        self.observed_selection_entries = [
            (self.obs_labels_per_compound[key], compound_type.formula)
            for compound_type, key in zip(self.obs_types, self.obs_keys)
        ]
        self.exclude_same_molecule = config.exclude_same_molecule
        self.r_cut = config.r_cut

        self.rebuild_runtime_state()
        if self.ref_indices.size == 0 or self.obs_indices.size == 0:
            raise ValueError("No atoms matched the given labels in the initial frame.")

        self.n_hist = Counter()
        self.total_ref_atoms = 0
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

        self.rebuild_runtime_state()
        self.observed_selection_entries = [
            (self.obs_labels_per_compound[key], compound_type.formula)
            for compound_type, key in zip(self.obs_types, self.obs_keys)
        ]
        if self.ref_indices.size == 0 or self.obs_indices.size == 0:
            return False
        return True

    def process_frame(self):
        if self.ref_indices.size == 0 or self.obs_indices.size == 0:
            return

        coords = self.traj.coords
        obs_coords = coords[self.obs_indices]
        tree = cKDTree(obs_coords, boxsize=self.traj.box_size)
        ref_coords = coords[self.ref_indices]
        neighbours = tree.query_ball_point(ref_coords, self.r_cut)
        obs_global = self.obs_indices
        atom_to_type_id = self.traj.topology_frame.atom_to_type_id
        atom_to_molecule_index = self.traj.topology_frame.atom_to_molecule_index

        self.total_ref_atoms += int(self.ref_indices.size)
        for ref_idx, nb_list in zip(self.ref_indices, neighbours):
            count = 0
            ref_type_id = int(atom_to_type_id[ref_idx])
            ref_molecule_index = int(atom_to_molecule_index[ref_idx])

            for nb in nb_list:
                obs_idx = obs_global[nb]
                if obs_idx == ref_idx:
                    continue
                if self.exclude_same_molecule:
                    if (
                        int(atom_to_type_id[obs_idx]) == ref_type_id
                        and int(atom_to_molecule_index[obs_idx]) == ref_molecule_index
                    ):
                        continue
                count += 1

            self.n_hist[count] += 1

    def postprocess(self):
        if self.total_ref_atoms == 0:
            console.warn("No reference atoms found - nothing to write.")
            return

        max_n = max(self.n_hist) if self.n_hist else 0
        probs = {n: self.n_hist[n] / self.total_ref_atoms for n in range(max_n + 1)}

        fname = build_output_filename(
            "ncount",
            [
                format_selection(self.ref_labels, self.ref_type.formula),
                format_selection_group(self.observed_selection_entries),
            ],
        )
        write_table(
            fname,
            headers=["n", "P(n)"],
            data=[[n, probs.get(n, 0.0)] for n in range(max_n + 1)],
        )

        console.success(f"Saved neighbour-count results to {fname}")
