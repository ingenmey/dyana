from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from scipy.spatial import cKDTree

from analysis_params import AtomLabelsParam, BoolParam, CompoundParam, FloatParam, ForEach
from analyses.base_analysis import BaseAnalysis
from analyses.selection import build_atom_to_molecule, collect_indices_for_compounds, collect_atom_indices


@dataclass(frozen=True)
class NeighborCountConfig:
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
    """
    Neighbour-count probability P(n) analysis.

    For each selected reference atom, counts how many observed atoms lie within
    a cutoff distance r_cut, and accumulates a histogram over all frames:
        n -> occurrences
    Then reports:
        P(n) = occurrences(n) / total_number_of_reference_atoms_seen

    Supports optional per-frame molecule recognition (update_compounds),
    in which case compound identity and matching atoms are re-evaluated in
    every frame. Frames where compounds / labels disappear are skipped.

    Optionally excludes observed atoms that belong to the same molecule as the
    reference atom.
    """

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
        self.config = config
        compounds = self.get_compounds()
        keys = list(self.traj.compounds.keys())

        try:
            self.ref_comp = compounds[config.ref_compound_index]
            self.ref_key = keys[config.ref_compound_index]
            self.obs_comps = [compounds[idx] for idx in config.obs_compound_indices]
            self.obs_keys = [keys[idx] for idx in config.obs_compound_indices]
        except IndexError as exc:
            raise ValueError("Neighbor-count compound index is out of range.") from exc

        self.ref_labels = list(config.ref_labels)
        self.obs_labels_per_compound = {
            keys[idx]: list(config.obs_labels_per_compound[idx])
            for idx in config.obs_compound_indices
        }
        self.exclude_same_molecule = config.exclude_same_molecule
        self.r_cut = config.r_cut

        self._update_indices()
        if not self.ref_indices or not self.obs_indices:
            raise ValueError("No atoms matched the given labels in the initial frame.")

        self.n_hist = Counter()
        self.total_ref_atoms = 0
        self.mark_configured()

    def _update_indices(self):
        self.ref_indices = collect_atom_indices(self.ref_comp, self.ref_labels)
        self.obs_indices = collect_indices_for_compounds(self.obs_comps, self.obs_labels_per_compound, self.obs_keys)
        self.ref_atom_to_mol = build_atom_to_molecule(self.ref_comp)
        self.obs_atom_to_mol = {}
        for comp in self.obs_comps:
            self.obs_atom_to_mol.update(build_atom_to_molecule(comp))

    def post_compound_update(self):
        try:
            self.ref_comp = self.traj.compounds[self.ref_key]
            self.obs_comps = [self.traj.compounds[k] for k in self.obs_keys]
        except KeyError:
            return False

        self._update_indices()
        if not self.ref_indices or not self.obs_indices:
            return False
        return True

    def process_frame(self):
        if not self.ref_indices or not self.obs_indices:
            return

        coords = self.traj.coords
        obs_coords = coords[self.obs_indices]
        tree = cKDTree(obs_coords, boxsize=self.traj.box_size)
        ref_coords = coords[self.ref_indices]
        neighbours = tree.query_ball_point(ref_coords, self.r_cut)
        obs_global = self.obs_indices

        self.total_ref_atoms += len(self.ref_indices)
        for ref_idx, nb_list in zip(self.ref_indices, neighbours):
            count = 0
            ref_mol = self.ref_atom_to_mol.get(ref_idx)

            for nb in nb_list:
                obs_idx = obs_global[nb]
                if obs_idx == ref_idx:
                    continue
                if self.exclude_same_molecule:
                    obs_mol = self.obs_atom_to_mol.get(obs_idx)
                    if obs_mol == ref_mol:
                        continue
                count += 1

            self.n_hist[count] += 1

    def postprocess(self):
        print()

        if self.total_ref_atoms == 0:
            print("No reference atoms found - nothing to write.")
            return

        max_n = max(self.n_hist) if self.n_hist else 0
        probs = {n: self.n_hist[n] / self.total_ref_atoms for n in range(max_n + 1)}

        fname = "ncount.dat"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"# P(n)   cutoff = {self.r_cut:.2f} Angstrom\n")
            f.write("#  n   P(n)\n")
            for n in range(max_n + 1):
                f.write(f"{n:3d}  {probs.get(n, 0.0):.6f}\n")

        print(f"Neighbour-count distribution written to {fname}")
