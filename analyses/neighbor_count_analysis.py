import numpy as np
from collections import Counter
from scipy.spatial import cKDTree

from analyses.base_analysis import BaseAnalysis
from utils import label_matches, prompt_yn


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

    def setup(self):
        # --- Reference selection ---
        self.ref_idx, self.ref_comp = self.compound_selection("reference")
        self.ref_labels = self.atom_selection("reference", compound=self.ref_comp)
        self.ref_key = list(self.traj.compounds.keys())[self.ref_idx]

        # --- Observed selection: allow multiple compounds ---
        obs_selection = self.compound_selection("observed", multi=True)
        self.obs_idxs = [idx for idx, _ in obs_selection]
        self.obs_comps = [comp for _, comp in obs_selection]
        self.obs_keys = [list(self.traj.compounds.keys())[idx] for idx in self.obs_idxs]

        # Prompt for observed labels separately for each selected observed compound
        self.obs_labels_per_compound = {}
        for key, comp in zip(self.obs_keys, self.obs_comps):
            self.obs_labels_per_compound[key] = self.atom_selection("observed", compound=comp)

        self.exclude_same_molecule = prompt_yn(
            "Exclude observed atoms that belong to the same molecule as the reference atom?",
            True
        )

        self.r_cut = self._prompt_cutoff()

        # Build first-frame index lists and molecule maps
        self._update_indices()

        if not self.ref_indices or not self.obs_indices:
            raise ValueError("No atoms matched the given labels in the initial frame.")

        # Histogram and accumulators
        self.n_hist = Counter()      # n -> occurrences
        self.total_ref_atoms = 0     # total number of reference atoms counted over all processed frames

    def _prompt_cutoff(self):
        from utils import prompt_float
        return prompt_float("Neighbour cutoff distance Å: ", 3.5, minval=0.1)

    # ---------- index / molecule helpers ----------

    def _get_indices(self, comp, labels):
        """
        Collect global atom indices for a given compound and list of label patterns.
        Uses label_matches(...) for wildcard support.
        """
        return [
            idx for mol in comp.members
            for lab, idx in mol.label_to_global_id.items()
            if any(label_matches(user_lab, lab) for user_lab in labels)
        ]

    def _build_atom_to_mol_map(self, comp):
        """
        Return dict mapping global atom index -> parent molecule object
        for all atoms in the given compound.
        """
        atom_to_mol = {}
        for mol in comp.members:
            for idx in mol.label_to_global_id.values():
                atom_to_mol[idx] = mol
        return atom_to_mol

    def _update_indices(self):
        """
        Rebuild reference/observed atom index lists and parent-molecule maps
        from the currently attached compounds and label selections.
        """
        self.ref_indices = self._get_indices(self.ref_comp, self.ref_labels)

        self.obs_indices = []
        for key, comp in zip(self.obs_keys, self.obs_comps):
            labels = self.obs_labels_per_compound[key]
            self.obs_indices.extend(self._get_indices(comp, labels))

        # Parent-molecule lookup maps for same-molecule filtering
        self.ref_atom_to_mol = self._build_atom_to_mol_map(self.ref_comp)
        self.obs_atom_to_mol = {}
        for comp in self.obs_comps:
            self.obs_atom_to_mol.update(self._build_atom_to_mol_map(comp))

    # ---------- BaseAnalysis interface methods ----------

    def post_compound_update(self):
        """
        Called when self.update_compounds is True.

        Re-attach the selected reference compound and all selected observed
        compounds in the current frame, then rebuild atom index lists.

        If any selected compound type is missing, or no atoms match, skip frame.
        """
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
        """
        For each processed frame:
          - build KDTree on observed atoms,
          - count neighbours within r_cut for each reference atom,
          - optionally exclude same-molecule atoms,
          - update histogram and total_ref_atoms.
        """
        if not self.ref_indices or not self.obs_indices:
            return

        coords = self.traj.coords

        # KD-tree on all observed atoms from all selected observed compounds
        obs_coords = coords[self.obs_indices]
        tree = cKDTree(obs_coords, boxsize=self.traj.box_size)

        # Reference atom coordinates
        ref_coords = coords[self.ref_indices]

        # Neighbour lists for each reference atom
        neighbours = tree.query_ball_point(ref_coords, self.r_cut)

        obs_global = self.obs_indices

        self.total_ref_atoms += len(self.ref_indices)
        for ref_idx, nb_list in zip(self.ref_indices, neighbours):
            count = 0
            ref_mol = self.ref_atom_to_mol.get(ref_idx)

            for nb in nb_list:
                obs_idx = obs_global[nb]

                # Always exclude exact self-match
                if obs_idx == ref_idx:
                    continue

                # Optionally exclude atoms from the same molecule
                if self.exclude_same_molecule:
                    obs_mol = self.obs_atom_to_mol.get(obs_idx)
                    if obs_mol == ref_mol:
                        continue

                count += 1

            self.n_hist[count] += 1

    def postprocess(self):
        """
        Compute P(n) and write ncount.dat.
        """
        print()

        if self.total_ref_atoms == 0:
            print("No reference atoms found — nothing to write.")
            return

        max_n = max(self.n_hist) if self.n_hist else 0
        probs = {n: self.n_hist[n] / self.total_ref_atoms for n in range(max_n + 1)}

        fname = "ncount.dat"
        with open(fname, "w") as f:
            f.write(f"# P(n)   cutoff = {self.r_cut:.2f} Å\n")
            f.write("#  n   P(n)\n")
            for n in range(max_n + 1):
                f.write(f"{n:3d}  {probs.get(n, 0.0):.6f}\n")

        print(f"Neighbour-count distribution written to {fname}")

