# analyses/neighbor_count.py

import numpy as np
from collections import Counter
from scipy.spatial import cKDTree

from analyses.base_analysis import BaseAnalysis
from utils import label_matches


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
    """

    def setup(self):
        # --- User setup: compound & atom selection ---
        self.ref_idx, self.ref_comp = self.compound_selection("reference")
        self.obs_idx, self.obs_comp = self.compound_selection("observed")

        self.ref_labels = self.atom_selection("reference", compound=self.ref_comp)
        self.obs_labels = self.atom_selection("observed", compound=self.obs_comp)

        self.r_cut = self._prompt_cutoff()

        # Keep stable keys so we can refind the same compound *type* in later frames
        keys = list(self.traj.compounds.keys())
        self.ref_key = keys[self.ref_idx]
        self.obs_key = keys[self.obs_idx]

        # Initial index lists for the first frame
        self._update_indices_initial()

        if not self.ref_indices or not self.obs_indices:
            raise ValueError("No atoms matched the given labels in the initial frame.")

        # Histogram and accumulators
        self.n_hist = Counter()      # n -> occurrences
        self.total_ref_atoms = 0     # total number of reference atoms counted over all processed frames

    def _prompt_cutoff(self):
        # small helper to keep setup() clean
        from utils import prompt_float
        return prompt_float("Neighbour cutoff distance Å: ", 3.5, minval=0.1)

    # ---------- index handling helpers ----------

    def _get_indices(self, comp, labels):
        """
        Collect global atom indices for a given compound and list of label patterns.
        Uses label_matches(...) for wildcard support.
        """
        return [
            idx for mol in comp.members
            for lab, idx in mol.label_to_global_id.items()
            if any(label_matches(lab_in, lab) for lab_in in labels)
        ]

    def _update_indices_initial(self):
        """
        Build ref/obs indices from the compounds selected in the first frame.
        """
        self.ref_indices = self._get_indices(self.ref_comp, self.ref_labels)
        self.obs_indices = self._get_indices(self.obs_comp, self.obs_labels)

    # ---------- BaseAnalysis interface methods ----------

    def post_compound_update(self):
        """
        Called when self.update_compounds is True (per BaseAnalysis.run()).

        We re-attach the selected compound *types* (via self.ref_key / self.obs_key)
        to the current frame's traj.compounds, and rebuild atom index lists.

        If the relevant compounds or matching atoms are missing in this frame,
        return False to skip the frame (no neighbour counts or histogram update).
        """
        try:
            self.ref_comp = self.traj.compounds[self.ref_key]
            self.obs_comp = self.traj.compounds[self.obs_key]
        except KeyError:
            # Selected compound type not present in this frame
            return False

        self.ref_indices = self._get_indices(self.ref_comp, self.ref_labels)
        self.obs_indices = self._get_indices(self.obs_comp, self.obs_labels)

        if not self.ref_indices or not self.obs_indices:
            # No matching atoms in this frame → skip
            return False

        return True

    def process_frame(self):
        """
        For each processed frame:
          - build KDTree on observed atoms,
          - count neighbours within r_cut for each reference atom,
          - update histogram and total_ref_atoms.
        """
        # ref_indices / obs_indices are:
        #   - static if update_compounds is False
        #   - updated in post_compound_update() if update_compounds is True
        if not self.ref_indices or not self.obs_indices:
            # Should only happen if update_compounds=False and nothing matched,
            # which would've already raised in setup(). Safe-guard anyway.
            return

        coords = self.traj.coords

        # KD-tree on observed atoms
        obs_coords = coords[self.obs_indices]
        tree = cKDTree(obs_coords, boxsize=self.traj.box_size)

        # Reference atom coordinates
        ref_coords = coords[self.ref_indices]

        # neighbour lists for each reference atom
        neighbours = tree.query_ball_point(ref_coords, self.r_cut)

        # Update histogram and count of reference atoms
        self.total_ref_atoms += len(self.ref_indices)
        for nb_list in neighbours:
            self.n_hist[len(nb_list)] += 1

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

