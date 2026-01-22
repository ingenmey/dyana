# analyses/adf_threebody_analysis.py

import numpy as np
from scipy.spatial import cKDTree
from analyses.base_analysis import BaseAnalysis
from utils import prompt, prompt_int, prompt_float, prompt_yn, label_matches
from analyses.histogram import HistogramND

class ADFThreeBody(BaseAnalysis):
    def setup(self):
        # --- User Input for Compounds and Atom Labels ---
        self.center_idx, self.center_comp = self.compound_selection("center")
        self.neigh1_idx, self.neigh1_comp = self.compound_selection("neighbor 1")
        self.neigh2_idx, self.neigh2_comp = self.compound_selection("neighbor 2")

        self.center_key = list(self.traj.compounds.keys())[self.center_idx]
        self.neigh1_key = list(self.traj.compounds.keys())[self.neigh1_idx]
        self.neigh2_key = list(self.traj.compounds.keys())[self.neigh2_idx]

        self.center_label = self.atom_selection("center", self.center_comp)
        self.neigh1_label = self.atom_selection("neighbor 1", self.neigh1_comp)
        self.neigh2_label = self.atom_selection("neighbor 2", self.neigh2_comp)

        self.cutoffs = np.zeros((2, 2))
        self.cutoffs[0, 0] = prompt_float("Minimum distance for center - neighbor1 (Å): ", 0.0)
        self.cutoffs[0, 1] = prompt_float("Maximum distance for center - neighbor1 (Å): ", 3.5)
        self.cutoffs[1, 0] = prompt_float("Minimum distance for center - neighbor2 (Å): ", 0.0)
        self.cutoffs[1, 1] = prompt_float("Maximum distance for center - neighbor2 (Å): ", 3.5)

        self.enforce_threebody = prompt_yn("Enforce that neighbor1 and neighbor2 come from different molecules?", True)
        self.bin_count = prompt_int("Number of bins for ADF histogram: ", 180, minval=1)

        self.n_center = len(self.center_comp.members)
        self.n_neigh1 = len(self.neigh1_comp.members)
        self.n_neigh2 = len(self.neigh2_comp.members)
        self.angle_edges = np.linspace(0, 180, self.bin_count + 1)
        self.hist = HistogramND([self.angle_edges], "linear")

        # Precompute atom indices
        self.update_selectors()

    def update_selectors(self):
        self.center_ids = self._get_indices(self.center_comp, self.center_label)
        self.neigh1_ids = self._get_indices(self.neigh1_comp, self.neigh1_label)
        self.neigh2_ids = self._get_indices(self.neigh2_comp, self.neigh2_label)

    def _get_indices(self, compound, labels):
        # Accepts labels as list or string
        if isinstance(labels, str):
            labels = [labels]
        return [
            idx for mol in compound.members
            for label, idx in mol.label_to_global_id.items()
            if any(label_matches(lab, label) for lab in labels)
        ]

    def post_compound_update(self):
        try:
            self.center_comp = self.traj.compounds[self.center_key]
            self.neigh1_comp = self.traj.compounds[self.neigh1_key]
            self.neigh2_comp = self.traj.compounds[self.neigh2_key]
        except KeyError:
            return False

        self.update_selectors()
        if not (len(self.center_ids) > 0 and len(self.neigh1_ids) > 0 and len(self.neigh2_ids) > 0):
            return False

        self.n_center = (self.n_center * self.processed_frames + len(self.center_comp.members))/(self.processed_frames + 1)
        self.n_neigh1 = (self.n_neigh1 * self.processed_frames + len(self.neigh1_comp.members))/(self.processed_frames + 1)
        self.n_neigh2 = (self.n_neigh2 * self.processed_frames + len(self.neigh2_comp.members))/(self.processed_frames + 1)
        return True

    def process_frame(self):
        vec1, vec2 = self.compute_threebody_vectors_fast(
            self.traj, self.center_ids, self.neigh1_ids, self.neigh2_ids,
            self.cutoffs, self.traj.box_size, self.enforce_threebody
        )

        if vec1.size and vec2.size:
            angles = self.calculate_angles(vec1, vec2)
            self.hist.add(angles)

    def postprocess(self):
        # Normalize by sin(theta)
        bin_centers = 0.5 * (self.angle_edges[1:] + self.angle_edges[:-1])
        radians = np.deg2rad(bin_centers)
        sin_weights = 1.0 / np.sin(radians)
        self.hist.counts *= sin_weights

        # Normalize per number of triplets/frames (optional: can add total normalization)
        if self.processed_frames > 0 and len(self.center_ids) > 0:
            self.hist.counts /= (self.processed_frames * self.n_center * self.n_neigh1 * self.n_neigh2)

        self.hist.normalize(method="total", total=self.bin_count * 100)
        self.hist.save_txt("adf_threebody.dat", ["Angle (deg)", "ADF"])

        print("Three-body ADF results saved to adf_threebody.dat / adf_threebody.npy")

    @staticmethod
    def compute_threebody_vectors_fast(traj, center_ids, neighbor1_ids, neighbor2_ids, cutoffs, box_size, enforce_threebody):
        coords = traj.coords
        box_size = np.asarray(box_size)
        center_coords = coords[center_ids]
        neighbor1_coords = coords[neighbor1_ids]
        neighbor2_coords = coords[neighbor2_ids]
        kdtree_n1 = cKDTree(neighbor1_coords, boxsize=box_size)
        kdtree_n2 = cKDTree(neighbor2_coords, boxsize=box_size)

        vectors1 = []
        vectors2 = []

        for idx_center, center in enumerate(center_coords):
            # Neighbor 1 search
            neighbor1_candidates = kdtree_n1.query_ball_point(center, cutoffs[0,1])
            neighbor1_indices = []
            for idx in neighbor1_candidates:
                disp = neighbor1_coords[idx] - center
                disp -= np.round(disp / box_size) * box_size
                dist = np.linalg.norm(disp)
                if dist >= cutoffs[0,0]:
                    neighbor1_indices.append(idx)

            # Neighbor 2 search
            neighbor2_candidates = kdtree_n2.query_ball_point(center, cutoffs[1,1])
            neighbor2_indices = []
            for idx in neighbor2_candidates:
                disp = neighbor2_coords[idx] - center
                disp -= np.round(disp / box_size) * box_size
                dist = np.linalg.norm(disp)
                if dist >= cutoffs[1,0]:
                    neighbor2_indices.append(idx)

            # Loop over all valid neighbor pairs
            for n1_idx in neighbor1_indices:
                for n2_idx in neighbor2_indices:
                    if neighbor1_ids[n1_idx] == neighbor2_ids[n2_idx]:
                        continue
                    idx1 = neighbor1_ids[n1_idx]
                    idx2 = neighbor2_ids[n2_idx]

                    if enforce_threebody:
                        mol1 = traj.atoms[idx1].parent_molecule
                        mol2 = traj.atoms[idx2].parent_molecule
                        if mol1 == mol2:
                            continue

                    neighbor1 = neighbor1_coords[n1_idx]
                    neighbor2 = neighbor2_coords[n2_idx]

                    vec1 = neighbor1 - center
                    vec1 -= np.round(vec1 / box_size) * box_size

                    vec2 = neighbor2 - center
                    vec2 -= np.round(vec2 / box_size) * box_size

                    vectors1.append(vec1)
                    vectors2.append(vec2)

        if vectors1 and vectors2:
            return np.array(vectors1), np.array(vectors2)
        else:
            return np.array([]), np.array([])

    @staticmethod
    def calculate_angles(vec1, vec2):
        # Normalize vectors
        vec1 /= np.linalg.norm(vec1, axis=1, keepdims=True)
        vec2 /= np.linalg.norm(vec2, axis=1, keepdims=True)
        # Compute angles (in degrees)
        cos_angles = np.sum(vec1 * vec2, axis=1)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles) * (180.0 / np.pi)
        return angles


