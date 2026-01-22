# analyses/tetrahedral_analysis.py

import numpy as np
from scipy.spatial import cKDTree
from analyses.base_analysis import BaseAnalysis
from analyses.histogram import HistogramND
from utils import prompt, prompt_int, prompt_yn, prompt_float, label_matches

class TetrahedralOrderAnalysis(BaseAnalysis):
    def setup(self):
        self.ref_idx, self.ref_comp = self.compound_selection("reference")
        self.ref_labels = self.atom_selection("reference")
        self.ref_key = list(self.traj.compounds.keys())[self.ref_idx]

        # Multi-compound observed selection
        obs_selection = self.compound_selection("observed", multi=True)
        self.obs_idxs = [idx for idx, _ in obs_selection]
        self.obs_comps = [comp for _, comp in obs_selection]
        self.obs_keys = [list(self.traj.compounds.keys())[idx] for idx in self.obs_idxs]

        # Prompt for atom labels per observed compound
        self.obs_labels_per_compound = {}
        for key, comp in zip(self.obs_keys, self.obs_comps):
            self.obs_labels_per_compound[key] = self.atom_selection("observed", comp)

        # Cutoff and histogram setup as before...
        self.use_cutoff = prompt_yn("Use a maximum distance cutoff for neighbor search?", False)
        self.cutoff = prompt_float("Enter the maximum cutoff distance (Å):", 5.0, minval=0.0) if self.use_cutoff else None

        self.bin_count_q = prompt_int("Enter the number of bins for angular tetrahedral order distribution q: ", 100, minval=1)
        self.bin_count_s = prompt_int("Enter the number of bins for radial tetrahedral order distribution S: ", 10000, minval=1)

        self.bin_edges_q = np.linspace(0, 1, self.bin_count_q + 1)
        self.hist_q = HistogramND([self.bin_edges_q], mode="linear")
        self.bin_edges_s = np.linspace(0, 1, self.bin_count_s + 1)
        self.hist_s = HistogramND([self.bin_edges_s], mode="linear")

        self.update_selectors()

    def update_selectors(self):
        self.ref_indices = self._get_indices(self.ref_comp, self.ref_labels)
        self.obs_indices = []
        for key, comp in zip(self.obs_keys, self.obs_comps):
            labels = self.obs_labels_per_compound[key]
            self.obs_indices.extend(self._get_indices(comp, labels))

    def _get_indices(self, compound, labels):
        return [
            idx for mol in compound.members
            for label, idx in mol.label_to_global_id.items()
            if any(label_matches(lab, label) for lab in labels)
        ]

    def post_compound_update(self):
        try:
            self.ref_comp = self.traj.compounds[self.ref_key]
            self.obs_comps = [self.traj.compounds[k] for k in self.obs_keys]
        except KeyError:
            return False
        self.update_selectors()
        self.n_ref = (self.n_ref * self.processed_frames + len(self.ref_indices)) / (self.processed_frames + 1) if hasattr(self, 'n_ref') else len(self.ref_indices)
        self.n_obs = (self.n_obs * self.processed_frames + len(self.obs_indices)) / (self.processed_frames + 1) if hasattr(self, 'n_obs') else len(self.obs_indices)
        return True

    def process_frame(self):
        coords = self.traj.coords
        box = self.traj.box_size
        obs_coords = coords[self.obs_indices]
        kdtree = cKDTree(obs_coords, boxsize=box)

        q_vals = []
        s_vals = []

        for r_idx in self.ref_indices:
            r_coord = coords[r_idx]
            if self.use_cutoff:
                neighbor_idxs = kdtree.query_ball_point(r_coord, self.cutoff)
                neighbor_idxs = [i for i in neighbor_idxs if self.obs_indices[i] != r_idx]
                if len(neighbor_idxs) < 4:
                    continue
                neighbors = obs_coords[neighbor_idxs]
                distances = np.linalg.norm(neighbors - r_coord, axis=1)
                nearest = np.argsort(distances)[:4]
                four_nearest = neighbors[nearest]
                four_dists = distances[nearest]
            else:
                distances, idxs = kdtree.query(r_coord, k=5)
                filtered = [(d, i) for d, i in zip(distances, idxs) if self.obs_indices[i] != r_idx]
                if len(filtered) < 4:
                    continue
                filtered = filtered[:4]
                four_nearest = obs_coords[[i for _, i in filtered]]
                four_dists = np.array([d for d, _ in filtered])

            # Compute q
            cosines = []
            for j in range(3):
                for k in range(j + 1, 4):
                    vj = four_nearest[j] - r_coord
                    vk = four_nearest[k] - r_coord
                    vj -= box * np.round(vj / box)
                    vk -= box * np.round(vk / box)
                    vj /= np.linalg.norm(vj)
                    vk /= np.linalg.norm(vk)
                    cos_angle = np.dot(vj, vk)
                    cosines.append(cos_angle)
            q = 1 - (3/8) * sum((cos + 1/3)**2 for cos in cosines)
            q_vals.append(q)

            # Compute S
            r_mean = np.mean(four_dists)
            if r_mean > 1e-8:  # avoid divide-by-zero
                S = 1 - (1/3) * np.sum((four_dists - r_mean)**2) / (4 * r_mean**2)
                s_vals.append(S)

        # Add values to histograms
        if q_vals:
            self.hist_q.add(np.array(q_vals))
        if s_vals:
            self.hist_s.add(np.array(s_vals))

    def postprocess(self):
        # Normalize and write q
        total_q = self.hist_q.counts.sum()
        if total_q > 0:
            self.hist_q.normalize("total", total=100)
            self.hist_q.save_txt("tetrahedral_q.dat", ["q", "P(q)"])
            print("\nTetrahedral orientational order distribution saved to tetrahedral_q.dat")
        else:
            print("No valid q values were accumulated.")

        # Normalize and write S
        total_s = self.hist_s.counts.sum()
        if total_s > 0:
            self.hist_s.normalize("total", total=100)
            self.hist_s.save_txt("tetrahedral_s.dat", ["S", "P(S)"])
            print("Tetrahedral translational order distribution saved to tetrahedral_s.dat")
        else:
            print("No valid S values were accumulated.")

