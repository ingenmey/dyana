# metrics.py

import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree


@dataclass
class Selector:
    """Stores global atom indices (or index tuples) for vectorized coordinate access."""
    indices: np.ndarray  # shape (N,) for RDF, or (N, k) for angle, etc.

    def coords(self, xyz: np.ndarray) -> np.ndarray:
        if self.indices.ndim == 1:
            return xyz[self.indices]
        else:
            return np.stack([xyz[self.indices[:, i]] for i in range(self.indices.shape[1])], axis=1)


class DistanceMetric:
    def __init__(self, selector_a: Selector, selector_b: Selector, box: np.ndarray, cutoff: float = None):
        self.sel_a = selector_a
        self.sel_b = selector_b
        self.box = box
        self.cutoff = cutoff
        self.cutoff_sq = cutoff ** 2 if cutoff else None

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Compute all distances between atoms in A and B within cutoff, using KDTree."""
        coords_a = self.sel_a.coords(coords)
        coords_b = self.sel_b.coords(coords)

        if len(coords_a) == 0 or len(coords_b) == 0:
            return np.array([])

        tree = cKDTree(coords_b, boxsize=self.box)

        if self.cutoff:
            # query_ball_point returns a list of lists of neighbors
            pairs = tree.query_ball_point(coords_a, r=self.cutoff)
            result = []
            for i, neighbor_ids in enumerate(pairs):
                if not neighbor_ids:
                    continue
                deltas = coords_b[neighbor_ids] - coords_a[i]
                deltas -= self.box * np.round(deltas / self.box)
                dists = np.linalg.norm(deltas, axis=1)
                result.extend(dists[dists > 0])  # Exclude self-match
            return np.array(result)
        else:
            # full distance matrix
            deltas = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
            deltas -= self.box * np.round(deltas / self.box)
            dists = np.linalg.norm(deltas, axis=2)
            return dists[dists > 0].flatten()

class AngleMetric:
    def __init__(self,
                 selector_ref_base: Selector,
                 selector_ref_tip: Selector,
                 selector_obs_base: Selector,
                 selector_obs_tip: Selector,
                 box: np.ndarray,
                 enforce_shared_atom: bool = False,
                 v1_cutoff: float = None,
                 v2_cutoff: float = None):
        self.ref_base = selector_ref_base
        self.ref_tip = selector_ref_tip
        self.obs_base = selector_obs_base
        self.obs_tip = selector_obs_tip
        self.box = box
        self.v1_cutoff_sq = v1_cutoff ** 2 if v1_cutoff else None
        self.v2_cutoff_sq = v2_cutoff ** 2 if v2_cutoff else None
        self.enforce_shared_atom = enforce_shared_atom

        # Filter index combinations if needed
        self._filter_indices()

    def _filter_indices(self):
        rb = self.ref_base.indices
        rt = self.ref_tip.indices
        ob = self.obs_base.indices
        ot = self.obs_tip.indices

        if self.enforce_shared_atom:
            mask = rt == ob
            self.ref_base.indices = rb[mask]
            self.ref_tip.indices = rt[mask]
            self.obs_base.indices = ob[mask]
            self.obs_tip.indices = ot[mask]

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        v1 = self.ref_tip.coords(coords) - self.ref_base.coords(coords)
        v2 = self.obs_tip.coords(coords) - self.obs_base.coords(coords)

        # Minimum image convention
        v1 -= self.box * np.round(v1 / self.box)
        v2 -= self.box * np.round(v2 / self.box)

        # v1 , v2 :   shape (N, 3)      (already wrapped by MIC)
        # v1_cutoff_sq , v2_cutoff_sq :  either a float (cut-off²) or None
        if (self.v1_cutoff_sq is not None) or (self.v2_cutoff_sq is not None):
            mask = np.ones(len(v1), dtype=bool)          # start with all True
            if self.v1_cutoff_sq is not None:                 # ‖v1‖² ≤ cut-off₁²
                mask &= np.einsum('ij,ij->i', v1, v1) <= self.v1_cutoff_sq
            if self.v2_cutoff_sq is not None:                 # ‖v2‖² ≤ cut-off₂²
                mask &= np.einsum('ij,ij->i', v2, v2) <= self.v2_cutoff_sq
            v1 = v1[mask]
            v2 = v2[mask]

        # Normalize vectors and compute angles
        v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
        v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
        cos_theta = np.sum(v1 * v2, axis=1)
        angles = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180 / np.pi)
        return angles

#class AngleMetric:
#    def __init__(self,
#                 ref_base_indices: np.ndarray,
#                 ref_tip_indices: np.ndarray,
#                 obs_base_indices: np.ndarray,
#                 obs_tip_indices: np.ndarray,
#                 box: np.ndarray,
#                 enforce_shared_atom: bool = False,
#                 cutoff: float = None):
#        """
#        Constructs AngleMetric for computing angles between two vectors: ref and obs.
#        The vectors are defined by global atom indices:
#          - ref = (ref_tip - ref_base)
#          - obs = (obs_tip - obs_base)
#        """
#        self.box = box
#        self.cutoff_sq = cutoff ** 2 if cutoff is not None else None
#        self.enforce_shared_atom = enforce_shared_atom
#
#        # Construct filtered index list
#        ref_base_list = []
#        ref_tip_list = []
#        obs_base_list = []
#        obs_tip_list = []
#
#        for i in range(len(ref_base_indices)):
#            rb = ref_base_indices[i]
#            rt = ref_tip_indices[i]
#            ob = obs_base_indices[i]
#            ot = obs_tip_indices[i]
#
#            if enforce_shared_atom and rt != ob:
#                continue
#            ref_base_list.append(rb)
#            ref_tip_list.append(rt)
#            obs_base_list.append(ob)
#            obs_tip_list.append(ot)
#
#        self.ref_base = np.array(ref_base_list)
#        self.ref_tip = np.array(ref_tip_list)
#        self.obs_base = np.array(obs_base_list)
#        self.obs_tip = np.array(obs_tip_list)
#
#    def __call__(self, coords: np.ndarray) -> np.ndarray:
#        """Computes angles in degrees between (ref_tip - ref_base) and (obs_tip - obs_base)."""
#        ref_vecs = coords[self.ref_tip] - coords[self.ref_base]
#        obs_vecs = coords[self.obs_tip] - coords[self.obs_base]
#
#        # Minimum image convention
#        ref_vecs -= self.box * np.round(ref_vecs / self.box)
#        obs_vecs -= self.box * np.round(obs_vecs / self.box)
#
#        if self.cutoff_sq is not None:
#            delta = coords[self.ref_tip] - coords[self.obs_tip]
#            delta -= self.box * np.round(delta / self.box)
#            dist_sq = np.sum(delta**2, axis=1)
#            mask = dist_sq <= self.cutoff_sq
#            ref_vecs = ref_vecs[mask]
#            obs_vecs = obs_vecs[mask]
#
#        # Normalize and compute angles
#        ref_vecs /= np.linalg.norm(ref_vecs, axis=1, keepdims=True)
#        obs_vecs /= np.linalg.norm(obs_vecs, axis=1, keepdims=True)
#
#        cos_theta = np.sum(ref_vecs * obs_vecs, axis=1)
#        cos_theta = np.clip(cos_theta, -1.0, 1.0)
#        angles = np.arccos(cos_theta) * (180 / np.pi)
#        return angles

