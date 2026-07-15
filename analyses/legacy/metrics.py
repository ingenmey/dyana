"""Shared geometric metric helpers for the supported analyses."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from core.geometry import minimum_image


@dataclass
class Selector:
    """Store global atom indices for vectorized coordinate access."""

    indices: np.ndarray

    def coords(self, xyz: np.ndarray) -> np.ndarray:
        if self.indices.ndim == 1:
            return xyz[self.indices]
        return np.stack([xyz[self.indices[:, i]] for i in range(self.indices.shape[1])], axis=1)


class DistanceMetric:
    """Distance metric between two atom selections under periodic boundaries."""

    def __init__(self, selector_a: Selector, selector_b: Selector, box: np.ndarray, cutoff: float = None):
        self.sel_a = selector_a
        self.sel_b = selector_b
        self.box = box
        self.cutoff = cutoff
        self.cutoff_sq = cutoff ** 2 if cutoff else None

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        coords_a = self.sel_a.coords(coords)
        coords_b = self.sel_b.coords(coords)

        if len(coords_a) == 0 or len(coords_b) == 0:
            return np.array([])

        tree = cKDTree(coords_b, boxsize=self.box)

        if self.cutoff:
            pairs = tree.query_ball_point(coords_a, r=self.cutoff)
            result = []
            for i, neighbor_ids in enumerate(pairs):
                if not neighbor_ids:
                    continue
                deltas = minimum_image(coords_b[neighbor_ids] - coords_a[i], self.box)
                dists = np.linalg.norm(deltas, axis=1)
                result.extend(dists[dists > 0])
            return np.array(result)

        deltas = minimum_image(coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :], self.box)
        dists = np.linalg.norm(deltas, axis=2)
        return dists[dists > 0].flatten()


class AngleMetric:
    """Angle metric between two vectors defined by four atom selections."""

    def __init__(
        self,
        selector_ref_base: Selector,
        selector_ref_tip: Selector,
        selector_obs_base: Selector,
        selector_obs_tip: Selector,
        box: np.ndarray,
        enforce_shared_atom: bool = False,
        v1_cutoff: float = None,
        v2_cutoff: float = None,
    ):
        self.ref_base = selector_ref_base
        self.ref_tip = selector_ref_tip
        self.obs_base = selector_obs_base
        self.obs_tip = selector_obs_tip
        self.box = box
        self.v1_cutoff_sq = v1_cutoff ** 2 if v1_cutoff else None
        self.v2_cutoff_sq = v2_cutoff ** 2 if v2_cutoff else None
        self.enforce_shared_atom = enforce_shared_atom
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
        v1 = minimum_image(self.ref_tip.coords(coords) - self.ref_base.coords(coords), self.box)
        v2 = minimum_image(self.obs_tip.coords(coords) - self.obs_base.coords(coords), self.box)

        if (self.v1_cutoff_sq is not None) or (self.v2_cutoff_sq is not None):
            mask = np.ones(len(v1), dtype=bool)
            if self.v1_cutoff_sq is not None:
                mask &= np.einsum("ij,ij->i", v1, v1) <= self.v1_cutoff_sq
            if self.v2_cutoff_sq is not None:
                mask &= np.einsum("ij,ij->i", v2, v2) <= self.v2_cutoff_sq
            v1 = v1[mask]
            v2 = v2[mask]

        v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
        v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
        cos_theta = np.sum(v1 * v2, axis=1)
        angles = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180 / np.pi)
        return angles
