"""Shared channel datatypes and helpers for supported reference analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial import cKDTree

from core.geometry import minimum_image
from analyses.common.reference_batch import ReferenceBatch


@dataclass(frozen=True)
class ChannelSamples:
    """Zero or more scalar samples for one reference molecule."""

    values: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.values).reshape(-1)
        object.__setattr__(self, "values", values)

    @property
    def size(self) -> int:
        return int(len(self.values))

    @property
    def is_empty(self) -> bool:
        return self.size == 0


@runtime_checkable
class ReferenceChannel(Protocol):
    """One scalar-valued shared channel evaluated per reference molecule."""

    output_name: str
    bin_edges: np.ndarray

    def prepare(self, traj, ref_compound_type) -> None: ...

    def rebuild_runtime_state(self, traj, ref_compound_type) -> None: ...

    def samples_for_reference(self, batch: ReferenceBatch, ref_molecule_index: int) -> ChannelSamples: ...

    def axis_normalization_factors(self) -> np.ndarray | None: ...


def radial_shell_volumes(bin_edges: np.ndarray) -> np.ndarray:
    """Return spherical shell volumes for RDF-style radial bins."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    return (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def angular_inverse_sin_weights(bin_edges: np.ndarray) -> np.ndarray:
    """Return inverse-sin weights at angular bin centers for ADF-style bins."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    centers = 0.5 * (edges[1:] + edges[:-1])
    radians = np.deg2rad(centers)
    sin_values = np.sin(radians)
    return np.divide(
        1.0,
        sin_values,
        out=np.zeros_like(sin_values, dtype=np.float64),
        where=sin_values != 0,
    )


@dataclass
class DistanceChannel:
    """Per-reference distance sample generator."""

    ref_key: tuple
    obs_key: tuple
    ref_local_indices: tuple[int, ...]
    obs_local_indices: tuple[int, ...]
    max_distance: float
    bin_edges: np.ndarray
    output_name: str = "r"

    def __post_init__(self):
        self.bin_edges = np.asarray(self.bin_edges, dtype=np.float64)
        self.ref_atom_ids = np.empty(0, dtype=np.int32)
        self.obs_atom_ids = np.empty(0, dtype=np.int32)
        self.ref_molecule_atom_ids = np.empty((0, 0), dtype=np.int32)
        self.ref_type = None
        self.obs_type = None
        self._cached_batch_token = None
        self._obs_tree = None
        self._obs_coords = None

    def prepare(self, traj, ref_compound_type) -> None:
        return None

    def rebuild_runtime_state(self, traj, ref_compound_type=None) -> None:
        topology_frame = traj.topology_frame
        if not topology_frame.has_compound_type_key(self.ref_key) or not topology_frame.has_compound_type_key(self.obs_key):
            self.ref_type = None
            self.obs_type = None
            self.ref_atom_ids = np.empty(0, dtype=np.int32)
            self.obs_atom_ids = np.empty(0, dtype=np.int32)
            self.ref_molecule_atom_ids = np.empty((0, len(self.ref_local_indices)), dtype=np.int32)
            return

        self.ref_type = topology_frame.get_compound_type_by_key(self.ref_key)
        self.obs_type = topology_frame.get_compound_type_by_key(self.obs_key)
        ref_molecule_atom_ids = topology_frame.get_molecule_atom_ids(self.ref_type)
        self.ref_molecule_atom_ids = ref_molecule_atom_ids
        self.ref_atom_ids = topology_frame.get_atom_ids_for_local_indices(self.ref_type, self.ref_local_indices)
        self.obs_atom_ids = topology_frame.get_atom_ids_for_local_indices(self.obs_type, self.obs_local_indices)
        self._cached_batch_token = None
        self._obs_tree = None
        self._obs_coords = None

    def build_batch(self, traj) -> ReferenceBatch:
        return ReferenceBatch(
            ref_compound_key=self.ref_key,
            ref_compound_type=self.ref_type,
            molecule_atom_ids=self.ref_molecule_atom_ids,
            coords=traj.coords,
            box=traj.box_size,
            topology_frame=traj.topology_frame,
        )

    def begin_batch(self, batch: ReferenceBatch) -> None:
        batch_token = (id(batch.coords), id(batch.box))
        if self._cached_batch_token == batch_token:
            return

        self._cached_batch_token = batch_token
        if self.obs_atom_ids.size == 0:
            self._obs_coords = np.empty((0, 3), dtype=np.float64)
            self._obs_tree = None
            return

        self._obs_coords = batch.coords[self.obs_atom_ids]
        self._obs_tree = cKDTree(self._obs_coords, boxsize=batch.box)

    def samples_for_reference(self, batch: ReferenceBatch, ref_molecule_index: int) -> ChannelSamples:
        if batch.n_references == 0 or self.obs_atom_ids.size == 0:
            return ChannelSamples(values=np.array([], dtype=np.float64))

        self.begin_batch(batch)
        ref_atom_ids = np.asarray(batch.molecule_atom_ids[ref_molecule_index, list(self.ref_local_indices)]).reshape(-1)
        if ref_atom_ids.size == 0 or self._obs_tree is None:
            return ChannelSamples(values=np.array([], dtype=np.float64))

        ref_coords = batch.coords[ref_atom_ids]
        neighbors_per_ref = self._obs_tree.query_ball_point(ref_coords, r=self.max_distance)
        values = []
        for ref_index, neighbor_ids in enumerate(neighbors_per_ref):
            if not neighbor_ids:
                continue
            deltas = minimum_image(self._obs_coords[neighbor_ids] - ref_coords[ref_index], batch.box)
            distances = np.linalg.norm(deltas, axis=1)
            for distance in distances:
                if distance <= 1e-12:
                    continue
                values.append(float(distance))
        return ChannelSamples(values=np.asarray(values, dtype=np.float64))

    def axis_normalization_factors(self) -> np.ndarray | None:
        return radial_shell_volumes(self.bin_edges)


@dataclass
class AngleChannel:
    """Per-reference angle sample generator."""

    ref_key: tuple
    obs_key: tuple
    ref_base_source: str
    ref_tip_source: str
    obs_base_source: str
    obs_tip_source: str
    ref_base_local_indices: tuple[int, ...]
    ref_tip_local_indices: tuple[int, ...]
    obs_base_local_indices: tuple[int, ...]
    obs_tip_local_indices: tuple[int, ...]
    bin_edges: np.ndarray
    output_name: str = "angle"
    enforce_shared_atom: bool = False
    v1_cutoff: float | None = None
    v2_cutoff: float | None = None

    def __post_init__(self):
        self.bin_edges = np.asarray(self.bin_edges, dtype=np.float64)
        self.ref_type = None
        self.obs_type = None
        self.ref_molecule_atom_ids = np.empty((0, 0), dtype=np.int32)
        self.ref_base_ids = np.empty(0, dtype=np.int32)
        self.ref_tip_ids = np.empty(0, dtype=np.int32)
        self.obs_base_ids = np.empty(0, dtype=np.int32)
        self.obs_tip_ids = np.empty(0, dtype=np.int32)
        self.vector_ids_by_reference = []

    def prepare(self, traj, ref_compound_type) -> None:
        return None

    def rebuild_runtime_state(self, traj, ref_compound_type=None) -> None:
        topology_frame = traj.topology_frame
        if not topology_frame.has_compound_type_key(self.ref_key) or not topology_frame.has_compound_type_key(self.obs_key):
            self.ref_type = None
            self.obs_type = None
            self.ref_molecule_atom_ids = np.empty((0, 0), dtype=np.int32)
            self.ref_base_ids = np.empty(0, dtype=np.int32)
            self.ref_tip_ids = np.empty(0, dtype=np.int32)
            self.obs_base_ids = np.empty(0, dtype=np.int32)
            self.obs_tip_ids = np.empty(0, dtype=np.int32)
            self.vector_ids_by_reference = []
            return

        self.ref_type = topology_frame.get_compound_type_by_key(self.ref_key)
        self.obs_type = topology_frame.get_compound_type_by_key(self.obs_key)
        self.ref_molecule_atom_ids = topology_frame.get_molecule_atom_ids(self.ref_type)
        self.vector_ids_by_reference = build_vector_lists_per_reference(
            topology_frame,
            self.ref_type,
            self.obs_type,
            self.ref_base_source,
            self.ref_tip_source,
            self.obs_base_source,
            self.obs_tip_source,
            self.ref_base_local_indices,
            self.ref_tip_local_indices,
            self.obs_base_local_indices,
            self.obs_tip_local_indices,
            self.enforce_shared_atom,
        )

        nonempty_groups = [group for group in self.vector_ids_by_reference if group[0].size > 0]
        if nonempty_groups:
            self.ref_base_ids = np.concatenate([group[0] for group in nonempty_groups])
            self.ref_tip_ids = np.concatenate([group[1] for group in nonempty_groups])
            self.obs_base_ids = np.concatenate([group[2] for group in nonempty_groups])
            self.obs_tip_ids = np.concatenate([group[3] for group in nonempty_groups])
        else:
            self.ref_base_ids = np.empty(0, dtype=np.int32)
            self.ref_tip_ids = np.empty(0, dtype=np.int32)
            self.obs_base_ids = np.empty(0, dtype=np.int32)
            self.obs_tip_ids = np.empty(0, dtype=np.int32)

    def build_batch(self, traj) -> ReferenceBatch:
        return ReferenceBatch(
            ref_compound_key=self.ref_key,
            ref_compound_type=self.ref_type,
            molecule_atom_ids=self.ref_molecule_atom_ids,
            coords=traj.coords,
            box=traj.box_size,
            topology_frame=traj.topology_frame,
        )

    def samples_for_reference(self, batch: ReferenceBatch, ref_molecule_index: int) -> ChannelSamples:
        if ref_molecule_index >= len(self.vector_ids_by_reference):
            return ChannelSamples(values=np.array([], dtype=np.float64))

        ref_base_ids, ref_tip_ids, obs_base_ids, obs_tip_ids = self.vector_ids_by_reference[ref_molecule_index]
        if ref_base_ids.size == 0:
            return ChannelSamples(values=np.array([], dtype=np.float64))

        v1 = minimum_image(batch.coords[ref_tip_ids] - batch.coords[ref_base_ids], batch.box)
        v2 = minimum_image(batch.coords[obs_tip_ids] - batch.coords[obs_base_ids], batch.box)

        if (self.v1_cutoff is not None) or (self.v2_cutoff is not None):
            mask = np.ones(len(v1), dtype=bool)
            if self.v1_cutoff is not None:
                mask &= np.einsum("ij,ij->i", v1, v1) <= self.v1_cutoff ** 2
            if self.v2_cutoff is not None:
                mask &= np.einsum("ij,ij->i", v2, v2) <= self.v2_cutoff ** 2
            v1 = v1[mask]
            v2 = v2[mask]

        if len(v1) == 0:
            return ChannelSamples(values=np.array([], dtype=np.float64))

        v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        cos_theta = np.sum(v1 * v2, axis=1)
        angles = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180.0 / np.pi)
        return ChannelSamples(values=np.asarray(angles, dtype=np.float64))

    def axis_normalization_factors(self) -> np.ndarray | None:
        inverse_weights = angular_inverse_sin_weights(self.bin_edges)
        return np.divide(
            1.0,
            inverse_weights,
            out=np.zeros(len(self.bin_edges) - 1, dtype=np.float64),
            where=inverse_weights != 0,
        )


def build_vector_lists_per_reference(
    topology_frame,
    ref_type,
    obs_type,
    ref_base_source,
    ref_tip_source,
    obs_base_source,
    obs_tip_source,
    ref_base_local_indices,
    ref_tip_local_indices,
    obs_base_local_indices,
    obs_tip_local_indices,
    enforce_shared_atom,
):
    """Build per-reference global atom-id arrays for all valid ADF vector combinations."""
    ref_molecule_atom_ids = topology_frame.get_molecule_atom_ids(ref_type)
    obs_molecule_atom_ids = topology_frame.get_molecule_atom_ids(obs_type)

    same_type = ref_type.key == obs_type.key
    per_reference = []

    for ref_molecule_index, ref_atom_ids in enumerate(ref_molecule_atom_ids):
        ref_base_ids = []
        ref_tip_ids = []
        obs_base_ids = []
        obs_tip_ids = []

        for obs_molecule_index, obs_atom_ids in enumerate(obs_molecule_atom_ids):
            if same_type and ref_molecule_index == obs_molecule_index:
                continue

            rb_source = ref_atom_ids if ref_base_source == "r" else obs_atom_ids
            rt_source = ref_atom_ids if ref_tip_source == "r" else obs_atom_ids
            ob_source = obs_atom_ids if obs_base_source == "o" else ref_atom_ids
            ot_source = obs_atom_ids if obs_tip_source == "o" else ref_atom_ids

            rb_ids = rb_source[list(ref_base_local_indices)]
            rt_ids = rt_source[list(ref_tip_local_indices)]
            ob_ids = ob_source[list(obs_base_local_indices)]
            ot_ids = ot_source[list(obs_tip_local_indices)]

            for rb in rb_ids:
                for rt in rt_ids:
                    for ob in ob_ids:
                        if enforce_shared_atom and rt != ob:
                            continue
                        for ot in ot_ids:
                            ref_base_ids.append(int(rb))
                            ref_tip_ids.append(int(rt))
                            obs_base_ids.append(int(ob))
                            obs_tip_ids.append(int(ot))

        per_reference.append(
            (
                np.asarray(ref_base_ids, dtype=np.int32),
                np.asarray(ref_tip_ids, dtype=np.int32),
                np.asarray(obs_base_ids, dtype=np.int32),
                np.asarray(obs_tip_ids, dtype=np.int32),
            )
        )

    return per_reference
