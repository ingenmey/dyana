from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from core.geometry import minimum_image
from framework.analysis_params import AtomLabelsParam, BoolParam, CompoundParam, FloatParam, ForEach, Group, IntParam, When


@dataclass(frozen=True)
class ObservedAtomGroupSpec:
    """One observed compound/label group used by a pair selector."""

    compound_index: int
    labels: list[str]

    def __post_init__(self):
        if self.compound_index < 0:
            raise ValueError("compound_index must be >= 0.")
        if not self.labels:
            raise ValueError("labels must not be empty.")


@dataclass(frozen=True)
class PairSelectorSpec:
    """Reference-to-observed atom-pair selection criteria."""

    ref_labels: list[str]
    observed_groups: list[ObservedAtomGroupSpec]
    min_distance: float | None = None
    max_distance: float | None = None
    min_rank: int | None = None
    max_rank: int | None = None

    def __post_init__(self):
        if not self.ref_labels:
            raise ValueError("ref_labels must not be empty.")
        if not self.observed_groups:
            raise ValueError("observed_groups must not be empty.")

        uses_distance = self.min_distance is not None or self.max_distance is not None
        uses_rank = self.min_rank is not None or self.max_rank is not None
        if not uses_distance and not uses_rank:
            raise ValueError("At least one distance or nearest-neighbour condition is required.")

        if uses_distance:
            if self.min_distance is None or self.max_distance is None:
                raise ValueError("min_distance and max_distance must both be set when using a distance condition.")
            if self.min_distance < 0 or self.max_distance < 0:
                raise ValueError("Distance limits must be >= 0.")
            if self.min_distance > self.max_distance:
                raise ValueError("min_distance must be <= max_distance.")

        if uses_rank:
            if self.min_rank is None or self.max_rank is None:
                raise ValueError("min_rank and max_rank must both be set when using a nearest-neighbour condition.")
            if self.min_rank < 1 or self.max_rank < 1:
                raise ValueError("Nearest-neighbour ranks must be >= 1.")
            if self.min_rank > self.max_rank:
                raise ValueError("min_rank must be <= max_rank.")

    @property
    def uses_distance(self) -> bool:
        return self.min_distance is not None

    @property
    def uses_rank(self) -> bool:
        return self.min_rank is not None

    @property
    def selected_rank_count(self) -> int | None:
        if not self.uses_rank:
            return None
        return int(self.max_rank - self.min_rank + 1)


def pair_selector_schema(
    name: str,
    label: str,
    ref_compound_field: str,
    *,
    default_use_distance: bool = False,
    default_distance_max: float = 3.5,
    default_use_rank: bool = False,
    default_min_rank: int = 1,
    default_max_rank: int = 1,
) -> Group:
    """Prompt schema for one reusable ref/obs pair selector."""

    return Group(
        name=name,
        config_class=_build_pair_selector_spec,
        steps=[
            AtomLabelsParam(
                name="ref_labels",
                role="reference",
                compound=ref_compound_field,
                prompt=f"Which reference atom(s) define {label}? (comma-separated) ",
            ),
            CompoundParam(
                name="observed_compound_indices",
                role="observed",
                multi=True,
            ),
            ForEach(
                source="observed_compound_indices",
                item_name="observed_compound_index",
                collect_as="observed_groups",
                collect_mode="list",
                config_class=ObservedAtomGroupSpec,
                include_item_as="compound_index",
                steps=[
                    AtomLabelsParam(
                        name="labels",
                        role="observed",
                        compound="observed_compound_index",
                    ),
                ],
            ),
            BoolParam(
                name="use_distance_condition",
                prompt="Use a distance condition?",
                default=default_use_distance,
            ),
            When(
                source="use_distance_condition",
                value=True,
                steps=[
                    FloatParam(
                        name="min_distance",
                        prompt="Enter the minimum distance (Angstrom): ",
                        default=0.0,
                        minval=0.0,
                    ),
                    FloatParam(
                        name="max_distance",
                        prompt="Enter the maximum distance (Angstrom): ",
                        default=default_distance_max,
                        minval=0.0,
                    ),
                ],
            ),
            BoolParam(
                name="use_rank_condition",
                prompt="Use a nearest-neighbour condition?",
                default=default_use_rank,
            ),
            When(
                source="use_rank_condition",
                value=True,
                steps=[
                    IntParam(
                        name="min_rank",
                        prompt="Use next neighbours from the n-th on: ",
                        default=default_min_rank,
                        minval=1,
                    ),
                    IntParam(
                        name="max_rank",
                        prompt="Use next neighbours up to the n-th: ",
                        default=default_max_rank,
                        minval=1,
                    ),
                ],
            ),
        ],
    )


class PairSelector:
    """Resolve and apply one reusable ref/obs pair selector."""

    def __init__(self, owner, ref_type, ref_key, spec: PairSelectorSpec):
        self.traj = owner.traj
        self.ref_key = ref_key
        self.ref_type = ref_type
        self.spec = spec

        topology_frame = self.traj.topology_frame
        self.ref_selection = topology_frame.resolve_selection(ref_type, spec.ref_labels)
        if len(self.ref_selection.local_indices) == 0:
            raise ValueError("Reference selector matched no atoms in the initial frame.")

        resolved_obs = owner.resolve_compound_types([group.compound_index for group in spec.observed_groups])
        self.obs_keys = [key for _, key in resolved_obs]
        self.obs_types = [compound_type for compound_type, _ in resolved_obs]
        self.obs_selections = []
        for group, compound_type in zip(spec.observed_groups, self.obs_types):
            selection = topology_frame.resolve_selection(compound_type, group.labels)
            if len(selection.local_indices) == 0:
                raise ValueError("Observed selector matched no atoms in the initial frame.")
            self.obs_selections.append(selection)

        self.observed_selection_entries = [
            (group.labels, compound_type.formula)
            for group, compound_type in zip(spec.observed_groups, self.obs_types)
        ]
        self.ref_indices = np.empty(0, dtype=np.int32)
        self.obs_indices = np.empty(0, dtype=np.int32)
        self._has_self_overlap = False
        self._cached_frame_token = None
        self._obs_coords = np.empty((0, 3), dtype=np.float64)
        self._obs_tree = None
        self.rebuild_runtime_state()
        if self.obs_indices.size == 0:
            raise ValueError("Observed selector matched no atoms in the initial frame.")

    def rebuild_runtime_state(self):
        topology_frame = self.traj.topology_frame
        self.ref_indices = topology_frame.get_atom_ids_for_local_indices(
            self.ref_type,
            self.ref_selection.local_indices,
        )

        obs_parts = [
            topology_frame.get_atom_ids_for_local_indices(compound_type, selection.local_indices)
            for compound_type, selection in zip(self.obs_types, self.obs_selections)
        ]
        if any(part.size > 0 for part in obs_parts):
            self.obs_indices = np.unique(np.concatenate([part for part in obs_parts if part.size > 0]).astype(np.int32))
        else:
            self.obs_indices = np.empty(0, dtype=np.int32)

        self._has_self_overlap = bool(np.intersect1d(self.ref_indices, self.obs_indices).size)
        self._cached_frame_token = None
        self._obs_coords = np.empty((0, 3), dtype=np.float64)
        self._obs_tree = None

    def reattach_and_rebuild(self) -> bool:
        topology_frame = self.traj.topology_frame
        if not topology_frame.has_compound_type_key(self.ref_key):
            return False
        if any(not topology_frame.has_compound_type_key(key) for key in self.obs_keys):
            return False

        self.ref_type = topology_frame.get_compound_type_by_key(self.ref_key)
        self.obs_types = [topology_frame.get_compound_type_by_key(key) for key in self.obs_keys]
        self.observed_selection_entries = [
            (group.labels, compound_type.formula)
            for group, compound_type in zip(self.spec.observed_groups, self.obs_types)
        ]
        try:
            self.rebuild_runtime_state()
        except KeyError:
            return False
        return self.ref_indices.size > 0 and self.obs_indices.size > 0

    def begin_frame(self):
        frame_token = (id(self.traj.coords), id(self.traj.box_size), int(self.obs_indices.size))
        if self._cached_frame_token == frame_token:
            return

        self._cached_frame_token = frame_token
        if self.obs_indices.size == 0:
            self._obs_coords = np.empty((0, 3), dtype=np.float64)
            self._obs_tree = None
            return

        self._obs_coords = self.traj.coords[self.obs_indices]
        self._obs_tree = cKDTree(self._obs_coords, boxsize=self.traj.box_size)

    def select_frame(self, include_deltas: bool = False) -> tuple[list[np.ndarray], list[np.ndarray] | None, list[np.ndarray]]:
        self.begin_frame()
        n_refs = int(self.ref_indices.size)
        empty_ids = np.empty(0, dtype=np.int32)
        empty_deltas = np.empty((0, 3), dtype=np.float64)
        empty_distances = np.empty(0, dtype=np.float64)
        obs_ids_by_ref = [empty_ids for _ in range(n_refs)]
        distances_by_ref = [empty_distances for _ in range(n_refs)]
        deltas_by_ref = [empty_deltas for _ in range(n_refs)] if include_deltas else None
        if n_refs == 0 or self._obs_tree is None:
            return obs_ids_by_ref, deltas_by_ref, distances_by_ref

        ref_coords = self.traj.coords[self.ref_indices]
        box = self.traj.box_size

        if self.spec.uses_rank:
            query_k = self.spec.max_rank + (1 if self._has_self_overlap else 0)
            query_k = min(query_k, len(self._obs_coords))
            distances_all, positions_all = self._obs_tree.query(ref_coords, k=query_k)
            distances_all = np.asarray(distances_all, dtype=np.float64)
            positions_all = np.asarray(positions_all, dtype=np.intp)
            if distances_all.ndim == 1:
                distances_all = distances_all[:, None]
                positions_all = positions_all[:, None]

            obs_ids_all = self.obs_indices[positions_all]
            structural = (
                (obs_ids_all != self.ref_indices[:, None])
                & np.isfinite(distances_all)
                & (distances_all > 1e-12)
            )
            rank_min = self.spec.min_rank
            rank_max = self.spec.max_rank

            for ref_position, ref_coord in enumerate(ref_coords):
                positions = positions_all[ref_position][structural[ref_position]]
                distances = distances_all[ref_position][structural[ref_position]]
                if positions.size == 0:
                    continue

                if rank_min != 1 or rank_max < distances.size:
                    ranks = np.arange(1, distances.size + 1, dtype=np.int32)
                    mask = (ranks >= rank_min) & (ranks <= rank_max)
                else:
                    mask = np.ones(distances.size, dtype=bool)

                if self.spec.uses_distance:
                    mask &= (distances >= self.spec.min_distance) & (distances <= self.spec.max_distance)

                positions = positions[mask]
                if positions.size == 0:
                    continue

                obs_ids_by_ref[ref_position] = self.obs_indices[positions]
                distances_by_ref[ref_position] = distances[mask]
                if include_deltas:
                    deltas_by_ref[ref_position] = minimum_image(self._obs_coords[positions] - ref_coord, box)
        else:
            positions_by_ref = self._obs_tree.query_ball_point(ref_coords, self.spec.max_distance)
            for ref_position, ref_coord in enumerate(ref_coords):
                positions = np.asarray(positions_by_ref[ref_position], dtype=np.intp)
                if positions.size == 0:
                    continue

                deltas = minimum_image(self._obs_coords[positions] - ref_coord, box)
                distances = np.linalg.norm(deltas, axis=1)
                order = np.argsort(distances, kind="stable")
                positions = positions[order]
                deltas = deltas[order]
                distances = distances[order]
                obs_ids = self.obs_indices[positions]

                structural = (obs_ids != self.ref_indices[ref_position]) & (distances > 1e-12)
                positions = positions[structural]
                deltas = deltas[structural]
                distances = distances[structural]
                if positions.size == 0:
                    continue

                if self.spec.min_distance > 0:
                    mask = distances >= self.spec.min_distance
                    positions = positions[mask]
                    deltas = deltas[mask]
                    distances = distances[mask]
                    if positions.size == 0:
                        continue

                obs_ids_by_ref[ref_position] = self.obs_indices[positions]
                distances_by_ref[ref_position] = distances
                if include_deltas:
                    deltas_by_ref[ref_position] = deltas

        return obs_ids_by_ref, deltas_by_ref, distances_by_ref


def _build_pair_selector_spec(**kwargs):
    kwargs.pop("observed_compound_indices", None)
    use_distance = kwargs.pop("use_distance_condition", False)
    use_rank = kwargs.pop("use_rank_condition", False)
    if not use_distance:
        kwargs.pop("min_distance", None)
        kwargs.pop("max_distance", None)
    if not use_rank:
        kwargs.pop("min_rank", None)
        kwargs.pop("max_rank", None)
    return PairSelectorSpec(**kwargs)
