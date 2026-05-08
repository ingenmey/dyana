from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import distance_squared, periodic_center
from utils import label_matches


@dataclass(frozen=True)
class CompoundType:
    type_id: int
    key: tuple
    rep: str
    canonical_labels: tuple[str, ...]
    label_to_local_index: dict[str, int]
    local_bonds: tuple[tuple[int, int], ...]
    local_elements: tuple[str, ...]
    atomic_masses: tuple[float, ...]

    @property
    def n_local_atoms(self) -> int:
        return len(self.canonical_labels)


@dataclass(frozen=True)
class SelectionSpec:
    type_key: tuple
    local_indices: tuple[int, ...]


class TypeRegistry:
    def __init__(self, compound_types):
        self.compound_types = tuple(compound_types)
        self._by_key = {compound_type.key: compound_type for compound_type in self.compound_types}

    def __iter__(self):
        return iter(self.compound_types)

    def __len__(self):
        return len(self.compound_types)

    def get_by_index(self, index: int) -> CompoundType:
        return self.compound_types[index]

    def get_by_key(self, key) -> CompoundType:
        return self._by_key[key]

    def has_key(self, key) -> bool:
        return key in self._by_key


class TopologyFrame:
    def __init__(
        self,
        registry: TypeRegistry,
        member_atom_ids_by_key: dict[tuple, np.ndarray],
        atom_to_type_id: np.ndarray,
        atom_to_member_index: np.ndarray,
        atom_to_local_index: np.ndarray,
    ):
        self.registry = registry
        self._member_atom_ids_by_key = member_atom_ids_by_key
        self.atom_to_type_id = atom_to_type_id
        self.atom_to_member_index = atom_to_member_index
        self.atom_to_local_index = atom_to_local_index

    def get_compound_types(self) -> tuple[CompoundType, ...]:
        return self.registry.compound_types

    def get_compound_type_by_index(self, index: int) -> CompoundType:
        return self.registry.get_by_index(index)

    def get_compound_type_by_key(self, key) -> CompoundType:
        return self.registry.get_by_key(key)

    def has_compound_type_key(self, key) -> bool:
        return self.registry.has_key(key)

    def get_member_atom_ids(self, compound_type_or_key) -> np.ndarray:
        compound_type = self._resolve_compound_type(compound_type_or_key)
        return self._member_atom_ids_by_key[compound_type.key]

    def get_member_count(self, compound_type_or_key) -> int:
        return int(self.get_member_atom_ids(compound_type_or_key).shape[0])

    def get_member_coms(
        self,
        compound_type_or_key,
        coords: np.ndarray,
        box_size: np.ndarray,
    ) -> np.ndarray:
        compound_type = self._resolve_compound_type(compound_type_or_key)
        member_atom_ids = self._member_atom_ids_by_key[compound_type.key]
        member_coms = np.zeros((len(member_atom_ids), 3), dtype=float)
        for member_index, atom_ids in enumerate(member_atom_ids):
            member_coms[member_index] = periodic_center(
                coords[atom_ids],
                box_size,
                weights=compound_type.atomic_masses,
            )
        return member_coms

    def get_average_bond_lengths(
        self,
        compound_type_or_key,
        coords: np.ndarray,
        box_size: np.ndarray,
    ) -> dict[str, float]:
        compound_type = self._resolve_compound_type(compound_type_or_key)
        member_atom_ids = self._member_atom_ids_by_key[compound_type.key]
        if len(member_atom_ids) == 0 or not compound_type.local_bonds:
            return {}

        bond_sum_sq: dict[str, float] = {}
        for local_a, local_b in compound_type.local_bonds:
            label_a = compound_type.canonical_labels[local_a]
            label_b = compound_type.canonical_labels[local_b]
            total_sq = 0.0
            for atom_ids in member_atom_ids:
                total_sq += distance_squared(coords[atom_ids[local_a]], coords[atom_ids[local_b]], box_size)
            mean_length = float(np.sqrt(total_sq / len(member_atom_ids)))
            bond_sum_sq[f"{label_a} {label_b}"] = mean_length
        return bond_sum_sq

    def resolve_selection(self, compound_type_or_key, labels) -> SelectionSpec:
        compound_type = self._resolve_compound_type(compound_type_or_key)
        local_indices = self._resolve_local_indices(compound_type, labels)
        return SelectionSpec(
            type_key=compound_type.key,
            local_indices=local_indices,
        )

    def _resolve_local_indices(self, compound_type_or_key, labels) -> tuple[int, ...]:
        compound_type = self._resolve_compound_type(compound_type_or_key)
        patterns = (labels,) if isinstance(labels, str) else tuple(labels)
        return tuple(
            compound_type.label_to_local_index[label]
            for label in compound_type.canonical_labels
            if any(label_matches(pattern, label) for pattern in patterns)
        )

    def get_atom_indices_for_local_indices(self, compound_type_or_key, local_indices) -> np.ndarray:
        member_atom_ids = self.get_member_atom_ids(compound_type_or_key)
        if len(member_atom_ids) == 0 or len(local_indices) == 0:
            return np.empty(0, dtype=np.int32)
        return member_atom_ids[:, list(local_indices)].reshape(-1)

    def get_member_coords(self, compound_type_or_key, coords: np.ndarray) -> np.ndarray:
        member_atom_ids = self.get_member_atom_ids(compound_type_or_key)
        return coords[member_atom_ids]

    def get_atom_location(self, atom_index: int) -> tuple[int, int, int]:
        return (
            int(self.atom_to_type_id[atom_index]),
            int(self.atom_to_member_index[atom_index]),
            int(self.atom_to_local_index[atom_index]),
        )

    def _resolve_compound_type(self, compound_type_or_key) -> CompoundType:
        if isinstance(compound_type_or_key, CompoundType):
            return compound_type_or_key
        return self.registry.get_by_key(compound_type_or_key)
