from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from scipy.spatial import cKDTree

from .app_config import load_app_config
from .atomic_properties import elem_covalent, elem_masses, elem_number, elem_vdW
from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from .geometry import distance_squared

config = load_app_config()
EXCLUDED_ELEMENTS = set(config["EXCLUDED_ELEMENTS"])
NEIGHBOR_SEARCH_SCALE = config.get("NEIGHBOR_SEARCH_SCALE", 1.164)
BOND_DISTANCE_SCALE = config.get("BOND_DISTANCE_SCALE", 1.4)


@dataclass(frozen=True)
class DetectedMolecule:
    """Connectivity-only molecule record used while building topology."""

    atom_ids: tuple[int, ...]
    elements: tuple[str, ...]
    local_bonds: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MoleculeGroup:
    """Equivalent detected molecules grouped into one structural type."""

    key: tuple
    formula: str
    members: tuple[DetectedMolecule, ...]


class BaseTrajectory(ABC):
    """
    Base class for trajectories (XYZ, LAMMPS, ...).

    Raw frame state:
      - n_atoms
      - symbols
      - coords
      - box_size

    Interpreted topology state:
      - topology_registry
      - topology_frame
    """

    def __init__(self, fin, box_size: np.ndarray):
        self.fin = fin
        self.box_size = np.array(box_size, dtype=float)
        self.half_box_size = self.box_size / 2.0

        self.box_x, self.box_y, self.box_z = self.box_size
        self.n_atoms = 0
        self.symbols: list[str] = []
        self.coords: np.ndarray | None = None

        self.topology_registry: CompoundTypeRegistry | None = None
        self.topology_frame: TopologyFrame | None = None

        # Global forbidden bonds (pairs of global atom indices)
        self.forbidden_bonds: set[tuple[int, int]] = set()

    @abstractmethod
    def read_frame(self):
        raise NotImplementedError

    def rewind_to_first_frame(self):
        self.fin.seek(0)
        self.read_frame()

    def are_connected(
        self,
        coord_i: np.ndarray,
        coord_j: np.ndarray,
        cov_radius_i: float,
        cov_radius_j: float,
    ) -> float | bool:
        distance_sq = distance_squared(coord_i, coord_j, self.box_size)
        threshold_sq = ((cov_radius_i + cov_radius_j) * BOND_DISTANCE_SCALE) ** 2
        return distance_sq if distance_sq < threshold_sq else False

    def rebuild_topology(self):
        """Build the authoritative runtime topology model for the current frame."""
        if self.n_atoms == 0:
            self.topology_registry = CompoundTypeRegistry([])
            self.topology_frame = TopologyFrame(
                registry=self.topology_registry,
                molecule_atom_ids_by_key={},
                atom_to_type_id=np.empty(0, dtype=np.int32),
                atom_to_molecule_index=np.empty(0, dtype=np.int32),
                atom_to_local_index=np.empty(0, dtype=np.int32),
            )
            return

        if self.coords is None:
            raise RuntimeError("coords are not set before calling rebuild_topology().")

        kdtree = cKDTree(self.coords, boxsize=self.box_size)
        molecules = self._identify_molecules(self.symbols, self.coords, kdtree)
        groups = self._group_molecules(molecules)
        self.topology_registry, self.topology_frame = self._build_registry_and_frame(groups)

    def _identify_molecules(
        self,
        frame_symbols: list[str],
        frame_coords: np.ndarray,
        kdtree: cKDTree,
    ) -> list[DetectedMolecule]:
        # Detect all admissible bonds first, then extract connected components.
        # This avoids the old order-dependent behavior where a bond could be
        # missed just because one of its atoms had already been assigned to an
        # earlier provisional molecule.
        bond_graph: list[set[int]] = [set() for _ in range(self.n_atoms)]

        for atom_index, atom_symbol in enumerate(frame_symbols):
            if atom_symbol in EXCLUDED_ELEMENTS:
                continue

            cov_radius_atom = elem_covalent.get(atom_symbol, 0.0)
            search_radius = elem_vdW.get(atom_symbol, 0.0) * NEIGHBOR_SEARCH_SCALE
            neighbor_indices = sorted(kdtree.query_ball_point(frame_coords[atom_index], r=search_radius))

            for neighbor_index in neighbor_indices:
                if neighbor_index <= atom_index:
                    continue

                neighbor_symbol = frame_symbols[neighbor_index]
                if neighbor_symbol in EXCLUDED_ELEMENTS:
                    continue

                bond_pair = (atom_index, neighbor_index)
                if bond_pair in self.forbidden_bonds:
                    continue

                cov_radius_neighbor = elem_covalent.get(neighbor_symbol, 0.0)
                distance_sq = self.are_connected(
                    frame_coords[atom_index],
                    frame_coords[neighbor_index],
                    cov_radius_atom,
                    cov_radius_neighbor,
                )
                if not distance_sq:
                    continue

                bond_graph[atom_index].add(neighbor_index)
                bond_graph[neighbor_index].add(atom_index)

        molecules: list[DetectedMolecule] = []
        visited_global_indices: set[int] = set()

        for seed_index in range(self.n_atoms):
            if seed_index in visited_global_indices:
                continue

            molecule_atom_indices: list[int] = []
            stack: list[int] = [seed_index]
            visited_global_indices.add(seed_index)

            while stack:
                current_global_idx = stack.pop()
                molecule_atom_indices.append(current_global_idx)

                for neighbor_global_idx in sorted(bond_graph[current_global_idx], reverse=True):
                    if neighbor_global_idx in visited_global_indices:
                        continue
                    visited_global_indices.add(neighbor_global_idx)
                    stack.append(neighbor_global_idx)

            global_to_local = {
                global_idx: local_idx
                for local_idx, global_idx in enumerate(molecule_atom_indices)
            }
            local_bonds: list[tuple[int, int]] = []
            for global_idx in molecule_atom_indices:
                for neighbor_global_idx in sorted(bond_graph[global_idx]):
                    if neighbor_global_idx not in global_to_local or global_idx >= neighbor_global_idx:
                        continue
                    local_bonds.append(
                        (global_to_local[global_idx], global_to_local[neighbor_global_idx])
                    )

            molecules.append(
                DetectedMolecule(
                    atom_ids=tuple(molecule_atom_indices),
                    elements=tuple(frame_symbols[gidx] for gidx in molecule_atom_indices),
                    local_bonds=tuple(local_bonds),
                )
            )

        return molecules

    def _group_molecules(self, molecules: list[DetectedMolecule]) -> list[MoleculeGroup]:
        grouped: dict[tuple, list[DetectedMolecule]] = {}

        for molecule in molecules:
            symbol_counts = Counter(molecule.elements)
            formula = "".join(
                f"{element}{count}" if count > 1 else element
                for element, count in sorted(symbol_counts.items())
            )

            bond_types = []
            for local_a, local_b in molecule.local_bonds:
                elem_a = molecule.elements[local_a]
                elem_b = molecule.elements[local_b]
                if elem_a > elem_b:
                    elem_a, elem_b = elem_b, elem_a
                bond_types.append((elem_a, elem_b))
            bond_types.sort()

            graph = self._build_graph(molecule)
            graph_hash = nx.weisfeiler_lehman_graph_hash(graph, node_attr="element")
            compound_key = (formula, tuple(bond_types), graph_hash)
            grouped.setdefault(compound_key, []).append(molecule)

        sorted_keys = sorted(grouped, key=self._compound_sort_key)
        return [
            MoleculeGroup(key=key, formula=key[0], members=tuple(grouped[key]))
            for key in sorted_keys
        ]

    def _build_registry_and_frame(self, groups: list[MoleculeGroup]) -> tuple[CompoundTypeRegistry, TopologyFrame]:
        compound_types: list[CompoundType] = []
        molecule_atom_ids_by_key: dict[tuple, np.ndarray] = {}
        atom_to_type_id = np.full(self.n_atoms, -1, dtype=np.int32)
        atom_to_molecule_index = np.full(self.n_atoms, -1, dtype=np.int32)
        atom_to_local_index = np.full(self.n_atoms, -1, dtype=np.int32)

        for type_id, group in enumerate(groups):
            template = group.members[0]
            template_id_to_label, template_label_to_local = self._initialize_connectivity_labels(template)
            canonical_labels = tuple(sorted(template_label_to_local))
            canonical_local_indices = [template_label_to_local[label] for label in canonical_labels]
            local_elements = tuple(template.elements[idx] for idx in canonical_local_indices)
            label_to_local_index = {
                label: local_index
                for local_index, label in enumerate(canonical_labels)
            }
            atomic_masses = tuple(elem_masses[element] for element in local_elements)

            local_bonds = []
            for local_a, local_b in template.local_bonds:
                label_a = template_id_to_label[local_a]
                label_b = template_id_to_label[local_b]
                local_bonds.append(
                    tuple(
                        sorted(
                            (
                                label_to_local_index[label_a],
                                label_to_local_index[label_b],
                            )
                        )
                    )
                )
            local_bonds.sort()

            compound_type = CompoundType(
                type_id=type_id,
                key=group.key,
                formula=group.formula,
                canonical_labels=canonical_labels,
                label_to_local_index=label_to_local_index,
                local_bonds=tuple(local_bonds),
                local_elements=local_elements,
                atomic_masses=atomic_masses,
            )
            compound_types.append(compound_type)

            molecule_atom_ids = np.zeros(
                (len(group.members), len(canonical_labels)),
                dtype=np.int32,
            )

            template_graph = self._build_graph(template)
            node_match = lambda attrs_t, attrs_m: attrs_t["element"] == attrs_m["element"]

            for molecule_index, molecule in enumerate(group.members):
                if molecule_index == 0:
                    template_to_molecule = {idx: idx for idx in range(len(template.atom_ids))}
                else:
                    molecule_graph = self._build_graph(molecule)
                    matcher = isomorphism.GraphMatcher(template_graph, molecule_graph, node_match=node_match)
                    if not matcher.is_isomorphic():
                        raise RuntimeError(
                            f"Detected molecule topology is not isomorphic to its template for compound {group.formula}."
                        )
                    template_to_molecule = next(matcher.isomorphisms_iter())

                # Every member row uses template-local column order so selections can
                # reuse canonical local indices across molecules and rebuilds.
                ordered_atom_ids = np.array(
                    [
                        molecule.atom_ids[template_to_molecule[template_index]]
                        for template_index in canonical_local_indices
                    ],
                    dtype=np.int32,
                )
                molecule_atom_ids[molecule_index] = ordered_atom_ids

                for local_index, atom_index in enumerate(ordered_atom_ids):
                    atom_to_type_id[atom_index] = type_id
                    atom_to_molecule_index[atom_index] = molecule_index
                    atom_to_local_index[atom_index] = local_index

            molecule_atom_ids_by_key[group.key] = molecule_atom_ids

        registry = CompoundTypeRegistry(compound_types)
        frame = TopologyFrame(
            registry=registry,
            molecule_atom_ids_by_key=molecule_atom_ids_by_key,
            atom_to_type_id=atom_to_type_id,
            atom_to_molecule_index=atom_to_molecule_index,
            atom_to_local_index=atom_to_local_index,
        )
        return registry, frame

    def _build_graph(self, molecule: DetectedMolecule) -> nx.Graph:
        graph = nx.Graph()
        for local_idx, element in enumerate(molecule.elements):
            graph.add_node(local_idx, element=element)
        for local_a, local_b in molecule.local_bonds:
            graph.add_edge(local_a, local_b)
        return graph

    def _initialize_connectivity_labels(
        self,
        molecule: DetectedMolecule,
    ) -> tuple[dict[int, str], dict[str, int]]:
        """Assign deterministic per-element labels within one detected molecule."""
        n_atoms = len(molecule.atom_ids)
        if n_atoms == 0:
            return {}, {}

        adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
        for local_a, local_b in molecule.local_bonds:
            adjacency[local_a].append(local_b)
            adjacency[local_b].append(local_a)

        ec_values = [
            elem_number[molecule.elements[i]] * 10 + len(adjacency[i])
            for i in range(n_atoms)
        ]

        # Refine local ranks until topology stops separating equivalent positions.
        while True:
            unique_ec = set(ec_values)
            if len(unique_ec) == n_atoms:
                break

            trial_ec = []
            for local_index in range(n_atoms):
                neighbor_sum = sum(ec_values[neighbor] for neighbor in adjacency[local_index])
                trial_ec.append(neighbor_sum + 5 * ec_values[local_index])

            unique_trial_ec = set(trial_ec)
            if len(unique_trial_ec) == len(unique_ec):
                ec_values = trial_ec
                break
            ec_values = trial_ec

        id_to_label: dict[int, str] = {}
        label_to_id: dict[str, int] = {}
        symbol_groups: dict[str, list[tuple[int, int]]] = {}
        for local_idx, ec_val in enumerate(ec_values):
            element = molecule.elements[local_idx]
            symbol_groups.setdefault(element, []).append((ec_val, local_idx))

        for element, group in symbol_groups.items():
            group.sort(reverse=True, key=lambda pair: pair[0])
            for label_index, (_, local_idx) in enumerate(group, start=1):
                label = f"{element}{label_index}"
                id_to_label[local_idx] = label
                label_to_id[label] = local_idx

        return id_to_label, label_to_id

    def _compound_sort_key(self, compound_key: tuple) -> tuple:
        formula, bond_types, graph_hash = compound_key
        return (formula, bond_types, graph_hash)


class XYZTrajectory(BaseTrajectory):
    def read_frame(self):
        natoms_line = self.fin.readline()
        if not natoms_line:
            raise ValueError("End of file reached while reading XYZ trajectory.")

        self.n_atoms = int(natoms_line.strip())
        self.fin.readline()

        symbols: list[str] = []
        coords_list: list[list[float]] = []

        for _ in range(self.n_atoms):
            parts = self.fin.readline().split()
            if len(parts) < 4:
                raise ValueError("Malformed XYZ line (expected at least 4 columns).")

            symbol_str, x_str, y_str, z_str = parts[:4]
            x_val, y_val, z_val = map(float, (x_str, y_str, z_str))

            x_val = x_val % self.box_x if self.box_x else x_val
            y_val = y_val % self.box_y if self.box_y else y_val
            z_val = z_val % self.box_z if self.box_z else z_val

            symbols.append(symbol_str.capitalize())
            coords_list.append([x_val, y_val, z_val])

        self.symbols = symbols
        self.coords = np.array(coords_list, dtype=float)


class LAMMPSTrajectory(BaseTrajectory):
    def read_frame(self):
        line = self.fin.readline().strip()
        while line and not line.startswith("ITEM: TIMESTEP"):
            line = self.fin.readline().strip()
        if not line:
            raise ValueError("End of file reached before finding TIMESTEP")
        self.timestep = int(self.fin.readline().strip())

        line = self.fin.readline().strip()
        while line and not line.startswith("ITEM: NUMBER OF ATOMS"):
            line = self.fin.readline().strip()
        if not line:
            raise ValueError("End of file reached before finding NUMBER OF ATOMS")
        self.n_atoms = int(self.fin.readline().strip())

        line = self.fin.readline().strip()
        while line and not line.startswith("ITEM: BOX BOUNDS"):
            line = self.fin.readline().strip()
        if not line:
            raise ValueError("End of file reached before finding BOX BOUNDS")

        box_lengths: list[float] = []
        for _ in range(3):
            bounds = list(map(float, self.fin.readline().strip().split()))
            if len(bounds) < 2:
                raise ValueError("Malformed BOX BOUNDS line (expected lower and upper).")
            lower, upper = bounds[:2]
            box_lengths.append(upper - lower)

        self.box_size = np.array(box_lengths, dtype=float)
        self.half_box_size = self.box_size / 2.0
        self.box_x, self.box_y, self.box_z = self.box_size

        line = self.fin.readline().strip()
        while line and not line.startswith("ITEM: ATOMS"):
            line = self.fin.readline().strip()
        if not line:
            raise ValueError("End of file reached before finding ATOMS header")

        columns = line.split()[2:]
        column_indices = {name: idx for idx, name in enumerate(columns)}

        has_unwrapped = {"xu", "yu", "zu"}.issubset(column_indices.keys())
        has_wrapped = {"x", "y", "z"}.issubset(column_indices.keys())
        if not (has_unwrapped or has_wrapped):
            raise ValueError(
                "Trajectory file missing required coordinate columns (xu,yu,zu) or (x,y,z)."
            )

        atom_rows = [self.fin.readline().strip().split() for _ in range(self.n_atoms)]
        if "id" in column_indices:
            atom_rows.sort(key=lambda row: int(row[column_indices["id"]]))

        symbols: list[str] = []
        coords_list: list[list[float]] = []

        for row in atom_rows:
            element_symbol = row[column_indices["element"]]

            if has_unwrapped:
                x_val = float(row[column_indices["xu"]])
                y_val = float(row[column_indices["yu"]])
                z_val = float(row[column_indices["zu"]])
            else:
                x_val = float(row[column_indices["x"]])
                y_val = float(row[column_indices["y"]])
                z_val = float(row[column_indices["z"]])

            x_val = x_val % self.box_x if self.box_x else x_val
            y_val = y_val % self.box_y if self.box_y else y_val
            z_val = z_val % self.box_z if self.box_z else z_val

            symbols.append(element_symbol.capitalize())
            coords_list.append([x_val, y_val, z_val])

        self.symbols = symbols
        self.coords = np.array(coords_list, dtype=float)


def load_trajectory(fin, traj_format: str, box_size: np.ndarray) -> BaseTrajectory:
    if traj_format == "xyz":
        return XYZTrajectory(fin, box_size)
    if traj_format == "lammps":
        return LAMMPSTrajectory(fin, box_size)
    raise ValueError(f"Unsupported trajectory format: {traj_format}")
