# core/trajectory_loader.py

import os
import json
import numpy as np
import itertools
import networkx as nx
from networkx.algorithms import isomorphism
from collections import Counter
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree

from atomic_properties import (
    elem_masses,
    elem_vdW,
    elem_covalent,
    elem_number,
    elem_color,
)
from utils import label_matches

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config_file_path = os.path.join(os.path.dirname(__file__), "../config.json")
with open(config_file_path, "r") as config_file:
    config = json.load(config_file)

EXCLUDED_ELEMENTS = set(config["EXCLUDED_ELEMENTS"])

# Optional tunables; fall back to current hard-coded values if not present
NEIGHBOR_SEARCH_SCALE = config.get("NEIGHBOR_SEARCH_SCALE", 1.164)
BOND_DISTANCE_SCALE = config.get("BOND_DISTANCE_SCALE", 1.4)


# ---------------------------------------------------------------------------
# Data model: Atom, Molecule, Compound
# ---------------------------------------------------------------------------

class Atom:
    """
    Canonical representation of an atom in the trajectory.

    There is exactly one Atom object per global atom index in BaseTrajectory.atoms.
    Molecules only hold references to these Atoms.

    Parameters
    ----------
    elem_number : int
        Atomic number.
    idx : int
        Global index of this atom in the trajectory (0..natoms-1).
    """

    def __init__(self, elem_number: int, idx: int):
        self.elem_number = elem_number
        self.idx = idx

        # These are updated every frame or whenever topology is rebuilt
        self.coord: np.ndarray | None = None
        self.parent_molecule: "Molecule | None" = None  # assigned in guess_molecules()


class Molecule:
    """
    Represents a molecule composed of multiple atoms.

    Parameters
    ----------
    mol_id : int
        Integer identifier for this molecule within a frame (0..n_molecules-1).
    atom_ids : list[int]
        Global atom indices belonging to this molecule.
    symbols : list[str]
        Element symbols corresponding to atom_ids (same length).
    """

    def __init__(self, mol_id: int, atom_ids: list[int], symbols: list[str]):
        self.mol_id = mol_id
        self.atom_ids = np.array(atom_ids, dtype=np.int32)
        self.symbols = symbols

        # Assigned when grouped into a Compound
        self.comp_id: int | None = None

        # Per-atom intrinsic properties (same order as atom_ids / symbols)
        self.atomic_masses = [elem_masses[s] for s in symbols]
        self.atomic_radii = [elem_vdW[s] for s in symbols]

        # Geometry / connectivity
        self.coords: np.ndarray | None = None   # shape (n_atoms, 3)
        self.bonds: list[tuple[int, int]] = []  # list of (local_idx_a, local_idx_b)
        self.bond_lengths_sq: list[float] = []  # squared bond distances

        self.com = np.zeros(3, dtype=float)     # center of mass
        self.atoms: list[Atom] = []             # Atom objects in local order

        # Label mappings
        self.id_to_label: dict[int, str] = {}        # local_id -> "O1", "H2", ...
        self.label_to_id: dict[str, int] = {}        # "O1" -> local_id
        self.label_to_global_id: dict[str, int] = {} # "O1" -> global atom index

        # Per-bond label-based lookup
        self.bond_lengths_table: dict[str, float] = {}  # "O1 H2" -> distance^2
        self.bond_lengths: dict[str, float] = {}        # "O1 H2" -> mean bond length (Å), filled in Compound

    # ------------------------------------------------------------------
    # Per-frame geometric updates
    # ------------------------------------------------------------------

    def update_coords(self, all_coords: np.ndarray, box_size: np.ndarray | None = None):
        """
        Update per-atom coordinates from the global coordinate array.

        Parameters
        ----------
        all_coords : np.ndarray
            Global coordinates array of shape (natoms, 3).
        box_size : np.ndarray or None
            Simulation box dimensions. If provided, coordinates are wrapped.
        """
        self.coords = all_coords[self.atom_ids]

        if box_size is not None:
            np.mod(self.coords, box_size, out=self.coords)

        # Update canonical Atom.coord for each Atom in this molecule
        for atom, position in zip(self.atoms, self.coords):
            atom.coord = position

    def update_center_of_mass(self, box_size: np.ndarray):
        """
        Update center of mass for this molecule using the minimum-image convention.
        """
        if self.coords is None or len(self.coords) == 0:
            self.com[:] = 0.0
            return

        reference_coord = self.coords[0]
        adjusted_coords = self.coords.copy()

        for i in range(1, len(adjusted_coords)):
            for dim in range(3):
                delta = adjusted_coords[i][dim] - reference_coord[dim]
                half_box = 0.5 * box_size[dim]
                if delta > half_box:
                    adjusted_coords[i][dim] -= box_size[dim]
                elif delta < -half_box:
                    adjusted_coords[i][dim] += box_size[dim]

        self.com = np.average(adjusted_coords, axis=0, weights=self.atomic_masses)
        self.com = np.mod(self.com, box_size)

    # ------------------------------------------------------------------
    # Extended connectivity & labelling (pure per-molecule logic)
    # ------------------------------------------------------------------

    def initialize_connectivity_labels(self):
        """
        Compute extended-connectivity-like labels for atoms in this molecule,
        assign labels such as O1, O2, H1, ...

        Populates:
          - id_to_label
          - label_to_id
          - label_to_global_id
          - bond_lengths_table
        """
        n_atoms = len(self.atom_ids)
        if n_atoms == 0:
            self.id_to_label.clear()
            self.label_to_id.clear()
            self.label_to_global_id.clear()
            self.bond_lengths_table.clear()
            return

        # Build adjacency list in local indices from bonds
        adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
        for local_a, local_b in self.bonds:
            adjacency[local_a].append(local_b)
            adjacency[local_b].append(local_a)

        # Extended connectivity algorithm (local, symbol-based)
        # Start with EC = Z * 10 + coordination
        ec_values = [
            elem_number[self.symbols[i]] * 10 + len(adjacency[i])
            for i in range(n_atoms)
        ]

        while True:
            unique_ec = set(ec_values)
            if len(unique_ec) == n_atoms:
                break

            trial_ec = []
            for i in range(n_atoms):
                neighbor_sum = sum(ec_values[neighbor] for neighbor in adjacency[i])
                trial_ec.append(neighbor_sum + 5 * ec_values[i])

            unique_trial_ec = set(trial_ec)
            # Stop if no further refinement is happening
            if len(unique_trial_ec) == len(unique_ec):
                ec_values = trial_ec
                break

            ec_values = trial_ec

        # Group by element symbol and assign labels in descending EC order
        self.id_to_label.clear()
        self.label_to_id.clear()
        self.label_to_global_id.clear()

        symbol_groups: dict[str, list[tuple[int, int]]] = {}
        for local_idx, ec_val in enumerate(ec_values):
            symbol = self.symbols[local_idx]
            symbol_groups.setdefault(symbol, []).append((ec_val, local_idx))

        for symbol, group in symbol_groups.items():
            # Sort by EC descending so "1" corresponds to highest EC
            group.sort(reverse=True, key=lambda pair: pair[0])
            for label_index, (_, local_idx) in enumerate(group, start=1):
                label = f"{symbol}{label_index}"
                self.id_to_label[local_idx] = label
                self.label_to_id[label] = local_idx
                self.label_to_global_id[label] = int(self.atom_ids[local_idx])

        # Build bond-length lookup by label pairs (squared length)
        self.bond_lengths_table.clear()
        for bond_idx, (local_a, local_b) in enumerate(self.bonds):
            label_a = self.id_to_label[local_a]
            label_b = self.id_to_label[local_b]
            bond_len_sq = self.bond_lengths_sq[bond_idx]

            self.bond_lengths_table[f"{label_a} {label_b}"] = bond_len_sq
            self.bond_lengths_table[f"{label_b} {label_a}"] = bond_len_sq

    def assign_labels_from_template(self,
                                    template_molecule: "Molecule",
                                    map_local_to_template: dict[int, int]):
        """
        Assign labels to this molecule by copying them from a template molecule.

        Parameters
        ----------
        template_molecule : Molecule
            Molecule that already has id_to_label / label_to_id filled.
        map_local_to_template : dict[int, int]
            Mapping from this molecule's local atom index -> template local index.
        """
        self.id_to_label.clear()
        self.label_to_id.clear()
        self.label_to_global_id.clear()

        # 1) Copy labels
        for local_idx, template_idx in map_local_to_template.items():
            label = template_molecule.id_to_label[template_idx]
            self.id_to_label[local_idx] = label
            self.label_to_id[label] = local_idx
            self.label_to_global_id[label] = int(self.atom_ids[local_idx])

        # 2) Build bond-length lookup by these labels
        self.bond_lengths_table = {}
        for bond_idx, (local_a, local_b) in enumerate(self.bonds):
            label_a = self.id_to_label[local_a]
            label_b = self.id_to_label[local_b]
            bond_len_sq = self.bond_lengths_sq[bond_idx]

            self.bond_lengths_table[f"{label_a} {label_b}"] = bond_len_sq
            self.bond_lengths_table[f"{label_b} {label_a}"] = bond_len_sq

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def draw_graph(self, compound_id_for_output: int = 0):
        """
        Draw the molecular graph and save it as a PDF named 'compound<id>.pdf'.
        """
        import matplotlib.pyplot as plt

        graph = nx.Graph()

        for label in self.label_to_id.keys():
            graph.add_node(label)

        for local_a, local_b in self.bonds:
            graph.add_edge(self.id_to_label[local_a], self.id_to_label[local_b])

        node_sizes = [
            elem_vdW.get(self.symbols[local_id], 1.0) * 2000
            for local_id in self.label_to_id.values()
        ]
        node_colors = [
            elem_color.get(self.symbols[local_id], "lightgray")
            for local_id in self.label_to_id.values()
        ]

        pos = nx.spring_layout(
            graph,
            k=0.2,
            iterations=300,
        )
        labels = {node: node for node in graph.nodes()}

        nx.draw(
            graph,
            pos,
            labels=labels,
            with_labels=True,
            node_size=node_sizes,
            node_color=node_colors,
            font_size=16,
            font_weight="bold",
            width=2.0,
        )

        plt.savefig(f"compound{compound_id_for_output}.pdf", format="pdf")
        plt.close()


class Compound:
    """
    Represents a chemical compound consisting of multiple molecules.

    Parameters
    ----------
    comp_id : int
        Integer id for this compound type.
    rep : str
        Representative formula string, e.g. "H2O" or "BF4".
    """

    def __init__(self, comp_id: int, rep: str):
        self.comp_id = comp_id
        self.rep = rep
        self.members: list[Molecule] = []
        self.atomic_masses: list[float] = []
        self.atomic_radii: list[float] = []

        # Optional: could store key if needed (formula, bond_types) for debugging
        self.key = None

    def update_centers_of_mass(self, box_size: np.ndarray):
        """
        Update center of mass for all molecules in this compound.
        """
        for molecule in self.members:
            molecule.update_center_of_mass(box_size)

    def get_coords(self, ref_label: str) -> np.ndarray:
        """
        Return an array of coordinates for all atoms in this compound that match
        the given label (or label pattern via label_matches).

        Parameters
        ----------
        ref_label : str
            Label or prefix-like pattern, e.g. "O1" or "O".

        Returns
        -------
        coords : np.ndarray
            Array of shape (n_matching_atoms, 3) containing coordinates.
        """
        if not self.members:
            return np.empty((0, 3), dtype=float)

        first_molecule = self.members[0]
        matching_labels = [
            label
            for label in first_molecule.label_to_id.keys()
            if label_matches(ref_label, label)
        ]

        coord_iterators = (
            (molecule.coords[molecule.label_to_id[label]] for label in matching_labels)
            for molecule in self.members
        )
        flattened_coords = itertools.chain.from_iterable(coord_iterators)
        return np.array(list(flattened_coords))

    def average_bond_lengths(self):
        """
        Compute average bond lengths (in Å) over all molecules of this compound
        for each labelled bond pair.
        """
        if not self.members:
            self.bond_lengths = {}
            return

        # Initialize accumulator from first molecule
        bond_sum_sq: dict[str, float] = {
            key: 0.0 for key in self.members[0].bond_lengths_table.keys()
        }

        for molecule in self.members:
            for bond_label, bond_len_sq in molecule.bond_lengths_table.items():
                bond_sum_sq[bond_label] += bond_len_sq

        n_molecules = len(self.members)
        self.bond_lengths = {
            key: np.sqrt(total_sq / n_molecules)
            for key, total_sq in bond_sum_sq.items()
        }


# ---------------------------------------------------------------------------
# Trajectory base class
# ---------------------------------------------------------------------------

class BaseTrajectory(ABC):
    """
    Base class for trajectories (XYZ, LAMMPS, ...).

    It stores per-frame atomic positions, and rebuilds molecular topology
    via guess_molecules(), which populates:

      - self.compounds: dict[compound_key -> Compound]
      - self.atoms: list[Atom], one per global atom index
    """

    def __init__(self, fin, box_size: np.ndarray):
        self.fin = fin
        self.box_size = np.array(box_size, dtype=float)
        self.half_box_size = self.box_size / 2.0

        self.dimx, self.dimy, self.dimz = self.box_size
        self.natoms = 0
        self.symbols: list[str] = []
        self.coords: np.ndarray | None = None

        # Canonical Atom list; index is the global atom index
        self.atoms: list[Atom] = []

        # Compounds: key=(formula, bond_types) -> Compound
        self.compounds: dict[tuple, Compound] = {}

        # Global forbidden bonds (pairs of global atom indices)
        self.forbidden_bonds: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Frame reading API
    # ------------------------------------------------------------------

    @abstractmethod
    def read_frame(self):
        """
        Read the next frame from the trajectory file and update
        self.natoms, self.symbols, self.coords, and (for LAMMPS) box_size.
        """
        raise NotImplementedError

    def reset_frame_idx(self):
        """
        Reset file pointer and reread the first frame.
        """
        self.fin.seek(0)
        self.read_frame()

    # ------------------------------------------------------------------
    # Per-frame geometric updates
    # ------------------------------------------------------------------

    def update_molecule_coords(self):
        """
        Update per-molecule coordinates and centers of mass based on
        the current global coords (self.coords).
        """
        if self.coords is None:
            return

        for compound in self.compounds.values():
            for molecule in compound.members:
                molecule.update_coords(self.coords)
            compound.update_centers_of_mass(self.box_size)

    # ------------------------------------------------------------------
    # Atom initialization & topology detection
    # ------------------------------------------------------------------

    def _compound_sort_key(self, compound_key: tuple) -> tuple:
        """
        Define a canonical sorting key for compound types.

        compound_key = (formula_str, bond_types_tuple, graph_hash)
        """
        formula_str, bond_types, graph_hash = compound_key
        return (formula_str, bond_types, graph_hash)


    def _initialize_atoms_for_frame(self):
        """
        Ensure self.atoms contains one Atom per global index for the current frame.
        Atoms are recreated each time guess_molecules() is called.

        This centralizes Atom creation in BaseTrajectory; Molecules only hold
        references to these Atoms.
        """
        self.atoms = []
        for global_index, symbol in enumerate(self.symbols):
            atomic_number = elem_number.get(symbol, 0)
            self.atoms.append(Atom(atomic_number, global_index))

    def are_connected(self, coord_i: np.ndarray, coord_j: np.ndarray,
                      cov_radius_i: float, cov_radius_j: float) -> float | bool:
        """
        Decide whether two atoms should be considered bonded.

        Uses a covalent radius criterion with minimum-image distance.

        Parameters
        ----------
        coord_i, coord_j : np.ndarray
            Coordinates of the two atoms.
        cov_radius_i, cov_radius_j : float
            Covalent radii of the two atoms.

        Returns
        -------
        distance_sq : float or False
            Squared distance between the atoms if they are within the
            bonding threshold, otherwise False.
        """
        delta = np.abs(coord_i - coord_j)
        delta = np.where(delta > self.half_box_size, np.abs(delta - self.box_size), delta)
        distance_sq = np.sum(delta ** 2)

        threshold_sq = ((cov_radius_i + cov_radius_j) * BOND_DISTANCE_SCALE) ** 2
        return distance_sq if distance_sq < threshold_sq else False

    def guess_molecules(self):
        """
        Rebuild molecular topology for the current frame.

        Steps:
          1. Create canonical Atom objects for this frame.
          2. Build a KDTree for coordinates.
          3. Identify molecules via connectivity search.
          4. Classify molecules into compounds by formula + bond types.
          5. For each molecule, link Atoms and compute connectivity labels.
        """
        if self.natoms == 0:
            self.atoms = []
            self.compounds = {}
            return

        self._initialize_atoms_for_frame()

        frame_symbols = self.symbols
        frame_coords = self.coords
        if frame_coords is None:
            raise RuntimeError("coords are not set before calling guess_molecules().")

        kdtree = cKDTree(frame_coords, boxsize=self.box_size)

        molecules = self._identify_molecules(frame_symbols, frame_coords, kdtree)
        self._classify_molecules(frame_symbols, molecules)

    def _identify_molecules(
        self,
        frame_symbols: list[str],
        frame_coords: np.ndarray,
        kdtree: cKDTree,
    ) -> list[Molecule]:
        """
        Identify molecules in the current frame by connectivity.

        Returns
        -------
        molecules : list[Molecule]
            Molecule objects with atom_ids, symbols, bonds, and bond_lengths_sq filled.
        """
        molecules: list[Molecule] = []
        visited_global_indices: set[int] = set()

        for seed_index in range(self.natoms):
            if seed_index in visited_global_indices:
                continue

            molecule_atom_indices: list[int] = [seed_index]
            stack: list[int] = [seed_index]
            visited_global_indices.add(seed_index)

            global_bonds: list[tuple[int, int]] = []
            bond_lengths_sq: list[float] = []

            # Only expand connectivity from non-excluded elements
            if frame_symbols[seed_index] not in EXCLUDED_ELEMENTS:
                while stack:
                    current_global_idx = stack.pop()
                    current_symbol = frame_symbols[current_global_idx]
                    cov_radius_current = elem_covalent.get(current_symbol, 0.0)

                    # Neighbor search radius based on vdW radius (heuristic)
                    search_radius = elem_vdW.get(current_symbol, 0.0) * NEIGHBOR_SEARCH_SCALE
                    neighbor_indices = sorted(
                        kdtree.query_ball_point(frame_coords[current_global_idx], r=search_radius)
                    )

                    for neighbor_global_idx in neighbor_indices:
                        neighbor_symbol = frame_symbols[neighbor_global_idx]
                        if neighbor_symbol in EXCLUDED_ELEMENTS:
                            continue

                        # Respect user-forbidden bonds
                        bond_pair = (
                            min(current_global_idx, neighbor_global_idx),
                            max(current_global_idx, neighbor_global_idx),
                        )
                        if bond_pair in self.forbidden_bonds:
                            continue

                        cov_radius_neighbor = elem_covalent.get(neighbor_symbol, 0.0)

                        neighbor_already_in_molecule = neighbor_global_idx in molecule_atom_indices
                        bond_already_recorded = (
                            (current_global_idx, neighbor_global_idx) in global_bonds
                            or (neighbor_global_idx, current_global_idx) in global_bonds
                        )

                        # We allow re-checking for atoms already in this molecule to add new bonds
                        should_consider_neighbor = (
                            neighbor_global_idx not in visited_global_indices
                            or (neighbor_already_in_molecule and not bond_already_recorded)
                        )

                        if not should_consider_neighbor:
                            continue

                        distance_sq = self.are_connected(
                            frame_coords[current_global_idx],
                            frame_coords[neighbor_global_idx],
                            cov_radius_current,
                            cov_radius_neighbor,
                        )
                        if not distance_sq:
                            continue

                        if neighbor_global_idx not in visited_global_indices:
                            stack.append(neighbor_global_idx)
                            molecule_atom_indices.append(neighbor_global_idx)
                            visited_global_indices.add(neighbor_global_idx)

                        if not bond_already_recorded:
                            global_bonds.append((current_global_idx, neighbor_global_idx))
                            bond_lengths_sq.append(distance_sq)

            # Build Molecule with remapped bonds in local indices
            mol_id = len(molecules)
            mol_symbols = [frame_symbols[gidx] for gidx in molecule_atom_indices]
            molecule = Molecule(mol_id, molecule_atom_indices, mol_symbols)

            global_to_local = {
                global_idx: local_idx
                for local_idx, global_idx in enumerate(molecule_atom_indices)
            }

            molecule.bonds = [
                (global_to_local[a_global], global_to_local[b_global])
                for (a_global, b_global) in global_bonds
            ]
            molecule.bond_lengths_sq = bond_lengths_sq

            molecules.append(molecule)

        return molecules

    def _classify_molecules(
        self,
        frame_symbols: list[str],
        molecules: list[Molecule],
    ):
        """
        Group molecules into compounds based on formula, bond types, and topology,
        and populate:

          - self.compounds[compound_key] = Compound
          - self.atoms[global_idx].parent_molecule
          - molecule.atoms (local Atom references)
          - connectivity labels (via template-based canonical labelling)
        """
        provisional_compounds: dict[tuple, Compound] = {}

        for molecule in molecules:
            # --- Stoichiometric formula from global symbols ---
            symbol_counts = Counter(frame_symbols[global_idx] for global_idx in molecule.atom_ids)
            sorted_symbol_counts = sorted(symbol_counts.items())
            formula_str = "".join(
                f"{element}{count}" if count > 1 else element
                for element, count in sorted_symbol_counts
            )

            # --- Bond-type multiset (sorted element pairs) ---
            bond_types: list[tuple[str, str]] = []
            for local_a, local_b in molecule.bonds:
                elem_a = molecule.symbols[local_a]
                elem_b = molecule.symbols[local_b]
                if elem_a > elem_b:
                    elem_a, elem_b = elem_b, elem_a
                bond_types.append((elem_a, elem_b))
            bond_types.sort()

            # --- Graph hash for topology ---
            G = nx.Graph()
            for local_idx, symbol in enumerate(molecule.symbols):
                G.add_node(local_idx, element=symbol)
            for local_a, local_b in molecule.bonds:
                G.add_edge(local_a, local_b)

            graph_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr="element")
            compound_key = (formula_str, tuple(bond_types), graph_hash)

            # --- Assign or create Compound for this key ---
            if compound_key not in provisional_compounds:
                compound = Compound(comp_id=len(provisional_compounds), rep=formula_str)
                compound.key = compound_key
                provisional_compounds[compound_key] = compound
            else:
                compound = provisional_compounds[compound_key]

            compound.members.append(molecule)

            # Link canonical Atom objects to this Molecule
            molecule.atoms = [self.atoms[gidx] for gidx in molecule.atom_ids]
            for atom in molecule.atoms:
                atom.parent_molecule = molecule

        # Canonicalize compound order (stable across frames)
        sorted_keys = sorted(provisional_compounds.keys(), key=self._compound_sort_key)

        canonical_compounds: dict[tuple, Compound] = {}
        for new_comp_id, compound_key in enumerate(sorted_keys):
            compound = provisional_compounds[compound_key]
            compound.comp_id = new_comp_id
            compound.key = compound_key

            # Update member molecules with canonical comp_id
            for molecule in compound.members:
                molecule.comp_id = new_comp_id

            # Assign consistent labels across all molecules of this compound
            self._assign_labels_for_compound(compound)

            canonical_compounds[compound_key] = compound

        self.compounds = canonical_compounds

    def _assign_labels_for_compound(self, compound: Compound):
        """
        For a given compound, choose the first molecule as a template,
        compute EC-based labels on it, and then use graph isomorphism
        to transfer those labels to all other member molecules.

        This ensures that for a given compound type, the same atom labels
        refer to the same topological positions across all molecules.
        """
        members = compound.members
        if not members:
            return

        # If there is only one molecule, just label it locally.
        if len(members) == 1:
            members[0].initialize_connectivity_labels()
            return

        # 1) Label the template molecule
        template = members[0]
        template.initialize_connectivity_labels()

        # Build template graph
        G_template = nx.Graph()
        for local_idx, symbol in enumerate(template.symbols):
            G_template.add_node(local_idx, element=symbol)
        for local_a, local_b in template.bonds:
            G_template.add_edge(local_a, local_b)

        # Node matching: element types must match
        node_match = lambda attrs_t, attrs_m: attrs_t["element"] == attrs_m["element"]

        # 2) Map all other molecules onto template
        for mol in members[1:]:
            if len(mol.atom_ids) != len(template.atom_ids):
                # Shouldn't happen if graph_hash is part of compound_key, but be defensive
                print(
                    f"Warning: molecule in compound {compound.rep} has different atom count "
                    "than template; using local labels."
                )
                mol.initialize_connectivity_labels()
                continue

            G_mol = nx.Graph()
            for local_idx, symbol in enumerate(mol.symbols):
                G_mol.add_node(local_idx, element=symbol)
            for local_a, local_b in mol.bonds:
                G_mol.add_edge(local_a, local_b)

            gm = isomorphism.GraphMatcher(G_template, G_mol, node_match=node_match)
            if not gm.is_isomorphic():
                # Again, should not happen if we hashed by topology, but better to not crash.
                print(
                    f"Warning: molecule topology not isomorphic to template for compound {compound.rep}; "
                    "using local labels."
                )
                mol.initialize_connectivity_labels()
                continue

            # GraphMatcher mapping: template_idx -> mol_idx
            mapping_template_to_mol = next(gm.isomorphisms_iter())
            # We want mol_local_idx -> template_local_idx
            map_mol_to_template = {
                mol_idx: template_idx
                for template_idx, mol_idx in mapping_template_to_mol.items()
            }

            mol.assign_labels_from_template(template, map_mol_to_template)

# ---------------------------------------------------------------------------
# Concrete trajectory classes
# ---------------------------------------------------------------------------

class XYZTrajectory(BaseTrajectory):
    """
    Simple XYZ trajectory reader.
    """

    def read_frame(self):
        """
        Read a single XYZ frame: natoms, comment, then natoms lines of (symbol, x, y, z).
        Box size is taken from the constructor and used only to wrap positions.
        """
        natoms_line = self.fin.readline()
        if not natoms_line:
            raise ValueError("End of file reached while reading XYZ trajectory.")

        self.natoms = int(natoms_line.strip())
        _comment = self.fin.readline().rstrip()

        symbols: list[str] = []
        coords_list: list[list[float]] = []

        for _ in range(self.natoms):
            parts = self.fin.readline().split()
            if len(parts) < 4:
                raise ValueError("Malformed XYZ line (expected at least 4 columns).")

            symbol_str, x_str, y_str, z_str = parts[:4]
            x_val, y_val, z_val = map(float, (x_str, y_str, z_str))

            x_val = x_val % self.dimx if self.dimx else x_val
            y_val = y_val % self.dimy if self.dimy else y_val
            z_val = z_val % self.dimz if self.dimz else z_val

            symbols.append(symbol_str.capitalize())
            coords_list.append([x_val, y_val, z_val])

        self.symbols = symbols
        self.coords = np.array(coords_list, dtype=float)


class LAMMPSTrajectory(BaseTrajectory):
    """
    LAMMPS dump trajectory reader with support for unwrapped or wrapped coordinates.

    Expects an "ITEM:" based dump with BOX BOUNDS and ATOMS including
    'element' and either (xu, yu, zu) or (x, y, z).
    """

    def read_frame(self):
        """
        Read the next LAMMPS dump frame and update box size and coordinates.
        """

        # --- TIMESTEP ---
        line = self.fin.readline().strip()
        while line and not line.startswith("ITEM: TIMESTEP"):
            line = self.fin.readline().strip()
        if not line:
            raise ValueError("End of file reached before finding TIMESTEP")
        self.timestep = int(self.fin.readline().strip())

        # --- NUMBER OF ATOMS ---
        line = self.fin.readline().strip()
        while line and not line.startswith("ITEM: NUMBER OF ATOMS"):
            line = self.fin.readline().strip()
        if not line:
            raise ValueError("End of file reached before finding NUMBER OF ATOMS")
        self.natoms = int(self.fin.readline().strip())

        # --- BOX BOUNDS ---
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
        self.dimx, self.dimy, self.dimz = self.box_size

        # --- ATOMS header ---
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
                "Trajectory file missing required coordinate columns "
                "(xu,yu,zu) or (x,y,z)."
            )

        atom_rows = [self.fin.readline().strip().split() for _ in range(self.natoms)]

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

            x_val = x_val % self.dimx if self.dimx else x_val
            y_val = y_val % self.dimy if self.dimy else y_val
            z_val = z_val % self.dimz if self.dimz else z_val

            symbols.append(element_symbol.capitalize())
            coords_list.append([x_val, y_val, z_val])

        self.symbols = symbols
        self.coords = np.array(coords_list, dtype=float)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_trajectory(fin, traj_format: str, box_size: np.ndarray) -> BaseTrajectory:
    """
    Factory function returning the appropriate trajectory reader.

    Parameters
    ----------
    fin :
        Open file-like object for the trajectory.
    traj_format : {"xyz", "lammps"}
        Supported trajectory format.
    box_size : np.ndarray
        Box dimensions for XYZ trajectories (ignored for LAMMPS, which reads box from file).

    Returns
    -------
    traj : BaseTrajectory
        Instance of XYZTrajectory or LAMMPSTrajectory.

    Raises
    ------
    ValueError
        If the trajectory format is not supported.
    """
    if traj_format == "xyz":
        return XYZTrajectory(fin, box_size)
    elif traj_format == "lammps":
        return LAMMPSTrajectory(fin, box_size)
    else:
        raise ValueError(f"Unsupported trajectory format: {traj_format}")

