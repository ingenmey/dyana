# core/trajectory_loader.py

import os
import json
import numpy as np
import itertools
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
from atomic_properties import elem_masses, elem_vdW, elem_covalent, elem_number, elem_color  # Import atomic properties
from utils import label_matches

# Load the configuration
config_file_path = os.path.join(os.path.dirname(__file__), "../config.json")
with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

EXCLUDED_ELEMENTS = config["EXCLUDED_ELEMENTS"]

class Atom:
    def __init__(self, elem_number, idx):
        self.elem_number = elem_number
        self.idx = idx
#        self.comp_id = None
        self.bonds = []
        self.ec = 0
        self.tec = 0
        self.coord = None
        self.parent_molecule = None

class Molecule:
    """
    Represents a molecule composed of multiple atoms.
    """
    def __init__(self, mol_id: int, atom_ids: list, symbols: list):
        self.mol_id = mol_id
        self.atom_ids = atom_ids
        self.symbols = symbols
        self.comp_id = None
        self.atomic_masses = [elem_masses[s] for s in symbols]
        self.atomic_radii = [elem_vdW[s] for s in symbols]
        self.coords = []
        self.bonds = []  # List of (atom_id1, atom_id2) tuples
        self.bond_lengths_sq = []
        self.com = [0, 0, 0]
        self.bonds_internal = []
        self.atoms = []
        self.id_to_label = {}
        self.label_to_id = {}
        self.label_to_global_id = {}

    def update_coords(self, coords: np.ndarray, box_size: np.ndarray=None):
        self.coords = coords[self.atom_ids]
        if (box_size):
            self.coords = np.mod(self.coords, box_size)  # Ensure coordinates are within the box

        # Update atom coords
        for i, atom in enumerate(self.atoms):
            atom.coord = self.coords[i]

    def update_com(self, box_size):
        base_coord = self.coords[0]
        adjusted_coords = self.coords.copy()

        for i in range(1, len(adjusted_coords)):
            for dim in range(3):
                delta = adjusted_coords[i][dim] - base_coord[dim]
                if delta > 0.5 * box_size[dim]:
                    adjusted_coords[i][dim] -= box_size[dim]
                elif delta < -0.5 * box_size[dim]:
                    adjusted_coords[i][dim] += box_size[dim]

        self.com = np.average(adjusted_coords, axis=0, weights=self.atomic_masses)
        self.com = np.mod(self.com, box_size)  # Ensure COM is within the box

    def convert_bonds_to_internal_ids(self):
        self.internal_ids = {global_id: i for i, global_id in enumerate(self.atom_ids)}
        self.bonds_internal = [(self.internal_ids[a], self.internal_ids[b]) for a, b in self.bonds]

    def calculate_extended_connectivity(self):
        # Create atoms with initial EC values
        self.convert_bonds_to_internal_ids()
        atoms = []

        for s,idx in zip(self.symbols, self.atom_ids):
            atoms.append(Atom(elem_number[s], idx))

        for a, b in self.bonds_internal:
            atoms[a].bonds.append(b)
            atoms[b].bonds.append(a)

        for atom in atoms:
            atom.ec = atom.elem_number * 10 + len(atom.bonds)
            atom.parent_molecule = self

        iteration = 0
        while True:
            ec_values = set(atom.ec for atom in atoms)
            if len(ec_values) == len(atoms):
                break

            # Calculate trial EC (TEC)
            for atom in atoms:
                atom.tec = sum(atoms[neighbor].ec for neighbor in atom.bonds) + 5 * atom.ec

            tec_values = set(atom.tec for atom in atoms)
            if len(tec_values) == len(ec_values):
                break

            # Update EC
            for atom in atoms:
                atom.ec = atom.tec
            iteration += 1

        # Assign labels based on EC values and element type
        symbol_groups = {}
        for i, atom in enumerate(atoms):
            symbol = self.symbols[i]
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append((atom.ec, i))

        for symbol, group in symbol_groups.items():
            group.sort(reverse=True, key=lambda x: x[0])
            for index, (_, internal_id) in enumerate(group, 1):
                self.id_to_label[internal_id] = f"{symbol}{index}"
                self.label_to_id[f"{symbol}{index}"] = internal_id
                self.label_to_global_id[f"{symbol}{index}"] = self.atom_ids[internal_id]

        self.bond_lengths_table = {}
        for i, (a, b) in enumerate(self.bonds_internal):
            label_a = self.id_to_label[a]
            label_b = self.id_to_label[b]
            bond_length_sq = self.bond_lengths_sq[i]
            self.bond_lengths_table[f"{label_a} {label_b}"] = bond_length_sq
            self.bond_lengths_table[f"{label_b} {label_a}"] = bond_length_sq

        self.atoms = atoms

    def draw_graph(self, comp_id=0):
        import networkx as nx
        import matplotlib.pyplot as plt

        g = nx.Graph()

        for label in self.label_to_id.keys():
            g.add_node(label)

        for a, b in self.bonds_internal:
            g.add_edge(self.id_to_label[a], self.id_to_label[b])

        node_sizes = [elem_vdW.get(self.symbols[id], 1) * 2000 for id in self.id_to_label.keys()]
        node_colors = [elem_color.get(self.symbols[id], 'lightgray') for id in self.id_to_label.keys()]
#        pos = nx.spring_layout(g)
        pos = nx.spring_layout(
        g,
        k=0.2,  # Optimal distance between nodes (default is ~0.1)
        iterations=300  # Number of iterations for layout refinement
        )
        labels = {node: node for node in g.nodes()}
        nx.draw(g, pos, labels=labels, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=16, font_weight="bold", width=2.0)
        plt.savefig(f"compound{comp_id}.pdf", format='pdf')
        plt.close()

class Compound:
    """
    Represents a chemical compound consisting of multiple molecules.
    """
    def __init__(self, comp_id: int, rep: str):
        self.comp_id = comp_id
        self.rep = rep
        self.members = []
        self.atomic_masses = []
        self.atomic_radii = []

    def update_coms(self, box_size):
        for mol in self.members:
            mol.update_com(box_size)

    def get_coords(self, ref_label):
        # Step 1: Precompute matching labels using the first molecule
        matching_labels = [
            label for label in self.members[0].label_to_id.keys()
            if label_matches(ref_label, label)
        ]

        # Step 2: Generator that yields coordinates directly
        coord_iterators = (
            (molecule.coords[molecule.label_to_id[label]] for label in matching_labels)
            for molecule in self.members
        )

        flattened_coords = itertools.chain.from_iterable(coord_iterators)

        return np.array(list(flattened_coords))

    def average_bond_lengths(self):
        sum_dict = {key: 0 for key in self.members[0].bond_lengths_table.keys()}

        for mol in self.members:
            for bond, bond_length in mol.bond_lengths_table.items():
                sum_dict[bond] += bond_length

        self.bond_lengths = {key: np.sqrt(sum_dict[key] / len(self.members)) for key in sum_dict}


class BaseTrajectory(ABC):
    def __init__(self, fin, box_size):
        self.fin = fin
        self.box_size = box_size
        self.dimx, self.dimy, self.dimz = box_size
        self.natoms = 0
        self.symbols = []
        self.coords = []
        self.atoms = {}
        self.compounds = {}

    @abstractmethod
    def read_frame(self):
        pass

    # TODO: Use pre-calculated bond distance list
    def are_connected(self, coord1: np.ndarray, coord2: np.ndarray, rad1: float, rad2: float) -> bool:
        dist = np.abs(coord1 - coord2)
        dist = np.where(dist > np.array([self.dimx, self.dimy, self.dimz]) / 2, np.abs(dist - np.array([self.dimx, self.dimy, self.dimz])), dist)
        distance_sq = np.sum(dist**2)
        return distance_sq if distance_sq < ((rad1 + rad2) * 1.4)**2 else False

    def guess_molecules(self):
        symbols = self.symbols
        coords = self.coords

        kdtree = cKDTree(coords, boxsize=self.box_size)
        molecules = self._identify_molecules(symbols, coords, kdtree)
        self._classify_molecules(symbols, molecules)

    def _identify_molecules(self, symbols: list, coords: np.ndarray, kdtree: cKDTree) -> list:
        molecules = []
        visited = set()

        for i in range(self.natoms):
            if i in visited:
                continue

            molecule = [i]
            stack = [i]
            visited.add(i)
            bonds = []
            bond_lengths_sq = []

            if symbols[i] not in EXCLUDED_ELEMENTS:
                while stack:
                    current_atom = stack.pop()
                    rad1 = elem_covalent.get(symbols[current_atom], 0.0)

                    neighbors = sorted(kdtree.query_ball_point(coords[current_atom], r=elem_vdW.get(symbols[current_atom], 0.0) ))
                    #neighbors = kdtree.query_ball_point(coords[current_atom], r=elem_vdW.get(symbols[current_atom], 0.0))

                    for neighbor in neighbors:
                        rad2 = elem_covalent.get(symbols[neighbor], 0.0)
                        if symbols[neighbor] not in EXCLUDED_ELEMENTS and (neighbor not in visited or (neighbor in molecule and (current_atom, neighbor) not in bonds)):
                            if r_sq := self.are_connected(coords[current_atom], coords[neighbor], rad1, rad2):
                                # Ensure atoms are only added once to `molecule`
                                if neighbor not in visited:
                                    stack.append(neighbor)
                                    molecule.append(neighbor)
                                    visited.add(neighbor)

                                # Ensure bonds are always recorded, even for cycles
                                if (current_atom, neighbor) not in bonds and (neighbor, current_atom) not in bonds:
                                    bonds.append((current_atom, neighbor))
                                    bond_lengths_sq.append(r_sq)


            mol = Molecule(len(molecules), molecule, [symbols[i] for i in molecule])
            mol.bonds = bonds
            mol.bond_lengths_sq = bond_lengths_sq
            mol.calculate_extended_connectivity()
            molecules.append(mol)
        return molecules

    def _classify_molecules(self, symbols: list, molecules: list):
        compounds = {}

        for mol in molecules:
            symbol_count = {}
            for atom_index in mol.atom_ids:
                symbol = symbols[atom_index]
                symbol_count[symbol] = symbol_count.get(symbol, 0) + 1

            sorted_symbols = sorted(symbol_count.items())
            form_str = ''.join([f"{symbol}{count}" if count > 1 else symbol for symbol, count in sorted_symbols])

            bond_str = tuple(sorted((symbols[a], symbols[b]) if symbols[a] <= symbols[b] else (symbols[b], symbols[a]) for a, b in mol.bonds))
            compound_key = (form_str, bond_str)

            if compound_key not in compounds:
                compounds[compound_key] = Compound(len(compounds), form_str)

            mol.comp_id = compounds[compound_key].comp_id
            compounds[compound_key].members.append(mol)

            for atom_index, atom in zip(mol.atom_ids, mol.atoms):
                self.atoms[atom_index] = atom

        self.compounds = compounds


    def update_molecule_coords(self):
        for comp in self.compounds.values():
            for mol in comp.members:
                mol.update_coords(self.coords)

class XYZTrajectory(BaseTrajectory):
    def read_frame(self):
        self.natoms = int(self.fin.readline().strip())
        self.fin.readline().rstrip()
        symbols = []
        coords = []

        for _ in range(self.natoms):
            data = self.fin.readline().split()
            symbol, x, y, z = data[:4]
            x, y, z = map(float, [x, y, z])
            x = x % self.dimx if self.dimx else x
            y = y % self.dimy if self.dimy else y
            z = z % self.dimz if self.dimz else z
            symbols.append(symbol.capitalize())
            coords.append([x, y, z])

        self.symbols = symbols
        self.coords = np.array(coords)

class LAMMPSTrajectory(BaseTrajectory):
    def read_frame(self):
        line = self.fin.readline().strip()
        while not line.startswith("ITEM: TIMESTEP"):
            if line == '':  # End of file check
                raise ValueError("End of file reached before finding TIMESTEP")
            line = self.fin.readline().strip()
        self.timestep = int(self.fin.readline().strip())

        line = self.fin.readline().strip()
        while not line.startswith("ITEM: NUMBER OF ATOMS"):
            if line == '':  # End of file check
                raise ValueError("End of file reached before finding NUMBER OF ATOMS")
            line = self.fin.readline().strip()
        self.natoms = int(self.fin.readline().strip())

        line = self.fin.readline().strip()
        while not line.startswith("ITEM: BOX BOUNDS"):
            if line == '':  # End of file check
                raise ValueError("End of file reached before finding BOX BOUNDS")
            line = self.fin.readline().strip()

        self.box_size = []
        for _ in range(3):
            dim = list(map(float, self.fin.readline().strip().split()))
            length = dim[1] - dim[0]
            self.box_size.append(length)
        self.box_size = np.array(self.box_size)
        self.dimx, self.dimy, self.dimz = self.box_size

        line = self.fin.readline().strip()
        while not line.startswith("ITEM: ATOMS"):
            if line == '':  # End of file check
                raise ValueError("End of file reached before finding ATOMS")
            line = self.fin.readline().strip()

        columns = line.split()[2:]
        atom_data = []
        for _ in range(self.natoms):
            atom_data.append(self.fin.readline().strip().split())

        column_indices = {name: idx for idx, name in enumerate(columns)}
        if not {'xu', 'yu', 'zu'}.issubset(column_indices.keys()) and not {'x', 'y', 'z'}.issubset(column_indices.keys()):
            raise ValueError("Trajectory file missing required coordinate columns")

        symbols = []
        coords = []
        for data in atom_data:
            symbol = data[column_indices['element']]
            if 'xu' in column_indices:
                x = float(data[column_indices['xu']])
                y = float(data[column_indices['yu']])
                z = float(data[column_indices['zu']])
            else:
                x = float(data[column_indices['x']])
                y = float(data[column_indices['y']])
                z = float(data[column_indices['z']])
            x = x % self.dimx if self.dimx else x
            y = y % self.dimy if self.dimy else y
            z = z % self.dimz if self.dimz else z
            symbols.append(symbol.capitalize())
            coords.append([x, y, z])

        self.symbols = symbols
        self.coords = np.array(coords)

def load_trajectory(fin, traj_format, box_size):
    if traj_format == 'xyz':
        return XYZTrajectory(fin, box_size)
    elif traj_format == 'lammps':
        return LAMMPSTrajectory(fin, box_size)
    else:
        raise ValueError(f"Unsupported trajectory format: {traj_format}")



