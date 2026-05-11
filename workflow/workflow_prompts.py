from __future__ import annotations

import numpy as np

from core.trajectory_loader import load_trajectory
from io_support.console import console
from io_support.input_providers import InteractiveInputProvider
from workflow.prepared_setup import (
    apply_prepared_setup,
    build_prepared_setup,
    load_prepared_setup,
    save_prepared_setup,
    validate_prepared_setup,
)


def draw_compound_graph(compound_type, compound_id_for_output: int = 0):
    """Render one compound-type graph to a PDF file."""
    import matplotlib.pyplot as plt
    import networkx as nx

    from core.atomic_properties import elem_color, elem_vdW

    graph = nx.Graph()
    for label in compound_type.canonical_labels:
        graph.add_node(label)
    for local_a, local_b in compound_type.local_bonds:
        graph.add_edge(compound_type.canonical_labels[local_a], compound_type.canonical_labels[local_b])

    node_sizes = [
        elem_vdW.get(element, 1.0) * 2000
        for element in compound_type.local_elements
    ]
    node_colors = [
        elem_color.get(element, "lightgray")
        for element in compound_type.local_elements
    ]

    pos = nx.spring_layout(graph, k=0.2, iterations=300)
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


class WorkflowPrompts:
    """Interactive workflow for preparing trajectory and topology state."""

    def __init__(self, input_provider=None):
        self.input_provider = input_provider or InteractiveInputProvider()

    def prompt_cell_vectors(self, traj_format, provider=None):
        input_provider = provider or self.input_provider
        if traj_format != "xyz":
            return np.array([0.0, 0.0, 0.0])

        box_x = input_provider.ask_float("Enter cell vector length in X dimension (in Angstrom): ", minval=0.0)
        box_y = input_provider.ask_float("Enter cell vector length in Y dimension (in Angstrom): ", minval=0.0)
        box_z = input_provider.ask_float("Enter cell vector length in Z dimension (in Angstrom): ", minval=0.0)
        return np.array([box_x, box_y, box_z], dtype=float)

    def prepare_trajectory(self, traj_file, traj_format, provider=None, save_prepared_setup_path=None):
        input_provider = provider or self.input_provider
        console.section("Topology Setup")
        cell_vectors = self.prompt_cell_vectors(traj_format, provider=input_provider)
        traj = self._load_initial_trajectory(traj_file, traj_format, cell_vectors)

        box_size = traj.box_size
        console.key_value(
            "Cell vectors",
            f"a={box_size[0]:.4f}  b={box_size[1]:.4f}  c={box_size[2]:.4f}",
        )

        self.process_compounds(traj, provider=input_provider)
        if save_prepared_setup_path is not None:
            prepared_setup = build_prepared_setup(traj, traj_file, traj_format, cell_vectors)
            save_prepared_setup(save_prepared_setup_path, prepared_setup)
            console.success(f"Saved prepared setup to {save_prepared_setup_path}")
        return traj

    def prepare_trajectory_from_setup(self, traj_file, prepared_setup_path):
        console.section("Topology Setup")
        console.info(f"Loading prepared setup from {prepared_setup_path}")
        prepared_setup = load_prepared_setup(prepared_setup_path)
        recipe = prepared_setup.recipe
        traj_format = recipe["trajectory_format"]
        cell_vectors = np.array(recipe["cell_vectors"], dtype=float)

        traj = self._load_initial_trajectory(traj_file, traj_format, cell_vectors)
        apply_prepared_setup(traj, prepared_setup)
        traj.rebuild_topology()
        validate_prepared_setup(traj, prepared_setup)
        console.key_value(
            "Cell vectors",
            f"a={traj.box_size[0]:.4f}  b={traj.box_size[1]:.4f}  c={traj.box_size[2]:.4f}",
        )
        console.success("Prepared setup matches the current trajectory topology.")
        return traj

    def process_compounds(self, traj, provider=None):
        input_provider = provider or self.input_provider
        frame_idx = 0
        traj.rebuild_topology()

        while True:
            compound_types = list(traj.topology_frame.get_compound_types())
            self._print_compound_summary(traj, compound_types)

            for i, compound_type in enumerate(compound_types):
                console.plain("")
                console.plain(f"Compound {i + 1} bond length matrix:")
                self._print_bond_length_matrix(traj, compound_type)

            should_draw_compounds = input_provider.ask_bool("Draw compounds to PDF?", False)
            if should_draw_compounds:
                for compound_type in compound_types:
                    draw_compound_graph(compound_type, compound_type.type_id + 1)

            is_keep_compounds = input_provider.ask_bool("Accept these molecules (y) or change something (n)", True)
            if is_keep_compounds:
                if frame_idx > 0:
                    traj.rewind_to_first_frame()
                console.success("Accepted compound list.")
                break

            should_break = input_provider.ask_int(
                "Break bonds (1) or repeat molecule recognition at specific frame (2)?",
                1,
            )
            if should_break == 1:
                self.break_bonds(traj, provider=input_provider)
            else:
                frame_idx = self.skip_to_frame(traj, frame_idx, provider=input_provider)

            traj.rebuild_topology()

    def break_bonds(self, traj, provider=None):
        input_provider = provider or self.input_provider
        while True:
            console.plain("")
            console.plain("Current compounds:")
            compound_types = list(traj.topology_frame.get_compound_types())
            for i, compound_type in enumerate(compound_types, start=1):
                count = traj.topology_frame.get_molecule_count(compound_type)
                console.key_value(str(i), f"{compound_type.formula} (Number: {count})", indent=2)

            comp_id = input_provider.ask_int("Which compound to modify?", -1, "[done]") - 1
            if comp_id < 0:
                break

            try:
                compound_type = compound_types[comp_id]
            except (ValueError, IndexError):
                console.warn("Invalid compound number.")
                continue

            atom1 = input_provider.ask_str("First atom label to break bond (e.g., O1): ").strip()
            atom2 = input_provider.ask_str("Second atom label to break bond (e.g., H2): ").strip()

            if atom1 not in compound_type.label_to_local_index or atom2 not in compound_type.label_to_local_index:
                console.warn("Invalid atom label(s). Try again.")
                continue

            local_idx1 = compound_type.label_to_local_index[atom1]
            local_idx2 = compound_type.label_to_local_index[atom2]
            molecule_atom_ids = traj.topology_frame.get_molecule_atom_ids(compound_type)
            for atom_ids in molecule_atom_ids:
                global_idx1 = int(atom_ids[local_idx1])
                global_idx2 = int(atom_ids[local_idx2])
                traj.forbidden_bonds.add((min(global_idx1, global_idx2), max(global_idx1, global_idx2)))

            console.success(f"Added forbidden bond between {atom1} and {atom2}.")

    def skip_to_frame(self, traj, frame_idx, provider=None):
        input_provider = provider or self.input_provider
        target_frame = input_provider.ask_int("Skip to which frame?", 0)

        if target_frame > frame_idx:
            nframes = target_frame - frame_idx
        else:
            traj.rewind_to_first_frame()
            nframes = target_frame
            frame_idx = 0

        for _ in range(nframes):
            frame_idx += 1
            traj.read_frame()

        return frame_idx

    def _load_initial_trajectory(self, traj_file, traj_format, cell_vectors):
        fin = open(traj_file, "r")
        traj = load_trajectory(fin, traj_format, cell_vectors)
        traj.read_frame()
        return traj

    def _print_compound_summary(self, traj, compound_types):
        console.plain("Detected compound types:")
        for i, compound_type in enumerate(compound_types):
            count = traj.topology_frame.get_molecule_count(compound_type)
            console.key_value(str(i + 1), f"{compound_type.formula} (Number: {count})", indent=2)

    def _print_bond_length_matrix(self, traj, compound_type):
        bond_lengths = traj.topology_frame.get_average_bond_lengths(
            compound_type,
            traj.coords,
            traj.box_size,
        )

        labels = list(compound_type.canonical_labels)
        size = len(labels)
        matrix = [["-  " for _ in range(size)] for _ in range(size)]
        label_to_index = {label: idx for idx, label in enumerate(labels)}

        for bond, length in bond_lengths.items():
            label1, label2 = bond.split()
            idx1, idx2 = label_to_index[label1], label_to_index[label2]
            matrix[idx1][idx2] = f"{length:.4f}"
            matrix[idx2][idx1] = f"{length:.4f}"

        header = " ".join(f"{label:>8}" for label in labels)
        console.plain(f"     {header}")
        for idx, label in enumerate(labels):
            row = " ".join(f"{val:>8}" for val in matrix[idx])
            console.plain(f"{label:>5} {row}")
