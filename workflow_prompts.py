from __future__ import annotations

import numpy as np

from input_providers import InteractiveInputProvider
from core.trajectory_loader import load_trajectory


class WorkflowPrompts:
    def __init__(self, input_provider=None):
        self.input_provider = input_provider or InteractiveInputProvider()

    def get_input_provider(self, provider=None):
        return provider or self.input_provider

    def prompt_cell_vectors(self, traj_format, provider=None):
        input_provider = self.get_input_provider(provider)
        if traj_format != "xyz":
            return np.array([0.0, 0.0, 0.0])

        dimx = input_provider.ask_float("Enter cell vector length in X dimension (in Angstrom): ", minval=0.0)
        dimy = input_provider.ask_float("Enter cell vector length in Y dimension (in Angstrom): ", minval=0.0)
        dimz = input_provider.ask_float("Enter cell vector length in Z dimension (in Angstrom): ", minval=0.0)
        return np.array([dimx, dimy, dimz], dtype=float)

    def prepare_trajectory(self, traj_file, traj_format, provider=None):
        input_provider = self.get_input_provider(provider)
        cell_vectors = self.prompt_cell_vectors(traj_format, provider=input_provider)

        with open(traj_file, "r") as fin:
            traj = load_trajectory(fin, traj_format, cell_vectors)
            traj.read_frame()

            box_size = traj.box_size
            print(f"\n\nCell vectors: a = {box_size[0]}, b = {box_size[1]}, c = {box_size[2]}\n")

            self.process_compounds(traj, provider=input_provider)
            return traj

    def process_compounds(self, traj, provider=None):
        input_provider = self.get_input_provider(provider)
        frame_idx = 0
        traj.guess_molecules()

        while True:
            for i, compound in enumerate(traj.compounds.values()):
                count = len(compound.members)
                print(f"Compound {i + 1}: {compound.rep}, Number: {count}")

            for i, compound in enumerate(traj.compounds.values()):
                compound.average_bond_lengths()
                print(f"\nCompound {i + 1} Bond Length Matrix:")

                labels = list(compound.members[0].label_to_id.keys())
                size = len(labels)
                matrix = [["-  " for _ in range(size)] for _ in range(size)]
                label_to_index = {label: idx for idx, label in enumerate(labels)}

                for bond, length in compound.bond_lengths.items():
                    label1, label2 = bond.split()
                    idx1, idx2 = label_to_index[label1], label_to_index[label2]
                    matrix[idx1][idx2] = f"{length:.4f}"
                    matrix[idx2][idx1] = f"{length:.4f}"

                header = " ".join(f"{label:>8}" for label in labels)
                print(f"     {header}")
                for idx, label in enumerate(labels):
                    row = " ".join(f"{val:>8}" for val in matrix[idx])
                    print(f"{label:>5} {row}")

            should_draw_compounds = input_provider.ask_bool("Draw compounds to PDF?", False)
            if should_draw_compounds:
                for compound in traj.compounds.values():
                    compound.members[0].draw_graph(compound.comp_id + 1)

            is_keep_compounds = input_provider.ask_bool("Accept these molecules (y) or change something (n)", True)
            if is_keep_compounds:
                if frame_idx > 0:
                    traj.reset_frame_idx()
                break

            should_break = input_provider.ask_int(
                "Break bonds (1) or repeat molecule recognition at specific frame (2)?",
                1,
            )
            if should_break == 1:
                self.break_bonds(traj, provider=input_provider)
            else:
                frame_idx = self.skip_to_frame(traj, frame_idx, provider=input_provider)

            traj.guess_molecules()

    def break_bonds(self, traj, provider=None):
        input_provider = self.get_input_provider(provider)
        while True:
            print("\nCurrent Compounds:")
            for i, compound in enumerate(traj.compounds.values(), start=1):
                print(f"{i}: {compound.rep}")

            comp_id = input_provider.ask_int("Which compound to modify?", -1, "[done]") - 1
            if comp_id < 0:
                break

            try:
                compound = list(traj.compounds.values())[comp_id]
            except (ValueError, IndexError):
                print("Invalid compound number.")
                continue

            atom1 = input_provider.ask_str("First atom label to break bond (e.g., O1): ").strip()
            atom2 = input_provider.ask_str("Second atom label to break bond (e.g., H2): ").strip()

            try:
                compound.members[0].label_to_global_id[atom1]
                compound.members[0].label_to_global_id[atom2]
            except KeyError:
                print("Invalid atom label(s). Try again.")
                continue

            for molecule in compound.members:
                global_idx1 = molecule.label_to_global_id[atom1]
                global_idx2 = molecule.label_to_global_id[atom2]
                traj.forbidden_bonds.add((min(global_idx1, global_idx2), max(global_idx1, global_idx2)))

            print(f"Added forbidden bond between {atom1} and {atom2}.")

    def skip_to_frame(self, traj, frame_idx, provider=None):
        input_provider = self.get_input_provider(provider)
        target_frame = input_provider.ask_int("Skip to which frame?", 0)

        if target_frame > frame_idx:
            nframes = target_frame - frame_idx
        else:
            traj.reset_frame_idx()
            nframes = target_frame
            frame_idx = 0

        for _ in range(nframes):
            frame_idx += 1
            traj.read_frame()

        return frame_idx
