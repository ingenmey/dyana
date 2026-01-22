# analyses/charge_msd_analysis.py

import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

from analyses.base_analysis import BaseAnalysis
from utils import (
    prompt,
    prompt_int,
    prompt_float,
    label_matches,
)


class ChargeMSDAnalysis(BaseAnalysis):
    """
    Charge transfer & charge MSD analysis.

    This tracks "charges" associated with molecules based on the number
    of protons bound to each molecule vs a user-specified neutral proton count.
    Charges move when protons hop between molecules.

    IMPORTANT:
      - This analysis assumes a *static* molecular topology.
      - Protons and acceptor atoms are selected once based on the initial
        molecule recognition (interactive preprocessing step).
      - Per-frame molecule recognition (guess_molecules) is DISABLED.
      - We only update coordinates per frame.

    The output consists of:
      - charge_trajectories.dat : charge ID, frame, molecule, sign
      - charge_msd.dat          : MSD of +1 and -1 charges vs lag time
    """

    def setup(self):
        print("\n--- Charge Transfer Analysis ---")
        print("NOTE: This analysis assumes that transferable protons have been")
        print("      cut off from acceptor molecules during the initial")
        print("      compound recognition and that the bonding topology is")
        print("      static. Per-frame molecule recognition is disabled.\n")

        # === Protons ===
        proton_selection = self.compound_selection(
            role="proton",
            multi=True,
            prompt_text="Choose the compounds with transferable protons (comma-separated numbers): "
        )

        protons_per_compound = {}
        for idx, comp in proton_selection:
            labels = self.atom_selection(
                role="proton",
                compound=comp,
                prompt_text=(
                    "Enter transferable proton labels for {role} compound "
                    "{compound_num} ({compound_name}) (comma-separated): "
                ),
            )
            protons_per_compound[comp] = labels

        # === Acceptors ===
        acceptor_selection = self.compound_selection(
            role="acceptor",
            multi=True,
            prompt_text="Choose the compounds that may accept protons (comma-separated numbers): "
        )

        acceptors_per_compound = {}
        self.neutral_proton_counts = {}
        for idx, comp in acceptor_selection:
            labels = self.atom_selection(
                role="acceptor",
                compound=comp,
                prompt_text=(
                    "Enter acceptor atom labels for {role} compound "
                    "{compound_num} ({compound_name}) (comma-separated): "
                ),
            )
            acceptors_per_compound[comp] = labels

            self.neutral_proton_counts[comp.comp_id] = prompt_int(
                f"Enter the number of protons compound {comp.rep} binds in its neutral state: ",
                minval=0,
            )

        # === Parameters ===
        self.bond_cutoff = prompt_float(
            "Enter maximum proton-acceptor bond distance (in Å): ",
            1.2,
            minval=0.1,
        )
        self.max_lag = prompt_int(
            "Enter time correlation depth (in frames): ",
            1000,
            minval=2,
        )

        # Build proton and acceptor global index lists (static)
        protons = [
            idx
            for comp, labels in protons_per_compound.items()
            for mol in comp.members
            for label, idx in mol.label_to_global_id.items()
            if any(label_matches(lab, label) for lab in labels)
        ]

        acceptors = [
            idx
            for comp, labels in acceptors_per_compound.items()
            for mol in comp.members
            for label, idx in mol.label_to_global_id.items()
            if any(label_matches(lab, label) for lab in labels)
        ]

        if not protons or not acceptors:
            raise ValueError("No protons or acceptors found with the given labels.")

        self.proton_indices = list(protons)
        self.acceptor_indices = list(acceptors)

        # Map acceptor atoms (global indices) -> Molecule
        self.atom_to_mol = {idx: self.traj.atoms[idx].parent_molecule for idx in self.acceptor_indices}

        # mol -> list of acceptor atom global indices (for position centers)
        self.mol_acceptor_atoms = defaultdict(list)
        for comp, labels in acceptors_per_compound.items():
            for mol in comp.members:
                for label in labels:
                    # Here we use exact label equality for acceptor-center atoms
                    if label in mol.label_to_global_id:
                        gid = mol.label_to_global_id[label]
                        self.mol_acceptor_atoms[mol].append(gid)

        # === Charge tracking state ===
        self.next_cid = 1  # next free charge ID
        self.mol_charge_state = {}               # mol -> integer charge state
        self.mol_charge_ids = defaultdict(list)  # mol -> list of charge IDs
        self.charge_signs = {}                   # cid -> +1 / -1
        self.charge_history = defaultdict(list)  # cid -> list of (frame_idx, mol_id)

        # Unwrapped positions of charges
        self.charge_unwrapped = defaultdict(list)   # cid -> list of positions
        self.charge_prev_wrapped = {}               # cid -> last wrapped position

        # Proton mapping from previous frame
        self.prev_proton_to_acceptor = {}  # p_idx -> acc_idx

        # Initialization flag: we lazily initialize charge states on the first processed frame
        self.initialized = False

        self.box = np.asarray(self.traj.box_size, dtype=float)

    def setup_frame_loop(self):
        """
        Override BaseAnalysis.setup_frame_loop to disable per-frame molecule recognition.
        We only ask for start_frame / nframes / frame_stride.
        """
        print("\nPer-frame molecule recognition is disabled for charge transfer analysis.")
        print("Molecules & proton/acceptor definitions are taken from the initial\n"
              "compound recognition and assumed to be static.\n")

        self.update_compounds = False  # IMPORTANT: we rely on static topology
        self.start_frame = prompt_int(
            "In which trajectory frame to start processing the trajectory?",
            1,
            minval=1,
        )
        self.nframes = prompt_int(
            "How many trajectory frames to read (from this position on)?",
            -1,
            "all",
        )
        self.frame_stride = prompt_int(
            "Use every n-th read trajectory frame for the analysis:",
            1,
            minval=1,
        )

    def post_compound_update(self):
        """
        No-op; required by BaseAnalysis, but we never update compounds per-frame.
        """
        return True

    def _initialize_on_first_frame(self):
        """
        Initialize charge states and starting positions on the first processed frame.
        Called from process_frame() the first time it's invoked.
        """
        coords = self.traj.coords

        proton_to_acceptor, proton_count = get_proton_count(
            coords,
            self.traj.box_size,
            self.proton_indices,
            self.acceptor_indices,
            self.bond_cutoff,
            self.atom_to_mol,
        )

        (self.mol_charge_state,
         self.mol_charge_ids,
         self.charge_signs,
         self.next_cid) = initialize_charge_states(
            proton_count,
            self.neutral_proton_counts,
            self.next_cid,
        )

        # Initialize unwrapped positions using acceptor-atom centers per charged molecule
        for mol, cids in self.mol_charge_ids.items():
            atom_indices = self.mol_acceptor_atoms.get(mol, [])
            if not atom_indices:
                continue
            center = np.mean(coords[atom_indices], axis=0)
            for cid in cids:
                self.charge_prev_wrapped[cid] = center.copy()
                self.charge_unwrapped[cid].append(center.copy())
                self.charge_history[cid].append((self.frame_idx, mol.mol_id))

        self.prev_proton_to_acceptor = proton_to_acceptor
        self.initialized = True

    def process_frame(self):
        """
        For each processed frame:
          - On the first frame: initialize charges and positions, no transfers.
          - On subsequent frames:
              * update unwrapped positions for existing charges
              * compute proton-to-acceptor mapping & proton counts
              * compute new charge states
              * process proton hops and move/create/annihilate charges
              * record charge locations
        """
        coords = self.traj.coords

        # Lazy initialization on first processed frame
        if not self.initialized:
            self._initialize_on_first_frame()
            return

        # --- Update unwrapped charge positions using current molecule centers ---
        for mol, cids in self.mol_charge_ids.items():
            atom_indices = self.mol_acceptor_atoms.get(mol, [])
            if not atom_indices:
                continue
            center = np.mean(coords[atom_indices], axis=0)
            for cid in cids:
                if cid in self.charge_prev_wrapped:
                    delta = center - self.charge_prev_wrapped[cid]
                    # minimum image convention
                    delta -= self.traj.box_size * np.round(delta / self.traj.box_size)
                    unwrapped = self.charge_unwrapped[cid][-1] + delta
                else:
                    # Should normally only happen at creation; start at center
                    unwrapped = center.copy()

                self.charge_prev_wrapped[cid] = center.copy()
                self.charge_unwrapped[cid].append(unwrapped)

        # --- Recompute proton mapping and counts for this frame ---
        proton_to_acceptor, proton_count = get_proton_count(
            coords,
            self.traj.box_size,
            self.proton_indices,
            self.acceptor_indices,
            self.bond_cutoff,
            self.atom_to_mol,
        )

        # New charge state from current proton counts
        new_charge_state = {
            mol: proton_count[mol] - self.neutral_proton_counts[mol.comp_id]
            for mol in proton_count
        }

        # Ensure we have previous states for all molecules active in THIS frame
        # (molecules that just appeared get previous state 0)
        for mol in new_charge_state.keys():
            if mol not in self.mol_charge_state:
                self.mol_charge_state[mol] = 0

        # --- Process proton hops and move/create/annihilate charges ---
        for p_idx, new_acc in proton_to_acceptor.items():
            old_acc = self.prev_proton_to_acceptor.get(p_idx)
            if old_acc is None or old_acc == new_acc:
                continue

            donor = self.atom_to_mol[old_acc]
            acceptor = self.atom_to_mol[new_acc]

            donor_ids = self.mol_charge_ids[donor]
            acceptor_ids = self.mol_charge_ids[acceptor]
            donor_state = self.mol_charge_state.get(donor, 0)
            acceptor_state = self.mol_charge_state.get(acceptor, 0)

            # Possible scenarios (same as original code)
            if donor_state == -1 and acceptor_state == 0 and donor_ids:
                # Move negative charge from donor to acceptor
                acceptor_ids.append(donor_ids.pop(0))

            elif donor_state == 0 and acceptor_state == -1 and acceptor_ids:
                # Move negative charge from acceptor to donor
                donor_ids.append(acceptor_ids.pop(0))

            elif donor_state == 1 and acceptor_state == 0 and donor_ids:
                # Move positive charge from donor to acceptor
                acceptor_ids.append(donor_ids.pop(0))

            elif donor_state == 0 and acceptor_state == 1 and acceptor_ids:
                # Move positive charge from acceptor to donor
                donor_ids.append(acceptor_ids.pop(0))

            elif donor_state == 0 and acceptor_state == 0:
                # Create +1 on donor and -1 on acceptor
                self.charge_signs[self.next_cid] = -1
                acceptor_ids.append(self.next_cid)
                self.next_cid += 1

                self.charge_signs[self.next_cid] = +1
                donor_ids.append(self.next_cid)
                self.next_cid += 1

            elif abs(donor_state) == 1 and acceptor_state == -donor_state:
                # Annihilate one +1 and one -1 charge if both are present
                if donor_state == 1:
                    pos_ids = [cid for cid in donor_ids if self.charge_signs.get(cid, 0) == +1]
                    neg_ids = [cid for cid in acceptor_ids if self.charge_signs.get(cid, 0) == -1]
                    if pos_ids and neg_ids:
                        donor_ids.remove(pos_ids[0])
                        acceptor_ids.remove(neg_ids[0])
                else:
                    pos_ids = [cid for cid in acceptor_ids if self.charge_signs.get(cid, 0) == +1]
                    neg_ids = [cid for cid in donor_ids if self.charge_signs.get(cid, 0) == -1]
                    if pos_ids and neg_ids:
                        donor_ids.remove(neg_ids[0])
                        acceptor_ids.remove(pos_ids[0])

        # --- Record current charge locations (after moves) ---
        for mol, cids in self.mol_charge_ids.items():
            for cid in cids:
                self.charge_history[cid].append((self.frame_idx, mol.mol_id))

        # Prepare for next frame
        self.prev_proton_to_acceptor = proton_to_acceptor
        self.mol_charge_state = new_charge_state

        if self.processed_frames % 100 == 0 and self.processed_frames > 0:
            print(f"Processed {self.processed_frames} frames (current frame {self.frame_idx+1})")

    def postprocess(self):
        # === Output trajectories ===
        with open("charge_trajectories.dat", "w") as f:
            f.write("# cid  frame  mol  sign\n")
            for cid, records in self.charge_history.items():
                f.write(f"# Charge ID {cid}\n")
                sign = self.charge_signs.get(cid, 0)
                last_mol_id = None
                for frame, mol_id in records:
                    if mol_id != last_mol_id:
                        f.write(f"{cid:<6} {frame:<6} {mol_id:<6} {sign:+d}\n")
                        last_mol_id = mol_id

        print("\nCharge tracking data written to charge_trajectories.dat.")
        print("Computing MSD...")

        compute_charge_msd(
            self.charge_unwrapped,
            self.charge_signs,
            self.max_lag,
            "charge_msd.dat",
        )


def get_proton_count(coords, boxsize, protons, acceptors, bond_cutoff, atom_to_mol):
    """
    Map protons to their nearest acceptor atom (within cutoff), then count
    the number of protons bound to each acceptor molecule.
    """
    tree = cKDTree(coords[acceptors], boxsize=boxsize)
    proton_to_acceptor = {}

    for p_idx in protons:
        p_coord = coords[p_idx]
        neighbors = tree.query_ball_point(p_coord, bond_cutoff)
        if neighbors:
            nearest = min(
                neighbors,
                key=lambda j: np.linalg.norm(p_coord - coords[acceptors[j]]),
            )
            proton_to_acceptor[p_idx] = acceptors[nearest]

    proton_count = defaultdict(int)
    for acc in proton_to_acceptor.values():
        proton_count[atom_to_mol[acc]] += 1

    return proton_to_acceptor, proton_count


def initialize_charge_states(proton_count, neutral_proton_counts, next_cid):
    """
    Given proton counts per molecule and neutral proton counts per compound type,
    initialize:
      - mol_charge_state:  mol -> integer charge
      - mol_charge_ids:    mol -> list of charge IDs
      - charge_signs:      cid -> +1 / -1
    """
    mol_charge_state = {}
    mol_charge_ids = defaultdict(list)
    charge_signs = {}

    for mol, count in proton_count.items():
        neutral = neutral_proton_counts[mol.comp_id]
        state = count - neutral
        mol_charge_state[mol] = state

        if state < 0:
            for _ in range(-state):
                cid = next_cid
                mol_charge_ids[mol].append(cid)
                charge_signs[cid] = -1
                next_cid += 1
        elif state > 0:
            for _ in range(state):
                cid = next_cid
                mol_charge_ids[mol].append(cid)
                charge_signs[cid] = +1
                next_cid += 1

    return mol_charge_state, mol_charge_ids, charge_signs, next_cid


def compute_charge_msd(charge_unwrapped, charge_signs, max_lag=1000, output_file="charge_msd.dat"):
    """
    Compute mean-square displacement of +1 and -1 charges separately, based
    on their unwrapped trajectories.
    """
    msd_pos = np.zeros(max_lag)
    msd_neg = np.zeros(max_lag)
    count_pos = np.zeros(max_lag, dtype=int)
    count_neg = np.zeros(max_lag, dtype=int)

    for cid, positions in charge_unwrapped.items():
        sign = charge_signs.get(cid, 0)
        traj = np.array(positions)  # shape: (T, 3)
        n = len(traj)
        if n < 2:
            continue

        usable_lag = min(n, max_lag)

        for dt in range(1, usable_lag):
            displacements = traj[dt:] - traj[:-dt]
            disp_sq = np.sum(displacements**2, axis=1)

            if sign == +1:
                msd_pos[dt] += np.sum(disp_sq)
                count_pos[dt] += len(disp_sq)
            elif sign == -1:
                msd_neg[dt] += np.sum(disp_sq)
                count_neg[dt] += len(disp_sq)

    with open(output_file, "w") as f:
        f.write("# t(Δframes)  MSD(+1)  MSD(-1)\n")
        for t in range(1, max_lag):
            n_pos = count_pos[t]
            n_neg = count_neg[t]
            msd_p = msd_pos[t] / n_pos if n_pos > 0 else 0.0
            msd_m = msd_neg[t] / n_neg if n_neg > 0 else 0.0
            f.write(f"{t:<4} {msd_p:.6f} {msd_m:.6f}\n")

    print(f"MSD written to {output_file}")

