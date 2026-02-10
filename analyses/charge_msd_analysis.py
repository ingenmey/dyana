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

        self.allow_compound_update = False

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
            "Enter default maximum proton-acceptor bond distance (in Å): ",
            1.2,
            minval=0.1,
        )
        self.max_lag = prompt_int(
            "Enter time correlation depth (in frames): ",
            1000,
            minval=2,
        )

        # Optional: per-acceptor-label cutoffs
        use_individual = prompt(
            "Use individual distance cutoffs for different acceptor labels? [y/N]: "
        ).strip().lower()
        per_label_cutoff = {}
        if use_individual in ("y", "yes"):
            print("\nEnter cutoffs per acceptor label pattern (press Enter to accept the default).")
            for comp, labels in acceptors_per_compound.items():
                for lab in labels:
                    if lab in per_label_cutoff:
                        # already asked (same pattern reused for another compound)
                        continue
                    cutoff = prompt_float(
                        f"  Cutoff distance for acceptor label pattern '{lab}' (Å): ",
                        self.bond_cutoff,
                        minval=0.1,
                    )
                    per_label_cutoff[lab] = cutoff
        self.per_label_acceptor_cutoff = per_label_cutoff

        # === Build proton global index list (static) ===
        protons = [
            idx
            for comp, labels in protons_per_compound.items()
            for mol in comp.members
            for label, idx in mol.label_to_global_id.items()
            if any(label_matches(lab, label) for lab in labels)
        ]
        if not protons:
            raise ValueError("No protons found with the given labels.")
        self.proton_indices = list(protons)

        # === Build acceptor global index list and per-atom cutoffs (static) ===
        self.acceptor_indices = []
        self.acceptor_cutoffs = {}           # acc_global_idx -> cutoff (Å)
        self.atom_to_mol = {}                # acc_global_idx -> Molecule
        self.mol_acceptor_atoms = defaultdict(list)  # Molecule -> list of acc_global_idx

        for comp, labels in acceptors_per_compound.items():
            for mol in comp.members:
                for atom_label, gid in mol.label_to_global_id.items():
                    matched_label = None
                    for user_label in labels:
                        if label_matches(user_label, atom_label):
                            matched_label = user_label
                            break
                    if matched_label is None:
                        continue

                    self.acceptor_indices.append(gid)
                    self.atom_to_mol[gid] = mol
                    self.mol_acceptor_atoms[mol].append(gid)

                    cutoff = self.per_label_acceptor_cutoff.get(matched_label, self.bond_cutoff)
                    self.acceptor_cutoffs[gid] = cutoff

        if not self.acceptor_indices:
            raise ValueError("No acceptor atoms found with the given labels.")

        # === Charge tracking state ===
        self.next_cid = 1  # next free charge ID
        self.mol_charge_state = {}               # mol -> integer charge state
        self.mol_charge_ids = defaultdict(list)  # mol -> list of charge IDs
        self.charge_signs = {}                   # cid -> +1 / -1
        self.charge_history = defaultdict(list)  # cid -> list of (frame_idx, mol_id)

        # Unwrapped positions of charges
        self.charge_unwrapped = defaultdict(list)   # cid -> list of positions
        self.charge_prev_wrapped = {}               # cid -> last wrapped position

        # Proton mapping from previous frame (filled in on first frame)
        self.prev_proton_to_acceptor = {}  # p_idx -> acc_idx

        # Initialization flag: we lazily initialize charge states on the first processed frame
        self.initialized = False

        self.box = np.asarray(self.traj.box_size, dtype=float)

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
            self.acceptor_cutoffs,
            self.atom_to_mol,
        )

        # Build proton counts for *all* acceptor molecules (including those with zero bound protons)
        all_mols = set(self.atom_to_mol.values())
        full_proton_count = {
            mol: proton_count.get(mol, 0)
            for mol in all_mols
        }

        (self.mol_charge_state,
         self.mol_charge_ids,
         self.charge_signs,
         self.next_cid) = initialize_charge_states(
            full_proton_count,
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

        # Save mapping for hop detection in the next frame
        self.prev_proton_to_acceptor = dict(proton_to_acceptor)
        self.initialized = True

    def process_frame(self):
        """
        For each processed frame:
          - On the first frame: initialize charges and positions, no transfers.
          - On subsequent frames:
              * update unwrapped positions for existing charges
              * compute proton-to-acceptor mapping & proton counts
              * for each proton hop, update integer charge states and move/create/
                annihilate charge IDs in a local, per-hop manner
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
            self.acceptor_cutoffs,
            self.atom_to_mol,
        )

        # Working copy of charge states that we update per hop
        local_charge_state = dict(self.mol_charge_state)

        # --- Process proton hops with generalized per-hop logic ---
        for p_idx, new_acc in proton_to_acceptor.items():
            old_acc = self.prev_proton_to_acceptor.get(p_idx)
            if old_acc is None or old_acc == new_acc:
                continue

            donor = self.atom_to_mol[old_acc]
            acceptor = self.atom_to_mol[new_acc]
            if donor is acceptor:
                continue  # shouldn't happen, but safe-guard

            donor_ids = self.mol_charge_ids[donor]
            acceptor_ids = self.mol_charge_ids[acceptor]

            qd = local_charge_state.get(donor, 0)
            qa = local_charge_state.get(acceptor, 0)

            # Case A: move +1 charge donor -> acceptor
            if qd > 0 and qa >= 0:
                cid = next((c for c in donor_ids if self.charge_signs.get(c, 0) == +1), None)
                if cid is None:
                    raise RuntimeError(f"No +1 charge ID on positively charged donor {donor}")
                donor_ids.remove(cid)
                acceptor_ids.append(cid)

            # Case B: move -1 charge acceptor -> donor
            elif qd <= 0 and qa < 0:
                cid = next((c for c in acceptor_ids if self.charge_signs.get(c, 0) == -1), None)
                if cid is None:
                    raise RuntimeError(f"No -1 charge ID on negatively charged acceptor {acceptor}")
                acceptor_ids.remove(cid)
                donor_ids.append(cid)

            # Case C: annihilate one (+1, -1) pair
            elif qd >= 1 and qa <= -1:
                pos_id = next((c for c in donor_ids if self.charge_signs.get(c, 0) == +1), None)
                neg_id = next((c for c in acceptor_ids if self.charge_signs.get(c, 0) == -1), None)
                if pos_id is None or neg_id is None:
                    raise RuntimeError(
                        f"Missing charge IDs for annihilation on donor {donor}, acceptor {acceptor}"
                    )
                donor_ids.remove(pos_id)
                acceptor_ids.remove(neg_id)

            # Case D: create a (-1, +1) pair
            elif qd <= 0 and qa >= 0:
                # new -1 on donor
                cid_neg = self.next_cid
                self.next_cid += 1
                self.charge_signs[cid_neg] = -1
                donor_ids.append(cid_neg)

                # new +1 on acceptor
                cid_pos = self.next_cid
                self.next_cid += 1
                self.charge_signs[cid_pos] = +1
                acceptor_ids.append(cid_pos)

            else:
                # In principle unreachable because the four cases cover all (qd, qa)
                raise RuntimeError(
                    f"Unclassified charge hop state: donor={qd}, acceptor={qa}"
                )

            # Update integer charge states
            local_charge_state[donor] = qd - 1
            local_charge_state[acceptor] = qa + 1

        # Optional sanity check: compare to what we'd get from proton_count
        # (can be commented out if too slow)
        # all_mols = set(self.atom_to_mol.values())
        # for mol in all_mols:
        #     expected = proton_count.get(mol, 0) - self.neutral_proton_counts[mol.comp_id]
        #     if local_charge_state.get(mol, 0) != expected:
        #         raise RuntimeError(
        #             f"Charge mismatch on {mol}: "
        #             f"state={local_charge_state.get(mol, 0)}, expected={expected}"
        #         )

        # --- Record current charge locations (after updates) ---
        for mol, cids in self.mol_charge_ids.items():
            for cid in cids:
                self.charge_history[cid].append((self.frame_idx, mol.mol_id))

        # Prepare for next frame
        self.mol_charge_state = local_charge_state
        self.prev_proton_to_acceptor = dict(proton_to_acceptor)

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


def get_proton_count(coords, boxsize, protons, acceptors, acceptor_cutoffs, atom_to_mol):
    """
    Map each proton to a bound acceptor atom and count protons per acceptor molecule.

    Behaviour:
      - Every proton is *always* assigned to some acceptor (no free protons).
      - If per-acceptor cutoffs are given, the "closest" acceptor is chosen
        by first restricting to acceptors within their individual cutoff
        distances; if none satisfy this, we fall back to the globally
        closest acceptor (ignoring cutoffs).
    """
    if not acceptors:
        return {}, defaultdict(int)

    acc_coords = coords[acceptors]
    tree = cKDTree(acc_coords, boxsize=boxsize)

    # Per-acceptor cutoff array aligned with `acceptors`
    acc_cut_array = np.array(
        [acceptor_cutoffs.get(idx, np.inf) for idx in acceptors],
        dtype=float,
    )
    finite_mask = np.isfinite(acc_cut_array)
    if np.any(finite_mask):
        max_cut = float(np.max(acc_cut_array[finite_mask]))
    else:
        max_cut = np.inf

    proton_to_acceptor = {}
    proton_count = defaultdict(int)

    for p_idx in protons:
        p_coord = coords[p_idx]

        # Candidate neighbours within the largest cutoff
        if np.isfinite(max_cut):
            neighbor_ids = tree.query_ball_point(p_coord, max_cut)
        else:
            # No finite cutoffs defined: consider all acceptors
            neighbor_ids = list(range(len(acceptors)))

        best_acc = None
        best_dist2 = None

        # First try to respect individual cutoffs
        for j in neighbor_ids:
            acc_idx = acceptors[j]
            cutoff = acc_cut_array[j]
            if not np.isfinite(cutoff):
                continue
            diff = p_coord - coords[acc_idx]
            d2 = np.dot(diff, diff)
            if d2 <= cutoff * cutoff and (best_dist2 is None or d2 < best_dist2):
                best_dist2 = d2
                best_acc = acc_idx

        if best_acc is None:
            # No acceptor within its own cutoff: fall back to globally closest acceptor
            _, j = tree.query(p_coord, k=1)
            best_acc = acceptors[j]

        proton_to_acceptor[p_idx] = best_acc
        proton_count[atom_to_mol[best_acc]] += 1

    return proton_to_acceptor, proton_count


def initialize_charge_states(proton_count, neutral_proton_counts, next_cid):
    """
    Given proton counts per molecule and neutral proton counts per compound type,
    initialize:
      - mol_charge_state:  mol -> integer charge
      - mol_charge_ids:    mol -> list of charge IDs
      - charge_signs:      cid -> +1 / -1

    This works for arbitrary integer charge states (not just -1, 0, +1).
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

