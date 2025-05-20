# analyses/charge_transfer.py

import os
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from utils import prompt, prompt_int, prompt_float, label_matches

def charge_transfer(traj):
    print("\n--- Charge Transfer Analysis ---")

    # === User input: compound and atom selections ===
    proton_comp_indices = [int(x.strip()) - 1 for x in prompt("Enter compound numbers with transferable protons (comma-separated): ").split(',')]
    proton_compounds = [list(traj.compounds.values())[i] for i in proton_comp_indices]
    protons_per_compound = {
        comp: [l.strip() for l in prompt(f"Enter transferable proton labels for compound {i+1} ({comp.rep}) (comma-separated): ").split(',')]
        for i, comp in zip(proton_comp_indices, proton_compounds)
    }

    acceptor_comp_indices = [int(x.strip()) - 1 for x in prompt("Enter compound numbers that may accept protons (comma-separated): ").split(',')]
    acceptor_compounds = [list(traj.compounds.values())[i] for i in acceptor_comp_indices]
    acceptors_per_compound = {
        comp: [l.strip() for l in prompt(f"Enter acceptor atom labels for compound {i+1} ({comp.rep}) (comma-separated): ").split(',')]
        for i, comp in zip(acceptor_comp_indices, acceptor_compounds)
    }

    neutral_proton_counts = {
        comp.comp_id: prompt_int(f"Enter the number of protons compound {comp.rep} binds in its neutral state: ", minval=0)
        for comp in acceptor_compounds
    }

    bond_cutoff = prompt_float("Enter maximum proton-acceptor bond distance (in Å): ", 1.2, minval=0.1)
    max_lag = prompt_int("Enter time correlation depth (in frames): ", 1000, minval=2)
    start_frame = prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)

    # === Build index lists ===
    protons = [idx for comp, labels in protons_per_compound.items()
               for mol in comp.members
               for label, idx in mol.label_to_global_id.items()
               if any(label_matches(lab, label) for lab in labels)]

    acceptors = [idx for comp, labels in acceptors_per_compound.items()
                 for mol in comp.members
                 for label, idx in mol.label_to_global_id.items()
                 if any(label_matches(lab, label) for lab in labels)]

    atom_to_mol = {idx: traj.atoms[idx].parent_molecule for idx in acceptors}

    # === Initialization ===
    frame_idx = 0
    next_cid = 1
    processed_frames = 0
    mol_charge_state = {}
    mol_charge_ids = defaultdict(list)
    charge_history = defaultdict(list)
    charge_unwrapped = defaultdict(list)  # cid → [unwrapped positions]
    charge_prev_wrapped = {}             # cid → previous wrapped coord


    # mol → list of acceptor atom indices (global!)
    mol_acceptor_atoms = defaultdict(list)
    for comp, labels in acceptors_per_compound.items():
        for mol in comp.members:
            for label in labels:
                if label in mol.label_to_global_id:
                    idx = mol.label_to_global_id[label]
                    mol_acceptor_atoms[mol].append(idx)


    traj.update_molecule_coords()
    coords = traj.coords

    proton_to_acceptor, proton_count = get_proton_count(coords, traj.box_size, protons, acceptors, bond_cutoff, atom_to_mol)
    mol_charge_state, mol_charge_ids, charge_signs, next_cid = initialize_charge_states(proton_count, neutral_proton_counts, next_cid)

    for mol, cids in mol_charge_ids.items():
        for cid in cids:
            charge_history[cid].append((frame_idx, mol.mol_id))

    prev_proton_to_acceptor = proton_to_acceptor

    # === Skip to start frame ===
    if start_frame > 1:
        print(f"Skipping to frame {start_frame}...")
        while frame_idx < start_frame - 1:
            traj.read_frame()
            frame_idx += 1

    # === Frame loop ===
    while nframes != 0:
        try:
            traj.update_molecule_coords()
            coords = traj.coords

#            for mol, cids in mol_charge_ids.items():
#                atom_indices = mol_acceptor_atoms.get(mol, [])
#                if not atom_indices:
#                    continue
#                center = np.mean(traj.coords[atom_indices], axis=0)
#                for cid in cids:
#                    if cid in charge_prev_wrapped:
#                        delta = center - charge_prev_wrapped[cid]
#                        # Apply minimum image convention (PBC correction)
#                        delta -= traj.box_size * np.round(delta / traj.box_size)
#                        unwrapped = charge_unwrapped[cid][-1] + delta
#                    else:
#                        unwrapped = center.copy()
#
#                    charge_prev_wrapped[cid] = center.copy()
#                    charge_unwrapped[cid].append(unwrapped)

            proton_to_acceptor, proton_count = get_proton_count(coords, traj.box_size, protons, acceptors, bond_cutoff, atom_to_mol)

            # Update charge states
            new_charge_state = {mol: proton_count[mol] - neutral_proton_counts[mol.comp_id] for mol in proton_count}
            mol_charge_state.update({mol: mol_charge_state.get(mol, 0) for mol in new_charge_state})

            # Process proton hops
            for p_idx, new_acc in proton_to_acceptor.items():
                old_acc = prev_proton_to_acceptor.get(p_idx)
                if old_acc is None or old_acc == new_acc:
                    continue

                donor = atom_to_mol[old_acc]
                acceptor = atom_to_mol[new_acc]
                donor_ids = mol_charge_ids[donor]
                acceptor_ids = mol_charge_ids[acceptor]
                donor_state = mol_charge_state[donor]
                acceptor_state = mol_charge_state[acceptor]

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
                    charge_signs[next_cid] = -1
                    acceptor_ids.append(next_cid)
                    next_cid += 1

                    charge_signs[next_cid] = +1
                    donor_ids.append(next_cid)
                    next_cid += 1
                    frame_name = f"charge_creation_frame{frame_idx}_cid{next_cid}.xyz"
                    save_xyz_snapshot(traj, filename=frame_name)
                elif abs(donor_state) == 1 and acceptor_state == -donor_state:
                    # Annihilate one +1 and one -1 charge if both are present
                    if donor_state == 1:
                        pos_ids = [cid for cid in donor_ids if charge_signs[cid] == +1]
                        neg_ids = [cid for cid in acceptor_ids if charge_signs[cid] == -1]
                        if pos_ids and neg_ids:
                            donor_ids.remove(pos_ids[0])
                            acceptor_ids.remove(neg_ids[0])
                    else:
                        pos_ids = [cid for cid in acceptor_ids if charge_signs[cid] == +1]
                        neg_ids = [cid for cid in donor_ids if charge_signs[cid] == -1]
                        if pos_ids and neg_ids:
                            donor_ids.remove(neg_ids[0])
                            acceptor_ids.remove(pos_ids[0])

            # Record current charge locations
            for mol, cids in mol_charge_ids.items():
                for cid in cids:
                    charge_history[cid].append((frame_idx, mol.mol_id))

            # Prepare next iteration
            prev_proton_to_acceptor = proton_to_acceptor
            mol_charge_state = new_charge_state
            processed_frames += 1
#            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx + 1})", end="")

            for _ in range(frame_stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()

        except ValueError:
            print("\nReached end of trajectory.")
            break
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user.")
            break

    # === Output ===
    with open("charge_trajectories.dat", "w") as f:
        f.write("# cid  frame  mol  sign\n")
        for cid, records in charge_history.items():
            f.write(f"# Charge ID {cid}\n")
            sign = charge_signs.get(cid, 0)
            last_mol_id = None
            for frame, mol_id in records:
                if mol_id != last_mol_id:
                    f.write(f"{cid:<6} {frame:<6} {mol_id:<6} {sign:+d}\n")
                    last_mol_id = mol_id


    print("\nCharge tracking data written to charge_trajectories.dat.")

#    print(f"Computing MSD...")

#    compute_charge_msd(charge_unwrapped, charge_signs, max_lag, "charge_msd.dat")

def save_xyz_snapshot(traj, filename="debug.xyz"):
    with open(filename, "w") as f:
        f.write(f"{traj.natoms}\n")
        f.write(f"Frame snapshot\n")
        for symbol, coord in zip(traj.symbols, traj.coords):
            f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


def compute_charge_msd(charge_unwrapped, charge_signs, max_lag=1000, output_file="charge_msd.dat"):
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

    # Write result
    with open(output_file, "w") as f:
        f.write("# t  MSD(+1)  MSD(-1)\n")
        for t in range(1, max_lag):
            n_pos = count_pos[t]
            n_neg = count_neg[t]
            msd_p = msd_pos[t] / n_pos if n_pos > 0 else 0.0
            msd_m = msd_neg[t] / n_neg if n_neg > 0 else 0.0
            f.write(f"{t:<4} {msd_p:.6f} {msd_m:.6f}\n")

    print(f"MSD written to {output_file}")


#    # Create directory for XYZ files
#    os.makedirs("charge_xyz", exist_ok=True)
#
#    for cid, positions in charge_unwrapped.items():
#        if len(positions) < 1:
#            continue
#
#        sign = charge_signs.get(cid, 0)
#        filename = f"charge_xyz/charge_{cid}_{'pos' if sign == 1 else 'neg'}.xyz"
#
#        with open(filename, "w") as f:
#            for t, pos in enumerate(positions):
#                f.write("1\n")
#                f.write(f"Charge ID {cid}, Frame {t}, Sign {sign:+d}\n")
#                f.write(f"X {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")


def get_proton_count(coords, boxsize, protons, acceptors, bond_cutoff, atom_to_mol):
    tree = cKDTree(coords[acceptors], boxsize=boxsize)
    proton_to_acceptor = {}
    for p_idx in protons:
        p_coord = coords[p_idx]
        neighbors = tree.query_ball_point(p_coord, bond_cutoff)
        if neighbors:
            nearest = min(neighbors, key=lambda j: np.linalg.norm(p_coord - coords[acceptors[j]]))
            proton_to_acceptor[p_idx] = acceptors[nearest]

    proton_count = defaultdict(int)
    for acc in proton_to_acceptor.values():
        proton_count[atom_to_mol[acc]] += 1

    return proton_to_acceptor, proton_count


def initialize_charge_states(proton_count, neutral_proton_counts, next_cid):
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

