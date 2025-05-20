# analyses/proton_coupling.py

import numpy as np
from collections import defaultdict, Counter, deque
from scipy.spatial import cKDTree
from utils import prompt, prompt_int, prompt_float, prompt_yn, label_matches

def proton_coupling(traj):
    print("\n--- Proton Coupling Correlation Function Analysis ---")

    # === Protons ===
    proton_comp_indices = prompt("Enter compound numbers with transferable protons (comma-separated): ").strip()
    proton_comp_indices = [int(x.strip()) - 1 for x in proton_comp_indices.split(',') if x.strip()]
    proton_compounds = [list(traj.compounds.values())[idx] for idx in proton_comp_indices]

    protons_per_compound = {}
    for idx, comp in zip(proton_comp_indices, proton_compounds):
        labels = prompt(f"Enter transferable proton labels for compound {idx+1} ({comp.rep}) (comma-separated): ").strip()
        labels = [l.strip() for l in labels.split(',') if l.strip()]
        protons_per_compound[comp] = labels

    # === Acceptors ===
    acceptor_comp_indices = prompt("Enter compound numbers that may accept protons (comma-separated): ").strip()
    acceptor_comp_indices = [int(x.strip()) - 1 for x in acceptor_comp_indices.split(',') if x.strip()]
    acceptor_compounds = [list(traj.compounds.values())[idx] for idx in acceptor_comp_indices]

    acceptors_per_compound = {}
    for idx, comp in zip(acceptor_comp_indices, acceptor_compounds):
        labels = prompt(f"Enter acceptor atom labels for compound {idx+1} ({comp.rep}) (comma-separated): ").strip()
        labels = [l.strip() for l in labels.split(',') if l.strip()]
        acceptors_per_compound[comp] = labels

    is_molecule_mode = prompt_yn("Track proton transfers by molecule [y] or atom [n]?", True)

    protons = []
    acceptors = []

    for comp, labels in protons_per_compound.items():
        for mol in comp.members:
            for label, idx in mol.label_to_global_id.items():
                if any(label_matches(lab, label) for lab in labels):
                    protons.append(idx)

    for comp, labels in acceptors_per_compound.items():
        for mol in comp.members:
            for label, idx in mol.label_to_global_id.items():
                if any(label_matches(lab, label) for lab in labels):
                    acceptors.append(idx)

    if not protons or not acceptors:
        print("No protons or acceptors found.")
        return []

    atom_to_mol = {}
    seen_molecules = {}
    acceptor_molecules = []

    for i,idx in enumerate(acceptors):  # `acceptors` contains global atom indices
        atom = traj.atoms[idx]
        mol = atom.parent_molecule

        if mol not in seen_molecules:
            matrix_col = len(acceptor_molecules)
            acceptor_molecules.append(mol)
            seen_molecules[mol] = matrix_col  # register new column index

        atom_to_mol[i] = seen_molecules[mol]


    # === Map proton/acceptor indices to their positions in traj.coords ===
    proton_indices = np.array(protons, dtype=int)
    acceptor_indices = np.array(acceptors, dtype=int)
    n_protons = len(protons)
    n_acceptors = len(seen_molecules) if is_molecule_mode else len(acceptors) # Isn't really necessary, as len(acceptors) is always larger

    # === Parameters ===
    bond_cutoff = prompt_float("Enter maximum proton-acceptor bond distance (in Å): ", 1.2, minval=0.1)
    dwell_threshold = prompt_int("Enter minimum dwell time on an acceptor (in frames): ", 2, minval=0)
    max_chain_gaps = prompt("Enter one or more max chain gaps (in frames, comma-separated): ", 100).strip()
    max_chain_gaps = [int(x.strip()) for x in max_chain_gaps.split(',') if x.strip().isdigit()]

    transfer_events = []  # (frame_idx, donor_acc, acceptor_acc)
    last_acceptor = np.full(n_protons, fill_value=-1, dtype=int)
    dwell_counters = np.zeros(n_protons, dtype=int)

    # === Prepare for trajectory iteration ===
    bond_matrices = []  # List of binary bond matrices per frame

    start_frame = prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)

    frame_idx = 0
    processed_frames = 0

    if start_frame > 1:
        print(f"Skipping forward to frame {start_frame}.")
        while frame_idx < start_frame - 1:
            traj.read_frame()
            frame_idx += 1

    while nframes != 0:
        try:
            traj.update_molecule_coords()
            coords = traj.coords


            # Build KDTree over acceptor coordinates
            acceptor_coords = coords[acceptor_indices]
            tree = cKDTree(acceptor_coords, boxsize=traj.box_size)

#            print(len(acceptor_coords), len(acceptor_indices), len(atom_to_mol))
#            input()
            # Initialize binary matrix
            bond_matrix = np.zeros((n_protons, n_acceptors), dtype=bool)

            for i, p_idx in enumerate(proton_indices):
                p_coord = coords[p_idx]
                neighbors = tree.query_ball_point(p_coord, bond_cutoff)

                # Choose nearest acceptor if multiple are within cutoff
                if neighbors:
                    dists = np.linalg.norm(acceptor_coords[neighbors] - p_coord, axis=1)
                    nearest = neighbors[np.argmin(dists)]
                    if is_molecule_mode:
                        acceptor_id = atom_to_mol[nearest]
                    else:
                        acceptor_id = nearest
                    bond_matrix[i, acceptor_id] = 1  # Proton i bonded to acceptor nearest

            for p_idx in range(n_protons):
                bonded = np.where(bond_matrix[p_idx])[0]
                if bonded.size == 0:
                    continue
                new_acceptor = bonded[0]
                if last_acceptor[p_idx] == -1:
                    last_acceptor[p_idx] = new_acceptor
                    dwell_counters[p_idx] = 1
                elif new_acceptor == last_acceptor[p_idx]:
                    dwell_counters[p_idx] += 1
                else:
                    if dwell_counters[p_idx] >= dwell_threshold:
                        donor = last_acceptor[p_idx]
                        transfer_events.append((frame_idx, donor, new_acceptor))
                    last_acceptor[p_idx] = new_acceptor
                    dwell_counters[p_idx] = 1


            processed_frames += 1

#            if (processed_frames % 10 == 0):
#                print(".", end="", flush=True)
#                if (processed_frames % 1000 == 0):
#                    print(f"\n{frame_idx+1}")

            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

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

    # === Post-processing: Track transfers and build chains ===
    print("\nAnalyzing transfer events and building chains...")

    acceptor_transfer_map = defaultdict(list)
    for frame, donor, acceptor in transfer_events:
        acceptor_transfer_map[donor].append((frame, acceptor))

    # Walk through chains
    chain_counts = np.zeros(20, dtype=int)  # Track up to 20-chain depth by default

    # Walk through chains with frame gap restriction
    for gap in max_chain_gaps:
        print(f"\nAnalyzing chains with max_chain_gap = {gap}...")
        acceptor_transfer_map = defaultdict(list)
        for frame, donor, acceptor in transfer_events:
            acceptor_transfer_map[donor].append((frame, acceptor))

        chain_counts = np.zeros(20, dtype=int)

        for frame, donor, acceptor in transfer_events:
            visited = set()
            current = acceptor
            depth = 1
            visited.add(donor)
            current_frame = frame

            while depth < len(chain_counts):
                next_links = [
                    (f, a) for (f, a) in acceptor_transfer_map.get(current, [])
                    if a not in visited and 0 < f - current_frame <= gap
                ]
                if not next_links:
                    break

                # Choose earliest valid transfer
                next_frame, next_acc = min(next_links, key=lambda t: t[0])
                visited.add(current)
                current = next_acc
                current_frame = next_frame
                depth += 1

            chain_counts[depth - 1] += 1

        # Normalize to get C(n)
        total_chains = np.sum(chain_counts)
        Cn = chain_counts / total_chains if total_chains > 0 else chain_counts

        fname = f"proton_chains_gap{gap}.dat"
        with open(fname, "w") as f:
            f.write("# n   C(n)\n")
            for i, val in enumerate(Cn):
                if val > 0:
                    f.write(f"{i+1} {val:.6f}\n")

        print(f"Wrote C(n) to {fname}")

