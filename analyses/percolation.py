# analyses/percolation.py

import numpy as np
from scipy.spatial import cKDTree
from collections import deque, defaultdict
from utils import prompt, prompt_int, prompt_float, prompt_yn, label_matches

def percolation(traj):
    print("\n--- Hydrogen Bond Percolation Pathway Analysis ---")

    # === User selects compounds and atom roles ===
    print("\nAvailable Compounds:")
    compounds = list(traj.compounds.values())
    for i, comp in enumerate(compounds, start=1):
        print(f"{i}: {comp.rep} (Number: {len(comp.members)})")

    comp_indices = prompt("Enter compound numbers to include in the analysis (comma-separated): ")
    comp_indices = [int(x.strip()) - 1 for x in comp_indices.split(',') if x.strip()]
    selected_comps = [compounds[i] for i in comp_indices]
    comp_keys = [next(k for k, c in traj.compounds.items() if c is comp) for comp in selected_comps]

    comp_acceptors = {}
    comp_protons = {}
    for idx, comp in zip(comp_indices, selected_comps):
        acc_labels = prompt(f"Enter acceptor atom labels for compound {idx + 1} ({comp.rep}) (comma-separated): ")
        proton_labels = prompt(f"Enter proton atom labels for compound {idx + 1} ({comp.rep}) (comma-separated): ")
        comp_key = next(k for k, c in traj.compounds.items() if c is comp)
        comp_acceptors[comp_key] = [l.strip() for l in acc_labels.split(',') if l.strip()]
        comp_protons[comp_key] = [l.strip() for l in proton_labels.split(',') if l.strip()]

    cutoff = prompt_float("Enter hydrogen bond cutoff distance (in Å): ", 2.5, minval=0.5)
    max_depth = prompt_int("Enter the maximum number of H-bond steps to consider: ", 5, minval=1)

    use_alternating = prompt_yn("Limit percolation pathways to alternating chains of accepted and donated hydrogen bonds?", False)

    update_compounds = prompt_yn("Perform molecule recognition and update compound list in each frame?", False)

    start_frame = prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)

    # === Build global lists ===
    all_acceptors, all_protons, mol_to_comp = build_atom_list(comp_keys, traj.compounds, comp_acceptors, comp_protons)

    if not all_acceptors or not all_protons:
        print("No atoms matched the given labels.")
        return

    # === Accumulators ===
    depth_counts = np.zeros(max_depth, dtype=np.float64)
    total_seeds = 0
    frame_idx = 0
    processed_frames = 0

    if start_frame > 1:
        print(f"Skipping forward to frame {start_frame}.")
        while frame_idx < start_frame - 1:
            traj.read_frame()
            frame_idx += 1

    while nframes != 0:
        try:
            if update_compounds:
                traj.guess_molecules()
                traj.update_molecule_coords()
                try:
                    # search and update compounds
                    selected_comps = [traj.compounds[key] for key in comp_keys]
                except KeyError:
                    # compound disappeared this frame – skip
                    frame_idx += 1
                    nframes -= 1
                    traj.read_frame()
                    continue

                # === Build global lists ===
                all_acceptors, all_protons, mol_to_comp = build_atom_list(comp_keys, traj.compounds, comp_acceptors, comp_protons)

                if not all_acceptors or not all_protons:
                    frame_idx += 1
                    nframes -= 1
                    traj.read_frame()
                    continue

            else:
                traj.update_molecule_coords()

            # === Build KD trees ===
            acc_coords = np.array([atom.coord for atom, _ in all_acceptors])
            proton_coords = np.array([atom.coord for atom, _ in all_protons])
            acc_tree = cKDTree(acc_coords, boxsize=traj.box_size)
            proton_tree = cKDTree(proton_coords, boxsize=traj.box_size)

            # === Molecule graph (undirected edges) ===
            neighbors = defaultdict(set)  # mol -> set of bonded mols

            for idx_a, (acc_atom, acc_mol) in enumerate(all_acceptors):
                nearby = proton_tree.query_ball_point(acc_atom.coord, r=cutoff)
                for idx_p in nearby:
                    prot_atom, prot_mol = all_protons[idx_p]
                    if prot_mol != acc_mol:
                        neighbors[acc_mol].add(prot_mol)
                        neighbors[prot_mol].add(acc_mol)

            for idx_p, (prot_atom, prot_mol) in enumerate(all_protons):
                nearby = acc_tree.query_ball_point(prot_atom.coord, r=cutoff)
                for idx_a in nearby:
                    acc_atom, acc_mol = all_acceptors[idx_a]
                    if prot_mol != acc_mol:
                        neighbors[prot_mol].add(acc_mol)
                        neighbors[acc_mol].add(prot_mol)

            # === Percolation depth counting ===
            seed_molecules = set(mol for _, mol in all_acceptors)

            # Build mol → list[Atom] maps for traversal
            mol_acceptors = defaultdict(list)
            mol_protons = defaultdict(list)
            for atom, mol in all_acceptors:
                mol_acceptors[mol].append(atom)
            for atom, mol in all_protons:
                mol_protons[mol].append(atom)

            for mol in seed_molecules:
                visited = {mol}
                if use_alternating:
                    frontier = deque([(mol, 0)])  # Start from acceptor
                    while frontier:
                        current_mol, depth = frontier.popleft()
                        if depth >= max_depth:
                            continue
                        depth_counts[depth] += 1
#                        for acc_atom in mol_acceptors[current_mol]:
#                            for idx in proton_tree.query_ball_point(acc_atom.coord, r=cutoff):
#                                neighbor_mol = all_protons[idx][1]
#                                if neighbor_mol not in visited:
#                                    visited.add(neighbor_mol)
#                                    frontier.append((neighbor_mol, depth + 1))
                        for prot_atom in mol_protons[current_mol]:
                            nearby = acc_tree.query_ball_point(prot_atom.coord, r=cutoff)
                            for idx in nearby:
                                _, neighbor_mol = all_acceptors[idx]
                                if neighbor_mol not in visited:
                                    visited.add(neighbor_mol)
                                    frontier.append((neighbor_mol, depth + 1))

                else:
                    frontier = deque([(mol, 0)])
                    while frontier:
                        current_mol, depth = frontier.popleft()
                        if depth >= max_depth:
                            continue
                        depth_counts[depth] += 1
                        for neighbor in neighbors[current_mol]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                frontier.append((neighbor, depth + 1))

                total_seeds += 1

            processed_frames += 1
            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

            for _ in range(frame_stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()

        except ValueError:
            print("\nEnd of trajectory reached.")
            break
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user.")
            break

    print()

    # === Normalize and report ===
    if total_seeds > 0 and processed_frames > 0:
        avg_counts = depth_counts / total_seeds
    else:
        avg_counts = np.zeros_like(depth_counts)

    print("\nAverage number of reachable molecules by H-bond depth:")
    for d, count in enumerate(avg_counts, 1):
        print(f"Depth {d}: {count:.4f}")

    with open("percolation_depths.dat", "w") as f:
        f.write("# Depth    AverageReachableMolecules\n")
        for d, count in enumerate(avg_counts, 1):
            f.write(f"{d} {count:.6f}\n")
    print("Results written to percolation_depths.dat")



def build_atom_list(comp_keys, traj_compounds, comp_acceptors, comp_protons):
    all_acceptors = []
    all_protons = []
    mol_to_comp = {}

    for key in comp_keys:
        comp = traj_compounds[key]
        for mol in comp.members:
            mol_acceptors = [
                mol.atoms[mol.label_to_id[label]]
                for label in mol.label_to_id
                if any(label_matches(lab, label) for lab in comp_acceptors[key])
            ]
            mol_protons = [
                mol.atoms[mol.label_to_id[label]]
                for label in mol.label_to_id
                if any(label_matches(lab, label) for lab in comp_protons[key])
            ]
            if mol_acceptors or mol_protons:
                all_acceptors.extend((atom, mol) for atom in mol_acceptors)
                all_protons.extend((atom, mol) for atom in mol_protons)
                mol_to_comp[mol] = comp  # map back for safety

    return all_acceptors, all_protons, mol_to_comp

