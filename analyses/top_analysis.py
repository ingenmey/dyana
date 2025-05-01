import numpy as np
from utils import prompt, prompt_int, prompt_float, prompt_yn
from utils import label_matches
from scipy.spatial import cKDTree

def weighted_linear_bin(value, hist, bin_edges, bin_width):
    """Distribute a value across two nearest bins using linear weighting."""
    if not (bin_edges[0] <= value <= bin_edges[-1]):
        return
    bin_index = (value - bin_edges[0]) / bin_width
    left = int(np.floor(bin_index))
    right = left + 1

    if left < 0:
        left = 0
    if right >= len(hist):
        right = len(hist) - 1

    w_right = bin_index - left
    w_left = 1.0 - w_right

    hist[left] += w_left
    if right != left:  # Avoid double-count at boundaries
        hist[right] += w_right

def tetrahedral_order(traj):
    print("\nTetrahedral Order Parameter Analysis")

    # List compounds
    print("\nAvailable Compounds:")
    for i, compound in enumerate(traj.compounds.values(), start=1):
        print(f"{i}: {compound.rep} (Number: {len(compound.members)})")

    ref_comp_idx = prompt_int("Choose the reference compound (number): ", 1, minval=1) - 1
    ref_compound = list(traj.compounds.values())[ref_comp_idx]

    obs_comp_indices = prompt("Enter the observed compounds (comma-separated numbers): ").strip()
    obs_comp_indices = [int(x.strip()) - 1 for x in obs_comp_indices.split(',') if x.strip()]
    obs_compounds = [list(traj.compounds.values())[idx] for idx in obs_comp_indices]

    ref_label = prompt("Enter the reference atom label (e.g., O): ").strip()
    obs_atoms_per_compound = {}
    for idx, comp in zip(obs_comp_indices, obs_compounds):
        labels = prompt(f"Enter observed atom labels for compound {idx+1} ({comp.rep}) (comma-separated): ").strip()
        labels = [l.strip() for l in labels.split(',') if l.strip()]
        obs_atoms_per_compound[comp] = labels

    # Collect reference and observed atoms (global indices)
    ref_atoms = []
    for mol in ref_compound.members:
        for label, idx in mol.label_to_global_id.items():
            if label_matches(ref_label, label):
                ref_atoms.append(idx)

    obs_atoms = []
    for comp, labels in obs_atoms_per_compound.items():
        for mol in comp.members:
            for label, idx in mol.label_to_global_id.items():
                if any(label_matches(obs_label, label) for obs_label in labels):
                    obs_atoms.append(idx)

    if len(obs_atoms) < 4:
        print("Not enough observed atoms to perform analysis.")
        return

    use_cutoff = prompt_yn("Use a maximum distance cutoff for neighbor search?", False)
    if use_cutoff:
        cutoff = prompt_float("Enter the maximum cutoff distance (Å):", 5.0, minval=0.0)
    else:
        cutoff = None

    bin_count_q = prompt_int("Enter the number of bins for angular tetrahedral order distribution q: ", 100, minval=1)
    bin_count_s = prompt_int("Enter the number of bins for radial tetrahedral order distribution S: ", 10000, minval=1)

    hist_q = np.zeros(bin_count_q)
    bin_edges_q = np.linspace(0, 1, bin_count_q + 1)
    bin_width_q = bin_edges_q[1] - bin_edges_q[0]

    hist_s = np.zeros(bin_count_s)
    bin_edges_s = np.linspace(0, 1, bin_count_s + 1)
    bin_width_s = bin_edges_s[1] - bin_edges_s[0]

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
            box = traj.box_size
            obs_coords = coords[obs_atoms]
            kdtree = cKDTree(obs_coords, boxsize=box)

            for r_idx in ref_atoms:
                r_coord = coords[r_idx]

                if use_cutoff:
                    neighbor_idxs = kdtree.query_ball_point(r_coord, cutoff)
                    neighbor_idxs = [i for i in neighbor_idxs if obs_atoms[i] != r_idx]
                    if len(neighbor_idxs) < 4:
                        continue
                    neighbors = obs_coords[neighbor_idxs]
                    distances = np.linalg.norm(neighbors - r_coord, axis=1)
                    nearest = np.argsort(distances)[:4]
                    four_nearest = neighbors[nearest]
                    four_dists = distances[nearest]
                else:
                    distances, idxs = kdtree.query(r_coord, k=5)
                    filtered = [(d, i) for d, i in zip(distances, idxs) if obs_atoms[i] != r_idx]
                    if len(filtered) < 4:
                        continue
                    filtered = filtered[:4]
                    four_nearest = obs_coords[[i for _, i in filtered]]
                    four_dists = np.array([d for d, _ in filtered])

                # Compute q
                cosines = []
                for j in range(3):
                    for k in range(j + 1, 4):
                        vj = four_nearest[j] - r_coord
                        vk = four_nearest[k] - r_coord
                        vj -= box * np.round(vj / box)
                        vk -= box * np.round(vk / box)
                        vj /= np.linalg.norm(vj)
                        vk /= np.linalg.norm(vk)
                        cos_angle = np.dot(vj, vk)
                        cosines.append(cos_angle)

                q = 1 - (3/8) * sum((cos + 1/3)**2 for cos in cosines)
                weighted_linear_bin(q, hist_q, bin_edges_q, bin_width_q)

                # Compute S
                r_mean = np.mean(four_dists)
                if r_mean > 1e-8:  # avoid divide-by-zero
                    S = 1 - (1/3) * np.sum((four_dists - r_mean)**2) / (4 * r_mean**2)
                    weighted_linear_bin(S, hist_s, bin_edges_s, bin_width_s)

            processed_frames += 1
            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

            for _ in range(frame_stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()

        except ValueError:
            break
        except KeyboardInterrupt:
            print("\nInterrupt received! Exiting main loop.")
            break

    print()

    # Normalize and write q
    total_q = np.sum(hist_q)
    if total_q > 0:
        with open("tetrahedral_q.dat", "w") as f:
            f.write("# q     P(q)\n")
            for i in range(bin_count_q):
                center = 0.5 * (bin_edges_q[i] + bin_edges_q[i + 1])
                probability = hist_q[i] / total_q * 100
                f.write(f"{center:.5f} {probability:.8f}\n")
        print("\nTetrahedral orientational order distribution saved to tetrahedral_q.dat")
    else:
        print("No valid q values were accumulated.")

    # Normalize and write S
    total_s = np.sum(hist_s)
    if total_s > 0:
        with open("tetrahedral_s.dat", "w") as f:
            f.write("# S     P(S)\n")
            for i in range(bin_count_s):
                center = 0.5 * (bin_edges_s[i] + bin_edges_s[i + 1])
                probability = hist_s[i] / total_s * 100
                f.write(f"{center:.5f} {probability:.8f}\n")
        print("Tetrahedral translational order distribution saved to tetrahedral_s.dat")
    else:
        print("No valid S values were accumulated.")

