import numpy as np
from scipy.spatial import cKDTree
from utils import prompt, prompt_int, prompt_float, prompt_yn, prompt_choice, label_matches

def adf_threebody(traj):
    # --- List compounds ---
    print("\nAvailable Compounds:")
    for i, compound in enumerate(traj.compounds.values(), start=1):
        print(f"{i}: {compound.rep} (Number: {len(compound.members)})")

    # --- User input ---
    center_index = prompt_int("Choose the center compound (number): ", 1, minval=1) - 1
    neighbor1_index = prompt_int("Choose the first neighbor compound (number): ", 1, minval=1) - 1
    neighbor2_index = prompt_int("Choose the second neighbor compound (number): ", 1, minval=1) - 1

    center_label = prompt("Label of center atom: ")
    neighbor1_label = prompt("Label of first neighbor atom: ")
    neighbor2_label = prompt("Label of second neighbor atom: ")

    cutoffs = np.zeros((2,2))
    cutoffs[0,0] = prompt_float("Minimum distance for center - neighbor1 (Å): ", 0.0)
    cutoffs[0,1] = prompt_float("Maximum distance for center - neighbor1 (Å): ", 3.5)
    cutoffs[1,0] = prompt_float("Minimum distance for center - neighbor2 (Å): ", 0.0)
    cutoffs[1,1] = prompt_float("Maximum distance for center - neighbor2 (Å): ", 3.5)

    enforce_threebody = prompt_yn("Enforce that neighbor1 and neighbor2 come from different molecules?", True)

    bin_count = prompt_int("Number of bins for ADF histogram: ", 180, minval=1)

    # --- Setup ---
    center_compound = list(traj.compounds.values())[center_index]
    neighbor1_compound = list(traj.compounds.values())[neighbor1_index]
    neighbor2_compound = list(traj.compounds.values())[neighbor2_index]
    box_size = traj.box_size

    # Find global ids matching the labels
    center_ids = []
    neighbor1_ids = []
    neighbor2_ids = []

    for mol in center_compound.members:
        center_ids.extend(find_matching_labels(mol, center_label))

    for mol in neighbor1_compound.members:
        neighbor1_ids.extend(find_matching_labels(mol, neighbor1_label))

    for mol in neighbor2_compound.members:
        neighbor2_ids.extend(find_matching_labels(mol, neighbor2_label))

    center_ids = np.array(center_ids)
    neighbor1_ids = np.array(neighbor1_ids)
    neighbor2_ids = np.array(neighbor2_ids)

    # Initialize
    adf_accumulator = np.zeros(bin_count)

    start_frame = prompt_int("In which trajectory frame to start processing?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame:", 1, minval=1)

    frame_idx = 0
    processed_frames = 0

    # --- Skip to start frame ---
    if start_frame > 1:
        print(f"Skipping to frame {start_frame}...")
        while frame_idx < start_frame - 1:
            traj.read_frame()
            frame_idx += 1

    # --- Main Loop ---
    while nframes != 0:
        try:
            traj.update_molecule_coords()

            vec1, vec2 = compute_threebody_vectors_fast(
                traj, center_ids, neighbor1_ids, neighbor2_ids,
                cutoffs, box_size, enforce_threebody
            )

            if vec1.any() and vec2.any():
                adf_result = calculate_adf(vec1, vec2, bin_count)
                adf_accumulator += adf_result

            processed_frames += 1
#            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

            # Move to next frame
            for _ in range(frame_stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()

        except ValueError:
            break  # End of trajectory

        except KeyboardInterrupt:
            print("\nInterrupted! Finishing early.")
            break

    print()

    # --- Normalize and Save ---
    total_triplets = processed_frames * len(center_ids)
    adf_average = adf_accumulator / total_triplets

    with open("adf_threebody.dat", "w") as f:
        f.write("# Angle (deg)    ADF\n")
        for i, value in enumerate(adf_average):
            angle = (i + 0.5) * (180 / bin_count)
            f.write(f"{angle:.4f} {value:.8f}\n")

    print("\nThree-body ADF results saved to adf_threebody.dat")


def find_matching_labels(mol, user_label):
    """Find all atom global IDs matching a user label."""
    return [atom_idx for label, atom_idx in mol.label_to_global_id.items() if label_matches(user_label, label)]


def compute_threebody_vectors_fast(traj, center_ids, neighbor1_ids, neighbor2_ids, cutoffs, box_size, enforce_threebody):
    """
    Fast KDTree-based threebody vector computation with correct triplet structure
    and min/max distance cutoff handling.
    """

    coords = traj.coords
    box_size = np.asarray(box_size)

    center_coords = coords[center_ids]
    neighbor1_coords = coords[neighbor1_ids]
    neighbor2_coords = coords[neighbor2_ids]

    kdtree_n1 = cKDTree(neighbor1_coords, boxsize=box_size)
    kdtree_n2 = cKDTree(neighbor2_coords, boxsize=box_size)

    vectors1 = []
    vectors2 = []

    for idx_center, center in enumerate(center_coords):
        # Neighbor 1 search
        neighbor1_candidates = kdtree_n1.query_ball_point(center, cutoffs[0,1])
        neighbor1_indices = []
        for idx in neighbor1_candidates:
            disp = neighbor1_coords[idx] - center
            disp -= np.round(disp / box_size) * box_size  # Minimum image convention
            dist = np.linalg.norm(disp)
            if dist >= cutoffs[0,0]:  # Apply min cutoff
                neighbor1_indices.append(idx)

        # Neighbor 2 search
        neighbor2_candidates = kdtree_n2.query_ball_point(center, cutoffs[1,1])
        neighbor2_indices = []
        for idx in neighbor2_candidates:
            disp = neighbor2_coords[idx] - center
            disp -= np.round(disp / box_size) * box_size
            dist = np.linalg.norm(disp)
            if dist >= cutoffs[1,0]:  # Apply min cutoff
                neighbor2_indices.append(idx)

        # Loop over all valid neighbor pairs
        for n1_idx in neighbor1_indices:
            for n2_idx in neighbor2_indices:
                if neighbor1_ids[n1_idx] == neighbor2_ids[n2_idx]:
                    continue  # Skip if neighbor1 and neighbor2 are the same atom

                idx1 = neighbor1_ids[n1_idx]
                idx2 = neighbor2_ids[n2_idx]

                if enforce_threebody:
                    mol1 = traj.atoms[idx1].parent_molecule
                    mol2 = traj.atoms[idx2].parent_molecule
                    if mol1 == mol2:
                        continue  # Skip if neighbor1 and neighbor2 are from same molecule

                neighbor1 = neighbor1_coords[n1_idx]
                neighbor2 = neighbor2_coords[n2_idx]

                # Build center-to-neighbor vectors
                vec1 = neighbor1 - center
                vec1 -= np.round(vec1 / box_size) * box_size

                vec2 = neighbor2 - center
                vec2 -= np.round(vec2 / box_size) * box_size

                vectors1.append(vec1)
                vectors2.append(vec2)

    return np.array(vectors1), np.array(vectors2)


def calculate_adf(vec1, vec2, bin_count):
    """Calculate ADF histogram from two sets of vectors."""
    bin_width = 180 / bin_count
    adf = np.zeros(bin_count)

    # Normalize
    vec1 /= np.linalg.norm(vec1, axis=1, keepdims=True)
    vec2 /= np.linalg.norm(vec2, axis=1, keepdims=True)

    # Angles
    cos_angles = np.sum(vec1 * vec2, axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    # Histogram
    bin_indices = np.floor(angles * (180 / np.pi) / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)
    np.add.at(adf, bin_indices, 1)

    # Normalize by sin(theta)
    angle_bin_centers = (np.arange(bin_count) + 0.5) * bin_width * (np.pi / 180)
    norm_factors = 1 / np.sin(angle_bin_centers)
    adf *= norm_factors

    return adf


