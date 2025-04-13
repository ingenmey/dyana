# analyses/adf_analysis.py

import numpy as np
from utils import prompt, prompt_int, prompt_float, prompt_yn, prompt_choice
from utils import label_matches

def adf(traj):
    # List compounds
    print("\nAvailable Compounds:")
    for i, compound in enumerate(traj.compounds.values(), start=1):
        print(f"{i}: {compound.rep} (Number: {len(compound.members)})")

    # Prompt user
    ref_index = prompt_int("Choose the reference compound (number): ", 1, minval=1) - 1
    obs_index = prompt_int("Choose the observed compound (number): ", 1, minval=1) - 1

    ref_base_source = prompt_choice("Base atom of first vector?", ["r", "o"], "r")
    ref_tip_source  = prompt_choice("Tip atom of first vector?", ["r", "o"], "r")
    ref_base_label  = prompt("Which atom is at the base of the first vector? ")
    ref_tip_label   = prompt("Which atom is at the tip of the first vector? ")

    obs_base_source = prompt_choice("Base atom of second vector?", ["r", "o"], "o")
    obs_tip_source  = prompt_choice("Tip atom of second vector?", ["r", "o"], "o")
    obs_base_label  = prompt("Which atom is at the base of the second vector? ")
    obs_tip_label   = prompt("Which atom is at the tip of the second vector? ")

    enforce_shared_atom = False
    if ref_tip_label == obs_base_label:
        enforce_shared_atom = prompt_yn("Should the tip atom of the reference vector and the base atom of the observed vector be the same atom?", True)

    bin_count = prompt_int("Enter the number of bins for ADF calculation: ", 180, minval=1)

    ref_compound = list(traj.compounds.values())[ref_index]
    obs_compound = list(traj.compounds.values())[obs_index]
    box_size = traj.box_size

    # Precompute atom indices ONCE
    ref_base_ids_per_mol = {}
    ref_tip_ids_per_mol = {}
    obs_base_ids_per_mol = {}
    obs_tip_ids_per_mol = {}

    for ref_mol in ref_compound.members:
        for obs_mol in obs_compound.members:
            if ref_mol == obs_mol:
                continue

            ref_base_mol = ref_mol if ref_base_source == "r" else obs_mol
            ref_tip_mol  = ref_mol if ref_tip_source  == "r" else obs_mol
            obs_base_mol = obs_mol if obs_base_source == "o" else ref_mol
            obs_tip_mol  = obs_mol if obs_tip_source  == "o" else ref_mol

            ref_base_ids_per_mol[(ref_mol, obs_mol)] = find_matching_labels(ref_base_mol, ref_base_label)
            ref_tip_ids_per_mol[(ref_mol, obs_mol)]  = find_matching_labels(ref_tip_mol, ref_tip_label)
            obs_base_ids_per_mol[(ref_mol, obs_mol)] = find_matching_labels(obs_base_mol, obs_base_label)
            obs_tip_ids_per_mol[(ref_mol, obs_mol)]  = find_matching_labels(obs_tip_mol, obs_tip_label)

    adf_accumulator = np.zeros(bin_count)

    use_distance_cutoff = prompt_yn("Use a distance cutoff between tip atoms?", False)
    if use_distance_cutoff:
        cutoff_distance = prompt_float("Enter the tip-tip distance cutoff (Å): ", 5.0, minval=0.0)
        cutoff_distance_sq = cutoff_distance**2
    else:
        cutoff_distance_sq = None

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

            # Compute vectors for the current frame
            ref_vectors, obs_vectors = compute_vectors_static(
                ref_compound, obs_compound,
                ref_base_ids_per_mol, ref_tip_ids_per_mol,
                obs_base_ids_per_mol, obs_tip_ids_per_mol,
                box_size, traj, enforce_shared_atom, cutoff_distance_sq
            )

            adf_result = calculate_adf(ref_vectors, obs_vectors, bin_count)
            adf_accumulator += adf_result

            processed_frames += 1
            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

            for _ in range(frame_stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()

        except ValueError:
            break  # End of trajectory

        except KeyboardInterrupt:
            print("\nInterrupt received! Exiting main loop.")
            break

    print()

    total_pairs = processed_frames * len(ref_compound.members) * len(obs_compound.members)
    adf_average = adf_accumulator / total_pairs

    with open("adf.dat", "w") as f:
        f.write("# Angle (deg)    ADF\n")
        for i, value in enumerate(adf_average):
            angle = (i + 0.5) * (180 / bin_count)
            f.write(f"{angle:.4f} {value:.8f}\n")
    print("\nADF results saved to adf.dat")


def find_matching_labels(mol, user_label):
    return [atom_idx for label, atom_idx in mol.label_to_global_id.items() if label_matches(user_label, label)]

def compute_vectors_static(ref_compound, obs_compound,
                            ref_base_ids_per_mol, ref_tip_ids_per_mol,
                            obs_base_ids_per_mol, obs_tip_ids_per_mol,
                            box_size, traj, enforce_shared_atom, cutoff_distance_sq):
    ref_base_list = []
    ref_tip_list = []
    obs_base_list = []
    obs_tip_list = []

    for ref_mol in ref_compound.members:
        for obs_mol in obs_compound.members:
            if ref_mol == obs_mol:
                continue

            ref_base_ids = ref_base_ids_per_mol[(ref_mol, obs_mol)]
            ref_tip_ids  = ref_tip_ids_per_mol[(ref_mol, obs_mol)]
            obs_base_ids = obs_base_ids_per_mol[(ref_mol, obs_mol)]
            obs_tip_ids  = obs_tip_ids_per_mol[(ref_mol, obs_mol)]

            for ref_base_id in ref_base_ids:
                for ref_tip_id in ref_tip_ids:
                    for obs_base_id in obs_base_ids:
                        if enforce_shared_atom and (ref_tip_id != obs_base_id):
                            continue
                        for obs_tip_id in obs_tip_ids:
                            ref_base_list.append(ref_base_id)
                            ref_tip_list.append(ref_tip_id)
                            obs_base_list.append(obs_base_id)
                            obs_tip_list.append(obs_tip_id)

    ref_base = traj.coords[np.array(ref_base_list)]
    ref_tip  = traj.coords[np.array(ref_tip_list)]
    obs_base = traj.coords[np.array(obs_base_list)]
    obs_tip  = traj.coords[np.array(obs_tip_list)]

    # Compute vectors
    ref_vectors = ref_tip - ref_base
    obs_vectors = obs_tip - obs_base

    # Apply minimum image convention
    ref_vectors -= np.round(ref_vectors / box_size) * box_size
    obs_vectors -= np.round(obs_vectors / box_size) * box_size

    # Apply tip-tip distance cutoff
    if cutoff_distance_sq is not None:
        delta = ref_tip - obs_tip
        delta -= np.round(delta / box_size) * box_size
        dist_sq = np.sum(delta**2, axis=1)
        mask = dist_sq <= cutoff_distance_sq

        ref_vectors = ref_vectors[mask]
        obs_vectors = obs_vectors[mask]

    return ref_vectors, obs_vectors

def calculate_adf(ref_vectors, obs_vectors, bin_count):
    bin_width = 180 / bin_count
    adf = np.zeros(bin_count)

    # Normalize vectors
    ref_vectors /= np.linalg.norm(ref_vectors, axis=1, keepdims=True)
    obs_vectors /= np.linalg.norm(obs_vectors, axis=1, keepdims=True)

    # Calculate angles
    cos_angles = np.sum(ref_vectors * obs_vectors, axis=1)
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


