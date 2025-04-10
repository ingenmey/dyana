# analyses/adf_analysis.py
import numpy as np
from utils import prompt

def adf(traj):
    # Prompt user for ADF parameters
    ref_index = int(prompt("Choose the reference compound (number): ")) - 1
    obs_index = int(prompt("Choose the observed compound (number): ")) - 1

    ref_base_source = prompt("Is the base atom of the first vector in the reference or observed compound? (r/o) ", "r").lower()
    ref_tip_source = prompt("Is the tip atom of the first vector in the reference or observed compound? (r/o) ", "r").lower()
    ref_base_label = prompt("Which atom is at the base of the first vector? ")
    ref_tip_label = prompt("Which atom is at the tip of the first vector? ")

    obs_base_source = prompt("Is the base atom of the second vector in the reference or observed compound? (r/o) ", "o").lower()
    obs_tip_source = prompt("Is the tip atom of the second vector in the reference or observed compound? (r/o) ", "o").lower()
    obs_base_label = prompt("Which atom is at the base of the second vector? ")
    obs_tip_label = prompt("Which atom is at the tip of the second vector? ")

    bin_count = int(prompt("Enter the number of bins for ADF calculation: ", 180))

    # Initialize the ADF accumulator
    adf_accumulator = np.zeros(bin_count)
    num_frames = 0

    ref_compound = list(traj.compounds.values())[ref_index]
    obs_compound = list(traj.compounds.values())[obs_index]
    box_size = traj.box_size

    # Loop through all frames
    while True:
        try:
            # Update the coordinates for each compound
            print("update")
            update_coords(traj, box_size)

            # Calculate the vectors for the current frame
            print("vectors")
            ref_vectors, obs_vectors = compute_vectors(
                ref_compound, obs_compound, ref_base_label, ref_tip_label,
                obs_base_label, obs_tip_label, ref_base_source, ref_tip_source,
                obs_base_source, obs_tip_source, box_size
            )

            # Calculate ADF for the current frame
            print("calculating")
            adf_result = calculate_adf(ref_vectors, obs_vectors, bin_count)

            # Accumulate the ADF results
            adf_accumulator += adf_result

            num_frames += 1
            print(num_frames)

            # Read the next frame
            traj.read_frame()
        except ValueError:
            # End of the trajectory file
            break

    # Average the ADF results over all frames
    adf_average = adf_accumulator / (num_frames * len(ref_compound.members) * len(obs_compound.members))

    # Output the averaged ADF results
    print("\nAveraged ADF Results:")
    for i, value in enumerate(adf_average):
        angle = (i + 0.5) * (180 / bin_count)
        print(f"Angle: {angle:.2f}°, ADF: {value:.8f}")

def update_coords(traj, box_size):
    for compound in traj.compounds.values():
        for molecule in compound.members:
            molecule.update_coords(traj.coords, box_size)

def compute_vectors(ref_compound, obs_compound, ref_base_label, ref_tip_label,
                    obs_base_label, obs_tip_label, ref_base_source, ref_tip_source,
                    obs_base_source, obs_tip_source, box_size):
    ref_vectors = []
    obs_vectors = []

    for ref_mol in ref_compound.members:
        for obs_mol in obs_compound.members:
            if ref_mol == obs_mol:
                continue
            ref_base_id = ref_mol.label_to_id[ref_base_label] if ref_base_source == "r" else obs_mol.label_to_id[ref_base_label]
            ref_tip_id  = ref_mol.label_to_id[ref_tip_label]  if ref_tip_source  == "r" else obs_mol.label_to_id[ref_tip_label]
            obs_base_id = obs_mol.label_to_id[obs_base_label] if obs_base_source == "o" else ref_mol.label_to_id[obs_base_label]
            obs_tip_id  = obs_mol.label_to_id[obs_tip_label]  if obs_tip_source  == "o" else ref_mol.label_to_id[obs_tip_label]

            ref_base = ref_mol.coords[ref_base_id] if ref_base_source == "r" else obs_mol.coords[ref_base_id]
            ref_tip  = ref_mol.coords[ref_tip_id]  if ref_tip_source  == "r" else obs_mol.coords[ref_tip_id]
            obs_base = obs_mol.coords[obs_base_id] if obs_base_source == "o" else ref_mol.coords[obs_base_id]
            obs_tip  = obs_mol.coords[obs_tip_id]  if obs_tip_source  == "o" else ref_mol.coords[obs_tip_id]

            ref_base = np.mod(ref_base, box_size)
            ref_tip  = np.mod(ref_tip, box_size)
            obs_base = np.mod(obs_base, box_size)
            obs_tip  = np.mod(obs_tip, box_size)

            # Apply minimum image convention
            ref_vector = ref_tip - ref_base
            ref_vector -= np.round(ref_vector / box_size) * box_size

            obs_vector = obs_tip - obs_base
            obs_vector -= np.round(obs_vector / box_size) * box_size

            ref_vectors.append(ref_vector)
            obs_vectors.append(obs_vector)

    return np.array(ref_vectors), np.array(obs_vectors)

def calculate_adf(ref_vectors, obs_vectors, bin_count):
    bin_width = 180 / bin_count
    adf = np.zeros(bin_count)

    # Normalize vectors
    ref_vectors /= np.linalg.norm(ref_vectors, axis=1, keepdims=True)
    obs_vectors /= np.linalg.norm(obs_vectors, axis=1, keepdims=True)

    # Calculate angles
    angles = []

    for ref_vector, obs_vector in zip(ref_vectors, obs_vectors):
        cos_angle = np.clip(np.dot(ref_vector, obs_vector), -1.0, 1.0)
        angles.append(np.arccos(cos_angle))  # in radians

    angles = np.array(angles)

    # Compute histogram
    bin_indices = np.floor(angles * (180 / np.pi) / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)
    np.add.at(adf, bin_indices, 1)

    # Calculate normalization factor
    angle_bin_centers = (np.arange(bin_count) + 0.5) * bin_width * (np.pi / 180)
    norm_factors = 1 / np.sin(angle_bin_centers)

    # Apply normalization factor
    adf *= norm_factors

    return adf

