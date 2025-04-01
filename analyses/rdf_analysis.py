# analyses/rdf_analysis.py
import numpy as np
from scipy.spatial import cKDTree
from utils import prompt

def rdf(traj):
    # Prompt user for RDF parameters
    ref_index = int(prompt("Choose the reference compound (number): ")) - 1
    obs_index = int(prompt("Choose the observed compound (number): ")) - 1
    ref_label = prompt("Which atom in reference compound? ")
    obs_label = prompt("Which atom in observed compound? ")
    max_distance = float(prompt("Enter the maximum distance for RDF calculation (in Å): ", 10.0))
    bin_count = int(prompt("Enter the number of bins for RDF calculation: ", 500))

    # Initialize the RDF accumulator
    rdf_accumulator = [0] * bin_count
    num_frames = 0

    # Loop through all frames
    while True:
        try:
            # Update the coordinates and COMs for each compound
            for compound in traj.compounds.values():
                for molecule in compound.members:
                    molecule.update_coords(traj.coords)
                compound.update_coms(traj.boxsize)

            # Perform the RDF calculation for the current frame
            ref_compound = list(traj.compounds.values())[ref_index]
            obs_compound = list(traj.compounds.values())[obs_index]
            rdf_result = calculate_rdf(ref_compound, obs_compound, ref_label, obs_label, traj.boxsize, max_distance, bin_count)

            # Accumulate the RDF results
            for i, (distance, value) in enumerate(rdf_result):
                rdf_accumulator[i] += value

            num_frames += 1
            print(num_frames)

            # Read the next frame
            traj.read_frame()
        except ValueError:
            # End of the trajectory file
            break

    # Average the RDF results over all frames
    rdf_average = [value / num_frames for value in rdf_accumulator]

    # Output the averaged RDF results
    print("\nAveraged RDF Results:")
    for i, value in enumerate(rdf_average):
        distance = (i + 0.5) * (max_distance / bin_count)
        print(f"Distance: {distance:.4f} Å, RDF: {value:.4f}")



def calculate_rdf(ref_compound, obs_compound, ref_label, obs_label, boxsize, max_distance, bin_count):
    dimx, dimy, dimz = boxsize
    bin_width = max_distance / bin_count
    rdf = np.zeros(bin_count)

    ref_coords = ref_compound.get_coords(ref_label)
    obs_coords = obs_compound.get_coords(obs_label)

    # Use KDTree for efficient distance calculations
    obs_tree = cKDTree(obs_coords, boxsize=boxsize)

    for ref_coord in ref_coords:
        distances, indices = obs_tree.query(ref_coord, k=len(obs_coords), distance_upper_bound=max_distance)
        distances = distances[distances < max_distance]

        # Avoid counting self-distances if ref and obs compounds are the same
        if ref_compound == obs_compound:
            distances = distances[distances > 0]

        # Compute histogram
        bin_indices = (distances / bin_width).astype(int)
        np.add.at(rdf, bin_indices, 1)

    # Normalize RDF
    volume = dimx * dimy * dimz
    norm_factor = (len(ref_coords) * len(obs_coords) * (4/3) * np.pi * ((np.arange(1, bin_count + 1) * bin_width)**3 - (np.arange(bin_count) * bin_width)**3)) / volume
    rdf /= norm_factor

    return list(zip(np.linspace(0, max_distance, bin_count, endpoint=False), rdf))

