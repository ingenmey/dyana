import numpy as np
from scipy.spatial import cKDTree
from utils import prompt

def percolation(traj):
    # Prompt user for percolation parameters
    compound_index = int(prompt("Choose the compound (number): ")) - 1
    acceptor_labels = prompt("Enter the accepting atoms (comma separated): ").split(',')
    donor_labels = prompt("Enter the donating atoms (comma separated): ").split(',')
    max_cutoff = int(prompt("Enter the maximum number of hydrogen bonds to consider: ", 5))
    max_distance = float(prompt("Enter the maximum distance for hydrogen bonds (in Å): ", 3.5))

    # Initialize the percolation accumulator
    percolation_accumulator = np.zeros(max_cutoff)
    num_frames = 0

    # Loop through all frames
    while True:
        try:
            # Update the coordinates and COMs for each compound
            for compound in traj.compounds.values():
                for molecule in compound.members:
                    molecule.update_coords(traj.coords)
                compound.update_coms(traj.boxsize)

            # Perform the percolation calculation for the current frame
            compound = list(traj.compounds.values())[compound_index]
            percolation_result = calculate_percolation(compound, acceptor_labels, donor_labels, traj.boxsize, max_cutoff, max_distance)

            if len(percolation_result) > 0:
                percolation_accumulator += percolation_result
                num_frames += 1
                print(f"Processed frame: {num_frames}")

            # Read the next frame
            traj.read_frame()
        except ValueError:
            # End of the trajectory file
            break

    if num_frames == 0:
        print("No frames processed. Check the input trajectory file and parameters.")
        return

    # Average the percolation results over all frames
    percolation_average = percolation_accumulator / num_frames

    # Output the averaged percolation results
    print("\nAveraged Percolation Pathways Results:")
    for i, value in enumerate(percolation_average):
        print(f"Depth: {i+1}, Count: {value:.4f}")

def calculate_percolation(compound, acceptor_labels, donor_labels, boxsize, max_cutoff, max_distance):
    dimx, dimy, dimz = boxsize

    acceptor_coords = np.array([molecule.coords[molecule.label_to_id[label.strip()]] for molecule in compound.members for label in acceptor_labels])
    donor_coords = np.array([molecule.coords[molecule.label_to_id[label.strip()]] for molecule in compound.members for label in donor_labels])

    if acceptor_coords.size == 0 or donor_coords.size == 0:
        print("No acceptor or donor atoms found in the specified labels.")
        return np.zeros(max_cutoff)

    acceptor_tree = cKDTree(acceptor_coords, boxsize=boxsize)
    donor_tree = cKDTree(donor_coords, boxsize=boxsize)

    results = []

    for acceptor_coord in acceptor_coords:
        visited = set()
        queue = [(acceptor_coord, 0)]
        depth_count = np.zeros(max_cutoff)

        while queue:
            current_coord, depth = queue.pop(0)
            if depth >= max_cutoff:
                continue

            if tuple(current_coord) in visited:
                continue
            visited.add(tuple(current_coord))

            depth_count[depth] += 1  # Increment the count for the current depth

            neighbors = donor_tree.query_ball_point(current_coord, max_distance)
            for neighbor in neighbors:
                neighbor_coord = donor_coords[neighbor]
                if tuple(neighbor_coord) not in visited:
                    if are_hbonded(current_coord, neighbor_coord, max_distance, dimx, dimy):
                        queue.append((neighbor_coord, depth + 1))

        results.append(depth_count)

    if len(results) == 0:
        return np.zeros(max_cutoff)

    # Average results across all molecules
    average_results = np.mean(results, axis=0)
    return average_results

def are_hbonded(coord1, coord2, max_distance, dimx, dimy):
    delta = coord1 - coord2

    if dimx:
        delta[0] -= np.rint(delta[0] / dimx) * dimx
    if dimy:
        delta[1] -= np.rint(delta[1] / dimy) * dimy

    distance = np.sum(delta**2)
    return distance < max_distance**2

