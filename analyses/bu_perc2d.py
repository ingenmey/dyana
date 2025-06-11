import numpy as np
from scipy.spatial import cKDTree
from utils import label_matches, prompt, prompt_int, prompt_float, prompt_yn

def percolation(traj):
    # Prompt user for percolation parameters
    compound_index = int(prompt("Choose the compound (number): ")) - 1
    acceptor_labels = prompt("Enter the accepting atoms (comma separated): ").split(',')
    donor_labels = prompt("Enter the donating atoms (comma separated): ").split(',')
    max_cutoff = int(prompt("Enter the maximum number of hydrogen bonds to consider: ", 5))
    max_distance = float(prompt("Enter the maximum distance for hydrogen bonds (in Å): ", 3.5))

    # Initialize the percolation accumulator
    percolation_accumulator = np.zeros(max_cutoff)


    start_frame = prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)

    # --- Frame loop ---
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

            # Perform the percolation calculation for the current frame
            compound = list(traj.compounds.values())[compound_index]
            percolation_result = calculate_percolation(compound, acceptor_labels, donor_labels, traj.box_size, max_cutoff, max_distance)

            if len(percolation_result) > 0:
                percolation_accumulator += percolation_result
                processed_frames += 1

            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

            for _ in range(frame_stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()

        except ValueError:
            # End of trajectory file
            break

        except KeyboardInterrupt:
            # Graceful exit when user presses Ctrl+C
            print("\nInterrupt received! Exiting main loop and post-processing data...")
            break

    print()

    # Average the percolation results over all frames
    percolation_average = percolation_accumulator / processed_frames

    # Output the averaged percolation results
    print("\nAveraged Percolation Pathways Results:")
    for i, value in enumerate(percolation_average):
        print(f"Depth: {i+1}, Count: {value:.4f}")

def calculate_percolation(compound, acceptor_labels, donor_labels, box_size, max_cutoff, max_distance):
    dimx, dimy, dimz = box_size

    acceptor_coords = np.array([molecule.coords[molecule.label_to_id[label.strip()]] for molecule in compound.members for label in acceptor_labels])
    donor_coords = np.array([molecule.coords[molecule.label_to_id[label.strip()]] for molecule in compound.members for label in donor_labels])

    if acceptor_coords.size == 0 or donor_coords.size == 0:
        print("No acceptor or donor atoms found in the specified labels.")
        return np.zeros(max_cutoff)

    acceptor_tree = cKDTree(acceptor_coords, boxsize=box_size)
    donor_tree = cKDTree(donor_coords, boxsize=box_size)

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

