# analyses/rdf_analysis.py
import numpy as np
from scipy.spatial import cKDTree
from utils import label_matches, prompt, prompt_int, prompt_float, prompt_yn, prompt_choice

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

    start_frame =  prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes =      prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)
    frame_idx = 0
    processed_frames = 0

    # Loop through all frames
    if (start_frame > 1):
        print(f"Skipping forward to frame {start_frame}.")
        while (frame_idx < start_frame - 1):
            traj.read_frame()
            frame_idx += 1


    while (nframes != 0):
        try:
            # Update the coordinates and COMs for each compound
            for compound in traj.compounds.values():
                for molecule in compound.members:
                    molecule.update_coords(traj.coords)
                compound.update_coms(traj.box_size)

            # Perform the RDF calculation for the current frame
            ref_compound = list(traj.compounds.values())[ref_index]
            obs_compound = list(traj.compounds.values())[obs_index]
            rdf_result = calculate_rdf(ref_compound, obs_compound, ref_label, obs_label, traj.box_size, max_distance, bin_count)

            # Accumulate the RDF results
            for i, (distance, value) in enumerate(rdf_result):
                rdf_accumulator[i] += value

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

    # Average the RDF results over all frames
    rdf_average = [value / processed_frames for value in rdf_accumulator]

    # Output the averaged RDF results
    with open(f"rdf_{ref_label}_{obs_label}.dat", "w") as f:
        f.write("r/pm   g(r)\n")
        for i, value in enumerate(rdf_average):
            distance = (i + 0.5) * (max_distance / bin_count)
            f.write(f"{distance:.4f} {value:.8f}\n")

    print(f"\nRDF results saved to rdf_{ref_label}_{obs_label}.dat")


def calculate_rdf(ref_compound, obs_compound, ref_label, obs_label, box_size, max_distance, bin_count):
    dimx, dimy, dimz = box_size
    bin_width = max_distance / bin_count
    rdf = np.zeros(bin_count)

    ref_coords = ref_compound.get_coords(ref_label)
    obs_coords = obs_compound.get_coords(obs_label)

    # Use KDTree for efficient distance calculations
    obs_tree = cKDTree(obs_coords, boxsize=box_size)

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

