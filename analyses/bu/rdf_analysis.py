# analyses/rdf_analysis.py
import traceback
import numpy as np
from scipy.spatial import cKDTree
from utils import label_matches, prompt, prompt_int, prompt_float, prompt_yn, prompt_choice

def rdf(traj):
    # Prompt user for RDF parameters
    ref_index = prompt_int("Choose the reference compound (number): ", 1, minval=1) - 1
    obs_index = prompt_int("Choose the observed compound (number): ", 1, minval=1) - 1
    ref_label = prompt("Which atom in reference compound? ")
    obs_label = prompt("Which atom in observed compound? ")
    max_distance = prompt_float("Enter the maximum distance for RDF calculation (in Å): ", 10.0)
    bin_count = prompt_int("Enter the number of bins for ADF calculation: ", 500, minval=1)

    # Initialize the RDF accumulator
    rdf_accumulator = np.zeros(bin_count)

    # Reference compounds
    ref_compound = list(traj.compounds.values())[ref_index]
    obs_compound = list(traj.compounds.values())[obs_index]

    # Precompute global atom indices for selected labels
    ref_indices = [
        idx for mol in ref_compound.members
        for label, idx in mol.label_to_global_id.items()
        if label_matches(ref_label, label)
    ]
    obs_indices = [
        idx for mol in obs_compound.members
        for label, idx in mol.label_to_global_id.items()
        if label_matches(obs_label, label)
    ]

    if not ref_indices or not obs_indices:
        print("No atoms matched the given labels.")
        return

    start_frame = prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
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
            traj.update_molecule_coords()

            # Perform the RDF calculation for the current frame
            ref_coords = traj.coords[ref_indices]
            obs_coords = traj.coords[obs_indices]

            rdf_result = calculate_rdf(ref_coords, obs_coords, traj.box_size, max_distance, bin_count)
            rdf_accumulator += rdf_result

            processed_frames += 1
            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

            for _ in range(frame_stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()

        except Exception as e:
            print(e)
            traceback.print_exc()
            break

        except ValueError:
            # End of trajectory file
            break

        except KeyboardInterrupt:
            # Graceful exit when user presses Ctrl+C
            print("\nInterrupt received! Exiting main loop and post-processing data...")
            break

    # Average the RDF results over all frames
    rdf_average = rdf_accumulator / processed_frames

    # Output the averaged RDF results
    with open(f"rdf_{ref_label}_{obs_label}.dat", "w") as f:
        f.write("r/pm   g(r)\n")
        for i, value in enumerate(rdf_average):
            distance = (i + 0.5) * (max_distance / bin_count)
            f.write(f"{distance:.4f} {value:.8f}\n")

    print(f"\nRDF results saved to rdf_{ref_label}_{obs_label}.dat")


def calculate_rdf(ref_coords, obs_coords, box_size, max_distance, bin_count):
    dimx, dimy, dimz = box_size
    bin_width = max_distance / bin_count
    rdf = np.zeros(bin_count)

    # Use KDTree for efficient distance calculations
    obs_tree = cKDTree(obs_coords, boxsize=box_size)

    for ref_coord in ref_coords:
        distances, _ = obs_tree.query(ref_coord, k=len(obs_coords), distance_upper_bound=max_distance)
        distances = distances[(distances < max_distance) & (distances > 0)]

        # Compute histogram
        bin_indices = (distances / bin_width).astype(int)
        np.add.at(rdf, bin_indices, 1)

    # Normalize RDF
    volume = dimx * dimy * dimz
    shell_volumes = (4/3) * np.pi * ((np.arange(1, bin_count + 1) * bin_width)**3 - (np.arange(bin_count) * bin_width)**3)
    norm_factor = len(ref_coords) * len(obs_coords) * shell_volumes / volume
    rdf /= norm_factor

    return rdf

