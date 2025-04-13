import numpy as np
from utils import prompt, prompt_int, prompt_float, prompt_choice

def density(traj):
    # Prompt user for density analysis parameters
    axis = prompt_choice("Choose the axis for density analysis", ["x", "y", "z"], "z")
    axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
    step_size = prompt_float("Enter the step size for density calculation (in Å): ", 0.1)

    box_length = traj.box_size[axis_index]
    num_bins = int(np.ceil(box_length / step_size))
    bin_centers = (np.arange(num_bins) + 0.5) * step_size

    # Initialize density accumulators
    densities = {comp_id: np.zeros(num_bins) for comp_id in traj.compounds.keys()}

    start_frame =  prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes =      prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)
    frame_idx = 0
    processed_frames = 0

    if (start_frame > 1):
        print(f"Skipping forward to frame {start_frame}.")
        while (frame_idx < start_frame - 1):
            traj.read_frame()
            frame_idx += 1

    while (nframes != 0):
        try:
            # Update coords
            for compound in traj.compounds.values():
                for molecule in compound.members:
                    molecule.update_coords(traj.coords)
                compound.update_coms(traj.box_size)

            # Precompute COMs
            coms_per_compound = {
                comp_id: np.array([mol.com[axis_index] for mol in comp.members])
                for comp_id, comp in traj.compounds.items()
            }

            frame_densities = calculate_density(coms_per_compound, num_bins, step_size)

            # Accumulate
            for comp_id in densities.keys():
                densities[comp_id] += frame_densities[comp_id]

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

    # Normalize
    for comp_id in densities.keys():
        densities[comp_id] /= processed_frames

    # Save to file
    with open("density.dat", "w") as f:
        header = "r/Å" + ''.join([f"    {traj.compounds[k].rep}" for k in sorted(densities.keys())])
        f.write(f"{header}\n")
        for i in range(num_bins):
            row = f"{bin_centers[i]:.4f}" + ''.join([f"    {densities[k][i]:.4f}" for k in sorted(densities.keys())])
            f.write(f"{row}\n")

    print("\nDensity data saved to 'density.dat'.")

def calculate_density(coms_per_compound, num_bins, step_size):
    density = {comp_id: np.zeros(num_bins) for comp_id in coms_per_compound.keys()}

    for comp_id, coms in coms_per_compound.items():
        lower_bins = np.floor(coms / step_size).astype(int)
        upper_bins = lower_bins + 1

        lower_bin_pos = lower_bins * step_size
        upper_bin_pos = upper_bins * step_size

        lower_weights = (upper_bin_pos - coms) / step_size
        upper_weights = (coms - lower_bin_pos) / step_size

        valid_lower = (lower_bins >= 0) & (lower_bins < num_bins)
        valid_upper = (upper_bins >= 0) & (upper_bins < num_bins)

        np.add.at(density[comp_id], lower_bins[valid_lower], lower_weights[valid_lower])
        np.add.at(density[comp_id], upper_bins[valid_upper], upper_weights[valid_upper])

    return density

