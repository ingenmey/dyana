import numpy as np
from utils import prompt

def density(traj):
    # Prompt user for density analysis parameters
    axis = prompt("Choose the axis for density analysis (x, y, z): ", "z").lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError("Invalid axis. Choose either 'x', 'y', or 'z'.")

    axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
    step_size = float(prompt("Enter the step size for density calculation (in Å): ", 0.1))

    # Initialize the density accumulator
    num_bins = int(np.ceil(traj.boxsize[axis_index] / step_size))
    densities = {compound_name: np.zeros(num_bins) for compound_name in traj.compounds.keys()}
    num_frames = 0

    # Loop through all frames
    while True:
        try:
            # Update the coordinates and COMs for each compound
            for compound in traj.compounds.values():
                for molecule in compound.members:
                    molecule.update_coords(traj.coords)
                compound.update_coms(traj.boxsize)

            # Perform the density calculation for the current frame
            frame_densities = calculate_density(traj, axis_index, step_size, num_bins)
            for compound_name in densities.keys():
                densities[compound_name] += frame_densities[compound_name]

            num_frames += 1
            print(f"Frame {num_frames} processed.")

            # Read the next frame
            traj.read_frame()
        except ValueError:
            # End of the trajectory file
            break

    # Average the density results over all frames
    for compound_name in densities.keys():
        densities[compound_name] /= num_frames

    # Output the averaged density results
    print("\nAveraged Density Results:")
    bin_positions = np.arange(num_bins) * step_size #+ step_size / 2
    header = "r (Å)" + ''.join([f"    {compound.rep}" for compound in traj.compounds.values()])
    print(header)
    for i in range(num_bins):
        row = f"{bin_positions[i]:.4f}" + ''.join([f"    {densities[compound_name][i]:.4f}" for compound_name in densities.keys()])
        print(row)

def calculate_density(traj, axis_index, step_size, num_bins):
    density = {compound_name: np.zeros(num_bins) for compound_name in traj.compounds.keys()}

    for compound_name, compound in traj.compounds.items():
        coms = np.array([molecule.com[axis_index] for molecule in compound.members])
        lower_bins = np.floor(coms / step_size).astype(int)
        upper_bins = lower_bins + 1

        lower_bin_pos = lower_bins * step_size
        upper_bin_pos = upper_bins * step_size

        lower_weights = (upper_bin_pos - coms) / step_size
        upper_weights = (coms - lower_bin_pos) / step_size

        valid_lower_bins = (lower_bins >= 0) & (lower_bins < num_bins)
        valid_upper_bins = (upper_bins >= 0) & (upper_bins < num_bins)

        np.add.at(density[compound_name], lower_bins[valid_lower_bins], lower_weights[valid_lower_bins])
        np.add.at(density[compound_name], upper_bins[valid_upper_bins], upper_weights[valid_upper_bins])

    return density

