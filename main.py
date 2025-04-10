import argparse
import os
import constants
from core.trajectory_loader import load_trajectory
from analyses.rdf_analysis import rdf
from analyses.adf_analysis import adf
from analyses.density_analysis import density
from analyses.surface_voronoi import surface_voronoi
from analyses.percolation import percolation
from analyses.cluster_analysis import cluster
from utils import prompt, set_input_file, set_log_file, close_log_file

AVAILABLE_ANALYSES = {
    'rdf': ('Radial distribution function analysis', rdf),
    'adf': ('Angular distribution function analysis', adf),
    'dens': ('Particle density analysis', density),
    'voro2d': ('2D surface voronoi analysis', surface_voronoi),
    'percolation': ('Hydrogen bond percolation analysis', percolation),
    'cluster': ('Cluster composition histogram', cluster)
}

def determine_traj_format(traj_file):
    _, ext = os.path.splitext(traj_file)
    ext = ext.lower()
    if ext in constants.EXT_XYZ:
        return 'xyz'
    elif ext in constants.EXT_LAMMPS:
        return 'lammps'
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def get_cell_vectors(traj_format):
    if traj_format == 'xyz':
        dimx = float(prompt("Enter cell vector length in X dimension (in Å): "))
        dimy = float(prompt("Enter cell vector length in Y dimension (in Å): "))
        dimz = float(prompt("Enter cell vector length in Z dimension (in Å): "))
        return [dimx, dimy, dimz]
    else:
        return [0, 0, 0]

def process_compounds(traj):
        traj.guess_molecules()

        # Print the list of different molecule types
        for i, compound in enumerate(traj.compounds.values()):
            count = len(compound.members)
            print(f"Compound {i + 1}: {compound.rep}, Number: {count}")

        should_draw_compounds = prompt("Draw compounds to PDF?", "No").lower().startswith('y')
        if (should_draw_compounds):
            for compound in traj.compounds.values():
                compound.members[0].draw_graph(compound.comp_id+1)

        for i, compound in enumerate(traj.compounds.values()):
            compound.average_bond_lengths()
            print(f"\nCompound {i + 1} Bond Length Matrix:")

            # Extract labels and initialize the matrix
            labels = list(compound.members[0].label_to_id.keys())
            size = len(labels)
            matrix = [["-  " for _ in range(size)] for _ in range(size)]
            label_to_index = {label: idx for idx, label in enumerate(labels)}

            # Fill the matrix with bond lengths
            for bond, length in compound.bond_lengths.items():
                label1, label2 = bond.split()
                idx1, idx2 = label_to_index[label1], label_to_index[label2]
                matrix[idx1][idx2] = f"{length:.4f}"
                matrix[idx2][idx1] = f"{length:.4f}"  # Ensure the matrix is symmetric

            # Print the matrix
            header = " ".join(f"{label:>8}" for label in labels)
            print(f"     {header}")
            for idx, label in enumerate(labels):
                row = " ".join(f"{val:>8}" for val in matrix[idx])
                print(f"{label:>5} {row}")


def choose_analysis():
    # List available analyses and prompt the user to choose one
    print("\nAvailable analyses:")
    for key, (description, _) in AVAILABLE_ANALYSES.items():
        print(f"{key}: {description}")

    while True:
        analysis_choice = prompt("\nChoose an analysis: ")
        if analysis_choice in AVAILABLE_ANALYSES:
            _, analysis_func = AVAILABLE_ANALYSES[analysis_choice]
            return analysis_func
        else:
            print("Invalid choice. Please choose an analysis from the above list.")


def main(traj_file):
    traj_format = determine_traj_format(traj_file)
    cell_vectors = get_cell_vectors(traj_format)

    with open(traj_file, 'r') as fin:
        traj = load_trajectory(fin, traj_format, cell_vectors)
        traj.read_frame()

        box_size = traj.box_size
        print(f"\n\nCell vectors: a = {box_size[0]}, b = {box_size[1]}, c = {box_size[2]}\n")

        process_compounds(traj)
        analysis_func = choose_analysis()
        analysis_func(traj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Molecular dynamics trajectory analyzer.")
    parser.add_argument('traj_file', type=str, help="Path to the trajectory file in XYZ format")
    parser.add_argument('-i', '--input', type=str, help="Path to the input file")
    parser.add_argument('-l', '--log', type=str, default='input.log', help="Path to the log file")
    args = parser.parse_args()

    if args.input is not None:
        set_input_file(args.input)

    if args.log is not None:
        set_log_file(args.log)

    try:
        main(args.traj_file)
    finally:
        close_log_file()
