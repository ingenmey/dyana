import argparse
import os
from core.trajectory_loader import load_trajectory
from analyses.rdf_analysis import rdf
from analyses.adf_analysis import adf
from analyses.density_analysis import density
from analyses.surface_voronoi import surface_voronoi
from analyses.percolation import percolation
from analyses.cluster_analysis import cluster
from utils import prompt, set_input_file, set_log_file, close_log_file

available_analyses = {
    'rdf': ('Radial distribution function analysis', rdf),
    'adf': ('Angular distribution function analysis', adf),
    'dens': ('Particle density analysis', density),
    'voro2d': ('2D surface voronoi analysis', surface_voronoi),
    'percolation': ('Hydrogen bond percolation analysis', percolation),
    'cluster': ('Cluster composition histogram', cluster)
}

def list_analyses():
    print("\nAvailable analyses:")
    for key, (description, _) in available_analyses.items():
        print(f"{key}: {description}")

def determine_format(traj_file):
    _, ext = os.path.splitext(traj_file)
    ext = ext.lower()
    if ext == '.xyz':
        return 'xyz'
    elif ext in ['.lmp', '.lammpstrj']:
        return 'lammps'
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def main(traj_file):
    format = determine_format(traj_file)

    if format == 'xyz':
        dimx = float(prompt("Enter cell vector length in X dimension (in Å): "))
        dimy = float(prompt("Enter cell vector length in Y dimension (in Å): "))
        dimz = float(prompt("Enter cell vector length in Z dimension (in Å): "))
        cell_vectors = [dimx, dimy, dimz]
    else:
        cell_vectors = [0, 0, 0]

    with open(traj_file, 'r') as fin:
        traj = load_trajectory(fin, format, cell_vectors)
        traj.read_frame()
        traj.guess_molecules()

        boxsize = traj.boxsize
        print(f"\n\nCell vectors: a = {boxsize[0]}, b = {boxsize[1]}, c = {boxsize[2]}\n")

        # Print the list of different molecule types
        for i, compound in enumerate(traj.compounds.values()):
            count = len(compound.members)
            print(f"Compound {i + 1}: {compound.rep}, Number: {count}")

        draw_compounds = prompt("Draw compounds to PDF?", "No").lower().startswith('y')
        if (draw_compounds):
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

        # List available analyses and prompt the user to choose one
        list_analyses()
        while True:
            analysis_choice = prompt("\nChoose an analysis: ")
            if analysis_choice in available_analyses:
                _, analysis_func = available_analyses[analysis_choice]
                analysis_func(traj)
                break
            else:
                print("Invalid choice. Please choose an analysis from the above list.")

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
