#!/usr/bin/env python3

import argparse
import os
import constants
import numpy as np
from core.trajectory_loader import load_trajectory
from analyses.rdf_analysis import RDF as rdf
from analyses.adf_analysis import ADF as adf
from analyses.adf3b_analysis import ADFThreeBody as adf3b
from analyses.density_analysis import DensityAnalysis as density
#from analyses.surface_voronoi import surface_voronoi
from analyses.percolation_analysis import PercolationAnalysis as percolation
from analyses.cluster_analysis import ClusterAnalysis as cluster
from analyses.dacf_analysis import DACFAnalysis as dacf
from analyses.top_analysis import TetrahedralOrderAnalysis as top
from analyses.pccf_analysis import PCCFAnalysis as pccf
from analyses.charge_msd_analysis import ChargeMSDAnalysis as cmsd
#from analyses.cdf_analysis import cdf
from analyses.neighbor_count_analysis import NeighborCountAnalysis as ncount
from utils import set_input_file, set_log_file, close_log_file
from utils import prompt, prompt_int, prompt_float, prompt_yn, prompt_choice

AVAILABLE_ANALYSES = {
    'rdf': ('Radial distribution function analysis', rdf),
    'adf': ('Angular distribution function analysis', adf),
    'adf3b': ('Threebody Angular distribution function analysis', adf3b),
    'dens': ('Particle density analysis', density),
#    'voro2d': ('2D surface voronoi analysis', surface_voronoi),
    'percolation': ('Hydrogen bond percolation analysis', percolation),
    'cluster': ('Cluster composition histogram', cluster),
    'dacf': ('Dimer existence auto-correlation function', dacf),
    'top': ('Tetrahedral order parameter', top),
    'pccf': ('Proton coupling correlation function', pccf),
    'cmsd': ('Charge mean square displacement', cmsd),
#    'cdf': ('Combined distribution function analysis', cdf),
    'ncount': ('Neighbour-count probability', ncount),
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
        dimx = prompt_float("Enter cell vector length in X dimension (in Å): ")
        dimy = prompt_float("Enter cell vector length in Y dimension (in Å): ")
        dimz = prompt_float("Enter cell vector length in Z dimension (in Å): ")
        return np.array([dimx, dimy, dimz])
    else:
        return np.array([0, 0, 0])

def process_compounds(traj):
        frame_idx = 0
        traj.guess_molecules()

        while True:
            # Print the list of different molecule types
            for i, compound in enumerate(traj.compounds.values()):
                count = len(compound.members)
                print(f"Compound {i + 1}: {compound.rep}, Number: {count}")

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

            should_draw_compounds = prompt_yn("Draw compounds to PDF?", False)
            if (should_draw_compounds):
                for compound in traj.compounds.values():
                    compound.members[0].draw_graph(compound.comp_id+1)

            is_keep_compounds = prompt_yn("Accept these molecules (y) or change something (n)", True)

            if is_keep_compounds:
                if frame_idx > 0:
                    traj.reset_frame_idx()
                break

            should_break = prompt_int("Break bonds (1) or repeat molecule recognition at specific frame (2)?", 1)

            if should_break == 1:
                # Otherwise allow user to break bonds
                break_bonds(traj)
            else:
                frame_idx = skip_to_frame(traj, frame_idx)

            # Re-guess molecules with forbidden bonds
            traj.guess_molecules()


def break_bonds(traj):
    while True:
        print("\nCurrent Compounds:")
        for i, compound in enumerate(traj.compounds.values(), start=1):
            print(f"{i}: {compound.rep}")

        comp_id = prompt_int("Which compound to modify?", -1, "[done]") - 1

        if comp_id < 0:
            break

        try:
            compound = list(traj.compounds.values())[comp_id]
        except (ValueError, IndexError):
            print("Invalid compound number.")
            continue

        atom1 = prompt("First atom label to break bond (e.g., O1): ").strip()
        atom2 = prompt("Second atom label to break bond (e.g., H2): ").strip()

        # Look up global atom indices
        try:
            idx1 = compound.members[0].label_to_global_id[atom1]
            idx2 = compound.members[0].label_to_global_id[atom2]
        except KeyError:
            print("Invalid atom label(s). Try again.")
            continue

        # Add forbidden bond for all molecules
        for molecule in compound.members:
            global_idx1 = molecule.label_to_global_id[atom1]
            global_idx2 = molecule.label_to_global_id[atom2]
            traj.forbidden_bonds.add((min(global_idx1, global_idx2), max(global_idx1, global_idx2)))

        print(f"Added forbidden bond between {atom1} and {atom2}.")


def skip_to_frame(traj, frame_idx):
    target_frame = prompt_int("Skip to which frame?", 0)

    if target_frame > frame_idx:
        nframes = target_frame - frame_idx
    else:
        traj.reset_frame_idx()
        nframes = target_frame
        frame_idx = 0

    for _ in range(nframes):
        frame_idx += 1
        traj.read_frame()

    return frame_idx


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
       # analysis_func(traj)
        analysis = analysis_func(traj)
        analysis.run()

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
