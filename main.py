#!/usr/bin/env python3

import argparse
import os

import constants

from analyses.adf3b_analysis import ADFThreeBody as adf3b
from analyses.adf_analysis import ADF as adf
from analyses.charge_msd_analysis import ChargeMSDAnalysis as cmsd
from analyses.cluster_analysis import ClusterAnalysis as cluster
from analyses.dacf_analysis import DACFAnalysis as dacf
from analyses.density_analysis import DensityAnalysis as density
from analyses.neighbor_count_analysis import NeighborCountAnalysis as ncount
from analyses.pccf_analysis import PCCFAnalysis as pccf
from analyses.percolation_analysis import PercolationAnalysis as percolation
from analyses.rdf_analysis import RDF as rdf
from analyses.top_analysis import TetrahedralOrderAnalysis as top
from input_providers import FileInputProvider, InteractiveInputProvider
from workflow_prompts import WorkflowPrompts

AVAILABLE_ANALYSES = {
    "rdf": ("Radial distribution function analysis", rdf),
    "adf": ("Angular distribution function analysis", adf),
    "adf3b": ("Threebody Angular distribution function analysis", adf3b),
    "dens": ("Particle density analysis", density),
    "percolation": ("Hydrogen bond percolation analysis", percolation),
    "cluster": ("Cluster composition histogram", cluster),
    "dacf": ("Dimer existence auto-correlation function", dacf),
    "top": ("Tetrahedral order parameter", top),
    "pccf": ("Proton coupling correlation function", pccf),
    "cmsd": ("Charge mean square displacement", cmsd),
    "ncount": ("Neighbour-count probability", ncount),
}


def determine_traj_format(traj_file):
    _, ext = os.path.splitext(traj_file)
    ext = ext.lower()
    if ext in constants.EXT_XYZ:
        return "xyz"
    if ext in constants.EXT_LAMMPS:
        return "lammps"
    raise ValueError(f"Unsupported file extension: {ext}")


def choose_analysis(input_provider):
    print("\nAvailable analyses:")
    for key, (description, _) in AVAILABLE_ANALYSES.items():
        print(f"{key}: {description}")

    while True:
        analysis_choice = input_provider.ask_str("\nChoose an analysis: ").strip()
        if analysis_choice in AVAILABLE_ANALYSES:
            _, analysis_func = AVAILABLE_ANALYSES[analysis_choice]
            return analysis_func
        print("Invalid choice. Please choose an analysis from the above list.")


def main(traj_file, input_provider=None, prepared_setup=None, save_prepared_setup_path=None):
    workflow_prompts = WorkflowPrompts(input_provider=input_provider)
    input_provider = workflow_prompts.input_provider
    traj = None

    try:
        if prepared_setup is not None:
            traj = workflow_prompts.prepare_trajectory_from_setup(traj_file, prepared_setup)
        else:
            traj_format = determine_traj_format(traj_file)
            traj = workflow_prompts.prepare_trajectory(
                traj_file,
                traj_format,
                provider=input_provider,
                save_prepared_setup_path=save_prepared_setup_path,
            )

        analysis_func = choose_analysis(input_provider)
        analysis = analysis_func(traj, input_provider=input_provider)
        analysis.run()
    finally:
        fin = getattr(traj, "fin", None)
        if fin is not None and not fin.closed:
            fin.close()


def cli():
    parser = argparse.ArgumentParser(description="Molecular dynamics trajectory analyzer.")
    parser.add_argument("traj_file", type=str, help="Path to the trajectory file in XYZ format")
    parser.add_argument("-i", "--input", type=str, help="Path to the input file")
    parser.add_argument("-l", "--log", type=str, default="input.log", help="Path to the log file")
    parser.add_argument("--prepared-setup", type=str, help="Path to a prepared setup JSON file")
    parser.add_argument("--save-prepared-setup", type=str, help="Write the accepted prepared setup to this JSON file")
    args = parser.parse_args()

    if args.input is not None:
        input_provider = FileInputProvider(
            file_path=args.input,
            fallback=InteractiveInputProvider(),
            log_path=args.log,
        )
    else:
        input_provider = InteractiveInputProvider()

    try:
        main(
            args.traj_file,
            input_provider=input_provider,
            prepared_setup=args.prepared_setup,
            save_prepared_setup_path=args.save_prepared_setup,
        )
    finally:
        close = getattr(input_provider, "close", None)
        if close:
            close()


if __name__ == "__main__":
    cli()
