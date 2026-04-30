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


def choose_analysis(workflow_prompts):
    input_provider = workflow_prompts.input_provider
    print("\nAvailable analyses:")
    for key, (description, _) in AVAILABLE_ANALYSES.items():
        print(f"{key}: {description}")

    while True:
        analysis_choice = input_provider.ask_str("\nChoose an analysis: ").strip()
        if analysis_choice in AVAILABLE_ANALYSES:
            _, analysis_func = AVAILABLE_ANALYSES[analysis_choice]
            return analysis_func
        print("Invalid choice. Please choose an analysis from the above list.")


def main(traj_file, input_provider=None):
    workflow_prompts = WorkflowPrompts(input_provider=input_provider)
    input_provider = workflow_prompts.input_provider

    traj_format = determine_traj_format(traj_file)
    traj = workflow_prompts.prepare_trajectory(traj_file, traj_format, provider=input_provider)
    analysis_func = choose_analysis(workflow_prompts)
    analysis = analysis_func(traj, input_provider=input_provider)
    analysis.run()


def cli():
    parser = argparse.ArgumentParser(description="Molecular dynamics trajectory analyzer.")
    parser.add_argument("traj_file", type=str, help="Path to the trajectory file in XYZ format")
    parser.add_argument("-i", "--input", type=str, help="Path to the input file")
    parser.add_argument("-l", "--log", type=str, default="input.log", help="Path to the log file")
    args = parser.parse_args()

    if args.input is not None or args.log is not None:
        input_provider = FileInputProvider(
            file_path=args.input,
            fallback=InteractiveInputProvider(),
            log_path=args.log,
        )
    else:
        input_provider = InteractiveInputProvider()

    try:
        main(args.traj_file, input_provider=input_provider)
    finally:
        close = getattr(input_provider, "close", None)
        if close:
            close()


if __name__ == "__main__":
    cli()
