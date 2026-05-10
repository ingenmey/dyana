#!/usr/bin/env python3
"""CLI entry point for Dyana's supported interactive workflow."""

import argparse
import importlib
import json
import os
from pathlib import Path

from core import constants
from io_support.input_providers import FileInputProvider, InteractiveInputProvider
from io_support.output_writer import configure_output, restore_output
from workflow.workflow_prompts import WorkflowPrompts

AVAILABLE_ANALYSES = {
    "rdf": ("Radial distribution function analysis", "analyses.rdf_analysis", "RDF"),
    "adf": ("Angular distribution function analysis", "analyses.adf_analysis", "ADF"),
    "dens": ("Particle density analysis", "analyses.density_analysis", "DensityAnalysis"),
    "ncount": ("Neighbour-count probability", "analyses.neighbor_count_analysis", "NeighborCountAnalysis"),
    # "adf3b": ("Threebody Angular distribution function analysis", "analyses.adf3b_analysis", "ADFThreeBody"),
    # "percolation": ("Hydrogen bond percolation analysis", "analyses.percolation_analysis", "PercolationAnalysis"),
    # "cluster": ("Cluster composition histogram", "analyses.cluster_analysis", "ClusterAnalysis"),
    # "dacf": ("Dimer existence auto-correlation function", "analyses.dacf_analysis", "DACFAnalysis"),
    # "top": ("Tetrahedral order parameter", "analyses.top_analysis", "TetrahedralOrderAnalysis"),
    # "pccf": ("Proton coupling correlation function", "analyses.pccf_analysis", "PCCFAnalysis"),
    # "cmsd": ("Charge mean square displacement", "analyses.charge_msd_analysis", "ChargeMSDAnalysis"),
}


def determine_traj_format(traj_file):
    """Infer trajectory format from the filename extension."""
    _, ext = os.path.splitext(traj_file)
    ext = ext.lower()
    if ext in constants.EXT_XYZ:
        return "xyz"
    if ext in constants.EXT_LAMMPS:
        return "lammps"
    raise ValueError(f"Unsupported file extension: {ext}")


def choose_analysis(input_provider):
    """Prompt for one of the currently supported analyses."""
    print("\nAvailable analyses:")
    for key, (description, _, _) in AVAILABLE_ANALYSES.items():
        print(f"{key}: {description}")

    while True:
        analysis_choice = input_provider.ask_str("\nChoose an analysis: ").strip()
        if analysis_choice in AVAILABLE_ANALYSES:
            _, module_name, class_name = AVAILABLE_ANALYSES[analysis_choice]
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        print("Invalid choice. Please choose an analysis from the above list.")


def _load_output_defaults(config_path: str | Path | None = None) -> dict[str, bool]:
    path = Path(config_path) if config_path is not None else Path(__file__).resolve().with_name("config.json")
    try:
        with open(path, "r", encoding="utf-8") as fin:
            raw = json.load(fin)
    except (OSError, json.JSONDecodeError):
        return {"force_overwrite": False}

    return {
        "force_overwrite": bool(raw.get("OUTPUT_FORCE_DEFAULT", False)),
    }


def _resolve_log_path(output_dir: str | Path, log_path: str | None) -> str:
    if log_path is not None:
        return log_path
    return str(Path(output_dir) / "input.log")


def main(
    traj_file,
    input_provider=None,
    prepared_setup=None,
    save_prepared_setup_path=None,
    output_dir=".",
    force_overwrite=False,
):
    """Run the interactive workflow for one trajectory file."""
    workflow_prompts = WorkflowPrompts(input_provider=input_provider)
    input_provider = workflow_prompts.input_provider
    traj = None
    previous_output_policy = configure_output(output_dir=output_dir, force_overwrite=force_overwrite)

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
        restore_output(previous_output_policy)


def cli():
    """Parse CLI arguments and run the supported Dyana entry path."""
    output_defaults = _load_output_defaults()
    parser = argparse.ArgumentParser(description="Molecular dynamics trajectory analyzer.")
    parser.add_argument("traj_file", type=str, help="Path to the trajectory file in XYZ format")
    parser.add_argument("-i", "--input", type=str, help="Path to the input file")
    parser.add_argument("-o", "--output-dir", type=str, default=".", help="Directory for managed analysis output files")
    parser.add_argument(
        "--force",
        action="store_true",
        default=output_defaults["force_overwrite"],
        help="Overwrite existing managed analysis output files instead of rotating older files",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default=None,
        help="Path to the input log file (defaults to <output-dir>/input.log)",
    )
    parser.add_argument("--prepared-setup", type=str, help="Path to a prepared setup JSON file")
    parser.add_argument("--save-prepared-setup", type=str, help="Write the accepted prepared setup to this JSON file")
    args = parser.parse_args()
    log_path = _resolve_log_path(args.output_dir, args.log)

    if args.input is not None:
        input_provider = FileInputProvider(
            file_path=args.input,
            fallback=InteractiveInputProvider(),
            log_path=log_path,
        )
    else:
        input_provider = InteractiveInputProvider(log_path=log_path)

    try:
        main(
            args.traj_file,
            input_provider=input_provider,
            prepared_setup=args.prepared_setup,
            save_prepared_setup_path=args.save_prepared_setup,
            output_dir=args.output_dir,
            force_overwrite=args.force,
        )
    finally:
        close = getattr(input_provider, "close", None)
        if close:
            close()


if __name__ == "__main__":
    cli()
