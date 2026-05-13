#!/usr/bin/env python3
"""CLI entry point for Dyana's supported interactive workflow."""

import argparse
import importlib
import os
from pathlib import Path

from core import constants
from core.app_config import load_app_config
from io_support.console import console
from io_support.input_providers import FileInputProvider, InteractiveInputProvider
from io_support.output_writer import configure_output, restore_output
from io_support.run_header import build_run_header, render_run_header
from workflow.workflow_prompts import WorkflowPrompts

AVAILABLE_ANALYSES = [
    ("s", "Pair Correlation"),
    ("a", "rdf", "Radial distribution function", "analyses.rdf_analysis", "RDF"),
    ("a", "adf", "Angular distribution function", "analyses.adf_analysis", "ADF"),

    ("s", "Density / Counting"),
    ("a", "dens", "Particle density", "analyses.density_analysis", "DensityAnalysis"),
    ("a", "ncount", "Neighbour-count probability", "analyses.neighbor_count_analysis", "NeighborCountAnalysis"),

    ("s", "Local Structure"),
    ("a", "top", "Tetrahedral order parameter", "analyses.top_analysis", "TetrahedralOrderAnalysis"),
    ("a", "lsi", "Local structure index", "analyses.lsi_analysis", "LSIAnalysis"),
    ("a", "q6", "Steinhardt q6/Q6 order parameter", "analyses.q6_analysis", "Q6Analysis"),

    # ("s", "Legacy"),
    # ("a", "adf3b", "Threebody Angular distribution function", "analyses.adf3b_analysis", "ADFThreeBody"),
    # ("a", "percolation", "Hydrogen bond percolation", "analyses.percolation_analysis", "PercolationAnalysis"),
    # ("a", "cluster", "Cluster composition histogram", "analyses.cluster_analysis", "ClusterAnalysis"),
    # ("a", "dacf", "Dimer existence auto-correlation function", "analyses.dacf_analysis", "DACFAnalysis"),
    # ("a", "pccf", "Proton coupling correlation function", "analyses.pccf_analysis", "PCCFAnalysis"),
    # ("a", "cmsd", "Charge mean square displacement", "analyses.charge_msd_analysis", "ChargeMSDAnalysis"),
]


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
    console.section("Available Analyses")
    analysis_lookup = {}
    for entry in AVAILABLE_ANALYSES:
        if entry[0] == "s":
            console.plain(entry[1], style="cyan")
            continue

        _, key, description, module_name, class_name = entry
        console.key_value(key, description, indent=2)
        analysis_lookup[key] = (module_name, class_name)

    while True:
        analysis_choice = input_provider.ask_str("\nChoose an analysis: ").strip()
        if analysis_choice in analysis_lookup:
            module_name, class_name = analysis_lookup[analysis_choice]
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        console.warn("Invalid choice. Please choose an analysis from the above list.")


def _load_output_defaults(config_path: str | Path | None = None) -> dict[str, bool]:
    raw = load_app_config(config_path)
    return {
        "force_overwrite": bool(raw.get("OUTPUT_FORCE_DEFAULT", False)),
    }


def main(
    traj_file,
    input_provider=None,
    prepared_setup=None,
    save_prepared_setup_path=None,
    output_dir=".",
    force_overwrite=False,
    console_log_path=None,
):
    """Run the interactive workflow for one trajectory file."""
    workflow_prompts = WorkflowPrompts(input_provider=input_provider)
    input_provider = workflow_prompts.input_provider
    traj = None
    previous_output_policy = configure_output(output_dir=output_dir, force_overwrite=force_overwrite)
    previous_console_state = console.capture_state()
    resolved_console_log_path = console_log_path or str(Path(output_dir) / "dyana.log")
    console.configure(log_path=resolved_console_log_path)
    if prepared_setup is None:
        traj_format = determine_traj_format(traj_file)
    else:
        try:
            traj_format = determine_traj_format(traj_file)
        except ValueError:
            traj_format = None

    try:
        input_log_path = getattr(input_provider, "log_file", None)
        header = build_run_header(
            traj_file,
            traj_format=traj_format,
            output_dir=output_dir,
            console_log_path=resolved_console_log_path,
            input_log_path=getattr(input_log_path, "name", None),
            prepared_setup=prepared_setup,
        )
        render_run_header(console, header)

        if prepared_setup is not None:
            traj = workflow_prompts.prepare_trajectory_from_setup(traj_file, prepared_setup)
        else:
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
        console.close()
        console.restore_state(previous_console_state)
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
    output_dir = args.output_dir
    log_path = args.log or str(Path(output_dir) / "input.log")
    console_log_path = str(Path(output_dir) / "dyana.log")

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
            output_dir=output_dir,
            force_overwrite=args.force,
            console_log_path=console_log_path,
        )
    finally:
        close = getattr(input_provider, "close", None)
        if close:
            close()


if __name__ == "__main__":
    cli()
