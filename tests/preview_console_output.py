from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from io_support.console import Console


def render_dyana_console_preview(console: Console) -> None:
    """Emit a manual preview of the planned interactive console style."""
    demo_dir = (REPO_ROOT / "demo_output").resolve()
    traj_path = (REPO_ROOT / "tests" / "fixtures" / "koh_h2o.xyz").resolve()
    console.header(
        "Dyana 0.1.0",
        lines=[
            "Started: 2026-05-11 14:32:10 CEST",
            f"Trajectory: {traj_path} (xyz)",
            f"Output dir: {demo_dir}",
            f"Console log: {demo_dir / 'dyana.log'}",
            f"Input log: {demo_dir / 'input.log'}",
        ],
    )

    console.section("Topology Setup")
    console.plain("")
    console.key_value("Cell vectors", "a=18.5000  b=18.5000  c=24.0000")
    console.plain("")
    console.info("Running compound recognition...")
    console.plain("")
    console.plain("Detected compound types:")
    console.key_value("1", "H2O (Number: 127)", indent=2)
    console.key_value("2", "OH- (Number: 1)", indent=2)
    console.key_value("3", "K+ (Number: 1)", indent=2)
    console.key_value("4", "H3O2 (Number: 1)", indent=2)

    console.plain("")
    console.plain("Compound 4 (H3O2) bond length matrix:")
    console.plain("          O1       H1       O2")
    console.plain("   O1      -    1.0162   2.4215")
    console.plain("   H1   1.0162      -    1.4058")
    console.plain("   O2   2.4215   1.4058      -")

    console.plain("")
    console.warn("Output file compound4.pdf already existed; moved the previous file to #1#compound4.pdf.")
    console.success("Accepted compound list.")
    console.plain("")
    console.info("Running compound recognition...")
    console.plain("")
    console.plain("Detected compound types:")
    console.key_value("1", "H2O (Number: 127)", indent=2)
    console.key_value("2", "OH- (Number: 1)", indent=2)
    console.key_value("3", "K+ (Number: 1)", indent=2)
    console.key_value("4", "H3O2 (Number: 1)", indent=2)

    console.section("Analysis Setup")
    console.plain("")
    console.plain("Available compounds:")
    console.key_value("1", "H3O2 (Number: 1)", indent=2)
    console.plain("")
    console.plain("Choose an analysis:  rdf")
    console.plain("Choose the reference compound (number):  2")
    console.plain("Which atom(s) in reference compound 2 (H3O2)? (comma-separated)  O")
    console.plain("")
    console.plain("Available compounds:")
    console.key_value("1", "H3O2 (Number: 1)", indent=2)
    console.plain("")
    console.plain("Choose the observed compound (number):  2")
    console.plain("Which atom(s) in observed compound 2 (H3O2)? (comma-separated)  H1")
    console.plain("Enter the maximum distance for RDF calculation (in Angstrom):  [10.0] 6.0")
    console.plain("Enter the number of bins for RDF calculation:  [1000] 600")

    console.section("Frame Loop")
    console.plain("Perform molecule recognition and update compound list in each frame? [No] y")
    console.plain("In which trajectory frame to start processing the trajectory? [1] 1")
    console.plain("How many trajectory frames to read (from this position on)? [all] ")
    console.plain("Use every n-th read trajectory frame for the analysis: [1] 1")

    console.section("Run")
    console.plain("Running analysis...")
    console.progress("Processed 100 frames (current frame 100)")
    console.progress("Processed 200 frames (current frame 200)")
    console.progress("Processed 300 frames (current frame 300)")
    console.warn("Skipped frame 314 because the selected compound type was not present after topology update.")
    console.error("Invalid atom label(s). Try again.")
    console.info("Retrying with the previous valid selection.")
    console.progress("Processed 400 frames (current frame 401)")
    console.success("Processed 487 frames total.")

    console.section("Results")
    console.success("Saved RDF results to rdf_O-H3O2_H1-H3O2.dat")
    console.key_value("Processed frames", 487)
    console.key_value("Output", "rdf_O-H3O2_H1-H3O2.dat", indent=2)
    console.key_value("Console log", demo_dir / "dyana.log", indent=2)


def main() -> None:
    log_path = Path("demo_output") / "dyana.log"
    console = Console(log_path=log_path, use_color=True)
    try:
        render_dyana_console_preview(console)
    finally:
        console.close()


if __name__ == "__main__":
    main()
