from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from io_support.console import Console


def render_dyana_console_preview(console: Console) -> None:
    """Emit a manual preview of the planned interactive console style."""
    console.header(
        "Dyana 0.1.0",
        lines=[
            "Trajectory: koh_h2o.xyz (xyz)",
            "Output dir: demo_output",
            "Input log: demo_output/input.log",
        ],
    )

    console.section("Topology Setup")
    console.info("Loaded first frame and prepared trajectory state.")
    console.key_value("Cell vectors", "a=18.5000  b=18.5000  c=24.0000")
    console.plain("Detected compound types:")
    console.key_value("1", "H2O   x127", indent=2)
    console.key_value("2", "OH-   x1", indent=2)
    console.key_value("3", "K+    x1", indent=2)
    console.key_value("4", "H3O2  x1", indent=2)

    console.plain("")
    console.plain("Compound 4 bond length matrix:")
    console.plain("          O1       H1       O2")
    console.plain("   O1      -    1.0162   2.4215")
    console.plain("   H1   1.0162      -    1.4058")
    console.plain("   O2   2.4215   1.4058      -")

    console.warn("Output file compound4.pdf already existed; moved the previous file to #1#compound4.pdf.")
    console.success("Accepted compound list.")

    console.section("Analysis Setup")
    console.key_value("Analysis", "RDF")
    console.key_value("Reference", "O in H3O2")
    console.key_value("Observed", "H1 in H3O2")
    console.key_value("Max distance", "6.0000 Angstrom")
    console.key_value("Bins", 600)

    console.section("Frame Loop")
    console.key_value("Start frame", 1)
    console.key_value("Frames", "all")
    console.key_value("Stride", 1)
    console.key_value("Per-frame topology update", "yes")
    console.info("Per-frame molecule recognition is enabled for this run.")

    console.section("Run")
    console.plain("Running RDF...")
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
    console.key_value("Console log", "demo_output/dyana.log", indent=2)


def main() -> None:
    log_path = Path("demo_output") / "dyana.log"
    console = Console(log_path=log_path, use_color=True)
    try:
        render_dyana_console_preview(console)
    finally:
        console.close()


if __name__ == "__main__":
    main()
