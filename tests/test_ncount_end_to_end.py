import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

from config_schema import FrameLoopConfig
from input_providers import FileInputProvider, NullInputProvider


FIXTURES = Path(__file__).resolve().parent / "fixtures"
NCOUNT_FIXTURES = FIXTURES / "ncount"
LAMMPS_FIXTURE = FIXTURES / "ca(bf4)2_thf.lmp"


def _required_deps_available():
    return (
        importlib.util.find_spec("numpy") is not None
        and importlib.util.find_spec("networkx") is not None
        and importlib.util.find_spec("scipy") is not None
    )


@unittest.skipUnless(_required_deps_available(), "numpy, networkx, and scipy are required for neighbor-count end-to-end tests")
class NeighborCountEndToEndTests(unittest.TestCase):
    maxDiff = None

    def test_scripted_input_log_reproduces_reference_ncount(self):
        generated = self._run_ncount(input_log=NCOUNT_FIXTURES / "input.log")
        reference = (NCOUNT_FIXTURES / "ncount_O-C4H8O_O-C4H8O.dat").read_text(encoding="utf-8")
        np.testing.assert_allclose(_parse_numeric_table(generated), _parse_numeric_table(reference))

    def test_programmatic_prepared_setup_path_reproduces_reference_ncount(self):
        from analyses.neighbor_count_analysis import NeighborCountAnalysis, NeighborCountConfig
        from workflow_prompts import WorkflowPrompts

        setup = json.loads((NCOUNT_FIXTURES / "setup.json").read_text(encoding="utf-8"))
        workflow = WorkflowPrompts()

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            traj = None
            try:
                os.chdir(tmp)
                traj = workflow.prepare_trajectory_from_setup(str(LAMMPS_FIXTURE), str(NCOUNT_FIXTURES / "setup.json"))
                analysis = NeighborCountAnalysis(traj)
                analysis.configure(
                    NeighborCountConfig(
                        ref_compound_index=1,
                        ref_labels=["O"],
                        obs_compound_indices=[1],
                        obs_labels_per_compound={1: ["O"]},
                        exclude_same_molecule=True,
                        r_cut=4.0,
                    )
                )
                analysis.configure_frame_loop(
                    FrameLoopConfig(
                        start_frame=1,
                        nframes=-1,
                        frame_stride=1,
                        update_compounds=False,
                    )
                )
                analysis.run()
                output = Path("ncount_O-C4H8O_O-C4H8O.dat")
                self.assertTrue(output.exists())
                generated = output.read_text(encoding="utf-8")
            finally:
                if traj is not None and getattr(traj, "fin", None) is not None and not traj.fin.closed:
                    traj.fin.close()
                os.chdir(cwd)

        reference = (NCOUNT_FIXTURES / "ncount_O-C4H8O_O-C4H8O.dat").read_text(encoding="utf-8")
        np.testing.assert_allclose(_parse_numeric_table(generated), _parse_numeric_table(reference))
        self.assertEqual(len(setup["compound_types"]), 3)

    def _run_ncount(self, input_log):
        import main as dyana_main

        provider = FileInputProvider(file_path=input_log, fallback=NullInputProvider())
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                dyana_main.main(
                    str(LAMMPS_FIXTURE),
                    input_provider=provider,
                )
                output = Path("ncount_O-C4H8O_O-C4H8O.dat")
                self.assertTrue(output.exists())
                return output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)
                provider.close()


def _parse_numeric_table(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(value) for value in stripped.split()])
    return np.array(rows, dtype=float)


if __name__ == "__main__":
    unittest.main()
