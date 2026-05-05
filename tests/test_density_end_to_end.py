import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from config_schema import FrameLoopConfig
from input_providers import FileInputProvider, NullInputProvider


FIXTURES = Path(__file__).resolve().parent / "fixtures"
DENSITY_FIXTURES = FIXTURES / "density"
LAMMPS_FIXTURE = FIXTURES / "ca(bf4)2_thf.lmp"


def _required_deps_available():
    return (
        importlib.util.find_spec("numpy") is not None
        and importlib.util.find_spec("networkx") is not None
        and importlib.util.find_spec("scipy") is not None
    )


@unittest.skipUnless(_required_deps_available(), "numpy, networkx, and scipy are required for density end-to-end tests")
class DensityEndToEndTests(unittest.TestCase):
    maxDiff = None

    def test_scripted_input_log_reproduces_reference_density(self):
        generated = self._run_density(input_log=DENSITY_FIXTURES / "input.log")
        reference = (DENSITY_FIXTURES / "density.dat").read_text(encoding="utf-8")
        self.assertEqual(generated, reference)

    def test_programmatic_prepared_setup_path_reproduces_reference_density(self):
        from analyses.density_analysis import DensityAnalysis, DensityConfig
        from workflow_prompts import WorkflowPrompts

        setup = json.loads((DENSITY_FIXTURES / "setup.json").read_text(encoding="utf-8"))
        workflow = WorkflowPrompts()

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            traj = None
            try:
                os.chdir(tmp)
                traj = workflow.prepare_trajectory_from_setup(str(LAMMPS_FIXTURE), str(DENSITY_FIXTURES / "setup.json"))
                analysis = DensityAnalysis(traj)
                analysis.configure(
                    DensityConfig(
                        axis="z",
                        step_size=0.1,
                        per_compound_normalization=False,
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
                output = Path("density.dat")
                self.assertTrue(output.exists())
                generated = output.read_text(encoding="utf-8")
            finally:
                if traj is not None and getattr(traj, "fin", None) is not None and not traj.fin.closed:
                    traj.fin.close()
                os.chdir(cwd)

        reference = (DENSITY_FIXTURES / "density.dat").read_text(encoding="utf-8")
        self.assertEqual(generated, reference)
        self.assertEqual(len(setup["compound_types"]), 3)

    def _run_density(self, input_log):
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
                output = Path("density.dat")
                self.assertTrue(output.exists())
                return output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)
                provider.close()


if __name__ == "__main__":
    unittest.main()
