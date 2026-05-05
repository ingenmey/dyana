import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from config_schema import FrameLoopConfig
from input_providers import FileInputProvider, NullInputProvider


FIXTURES = Path(__file__).resolve().parent / "fixtures"
RDF_FIXTURES = FIXTURES / "rdf"
WATER_FIXTURE = FIXTURES / "water128.xyz"


def _required_deps_available():
    return (
        importlib.util.find_spec("numpy") is not None
        and importlib.util.find_spec("networkx") is not None
        and importlib.util.find_spec("scipy") is not None
    )


@unittest.skipUnless(_required_deps_available(), "networkx and scipy are required for RDF end-to-end tests")
class RDFEndToEndTests(unittest.TestCase):
    maxDiff = None

    def test_scripted_input_log_reproduces_reference_rdf(self):
        generated = self._run_rdf(
            input_log=RDF_FIXTURES / "input.log",
        )
        reference = (RDF_FIXTURES / "rdf_O_H.dat").read_text(encoding="utf-8")
        self.assertEqual(generated, reference)

    def test_programmatic_prepared_setup_path_reproduces_reference_rdf(self):
        from analyses.rdf_analysis import RDF, RDFConfig
        from workflow_prompts import WorkflowPrompts

        setup = json.loads((RDF_FIXTURES / "setup.json").read_text(encoding="utf-8"))
        workflow = WorkflowPrompts()

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            traj = None
            try:
                os.chdir(tmp)
                traj = workflow.prepare_trajectory_from_setup(str(WATER_FIXTURE), str(RDF_FIXTURES / "setup.json"))
                analysis = RDF(traj)
                analysis.configure(
                    RDFConfig(
                        ref_compound_index=0,
                        obs_compound_index=0,
                        ref_labels=["O"],
                        obs_labels=["H"],
                        max_distance=10.0,
                        bin_count=1000,
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
                output = Path("rdf_O_H.dat")
                self.assertTrue(output.exists())
                generated = output.read_text(encoding="utf-8")
            finally:
                if traj is not None and getattr(traj, "fin", None) is not None and not traj.fin.closed:
                    traj.fin.close()
                os.chdir(cwd)

        reference = (RDF_FIXTURES / "rdf_O_H.dat").read_text(encoding="utf-8")
        self.assertEqual(generated, reference)

        self.assertEqual(len(setup["compound_types"]), 1)

    def _run_rdf(self, input_log):
        import main as dyana_main

        provider = FileInputProvider(file_path=input_log, fallback=NullInputProvider())
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                dyana_main.main(
                    str(WATER_FIXTURE),
                    input_provider=provider,
                )
                output = Path("rdf_O_H.dat")
                self.assertTrue(output.exists())
                return output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)
                provider.close()


if __name__ == "__main__":
    unittest.main()
