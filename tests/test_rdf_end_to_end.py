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
RDF_FIXTURES = FIXTURES / "rdf"
RDF_DYNAMIC_FIXTURES = FIXTURES / "rdf_dt"
WATER_FIXTURE = FIXTURES / "water128.xyz"
KOH_FIXTURE = FIXTURES / "koh_h2o.xyz"


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
        reference = (RDF_FIXTURES / "rdf_O-H2O_H-H2O.dat").read_text(encoding="utf-8")
        np.testing.assert_allclose(_parse_numeric_table(generated), _parse_numeric_table(reference))

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
                output = Path("rdf_O-H2O_H-H2O.dat")
                self.assertTrue(output.exists())
                generated = output.read_text(encoding="utf-8")
            finally:
                if traj is not None and getattr(traj, "fin", None) is not None and not traj.fin.closed:
                    traj.fin.close()
                os.chdir(cwd)

        reference = (RDF_FIXTURES / "rdf_O-H2O_H-H2O.dat").read_text(encoding="utf-8")
        np.testing.assert_allclose(_parse_numeric_table(generated), _parse_numeric_table(reference))

        self.assertEqual(len(setup["compound_types"]), 1)

    def test_dynamic_topology_rdf_matches_reference_for_h3o2_fixture(self):
        generated = self._run_main_workflow(
            traj_path=KOH_FIXTURE,
            input_log=RDF_DYNAMIC_FIXTURES / "input.log",
            output_name="rdf_O-H3O2_H1-H3O2.dat",
        )
        reference = (RDF_DYNAMIC_FIXTURES / "rdf_O-H3O2_H1-H3O2.dat").read_text(encoding="utf-8")
        np.testing.assert_allclose(_parse_numeric_table(generated), _parse_numeric_table(reference))

    def test_static_topology_h3o2_rdf_differs_from_dynamic_reference(self):
        from analyses.rdf_analysis import RDF, RDFConfig
        from core.trajectory_loader import load_trajectory

        generated = None
        traj = None
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with open(KOH_FIXTURE, "r", encoding="utf-8") as fin:
                    traj = load_trajectory(fin, "xyz", np.array([22.7274, 22.7274, 22.7274]))
                    traj.read_frame()
                    traj.rebuild_topology()

                    analysis = RDF(traj)
                    analysis.configure(
                        RDFConfig(
                            ref_compound_index=1,
                            obs_compound_index=1,
                            ref_labels=["O"],
                            obs_labels=["H1"],
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
                    output = Path("rdf_O-H3O2_H1-H3O2.dat")
                    self.assertTrue(output.exists())
                    generated = output.read_text(encoding="utf-8")
            finally:
                if traj is not None and getattr(traj, "fin", None) is not None and not traj.fin.closed:
                    traj.fin.close()
                os.chdir(cwd)

        static_table = _parse_numeric_table(generated)
        dynamic_reference = _parse_numeric_table(
            (RDF_DYNAMIC_FIXTURES / "rdf_O-H3O2_H1-H3O2.dat").read_text(encoding="utf-8")
        )
        self.assertGreater(np.max(np.abs(static_table - dynamic_reference)), 10.0)

    def _run_rdf(self, input_log):
        return self._run_main_workflow(
            traj_path=WATER_FIXTURE,
            input_log=input_log,
            output_name="rdf_O-H2O_H-H2O.dat",
        )

    def _run_main_workflow(self, traj_path, input_log, output_name):
        import main as dyana_main

        provider = FileInputProvider(file_path=input_log, fallback=NullInputProvider())
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                dyana_main.main(
                    str(traj_path),
                    input_provider=provider,
                )
                output = Path(output_name)
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
