import importlib.util
import os
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from config_schema import FrameLoopConfig
from input_providers import FileInputProvider, NullInputProvider

if importlib.util.find_spec("scipy") is None:
    RDFConfig = None
else:
    from analyses.rdf_analysis import RDF, RDFConfig


class DummyMolecule:
    def __init__(self):
        self.label_to_global_id = {"O1": 0, "H1": 1}


class DummyCompound:
    def __init__(self):
        self.members = [DummyMolecule()]
        self.rep = "OH"
        self.comp_id = 0


class DummyTrajectory:
    def __init__(self):
        self.box_size = np.array([10.0, 10.0, 10.0])
        self.coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.compounds = OrderedDict([(("OH", (), "hash"), DummyCompound())])
        self.read_calls = 0

    def update_molecule_coords(self):
        return None

    def read_frame(self):
        self.read_calls += 1
        raise ValueError("End of trajectory")


@unittest.skipIf(RDFConfig is None, "scipy is not installed")
class RDFConfigTests(unittest.TestCase):
    def test_rdf_config_validates_inputs(self):
        RDFConfig(
            ref_compound_index=0,
            obs_compound_index=0,
            ref_labels=["O"],
            obs_labels=["H"],
        )

        with self.assertRaises(ValueError):
            RDFConfig(
                ref_compound_index=-1,
                obs_compound_index=0,
                ref_labels=["O"],
                obs_labels=["H"],
            )
        with self.assertRaises(ValueError):
            RDFConfig(
                ref_compound_index=0,
                obs_compound_index=0,
                ref_labels=[],
                obs_labels=["H"],
            )
        with self.assertRaises(ValueError):
            RDFConfig(
                ref_compound_index=0,
                obs_compound_index=0,
                ref_labels=["O"],
                obs_labels=["H"],
                max_distance=0.0,
            )

    def test_configure_sets_up_selectors_and_histogram(self):
        rdf = RDF(DummyTrajectory())

        rdf.configure(
            RDFConfig(
                ref_compound_index=0,
                obs_compound_index=0,
                ref_labels=["O"],
                obs_labels=["H"],
                max_distance=2.0,
                bin_count=2,
            )
        )

        self.assertEqual(rdf.ref_indices, [0])
        self.assertEqual(rdf.obs_indices, [1])
        np.testing.assert_allclose(rdf.hist.bin_edges[0], [0.0, 1.0, 2.0])

    def test_prompt_config_uses_shared_schema_and_provider(self):
        provider = FileInputProvider(lines=["1", "1", "O", "H", "2.5", "4"], fallback=NullInputProvider())
        rdf = RDF(DummyTrajectory(), input_provider=provider)

        config = rdf.prompt_config()

        self.assertEqual(
            config,
            RDFConfig(
                ref_compound_index=0,
                obs_compound_index=0,
                ref_labels=["O"],
                obs_labels=["H"],
                max_distance=2.5,
                bin_count=4,
            ),
        )

    def test_configure_frame_loop_accepts_shared_config(self):
        rdf = RDF(DummyTrajectory())

        rdf.configure_frame_loop(FrameLoopConfig(start_frame=3, nframes=-1, frame_stride=2, update_compounds=True))

        self.assertEqual(rdf.start_frame, 3)
        self.assertEqual(rdf.nframes, -1)
        self.assertEqual(rdf.frame_stride, 2)
        self.assertTrue(rdf.update_compounds)

    def test_prompt_frame_loop_config_builds_shared_config(self):
        provider = FileInputProvider(lines=["y", "3", "-1", "2"], fallback=NullInputProvider())
        rdf = RDF(DummyTrajectory(), input_provider=provider)

        frame_loop = rdf.prompt_frame_loop_config()

        self.assertEqual(frame_loop, FrameLoopConfig(start_frame=3, nframes=-1, frame_stride=2, update_compounds=True))

    def test_process_frame_and_postprocess_write_histogram_output(self):
        rdf = RDF(DummyTrajectory())
        rdf.configure(
            RDFConfig(
                ref_compound_index=0,
                obs_compound_index=0,
                ref_labels=["O"],
                obs_labels=["H"],
                max_distance=2.0,
                bin_count=2,
            )
        )

        rdf.process_frame()
        rdf.processed_frames = 1

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                rdf.postprocess()
                output = Path("rdf_O_H.dat")
                self.assertTrue(output.exists())
                text = output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("r/Angstrom", text)
        self.assertGreater(rdf.hist.counts[1], 0.0)

    def test_run_uses_programmatic_configuration_without_prompting(self):
        rdf = RDF(DummyTrajectory(), input_provider=NullInputProvider())
        rdf.configure(
            RDFConfig(
                ref_compound_index=0,
                obs_compound_index=0,
                ref_labels=["O"],
                obs_labels=["H"],
                max_distance=2.0,
                bin_count=2,
            )
        )
        rdf.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                rdf.run()
                self.assertTrue(Path("rdf_O_H.dat").exists())
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
