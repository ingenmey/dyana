import os
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from analyses.density_analysis import DensityAnalysis, DensityConfig
from config_schema import FrameLoopConfig
from input_providers import FileInputProvider, NullInputProvider


class DummyMolecule:
    def __init__(self, com, labels=None):
        self.com = np.array(com, dtype=float)
        self.label_to_id = labels or {"O1": 0}


class DummyCompound:
    def __init__(self, rep, comp_id, coms):
        self.rep = rep
        self.comp_id = comp_id
        self.members = [DummyMolecule(com) for com in coms]


class DummyTrajectory:
    def __init__(self):
        self.box_size = np.array([4.0, 5.0, 6.0], dtype=float)
        self.compounds = OrderedDict(
            [
                (("H2O", (), "water"), DummyCompound("H2O", 0, [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])),
                (("Na", (), "na"), DummyCompound("Na", 1, [[0.0, 0.0, 3.0]])),
            ]
        )

    def update_molecule_coords(self):
        return None

    def read_frame(self):
        raise ValueError("End of trajectory")


class DensityConfigTests(unittest.TestCase):
    def test_density_config_validates_inputs(self):
        DensityConfig(axis="z", step_size=0.1, per_compound_normalization=False)

        with self.assertRaises(ValueError):
            DensityConfig(axis="q", step_size=0.1)
        with self.assertRaises(ValueError):
            DensityConfig(axis="z", step_size=0.0)

    def test_prompt_config_uses_shared_schema_and_provider(self):
        provider = FileInputProvider(lines=["x", "0.5", "y"], fallback=NullInputProvider())
        density = DensityAnalysis(DummyTrajectory(), input_provider=provider)

        config = density.prompt_config()

        self.assertEqual(
            config,
            DensityConfig(axis="x", step_size=0.5, per_compound_normalization=True),
        )

    def test_configure_sets_up_histogram(self):
        density = DensityAnalysis(DummyTrajectory())

        density.configure(DensityConfig(axis="z", step_size=2.0, per_compound_normalization=False))

        self.assertEqual(density.axis_index, 2)
        np.testing.assert_allclose(density.edges, [0.0, 2.0, 4.0, 6.0])
        self.assertIn("H2O", density.hist.data)
        self.assertIn("Na", density.hist.data)

    def test_run_uses_programmatic_configuration_without_prompting(self):
        density = DensityAnalysis(DummyTrajectory(), input_provider=NullInputProvider())
        density.configure(DensityConfig(axis="z", step_size=2.0, per_compound_normalization=False))
        density.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                density.run()
                output = Path("density.dat")
                self.assertTrue(output.exists())
                text = output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("r/Angstrom", text)
        self.assertIn("H2O", text)
        self.assertIn("Na", text)


if __name__ == "__main__":
    unittest.main()
