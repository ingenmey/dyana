import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyses.density_analysis import DensityAnalysis, DensityConfig
from framework.config_schema import FrameLoopConfig
from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from io_support.input_providers import FileInputProvider, NullInputProvider


class DummyTrajectory:
    def __init__(self):
        self.box_size = np.array([4.0, 5.0, 6.0], dtype=float)
        self.coords = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=float,
        )
        water_type = CompoundType(
            type_id=0,
            key=("H2O", (), "water"),
            formula="H2O",
            canonical_labels=("O1",),
            label_to_local_index={"O1": 0},
            local_bonds=tuple(),
            local_elements=("O",),
            atomic_masses=(16.0,),
        )
        na_type = CompoundType(
            type_id=1,
            key=("Na", (), "na"),
            formula="Na",
            canonical_labels=("Na1",),
            label_to_local_index={"Na1": 0},
            local_bonds=tuple(),
            local_elements=("Na",),
            atomic_masses=(23.0,),
        )
        self.topology_registry = CompoundTypeRegistry([water_type, na_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={
                ("H2O", (), "water"): np.array([[0], [1]], dtype=np.int32),
                ("Na", (), "na"): np.array([[2]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 0, 1], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 1, 0], dtype=np.int32),
            atom_to_local_index=np.array([0, 0, 0], dtype=np.int32),
        )

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
