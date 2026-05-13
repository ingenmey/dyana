import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from framework.config_schema import FrameLoopConfig
from io_support.input_providers import FileInputProvider, NullInputProvider

if importlib.util.find_spec("scipy") is None:
    LSIAnalysis = None
    LSIConfig = None
else:
    from analyses.lsi_analysis import LSIAnalysis, LSIConfig


class DummyTrajectory:
    def __init__(self):
        self.box_size = np.array([20.0, 20.0, 20.0], dtype=float)
        self.coords = np.array(
            [
                [10.0, 10.0, 10.0],
                [12.1, 10.0, 10.0],
                [10.0, 12.4, 10.0],
                [10.0, 10.0, 12.9],
                [12.9, 12.9, 10.0],
            ],
            dtype=float,
        )
        compound_type = CompoundType(
            type_id=0,
            key=("O", (), "sites"),
            formula="O",
            canonical_labels=("O1",),
            label_to_local_index={"O1": 0},
            local_bonds=(),
            local_elements=("O",),
            atomic_masses=(16.0,),
        )
        self.topology_registry = CompoundTypeRegistry([compound_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={("O", (), "sites"): np.array([[0], [1], [2], [3], [4]], dtype=np.int32)},
            atom_to_type_id=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 1, 2, 3, 4], dtype=np.int32),
            atom_to_local_index=np.array([0, 0, 0, 0, 0], dtype=np.int32),
        )

    def read_frame(self):
        raise ValueError("End of trajectory")


@unittest.skipIf(LSIConfig is None, "scipy is not installed")
class LSIConfigTests(unittest.TestCase):
    def test_lsi_config_validates_inputs(self):
        LSIConfig(
            compound_index=0,
            site_labels=["O"],
            cutoff=3.7,
            bin_count_local=50,
            histogram_min=0.0,
            histogram_max=0.4,
            output_frame_mean=True,
            optional_threshold=0.13,
        )

        with self.assertRaises(ValueError):
            LSIConfig(compound_index=-1, site_labels=["O"])
        with self.assertRaises(ValueError):
            LSIConfig(compound_index=0, site_labels=[])
        with self.assertRaises(ValueError):
            LSIConfig(compound_index=0, site_labels=["O"], cutoff=0.0)
        with self.assertRaises(ValueError):
            LSIConfig(compound_index=0, site_labels=["O"], bin_count_local=0)
        with self.assertRaises(ValueError):
            LSIConfig(compound_index=0, site_labels=["O"], histogram_min=0.4, histogram_max=0.4)
        with self.assertRaises(ValueError):
            LSIConfig(compound_index=0, site_labels=["O"], optional_threshold=-0.1)

    def test_prompt_config_builds_custom_config(self):
        provider = FileInputProvider(
            lines=["1", "O", "4.2", "12", "0.0", "0.5", "y", "0.13"],
            fallback=NullInputProvider(),
        )
        analysis = LSIAnalysis(DummyTrajectory(), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            LSIConfig(
                compound_index=0,
                site_labels=["O"],
                cutoff=4.2,
                bin_count_local=12,
                histogram_min=0.0,
                histogram_max=0.5,
                output_frame_mean=True,
                optional_threshold=0.13,
            ),
        )

    def test_configure_sets_up_runtime_site_selection(self):
        analysis = LSIAnalysis(DummyTrajectory())
        analysis.configure(
            LSIConfig(
                compound_index=0,
                site_labels=["O"],
                cutoff=4.2,
                bin_count_local=10,
            )
        )

        self.assertEqual(analysis.site_indices.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(analysis.site_selection.local_indices, (0,))
        self.assertEqual(
            tuple(analysis.compound_type.canonical_labels[i] for i in analysis.site_selection.local_indices),
            ("O1",),
        )

    def test_process_frame_includes_first_neighbor_outside_cutoff(self):
        analysis = LSIAnalysis(DummyTrajectory(), input_provider=NullInputProvider())
        analysis.configure(
            LSIConfig(
                compound_index=0,
                site_labels=["O"],
                cutoff=3.7,
                bin_count_local=10,
                histogram_min=0.0,
                histogram_max=0.4,
                output_frame_mean=True,
            )
        )

        analysis.process_frame()

        self.assertEqual(len(analysis.frame_rows), 1)
        frame, mean_lsi, std_lsi, count = analysis.frame_rows[0]
        self.assertEqual(frame, 1)
        self.assertAlmostEqual(mean_lsi, 0.12419596437396965)
        self.assertAlmostEqual(std_lsi, 0.09288425291790557)
        self.assertEqual(count, 4)
        self.assertEqual(int(analysis.local_hist.counts.sum()), 4)

    def test_run_writes_local_and_global_lsi_outputs(self):
        analysis = LSIAnalysis(DummyTrajectory(), input_provider=NullInputProvider())
        analysis.configure(
            LSIConfig(
                compound_index=0,
                site_labels=["O"],
                cutoff=4.5,
                bin_count_local=10,
                histogram_min=0.0,
                histogram_max=0.4,
                output_frame_mean=True,
                optional_threshold=0.05,
            )
        )
        analysis.configure_frame_loop(
            FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False)
        )

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                local_output = Path("lsi_local_O-O.dat")
                global_output = Path("lsi_global_O-O.dat")
                self.assertTrue(local_output.exists())
                self.assertTrue(global_output.exists())
                local_text = local_output.read_text(encoding="utf-8")
                global_text = global_output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("LSI/Angstrom^2", local_text)
        self.assertIn("P(LSI)", local_text)
        self.assertIn("frame", global_text)
        self.assertIn("mean_LSI", global_text)
        self.assertIn("std_LSI", global_text)
        self.assertIn("count", global_text)
        self.assertIn("fraction_below_threshold", global_text)
        self.assertIn("fraction_above_threshold", global_text)


if __name__ == "__main__":
    unittest.main()
