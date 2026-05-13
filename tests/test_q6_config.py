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
    Q6Analysis = None
    Q6Config = None
else:
    from analyses.q6_analysis import Q6Analysis, Q6Config


class DummyTrajectory:
    def __init__(self):
        self.box_size = np.array([20.0, 20.0, 20.0], dtype=float)
        self.coords = np.array(
            [
                [10.0, 10.0, 10.0],
                [11.0, 11.0, 11.0],
                [9.0, 9.0, 11.0],
                [9.0, 11.0, 9.0],
                [11.0, 9.0, 9.0],
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


@unittest.skipIf(Q6Config is None, "scipy is not installed")
class Q6ConfigTests(unittest.TestCase):
    def test_q6_config_validates_inputs(self):
        Q6Config(compound_index=0, site_labels=["O"], cutoff=3.5, bin_count_local=50)

        with self.assertRaises(ValueError):
            Q6Config(compound_index=-1, site_labels=["O"], cutoff=3.5, bin_count_local=50)
        with self.assertRaises(ValueError):
            Q6Config(compound_index=0, site_labels=[], cutoff=3.5, bin_count_local=50)
        with self.assertRaises(ValueError):
            Q6Config(compound_index=0, site_labels=["O"], cutoff=0.0, bin_count_local=50)
        with self.assertRaises(ValueError):
            Q6Config(compound_index=0, site_labels=["O"], cutoff=3.5, bin_count_local=0)

    def test_prompt_config_builds_custom_config(self):
        provider = FileInputProvider(
            lines=["1", "O", "4.5", "12"],
            fallback=NullInputProvider(),
        )
        analysis = Q6Analysis(DummyTrajectory(), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            Q6Config(
                compound_index=0,
                site_labels=["O"],
                cutoff=4.5,
                bin_count_local=12,
            ),
        )

    def test_configure_sets_up_runtime_site_selection(self):
        analysis = Q6Analysis(DummyTrajectory())
        analysis.configure(
            Q6Config(
                compound_index=0,
                site_labels=["O"],
                cutoff=4.5,
                bin_count_local=10,
            )
        )

        self.assertEqual(analysis.site_indices.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(analysis.site_selection.local_indices, (0,))
        self.assertEqual(tuple(analysis.compound_type.canonical_labels[i] for i in analysis.site_selection.local_indices), ("O1",))

    def test_run_writes_q6_qbar6_and_global_q6_outputs(self):
        analysis = Q6Analysis(DummyTrajectory(), input_provider=NullInputProvider())
        analysis.configure(
            Q6Config(
                compound_index=0,
                site_labels=["O"],
                cutoff=4.5,
                bin_count_local=10,
            )
        )
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                local_output = Path("q6_local_O-O.dat")
                qbar_local_output = Path("qbar6_local_O-O.dat")
                global_output = Path("q6_global_O-O.dat")
                self.assertTrue(local_output.exists())
                self.assertTrue(qbar_local_output.exists())
                self.assertTrue(global_output.exists())
                local_text = local_output.read_text(encoding="utf-8")
                qbar_local_text = qbar_local_output.read_text(encoding="utf-8")
                global_text = global_output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("q6", local_text)
        self.assertIn("P(q6)", local_text)
        self.assertIn("qbar6", qbar_local_text)
        self.assertIn("P(qbar6)", qbar_local_text)
        self.assertIn("frame", global_text)
        self.assertIn("Q6", global_text)


if __name__ == "__main__":
    unittest.main()
