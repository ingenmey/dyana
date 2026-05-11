import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from framework.config_schema import FrameLoopConfig
from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from io_support.input_providers import FileInputProvider, NullInputProvider

if importlib.util.find_spec("scipy") is None:
    TetrahedralOrderAnalysis = None
    TetrahedralOrderConfig = None
else:
    from analyses.top_analysis import TetrahedralOrderAnalysis, TetrahedralOrderConfig


class DummyTrajectory:
    def __init__(self):
        self.box_size = np.array([20.0, 20.0, 20.0], dtype=float)
        self.coords = np.array(
            [
                [10.0, 10.0, 10.0],  # O1
                [11.1, 11.0, 11.0],  # H1
                [9.0, 9.0, 11.0],    # H2
                [9.0, 11.0, 9.0],    # H3
                [11.0, 9.0, 9.0],    # H4
            ],
            dtype=float,
        )
        compound_type = CompoundType(
            type_id=0,
            key=("H4O", (), "tetra"),
            formula="H4O",
            canonical_labels=("H1", "H2", "H3", "H4", "O1"),
            label_to_local_index={"H1": 0, "H2": 1, "H3": 2, "H4": 3, "O1": 4},
            local_bonds=((0, 4), (1, 4), (2, 4), (3, 4)),
            local_elements=("H", "H", "H", "H", "O"),
            atomic_masses=(1.0, 1.0, 1.0, 1.0, 16.0),
        )
        self.topology_registry = CompoundTypeRegistry([compound_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={("H4O", (), "tetra"): np.array([[1, 2, 3, 4, 0]], dtype=np.int32)},
            atom_to_type_id=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            atom_to_local_index=np.array([4, 0, 1, 2, 3], dtype=np.int32),
        )

    def read_frame(self):
        raise ValueError("End of trajectory")


@unittest.skipIf(TetrahedralOrderConfig is None, "scipy is not installed")
class TetrahedralOrderConfigTests(unittest.TestCase):
    def test_top_config_validates_inputs(self):
        TetrahedralOrderConfig(
            ref_compound_index=0,
            ref_labels=["O"],
            obs_compound_indices=[0],
            obs_labels_per_compound={0: ["H"]},
        )

        with self.assertRaises(ValueError):
            TetrahedralOrderConfig(
                ref_compound_index=-1,
                ref_labels=["O"],
                obs_compound_indices=[0],
                obs_labels_per_compound={0: ["H"]},
            )
        with self.assertRaises(ValueError):
            TetrahedralOrderConfig(
                ref_compound_index=0,
                ref_labels=["O"],
                obs_compound_indices=[0],
                obs_labels_per_compound={},
            )
        with self.assertRaises(ValueError):
            TetrahedralOrderConfig(
                ref_compound_index=0,
                ref_labels=["O"],
                obs_compound_indices=[0],
                obs_labels_per_compound={0: ["H"]},
                use_cutoff=True,
                cutoff=None,
            )

    def test_prompt_config_builds_custom_config(self):
        provider = FileInputProvider(
            lines=["1", "O", "1", "H", "y", "4.5", "12", "20"],
            fallback=NullInputProvider(),
        )
        analysis = TetrahedralOrderAnalysis(DummyTrajectory(), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            TetrahedralOrderConfig(
                ref_compound_index=0,
                ref_labels=["O"],
                obs_compound_indices=[0],
                obs_labels_per_compound={0: ["H"]},
                use_cutoff=True,
                cutoff=4.5,
                bin_count_q=12,
                bin_count_s=20,
            ),
        )

    def test_configure_sets_up_runtime_selections(self):
        analysis = TetrahedralOrderAnalysis(DummyTrajectory())
        analysis.configure(
            TetrahedralOrderConfig(
                ref_compound_index=0,
                ref_labels=["O"],
                obs_compound_indices=[0],
                obs_labels_per_compound={0: ["H"]},
                use_cutoff=False,
                bin_count_q=8,
                bin_count_s=10,
            )
        )

        self.assertEqual(analysis.ref_indices.tolist(), [0])
        self.assertEqual(sorted(analysis.obs_indices.tolist()), [1, 2, 3, 4])
        self.assertEqual(analysis.ref_selection.local_indices, (4,))
        self.assertEqual(analysis.obs_selections_by_key[("H4O", (), "tetra")].local_indices, (0, 1, 2, 3))
        self.assertEqual(tuple(analysis.ref_type.canonical_labels[i] for i in analysis.ref_selection.local_indices), ("O1",))

    def test_run_writes_q_and_s_outputs(self):
        analysis = TetrahedralOrderAnalysis(DummyTrajectory(), input_provider=NullInputProvider())
        analysis.configure(
            TetrahedralOrderConfig(
                ref_compound_index=0,
                ref_labels=["O"],
                obs_compound_indices=[0],
                obs_labels_per_compound={0: ["H"]},
                use_cutoff=False,
                bin_count_q=10,
                bin_count_s=10,
            )
        )
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                q_output = Path("top_q_O-H4O_H-H4O.dat")
                s_output = Path("top_s_O-H4O_H-H4O.dat")
                self.assertTrue(q_output.exists())
                self.assertTrue(s_output.exists())
                q_text = q_output.read_text(encoding="utf-8")
                s_text = s_output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("q", q_text)
        self.assertIn("P(q)", q_text)
        self.assertIn("S", s_text)
        self.assertIn("P(S)", s_text)


if __name__ == "__main__":
    unittest.main()
