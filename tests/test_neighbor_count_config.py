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
    NeighborCountAnalysis = None
    NeighborCountConfig = None
else:
    from analyses.neighbor_count_analysis import NeighborCountAnalysis, NeighborCountConfig


class DummyMolecule:
    def __init__(self, global_ids, com=(0.0, 0.0, 0.0)):
        self.label_to_global_id = dict(global_ids)
        self.label_to_id = {label: i for i, label in enumerate(global_ids)}
        self.com = np.array(com, dtype=float)


class DummyCompound:
    def __init__(self, rep, comp_id, molecules):
        self.rep = rep
        self.comp_id = comp_id
        self.members = molecules


class DummyTrajectory:
    def __init__(self):
        self.box_size = np.array([10.0, 10.0, 10.0], dtype=float)
        self.coords = np.array(
            [
                [0.0, 0.0, 0.0],  # O1
                [0.9, 0.0, 0.0],  # H1
                [0.0, 0.9, 0.0],  # H2
                [2.0, 0.0, 0.0],  # Na1
            ],
            dtype=float,
        )
        self.compounds = OrderedDict(
            [
                (
                    ("H2O", (), "water"),
                    DummyCompound(
                        "H2O",
                        0,
                        [DummyMolecule([("O1", 0), ("H1", 1), ("H2", 2)])],
                    ),
                ),
                (
                    ("Na", (), "na"),
                    DummyCompound(
                        "Na",
                        1,
                        [DummyMolecule([("Na1", 3)])],
                    ),
                ),
            ]
        )

    def update_molecule_coords(self):
        return None

    def read_frame(self):
        raise ValueError("End of trajectory")


@unittest.skipIf(NeighborCountConfig is None, "scipy is not installed")
class NeighborCountConfigTests(unittest.TestCase):
    def test_neighbor_count_config_validates_inputs(self):
        NeighborCountConfig(
            ref_compound_index=0,
            ref_labels=["O"],
            obs_compound_indices=[0, 1],
            obs_labels_per_compound={0: ["H"], 1: ["Na"]},
        )

        with self.assertRaises(ValueError):
            NeighborCountConfig(
                ref_compound_index=-1,
                ref_labels=["O"],
                obs_compound_indices=[0],
                obs_labels_per_compound={0: ["H"]},
            )
        with self.assertRaises(ValueError):
            NeighborCountConfig(
                ref_compound_index=0,
                ref_labels=["O"],
                obs_compound_indices=[],
                obs_labels_per_compound={},
            )
        with self.assertRaises(ValueError):
            NeighborCountConfig(
                ref_compound_index=0,
                ref_labels=["O"],
                obs_compound_indices=[0],
                obs_labels_per_compound={},
            )

    def test_prompt_config_builds_custom_dynamic_config(self):
        provider = FileInputProvider(
            lines=["1", "O", "1,2", "H", "Na", "y", "3.0"],
            fallback=NullInputProvider(),
        )
        analysis = NeighborCountAnalysis(DummyTrajectory(), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            NeighborCountConfig(
                ref_compound_index=0,
                ref_labels=["O"],
                obs_compound_indices=[0, 1],
                obs_labels_per_compound={0: ["H"], 1: ["Na"]},
                exclude_same_molecule=True,
                r_cut=3.0,
            ),
        )

    def test_configure_sets_up_indices(self):
        analysis = NeighborCountAnalysis(DummyTrajectory())
        analysis.configure(
            NeighborCountConfig(
                ref_compound_index=0,
                ref_labels=["O"],
                obs_compound_indices=[0, 1],
                obs_labels_per_compound={0: ["H"], 1: ["Na"]},
                exclude_same_molecule=True,
                r_cut=3.0,
            )
        )

        self.assertEqual(analysis.ref_indices, [0])
        self.assertEqual(sorted(analysis.obs_indices), [1, 2, 3])

    def test_run_uses_programmatic_configuration_without_prompting(self):
        analysis = NeighborCountAnalysis(DummyTrajectory(), input_provider=NullInputProvider())
        analysis.configure(
            NeighborCountConfig(
                ref_compound_index=0,
                ref_labels=["O"],
                obs_compound_indices=[0, 1],
                obs_labels_per_compound={0: ["H"], 1: ["Na"]},
                exclude_same_molecule=True,
                r_cut=3.0,
            )
        )
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                output = Path("ncount.dat")
                self.assertTrue(output.exists())
                text = output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("P(n)", text)
        self.assertIn("1  1.000000", text)


if __name__ == "__main__":
    unittest.main()
