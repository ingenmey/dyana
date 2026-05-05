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
    ADF = None
    ADFConfig = None
else:
    from analyses.adf_analysis import ADF, ADFConfig


class DummyMolecule:
    def __init__(self, global_ids):
        self.label_to_global_id = dict(global_ids)
        self.label_to_id = {label: i for i, label in enumerate(global_ids)}


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
                [0.0, 0.0, 0.0],   # O1 ref
                [1.0, 0.0, 0.0],   # H1 ref
                [2.0, 0.0, 0.0],   # O1 obs
                [2.0, 1.0, 0.0],   # H1 obs
            ],
            dtype=float,
        )
        self.compounds = OrderedDict(
            [
                (
                    ("H2O", (), "ref"),
                    DummyCompound("H2O", 0, [DummyMolecule([("O1", 0), ("H1", 1)])]),
                ),
                (
                    ("OH", (), "obs"),
                    DummyCompound("OH", 1, [DummyMolecule([("O1", 2), ("H1", 3)])]),
                ),
            ]
        )

    def update_molecule_coords(self):
        return None

    def read_frame(self):
        raise ValueError("End of trajectory")


@unittest.skipIf(ADFConfig is None, "scipy is not installed")
class ADFConfigTests(unittest.TestCase):
    def test_adf_config_validates_inputs(self):
        ADFConfig(
            ref_compound_index=0,
            obs_compound_index=1,
            ref_base_source="r",
            ref_tip_source="r",
            ref_base_labels=["O"],
            ref_tip_labels=["H"],
            obs_base_source="o",
            obs_tip_source="o",
            obs_base_labels=["O"],
            obs_tip_labels=["H"],
        )

        with self.assertRaises(ValueError):
            ADFConfig(
                ref_compound_index=-1,
                obs_compound_index=1,
                ref_base_source="r",
                ref_tip_source="r",
                ref_base_labels=["O"],
                ref_tip_labels=["H"],
                obs_base_source="o",
                obs_tip_source="o",
                obs_base_labels=["O"],
                obs_tip_labels=["H"],
            )
        with self.assertRaises(ValueError):
            ADFConfig(
                ref_compound_index=0,
                obs_compound_index=1,
                ref_base_source="x",
                ref_tip_source="r",
                ref_base_labels=["O"],
                ref_tip_labels=["H"],
                obs_base_source="o",
                obs_tip_source="o",
                obs_base_labels=["O"],
                obs_tip_labels=["H"],
            )

    def test_prompt_config_uses_schema_when_and_optional_cutoffs(self):
        provider = FileInputProvider(
            lines=["1", "2", "r", "r", "O,C", "H", "o", "o", "H", "H,O", "y", "12", "", "3.0"],
            fallback=NullInputProvider(),
        )
        analysis = ADF(DummyTrajectory(), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            ADFConfig(
                ref_compound_index=0,
                obs_compound_index=1,
                ref_base_source="r",
                ref_tip_source="r",
                ref_base_labels=["O", "C"],
                ref_tip_labels=["H"],
                obs_base_source="o",
                obs_tip_source="o",
                obs_base_labels=["H"],
                obs_tip_labels=["H", "O"],
                enforce_shared_atom=True,
                bin_count=12,
                v1_cutoff=None,
                v2_cutoff=3.0,
            ),
        )

    def test_configure_sets_up_metric(self):
        analysis = ADF(DummyTrajectory())
        analysis.configure(
            ADFConfig(
                ref_compound_index=0,
                obs_compound_index=1,
                ref_base_source="r",
                ref_tip_source="r",
                ref_base_labels=["O"],
                ref_tip_labels=["H"],
                obs_base_source="o",
                obs_tip_source="o",
                obs_base_labels=["O"],
                obs_tip_labels=["H"],
                bin_count=18,
            )
        )

        self.assertTrue(len(analysis.ref_base_ids) > 0)
        self.assertEqual(len(analysis.angle_edges), 19)

    def test_run_uses_programmatic_configuration_without_prompting(self):
        analysis = ADF(DummyTrajectory(), input_provider=NullInputProvider())
        analysis.configure(
            ADFConfig(
                ref_compound_index=0,
                obs_compound_index=1,
                ref_base_source="r",
                ref_tip_source="r",
                ref_base_labels=["O"],
                ref_tip_labels=["H"],
                obs_base_source="o",
                obs_tip_source="o",
                obs_base_labels=["O"],
                obs_tip_labels=["H"],
                bin_count=18,
            )
        )
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                output = Path("adf.dat")
                self.assertTrue(output.exists())
                text = output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("bin_0", text)


if __name__ == "__main__":
    unittest.main()
