import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config_schema import FrameLoopConfig
from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from input_providers import FileInputProvider, NullInputProvider

if importlib.util.find_spec("scipy") is None:
    ADF = None
    ADFConfig = None
else:
    from analyses.adf_analysis import ADF, ADFConfig

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
        ref_type = CompoundType(
            type_id=0,
            key=("H2O", (), "ref"),
            formula="H2O",
            canonical_labels=("H1", "O1"),
            label_to_local_index={"H1": 0, "O1": 1},
            local_bonds=((0, 1),),
            local_elements=("H", "O"),
            atomic_masses=(1.0, 16.0),
        )
        obs_type = CompoundType(
            type_id=1,
            key=("OH", (), "obs"),
            formula="OH",
            canonical_labels=("H1", "O1"),
            label_to_local_index={"H1": 0, "O1": 1},
            local_bonds=((0, 1),),
            local_elements=("H", "O"),
            atomic_masses=(1.0, 16.0),
        )
        self.topology_registry = CompoundTypeRegistry([ref_type, obs_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={
                ("H2O", (), "ref"): np.array([[1, 0]], dtype=np.int32),
                ("OH", (), "obs"): np.array([[3, 2]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 0, 1, 1], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 0, 0, 0], dtype=np.int32),
            atom_to_local_index=np.array([1, 0, 1, 0], dtype=np.int32),
        )

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
        self.assertEqual(analysis.ref_base_selection.local_indices, (1,))
        self.assertEqual(analysis.ref_tip_selection.local_indices, (0,))
        self.assertEqual(tuple(analysis.ref_type.canonical_labels[i] for i in analysis.ref_base_selection.local_indices), ("O1",))
        self.assertEqual(tuple(analysis.ref_type.canonical_labels[i] for i in analysis.ref_tip_selection.local_indices), ("H1",))
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
                output = Path("adf_O-H2O_H-H2O_O-OH_H-OH.dat")
                self.assertTrue(output.exists())
                text = output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("angle/deg", text)
        self.assertIn("ADF", text)


if __name__ == "__main__":
    unittest.main()
