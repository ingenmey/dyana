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
    NeighborCountAnalysis = None
    NeighborCountConfig = None
else:
    from analyses.neighbor_count_analysis import NeighborCountAnalysis, NeighborCountConfig

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
        water_type = CompoundType(
            type_id=0,
            key=("H2O", (), "water"),
            formula="H2O",
            canonical_labels=("H1", "H2", "O1"),
            label_to_local_index={"H1": 0, "H2": 1, "O1": 2},
            local_bonds=((0, 2), (1, 2)),
            local_elements=("H", "H", "O"),
            atomic_masses=(1.0, 1.0, 16.0),
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
                ("H2O", (), "water"): np.array([[1, 2, 0]], dtype=np.int32),
                ("Na", (), "na"): np.array([[3]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 0, 0, 1], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 0, 0, 0], dtype=np.int32),
            atom_to_local_index=np.array([2, 0, 1, 0], dtype=np.int32),
        )

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

        self.assertEqual(analysis.ref_indices.tolist(), [0])
        self.assertEqual(sorted(analysis.obs_indices.tolist()), [1, 2, 3])
        self.assertEqual(analysis.ref_selection.local_indices, (2,))
        self.assertEqual(analysis.obs_selections_by_key[("H2O", (), "water")].local_indices, (0, 1))
        self.assertEqual(tuple(analysis.ref_type.canonical_labels[i] for i in analysis.ref_selection.local_indices), ("O1",))
        water_type = analysis.obs_types[0]
        self.assertEqual(
            tuple(water_type.canonical_labels[i] for i in analysis.obs_selections_by_key[("H2O", (), "water")].local_indices),
            ("H1", "H2"),
        )

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
                output = Path("ncount_O-H2O_H-H2O+Na-Na.dat")
                self.assertTrue(output.exists())
                text = output.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        rows = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([float(value) for value in stripped.split()])

        self.assertIn("P(n)", text)
        self.assertEqual(rows, [[0.0, 0.0], [1.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
