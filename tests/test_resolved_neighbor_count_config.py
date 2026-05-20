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
    ObservableConfig = None
    ObservedSiteConfig = None
    ResolvedNeighborCountAnalysis = None
    ResolvedNeighborCountConfig = None
else:
    from analyses.resolved_neighbor_count_analysis import (
        ObservableConfig,
        ObservedSiteConfig,
        ResolvedNeighborCountAnalysis,
        ResolvedNeighborCountConfig,
    )


class DummyTrajectory:
    def __init__(self):
        self.box_size = np.array([20.0, 20.0, 20.0], dtype=float)
        self.coords = np.array(
            [
                [0.0, 0.0, 0.0],   # K1
                [5.0, 0.0, 0.0],   # K2
                [1.0, 0.0, 0.0],   # O1 (H2O near K1)
                [1.0, 0.9, 0.0],   # H1
                [1.0, -0.9, 0.0],  # H2
                [1.4, 0.0, 0.0],   # O2 (HO near K1)
                [1.4, 0.9, 0.0],   # H3
                [0.8, 0.8, 0.0],   # Cl1 near K1
                [5.8, 0.0, 0.0],   # O3 (H2O near K2)
                [5.8, 0.9, 0.0],   # H4
                [5.8, -0.9, 0.0],  # H5
            ],
            dtype=float,
        )

        k_type = CompoundType(
            type_id=0,
            key=("K", (), "k"),
            formula="K",
            canonical_labels=("K1",),
            label_to_local_index={"K1": 0},
            local_bonds=tuple(),
            local_elements=("K",),
            atomic_masses=(39.0,),
        )
        water_type = CompoundType(
            type_id=1,
            key=("H2O", (), "water"),
            formula="H2O",
            canonical_labels=("H1", "H2", "O1"),
            label_to_local_index={"H1": 0, "H2": 1, "O1": 2},
            local_bonds=((0, 2), (1, 2)),
            local_elements=("H", "H", "O"),
            atomic_masses=(1.0, 1.0, 16.0),
        )
        ho_type = CompoundType(
            type_id=2,
            key=("HO", (), "hydroxide"),
            formula="HO",
            canonical_labels=("H1", "O1"),
            label_to_local_index={"H1": 0, "O1": 1},
            local_bonds=((0, 1),),
            local_elements=("H", "O"),
            atomic_masses=(1.0, 16.0),
        )
        cl_type = CompoundType(
            type_id=3,
            key=("Cl", (), "chloride"),
            formula="Cl",
            canonical_labels=("Cl1",),
            label_to_local_index={"Cl1": 0},
            local_bonds=tuple(),
            local_elements=("Cl",),
            atomic_masses=(35.0,),
        )
        self.topology_registry = CompoundTypeRegistry([k_type, water_type, ho_type, cl_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={
                ("K", (), "k"): np.array([[0], [1]], dtype=np.int32),
                ("H2O", (), "water"): np.array([[3, 4, 2], [9, 10, 8]], dtype=np.int32),
                ("HO", (), "hydroxide"): np.array([[6, 5]], dtype=np.int32),
                ("Cl", (), "chloride"): np.array([[7]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 0, 1, 1, 1, 2, 2, 3, 1, 1, 1], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32),
            atom_to_local_index=np.array([0, 0, 2, 0, 1, 1, 0, 0, 2, 0, 1], dtype=np.int32),
        )

    def read_frame(self):
        raise ValueError("End of trajectory")


@unittest.skipIf(ResolvedNeighborCountConfig is None, "scipy is not installed")
class ResolvedNeighborCountConfigTests(unittest.TestCase):
    def test_config_validates_nested_inputs(self):
        ResolvedNeighborCountConfig(
            ref_compound_index=0,
            observables=[
                ObservableConfig(
                    ref_labels=["K1"],
                    observed_sites=[
                        ObservedSiteConfig(compound_index=1, labels=["O"], cutoff=1.5),
                    ],
                ),
            ],
        )

        with self.assertRaises(ValueError):
            ResolvedNeighborCountConfig(ref_compound_index=-1, observables=[])
        with self.assertRaises(ValueError):
            ObservableConfig(ref_labels=["K1", "K2"], observed_sites=[ObservedSiteConfig(1, ["O"], 1.5)])
        with self.assertRaises(ValueError):
            ObservedSiteConfig(compound_index=1, labels=[], cutoff=1.5)

    def test_prompt_config_builds_nested_configuration(self):
        provider = FileInputProvider(
            lines=[
                "1",
                "K1",
                "2,3",
                "O",
                "1.5",
                "",
                "O",
                "2.0",
                "",
                "y",
                "K1",
                "4",
                "Cl",
                "1.5",
                "",
                "n",
            ],
            fallback=NullInputProvider(),
        )
        analysis = ResolvedNeighborCountAnalysis(DummyTrajectory(), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            ResolvedNeighborCountConfig(
                ref_compound_index=0,
                observables=[
                    ObservableConfig(
                        ref_labels=["K1"],
                        observed_sites=[
                            ObservedSiteConfig(compound_index=1, labels=["O"], cutoff=1.5, exclude_same_molecule=True),
                            ObservedSiteConfig(compound_index=2, labels=["O"], cutoff=2.0, exclude_same_molecule=True),
                        ],
                    ),
                    ObservableConfig(
                        ref_labels=["K1"],
                        observed_sites=[
                            ObservedSiteConfig(compound_index=3, labels=["Cl"], cutoff=1.5, exclude_same_molecule=True),
                        ],
                    ),
                ],
            ),
        )

    def test_run_writes_joint_and_marginal_outputs(self):
        analysis = ResolvedNeighborCountAnalysis(DummyTrajectory(), input_provider=NullInputProvider())
        analysis.configure(
            ResolvedNeighborCountConfig(
                ref_compound_index=0,
                observables=[
                    ObservableConfig(
                        ref_labels=["K1"],
                        observed_sites=[
                            ObservedSiteConfig(compound_index=1, labels=["O"], cutoff=1.5),
                            ObservedSiteConfig(compound_index=2, labels=["O"], cutoff=2.0),
                        ],
                    ),
                    ObservableConfig(
                        ref_labels=["K1"],
                        observed_sites=[
                            ObservedSiteConfig(compound_index=3, labels=["Cl"], cutoff=1.5),
                        ],
                    ),
                ],
            )
        )
        analysis.configure_frame_loop(FrameLoopConfig(start_frame=1, nframes=1, frame_stride=1, update_compounds=False))

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.run()
                joint_output = Path("rncount_joint_K1-K_O-H2O+O-HO_K1-K_Cl-Cl.dat")
                marginal_1 = Path("rncount_obs1_K1-K_O-H2O+O-HO.dat")
                marginal_2 = Path("rncount_obs2_K1-K_Cl-Cl.dat")
                self.assertTrue(joint_output.exists())
                self.assertTrue(marginal_1.exists())
                self.assertTrue(marginal_2.exists())
                joint_text = joint_output.read_text(encoding="utf-8")
                marginal_1_text = marginal_1.read_text(encoding="utf-8")
                marginal_2_text = marginal_2.read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("N1", joint_text)
        self.assertIn("N2", joint_text)
        self.assertIn("P", joint_text)
        self.assertIn("P(N1)", marginal_1_text)
        self.assertIn("P(N2)", marginal_2_text)

        joint_rows = _parse_numeric_table(joint_text)
        marginal_1_rows = _parse_numeric_table(marginal_1_text)
        marginal_2_rows = _parse_numeric_table(marginal_2_text)

        np.testing.assert_allclose(joint_rows, np.array([[1.0, 0.0, 0.5], [2.0, 1.0, 0.5]]))
        np.testing.assert_allclose(marginal_1_rows, np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.5]]))
        np.testing.assert_allclose(marginal_2_rows, np.array([[0.0, 0.5], [1.0, 0.5]]))


def _parse_numeric_table(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(value) for value in stripped.split()])
    return np.array(rows, dtype=float)


if __name__ == "__main__":
    unittest.main()
