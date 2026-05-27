import importlib.util
import unittest

import numpy as np

from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from io_support.input_providers import FileInputProvider, NullInputProvider

if importlib.util.find_spec("scipy") is None:
    AngleAxisConfig = None
    CDFAnalysis = None
    CDFConfig = None
    DistanceAxisConfig = None
else:
    from analyses.cdf_analysis import AngleAxisConfig, CDFAnalysis, CDFConfig, DistanceAxisConfig


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


@unittest.skipIf(CDFConfig is None, "scipy is not installed")
class CDFConfigTests(unittest.TestCase):
    def test_cdf_config_validates_tuple_mode(self):
        CDFConfig(
            ref_compound_index=0,
            x_axis=DistanceAxisConfig(obs_compound_index=1, ref_labels=["O"], obs_labels=["H"]),
            y_axis=DistanceAxisConfig(obs_compound_index=1, ref_labels=["O"], obs_labels=["H"]),
        )

        with self.assertRaises(ValueError):
            CDFConfig(
                ref_compound_index=0,
                x_axis=DistanceAxisConfig(obs_compound_index=1, ref_labels=["O"], obs_labels=["H"]),
                y_axis=DistanceAxisConfig(obs_compound_index=1, ref_labels=["O"], obs_labels=["H"]),
                tuple_mode="diagonal",
            )

    def test_prompt_config_uses_shared_schema_with_distance_and_angle_axes(self):
        provider = FileInputProvider(
            lines=[
                "1",
                "distance", "2", "O", "H", "5.0", "5",
                "angle", "2", "r", "r", "O", "H", "o", "o", "H", "H,O", "y", "18", "", "3.0",
                "second_context", "y",
            ],
            fallback=NullInputProvider(),
        )
        analysis = CDFAnalysis(DummyTrajectory(), input_provider=provider)

        config = analysis.prompt_config()

        self.assertEqual(
            config,
            CDFConfig(
                ref_compound_index=0,
                x_axis=DistanceAxisConfig(
                    obs_compound_index=1,
                    ref_labels=["O"],
                    obs_labels=["H"],
                    max_distance=5.0,
                    bin_count=5,
                ),
                y_axis=AngleAxisConfig(
                    obs_compound_index=1,
                    ref_base_source="r",
                    ref_tip_source="r",
                    ref_base_labels=["O"],
                    ref_tip_labels=["H"],
                    obs_base_source="o",
                    obs_tip_source="o",
                    obs_base_labels=["H"],
                    obs_tip_labels=["H", "O"],
                    enforce_shared_atom=True,
                    bin_count=18,
                    v1_cutoff=None,
                    v2_cutoff=3.0,
                ),
                tuple_mode="second_context",
                exclude_identical_contexts=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
