import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyses.cdf_analysis import (
    AngleAxisConfig,
    CDFAnalysis,
    CDFConfig,
    DistanceAxisConfig,
    build_2d_tuples,
)
from analyses.common.reference_channels import ContextSamples, ReferenceSamples
from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame


class DistanceTrajectory:
    def __init__(self):
        self.box_size = np.array([10.0, 10.0, 10.0], dtype=float)
        self.coords = np.array(
            [
                [0.0, 0.0, 0.0],  # O mol 0
                [1.0, 0.0, 0.0],  # H mol 0
                [3.0, 0.0, 0.0],  # O mol 1
                [4.0, 0.0, 0.0],  # H mol 1
            ],
            dtype=float,
        )
        compound_type = CompoundType(
            type_id=0,
            key=("OH", (), "dist"),
            formula="OH",
            canonical_labels=("H1", "O1"),
            label_to_local_index={"H1": 0, "O1": 1},
            local_bonds=((0, 1),),
            local_elements=("H", "O"),
            atomic_masses=(1.0, 16.0),
        )
        self.topology_registry = CompoundTypeRegistry([compound_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={("OH", (), "dist"): np.array([[1, 0], [3, 2]], dtype=np.int32)},
            atom_to_type_id=np.array([0, 0, 0, 0], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 0, 1, 1], dtype=np.int32),
            atom_to_local_index=np.array([1, 0, 1, 0], dtype=np.int32),
        )

    def read_frame(self):
        raise ValueError("End of trajectory")


class AngleTrajectory:
    def __init__(self):
        self.box_size = np.array([10.0, 10.0, 10.0], dtype=float)
        self.coords = np.array(
            [
                [0.0, 0.0, 0.0],   # O1 ref
                [1.0, 0.0, 0.0],   # H1 ref
                [2.0, 0.0, 0.0],   # O1 obs mol 0
                [2.0, 1.0, 0.0],   # H1 obs mol 0
                [3.0, 0.0, 0.0],   # O1 obs mol 1
                [4.0, 0.0, 0.0],   # H1 obs mol 1
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
                ("OH", (), "obs"): np.array([[3, 2], [5, 4]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 0, 1, 1, 1, 1], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
            atom_to_local_index=np.array([1, 0, 1, 0, 1, 0], dtype=np.int32),
        )

    def read_frame(self):
        raise ValueError("End of trajectory")


class TwoFamilyDistanceTrajectory:
    def __init__(self):
        self.box_size = np.array([20.0, 20.0, 20.0], dtype=float)
        self.coords = np.array(
            [
                [0.0, 0.0, 0.0],  # Na
                [1.0, 0.0, 0.0],  # Cl 0
                [4.0, 0.0, 0.0],  # Cl 1
                [2.0, 0.0, 0.0],  # O 0
                [5.0, 0.0, 0.0],  # O 1
            ],
            dtype=float,
        )
        ref_type = CompoundType(
            type_id=0,
            key=("Na", (), "ref"),
            formula="Na",
            canonical_labels=("Na1",),
            label_to_local_index={"Na1": 0},
            local_bonds=(),
            local_elements=("Na",),
            atomic_masses=(23.0,),
        )
        cl_type = CompoundType(
            type_id=1,
            key=("Cl", (), "obs-cl"),
            formula="Cl",
            canonical_labels=("Cl1",),
            label_to_local_index={"Cl1": 0},
            local_bonds=(),
            local_elements=("Cl",),
            atomic_masses=(35.5,),
        )
        o_type = CompoundType(
            type_id=2,
            key=("O", (), "obs-o"),
            formula="O",
            canonical_labels=("O1",),
            label_to_local_index={"O1": 0},
            local_bonds=(),
            local_elements=("O",),
            atomic_masses=(16.0,),
        )
        self.topology_registry = CompoundTypeRegistry([ref_type, cl_type, o_type])
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key={
                ("Na", (), "ref"): np.array([[0]], dtype=np.int32),
                ("Cl", (), "obs-cl"): np.array([[1], [2]], dtype=np.int32),
                ("O", (), "obs-o"): np.array([[3], [4]], dtype=np.int32),
            },
            atom_to_type_id=np.array([0, 1, 1, 2, 2], dtype=np.int32),
            atom_to_molecule_index=np.array([0, 1, 2, 3, 4], dtype=np.int32),
            atom_to_local_index=np.array([0, 0, 0, 0, 0], dtype=np.int32),
        )

    def read_frame(self):
        raise ValueError("End of trajectory")


class CDFAnalysisTests(unittest.TestCase):
    def test_build_2d_tuples_rejects_diagonal_only_second_context_matching(self):
        samples = ReferenceSamples(
            contexts=(
                ContextSamples(context_id=0, values=[1.0]),
                ContextSamples(context_id=1, values=[4.0]),
            )
        )

        x_values, y_values = build_2d_tuples(
            samples,
            samples,
            mode="second_context",
            exclude_identical_contexts=True,
        )

        np.testing.assert_allclose(x_values, [1.0, 4.0])
        np.testing.assert_allclose(y_values, [4.0, 1.0])

    def test_build_2d_tuples_crosses_two_context_families(self):
        x_samples = ReferenceSamples(
            contexts=(
                ContextSamples(context_id=0, values=[1.0]),
                ContextSamples(context_id=1, values=[2.0]),
            )
        )
        y_samples = ReferenceSamples(
            contexts=(
                ContextSamples(context_id=0, values=[10.0]),
                ContextSamples(context_id=1, values=[20.0]),
            )
        )

        x_values, y_values = build_2d_tuples(x_samples, y_samples, mode="second_context")

        np.testing.assert_allclose(x_values, [1.0, 1.0, 2.0, 2.0])
        np.testing.assert_allclose(y_values, [10.0, 20.0, 10.0, 20.0])

    def test_build_2d_tuples_supports_distance_angle_same_context(self):
        x_samples = ReferenceSamples(
            contexts=(
                ContextSamples(context_id=0, values=[1.0, 2.0]),
                ContextSamples(context_id=1, values=[3.0]),
            )
        )
        y_samples = ReferenceSamples(
            contexts=(
                ContextSamples(context_id=0, values=[90.0]),
                ContextSamples(context_id=1, values=[45.0, 60.0]),
            )
        )

        x_values, y_values = build_2d_tuples(x_samples, y_samples, mode="same_context")

        np.testing.assert_allclose(x_values, [1.0, 2.0, 3.0, 3.0])
        np.testing.assert_allclose(y_values, [90.0, 90.0, 45.0, 60.0])

    def test_build_2d_tuples_supports_angle_angle_same_context(self):
        x_samples = ReferenceSamples(
            contexts=(
                ContextSamples(context_id=0, values=[90.0]),
                ContextSamples(context_id=1, values=[120.0]),
            )
        )
        y_samples = ReferenceSamples(
            contexts=(
                ContextSamples(context_id=0, values=[45.0]),
                ContextSamples(context_id=1, values=[60.0]),
            )
        )

        x_values, y_values = build_2d_tuples(x_samples, y_samples, mode="same_context")

        np.testing.assert_allclose(x_values, [90.0, 120.0])
        np.testing.assert_allclose(y_values, [45.0, 60.0])

    def test_cdf_analysis_distance_distance_second_context_accumulates_off_diagonal_bins(self):
        analysis = CDFAnalysis(DistanceTrajectory())
        analysis.configure(
            CDFConfig(
                ref_compound_index=0,
                x_axis=DistanceAxisConfig(
                    obs_compound_index=0,
                    ref_labels=["O"],
                    obs_labels=["H"],
                    max_distance=5.0,
                    bin_count=5,
                ),
                y_axis=DistanceAxisConfig(
                    obs_compound_index=0,
                    ref_labels=["O"],
                    obs_labels=["H"],
                    max_distance=5.0,
                    bin_count=5,
                ),
                tuple_mode="second_context",
                exclude_identical_contexts=True,
            )
        )

        self.assertIs(analysis.x_channel, analysis.y_channel)
        analysis.process_frame()

        self.assertEqual(int(analysis.hist.counts.sum()), 4)
        self.assertEqual(analysis.hist.counts[1, 4], 1.0)
        self.assertEqual(analysis.hist.counts[4, 1], 1.0)
        self.assertEqual(analysis.hist.counts[2, 1], 1.0)
        self.assertEqual(analysis.hist.counts[1, 2], 1.0)

    def test_cdf_analysis_distance_distance_two_families_cross_combine_per_reference(self):
        analysis = CDFAnalysis(TwoFamilyDistanceTrajectory())
        analysis.configure(
            CDFConfig(
                ref_compound_index=0,
                x_axis=DistanceAxisConfig(
                    obs_compound_index=1,
                    ref_labels=["Na"],
                    obs_labels=["Cl"],
                    max_distance=6.0,
                    bin_count=6,
                ),
                y_axis=DistanceAxisConfig(
                    obs_compound_index=2,
                    ref_labels=["Na"],
                    obs_labels=["O"],
                    max_distance=6.0,
                    bin_count=6,
                ),
                tuple_mode="second_context",
            )
        )

        analysis.process_frame()

        self.assertEqual(int(analysis.hist.counts.sum()), 4)
        self.assertEqual(analysis.hist.counts[1, 2], 1.0)
        self.assertEqual(analysis.hist.counts[1, 5], 1.0)
        self.assertEqual(analysis.hist.counts[4, 2], 1.0)
        self.assertEqual(analysis.hist.counts[4, 5], 1.0)

    def test_cdf_analysis_distance_angle_postprocess_writes_joint_and_marginals(self):
        analysis = CDFAnalysis(AngleTrajectory())
        analysis.configure(
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
                    obs_base_labels=["O"],
                    obs_tip_labels=["H"],
                    bin_count=18,
                ),
                tuple_mode="same_context",
            )
        )

        analysis.process_frame()
        analysis.processed_frames = 1

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                analysis.postprocess()
                joint_files = list(Path(tmp).glob("cdf_joint*.dat"))
                x_files = list(Path(tmp).glob("cdf_x*.dat"))
                y_files = list(Path(tmp).glob("cdf_y*.dat"))
                self.assertEqual(len(joint_files), 1)
                self.assertEqual(len(x_files), 1)
                self.assertEqual(len(y_files), 1)
                joint_text = joint_files[0].read_text(encoding="utf-8")
            finally:
                os.chdir(cwd)

        self.assertIn("tuple_mode=same_context", joint_text)
        self.assertIn("r/Angstrom", joint_text)
        self.assertIn("angle/deg", joint_text)
        self.assertAlmostEqual(float(analysis.hist.counts.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
