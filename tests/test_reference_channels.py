import unittest

import numpy as np

from analyses.common.reference_channels import (
    AngleChannel,
    ContextSamples,
    DistanceChannel,
    ReferenceSamples,
    angular_inverse_sin_weights,
    radial_shell_volumes,
)
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


class ReferenceChannelTests(unittest.TestCase):
    def test_radial_shell_volumes(self):
        volumes = radial_shell_volumes(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(volumes, [(4.0 / 3.0) * np.pi, (4.0 / 3.0) * np.pi * 7.0])

    def test_angular_inverse_sin_weights(self):
        weights = angular_inverse_sin_weights(np.array([0.0, 90.0, 180.0]))
        self.assertEqual(len(weights), 2)
        self.assertTrue(np.all(weights > 0))

    def test_observed_context_samples_validate_optional_array_lengths(self):
        with self.assertRaises(ValueError):
            ContextSamples(context_id=0, values=[1.0, 2.0], enabled=[True])

        with self.assertRaises(ValueError):
            ContextSamples(context_id=0, values=[1.0, 2.0], combination_ids=[0])

    def test_grouped_channel_samples_reject_duplicates_and_collect_enabled_values(self):
        with self.assertRaises(ValueError):
            ReferenceSamples(
                contexts=(
                    ContextSamples(context_id=0, values=[1.0]),
                    ContextSamples(context_id=0, values=[2.0]),
                )
            )

        grouped = ReferenceSamples(
            contexts=(
                ContextSamples(context_id=0, values=[1.0, 2.0], enabled=[True, False]),
                ContextSamples(context_id=1, values=[3.0]),
            )
        )

        np.testing.assert_allclose(grouped.values, [1.0, 3.0])

    def test_distance_channel_returns_grouped_samples_per_observed_context(self):
        traj = DistanceTrajectory()
        channel = DistanceChannel(
            ref_key=("OH", (), "dist"),
            obs_key=("OH", (), "dist"),
            ref_local_indices=(1,),
            obs_local_indices=(0,),
            max_distance=4.1,
            bin_edges=np.array([0.0, 1.0, 2.0, 5.0]),
        )
        channel.rebuild_runtime_state(traj)
        batch = channel.build_batch(traj)
        channel.begin_batch(batch)

        first = channel.samples_for_reference(batch, 0)
        second = channel.samples_for_reference(batch, 1)

        self.assertEqual([context.context_id for context in first.contexts], [0, 1])
        np.testing.assert_allclose(first.contexts[0].values, [1.0])
        np.testing.assert_allclose(first.contexts[1].values, [4.0])

        self.assertEqual([context.context_id for context in second.contexts], [0, 1])
        np.testing.assert_allclose(second.contexts[0].values, [2.0])
        np.testing.assert_allclose(second.contexts[1].values, [1.0])

    def test_distance_channel_collects_all_grouped_values_for_reference_molecule(self):
        traj = DistanceTrajectory()
        channel = DistanceChannel(
            ref_key=("OH", (), "dist"),
            obs_key=("OH", (), "dist"),
            ref_local_indices=(1,),
            obs_local_indices=(0,),
            max_distance=4.1,
            bin_edges=np.array([0.0, 1.0, 2.0, 5.0]),
        )
        channel.rebuild_runtime_state(traj)
        batch = channel.build_batch(traj)
        channel.begin_batch(batch)

        first = channel.samples_for_reference(batch, 0).values
        second = channel.samples_for_reference(batch, 1).values

        np.testing.assert_allclose(np.sort(first), [1.0, 4.0])
        np.testing.assert_allclose(np.sort(second), [1.0, 2.0])

    def test_distance_channel_flat_values_match_grouped_values(self):
        traj = DistanceTrajectory()
        channel = DistanceChannel(
            ref_key=("OH", (), "dist"),
            obs_key=("OH", (), "dist"),
            ref_local_indices=(1,),
            obs_local_indices=(0,),
            max_distance=4.1,
            bin_edges=np.array([0.0, 1.0, 2.0, 5.0]),
        )
        channel.rebuild_runtime_state(traj)
        batch = channel.build_batch(traj)
        channel.begin_batch(batch)

        flat = channel.values_for_reference(batch, 0)
        grouped = channel.samples_for_reference(batch, 0).values

        np.testing.assert_allclose(np.sort(flat), np.sort(grouped))

    def test_distance_channel_excludes_zero_length_self_distances(self):
        traj = DistanceTrajectory()
        channel = DistanceChannel(
            ref_key=("OH", (), "dist"),
            obs_key=("OH", (), "dist"),
            ref_local_indices=(1,),
            obs_local_indices=(1,),
            max_distance=4.1,
            bin_edges=np.array([0.0, 1.0, 2.0, 5.0]),
        )
        channel.rebuild_runtime_state(traj)
        batch = channel.build_batch(traj)
        channel.begin_batch(batch)

        first = channel.samples_for_reference(batch, 0)
        second = channel.samples_for_reference(batch, 1)

        self.assertEqual([context.context_id for context in first.contexts], [1])
        np.testing.assert_allclose(first.contexts[0].values, [3.0])
        self.assertEqual([context.context_id for context in second.contexts], [0])
        np.testing.assert_allclose(second.contexts[0].values, [3.0])

    def test_angle_channel_returns_grouped_samples_per_observed_context(self):
        traj = AngleTrajectory()
        channel = AngleChannel(
            ref_key=("H2O", (), "ref"),
            obs_key=("OH", (), "obs"),
            ref_base_source="r",
            ref_tip_source="r",
            obs_base_source="o",
            obs_tip_source="o",
            ref_base_local_indices=(1,),
            ref_tip_local_indices=(0,),
            obs_base_local_indices=(1,),
            obs_tip_local_indices=(0,),
            bin_edges=np.linspace(0.0, 180.0, 7),
        )
        channel.rebuild_runtime_state(traj)
        batch = channel.build_batch(traj)

        grouped = channel.samples_for_reference(batch, 0)

        self.assertEqual([context.context_id for context in grouped.contexts], [0, 1])
        np.testing.assert_allclose(grouped.contexts[0].values, [90.0])
        np.testing.assert_allclose(grouped.contexts[1].values, [0.0])

    def test_angle_channel_applies_vector_cutoffs_inside_grouped_output(self):
        traj = AngleTrajectory()
        channel = AngleChannel(
            ref_key=("H2O", (), "ref"),
            obs_key=("OH", (), "obs"),
            ref_base_source="r",
            ref_tip_source="r",
            obs_base_source="o",
            obs_tip_source="o",
            ref_base_local_indices=(1,),
            ref_tip_local_indices=(0,),
            obs_base_local_indices=(1,),
            obs_tip_local_indices=(0,),
            bin_edges=np.linspace(0.0, 180.0, 7),
            v2_cutoff=0.5,
        )
        channel.rebuild_runtime_state(traj)
        batch = channel.build_batch(traj)

        grouped = channel.samples_for_reference(batch, 0)

        self.assertTrue(grouped.is_empty)
        self.assertEqual(grouped.values.size, 0)

    def test_angle_channel_collects_all_grouped_values_for_reference_molecule(self):
        traj = AngleTrajectory()
        channel = AngleChannel(
            ref_key=("H2O", (), "ref"),
            obs_key=("OH", (), "obs"),
            ref_base_source="r",
            ref_tip_source="r",
            obs_base_source="o",
            obs_tip_source="o",
            ref_base_local_indices=(1,),
            ref_tip_local_indices=(0,),
            obs_base_local_indices=(1,),
            obs_tip_local_indices=(0,),
            bin_edges=np.linspace(0.0, 180.0, 7),
        )
        channel.rebuild_runtime_state(traj)
        batch = channel.build_batch(traj)

        angles = channel.samples_for_reference(batch, 0).values

        np.testing.assert_allclose(np.sort(angles), [0.0, 90.0])

    def test_angle_channel_flat_values_match_grouped_values(self):
        traj = AngleTrajectory()
        channel = AngleChannel(
            ref_key=("H2O", (), "ref"),
            obs_key=("OH", (), "obs"),
            ref_base_source="r",
            ref_tip_source="r",
            obs_base_source="o",
            obs_tip_source="o",
            ref_base_local_indices=(1,),
            ref_tip_local_indices=(0,),
            obs_base_local_indices=(1,),
            obs_tip_local_indices=(0,),
            bin_edges=np.linspace(0.0, 180.0, 7),
        )
        channel.rebuild_runtime_state(traj)
        batch = channel.build_batch(traj)

        flat = channel.values_for_reference(batch, 0)
        grouped = channel.samples_for_reference(batch, 0).values

        np.testing.assert_allclose(np.sort(flat), np.sort(grouped))


if __name__ == "__main__":
    unittest.main()
