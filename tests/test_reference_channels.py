import unittest

import numpy as np

from analyses.common.reference_channels import (
    AngleChannel,
    DistanceChannel,
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


class ReferenceChannelTests(unittest.TestCase):
    def test_radial_shell_volumes(self):
        volumes = radial_shell_volumes(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(volumes, [(4.0 / 3.0) * np.pi, (4.0 / 3.0) * np.pi * 7.0])

    def test_angular_inverse_sin_weights(self):
        weights = angular_inverse_sin_weights(np.array([0.0, 90.0, 180.0]))
        self.assertEqual(len(weights), 2)
        self.assertTrue(np.all(weights > 0))

    def test_distance_channel_returns_samples_per_reference_molecule(self):
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

    def test_angle_channel_returns_samples_per_reference_molecule(self):
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

        np.testing.assert_allclose(angles, [90.0])


if __name__ == "__main__":
    unittest.main()
