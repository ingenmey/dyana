import importlib.util
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("networkx") is None:
    load_trajectory = None
else:
    from core.trajectory_loader import load_trajectory


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@unittest.skipIf(load_trajectory is None, "networkx is not installed")
class TrajectoryFixtureTests(unittest.TestCase):
    def test_xyz_fixture_reads_first_frame(self):
        with open(FIXTURES / "water128.xyz", "r") as fin:
            traj = load_trajectory(fin, "xyz", np.array([15.67, 15.67, 15.67]))
            traj.read_frame()

        self.assertEqual(traj.natoms, 384)
        self.assertEqual(len(traj.symbols), 384)
        self.assertEqual(traj.coords.shape, (384, 3))
        self.assertEqual(traj.symbols[:3], ["H", "H", "O"])

    def test_lammps_fixture_reads_first_frame(self):
        with open(FIXTURES / "ca(bf4)2_thf.lmp", "r") as fin:
            traj = load_trajectory(fin, "lammps", np.zeros(3))
            traj.read_frame()

        self.assertEqual(traj.natoms, 2622)
        self.assertEqual(len(traj.symbols), 2622)
        self.assertEqual(traj.coords.shape, (2622, 3))
        np.testing.assert_allclose(traj.box_size, [30.5247410713473] * 3)
        self.assertEqual(traj.symbols[:2], ["Ca", "Ca"])

    def test_water_fixture_builds_topology_registry_and_frame(self):
        with open(FIXTURES / "water128.xyz", "r") as fin:
            traj = load_trajectory(fin, "xyz", np.array([15.67, 15.67, 15.67]))
            traj.read_frame()
            traj.guess_molecules()

        self.assertIsNotNone(traj.topology_registry)
        self.assertIsNotNone(traj.topology_frame)

        compound_types = traj.topology_frame.get_compound_types()
        self.assertEqual(len(compound_types), 1)
        water_type = compound_types[0]
        self.assertEqual(water_type.rep, "H2O")
        self.assertEqual(set(water_type.canonical_labels), {"H1", "H2", "O1"})

        member_atom_ids = traj.topology_frame.get_member_atom_ids(water_type)
        self.assertEqual(member_atom_ids.shape, (128, 3))
        o_selection = traj.topology_frame.resolve_selection(water_type, ["O"])
        h_selection = traj.topology_frame.resolve_selection(water_type, ["H"])
        self.assertEqual(len(traj.topology_frame.get_atom_indices_for_local_indices(water_type, o_selection.local_indices)), 128)
        self.assertEqual(len(traj.topology_frame.get_atom_indices_for_local_indices(water_type, h_selection.local_indices)), 256)

        atom_type_id, member_index, local_index = traj.topology_frame.get_atom_location(int(member_atom_ids[0, 0]))
        self.assertEqual(atom_type_id, water_type.type_id)
        self.assertEqual(member_index, 0)
        self.assertEqual(local_index, 0)


if __name__ == "__main__":
    unittest.main()
