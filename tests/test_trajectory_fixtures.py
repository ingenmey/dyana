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

    def test_water_fixture_recognizes_one_h2o_compound(self):
        with open(FIXTURES / "water128.xyz", "r") as fin:
            traj = load_trajectory(fin, "xyz", np.array([15.67, 15.67, 15.67]))
            traj.read_frame()
            traj.guess_molecules()

        compounds = list(traj.compounds.values())
        self.assertEqual(len(compounds), 1)
        self.assertEqual(compounds[0].rep, "H2O")
        self.assertEqual(len(compounds[0].members), 128)

        labels = set(compounds[0].members[0].label_to_id)
        self.assertEqual(labels, {"H1", "H2", "O1"})


if __name__ == "__main__":
    unittest.main()
