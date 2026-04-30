import io
import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("networkx") is None:
    load_trajectory = None
else:
    from core.trajectory_loader import load_trajectory


@unittest.skipIf(load_trajectory is None, "networkx is not installed")
class LammpsLoaderTests(unittest.TestCase):
    def test_rows_are_sorted_by_atom_id_when_id_column_exists(self):
        dump = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id element x y z
2 H 2.0 0.0 0.0
1 O 1.0 0.0 0.0
"""
        traj = load_trajectory(io.StringIO(dump), "lammps", np.zeros(3))
        traj.read_frame()

        self.assertEqual(traj.symbols, ["O", "H"])
        np.testing.assert_allclose(traj.coords, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    def test_missing_coordinate_columns_raise_value_error(self):
        dump = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id element q
1 O -0.8
"""
        traj = load_trajectory(io.StringIO(dump), "lammps", np.zeros(3))

        with self.assertRaises(ValueError):
            traj.read_frame()


if __name__ == "__main__":
    unittest.main()
