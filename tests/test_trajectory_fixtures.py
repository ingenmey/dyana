import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("networkx") is None:
    load_trajectory = None
    trajectory_loader = None
else:
    import core.trajectory_loader as trajectory_loader
    from core.trajectory_loader import load_trajectory


FIXTURES = Path(__file__).resolve().parent / "fixtures"
KOH_FIXTURE = FIXTURES / "koh_h2o.xyz"
PAIRED_PEROXIDE = """8
paired peroxide
O 0.000 0.000 0.000
H -0.960 0.000 0.000
O 1.450 0.000 0.000
H 2.410 0.000 0.000
H 7.410 0.000 0.000
O 6.450 0.000 0.000
H 4.040 0.000 0.000
O 5.000 0.000 0.000
"""
STRETCHED_OH = """2
stretched oh
O 0.000 0.000 0.000
H 1.330 0.000 0.000
"""


@unittest.skipIf(load_trajectory is None, "networkx is not installed")
class TrajectoryFixtureTests(unittest.TestCase):
    def test_xyz_fixture_reads_first_frame(self):
        with open(FIXTURES / "water128.xyz", "r") as fin:
            traj = load_trajectory(fin, "xyz", np.array([15.67, 15.67, 15.67]))
            traj.read_frame()

        self.assertEqual(traj.n_atoms, 384)
        self.assertEqual(len(traj.symbols), 384)
        self.assertEqual(traj.coords.shape, (384, 3))
        self.assertEqual(traj.symbols[:3], ["H", "H", "O"])

    def test_lammps_fixture_reads_first_frame(self):
        with open(FIXTURES / "ca(bf4)2_thf.lmp", "r") as fin:
            traj = load_trajectory(fin, "lammps", np.zeros(3))
            traj.read_frame()

        self.assertEqual(traj.n_atoms, 2622)
        self.assertEqual(len(traj.symbols), 2622)
        self.assertEqual(traj.coords.shape, (2622, 3))
        np.testing.assert_allclose(traj.box_size, [30.5247410713473] * 3)
        self.assertEqual(traj.symbols[:2], ["Ca", "Ca"])

    def test_water_fixture_builds_topology_registry_and_frame(self):
        with open(FIXTURES / "water128.xyz", "r") as fin:
            traj = load_trajectory(fin, "xyz", np.array([15.67, 15.67, 15.67]))
            traj.read_frame()
            traj.rebuild_topology()

        self.assertIsNotNone(traj.topology_registry)
        self.assertIsNotNone(traj.topology_frame)

        compound_types = traj.topology_frame.get_compound_types()
        self.assertEqual(len(compound_types), 1)
        water_type = compound_types[0]
        self.assertEqual(water_type.formula, "H2O")
        self.assertEqual(set(water_type.canonical_labels), {"H1", "H2", "O1"})

        molecule_atom_ids = traj.topology_frame.get_molecule_atom_ids(water_type)
        self.assertEqual(molecule_atom_ids.shape, (128, 3))
        o_selection = traj.topology_frame.resolve_selection(water_type, ["O"])
        h_selection = traj.topology_frame.resolve_selection(water_type, ["H"])
        self.assertEqual(len(traj.topology_frame.get_atom_ids_for_local_indices(water_type, o_selection.local_indices)), 128)
        self.assertEqual(len(traj.topology_frame.get_atom_ids_for_local_indices(water_type, h_selection.local_indices)), 256)

        atom_type_id, molecule_index, local_index = traj.topology_frame.get_atom_location(int(molecule_atom_ids[0, 0]))
        self.assertEqual(atom_type_id, water_type.type_id)
        self.assertEqual(molecule_index, 0)
        self.assertEqual(local_index, 0)

    def test_equivalent_subgroup_mapping_preserves_paired_bonds_across_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj_path = Path(tmp) / "paired_peroxide.xyz"
            traj_path.write_text(PAIRED_PEROXIDE, encoding="utf-8")

            with open(traj_path, "r", encoding="utf-8") as fin:
                traj = load_trajectory(fin, "xyz", np.array([20.0, 20.0, 20.0]))
                traj.read_frame()
                traj.rebuild_topology()

        compound_types = traj.topology_frame.get_compound_types()
        self.assertEqual(len(compound_types), 1)
        peroxide_type = compound_types[0]
        self.assertEqual(peroxide_type.formula, "H2O2")

        molecule_atom_ids = traj.topology_frame.get_molecule_atom_ids(peroxide_type)
        self.assertEqual(molecule_atom_ids.shape, (2, 4))

        h1 = peroxide_type.label_to_local_index["H1"]
        h2 = peroxide_type.label_to_local_index["H2"]
        o1 = peroxide_type.label_to_local_index["O1"]
        o2 = peroxide_type.label_to_local_index["O2"]

        for atom_ids in molecule_atom_ids:
            coords = traj.coords[atom_ids]
            d_o1_h1 = _periodic_distance(coords[o1], coords[h1], traj.box_size)
            d_o2_h2 = _periodic_distance(coords[o2], coords[h2], traj.box_size)
            d_o1_h2 = _periodic_distance(coords[o1], coords[h2], traj.box_size)
            d_o2_h1 = _periodic_distance(coords[o2], coords[h1], traj.box_size)

            self.assertLess(d_o1_h1, 1.2)
            self.assertLess(d_o2_h2, 1.2)
            self.assertGreater(d_o1_h2, 1.5)
            self.assertGreater(d_o2_h1, 1.5)

    def test_forbidding_all_detected_oh_bonds_atomizes_koh_fixture(self):
        original_scale = trajectory_loader.BOND_DISTANCE_SCALE
        original_overrides = dict(trajectory_loader.BOND_DISTANCE_OVERRIDES)
        try:
            trajectory_loader.BOND_DISTANCE_SCALE = 1.4
            trajectory_loader.BOND_DISTANCE_OVERRIDES.clear()

            with open(KOH_FIXTURE, "r", encoding="utf-8") as fin:
                traj = load_trajectory(fin, "xyz", np.array([22.7274, 22.7274, 22.7274]))
                traj.read_frame()
                traj.rebuild_topology()

            initial_counts = {
                compound_type.formula: traj.topology_frame.get_molecule_count(compound_type)
                for compound_type in traj.topology_frame.get_compound_types()
            }
            self.assertEqual(initial_counts, {"H2O": 123, "H3O2": 27, "HO": 123, "K": 150})

            for compound_type in traj.topology_frame.get_compound_types():
                if compound_type.formula not in {"H2O", "H3O2", "HO"}:
                    continue
                molecule_atom_ids = traj.topology_frame.get_molecule_atom_ids(compound_type)
                for atom_ids in molecule_atom_ids:
                    for local_a, local_b in compound_type.local_bonds:
                        global_a = int(atom_ids[local_a])
                        global_b = int(atom_ids[local_b])
                        traj.forbidden_bonds.add((min(global_a, global_b), max(global_a, global_b)))

            traj.rebuild_topology()
            final_counts = {
                compound_type.formula: traj.topology_frame.get_molecule_count(compound_type)
                for compound_type in traj.topology_frame.get_compound_types()
            }
            self.assertEqual(final_counts, {"H": 450, "K": 150, "O": 300})
        finally:
            trajectory_loader.BOND_DISTANCE_SCALE = original_scale
            trajectory_loader.BOND_DISTANCE_OVERRIDES.clear()
            trajectory_loader.BOND_DISTANCE_OVERRIDES.update(original_overrides)

    def test_pair_specific_oh_cutoff_override_can_split_a_stretched_oh_pair(self):
        original_overrides = dict(trajectory_loader.BOND_DISTANCE_OVERRIDES)
        original_scale = trajectory_loader.BOND_DISTANCE_SCALE
        try:
            trajectory_loader.BOND_DISTANCE_SCALE = 1.4
            trajectory_loader.BOND_DISTANCE_OVERRIDES.clear()
            with tempfile.TemporaryDirectory() as tmp:
                traj_path = Path(tmp) / "stretched_oh.xyz"
                traj_path.write_text(STRETCHED_OH, encoding="utf-8")

                with open(traj_path, "r", encoding="utf-8") as fin:
                    traj = load_trajectory(fin, "xyz", np.array([20.0, 20.0, 20.0]))
                    traj.read_frame()
                    traj.rebuild_topology()

                initial_counts = {
                    compound_type.formula: traj.topology_frame.get_molecule_count(compound_type)
                    for compound_type in traj.topology_frame.get_compound_types()
                }
                self.assertEqual(initial_counts, {"HO": 1})

                trajectory_loader.BOND_DISTANCE_OVERRIDES.clear()
                trajectory_loader.BOND_DISTANCE_OVERRIDES["H-O"] = 1.28

                with open(traj_path, "r", encoding="utf-8") as fin:
                    traj = load_trajectory(fin, "xyz", np.array([20.0, 20.0, 20.0]))
                    traj.read_frame()
                    traj.rebuild_topology()

                override_counts = {
                    compound_type.formula: traj.topology_frame.get_molecule_count(compound_type)
                    for compound_type in traj.topology_frame.get_compound_types()
                }
                self.assertEqual(override_counts, {"H": 1, "O": 1})
        finally:
            trajectory_loader.BOND_DISTANCE_SCALE = original_scale
            trajectory_loader.BOND_DISTANCE_OVERRIDES.clear()
            trajectory_loader.BOND_DISTANCE_OVERRIDES.update(original_overrides)
def _periodic_distance(coord_a: np.ndarray, coord_b: np.ndarray, box_size: np.ndarray) -> float:
    delta = coord_a - coord_b
    delta -= np.round(delta / box_size) * box_size
    return float(np.linalg.norm(delta))


if __name__ == "__main__":
    unittest.main()
