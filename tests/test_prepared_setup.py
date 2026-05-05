import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

from prepared_setup import (
    PreparedSetupValidationError,
    apply_prepared_setup_recipe,
    build_prepared_setup,
    load_prepared_setup,
    save_prepared_setup,
    validate_prepared_setup,
)


class DummyMember:
    def __init__(self, labels):
        self.label_to_id = {label: i for i, label in enumerate(labels)}


class DummyCompound:
    def __init__(self, rep, key, labels, count):
        self.rep = rep
        self.key = key
        self.members = [DummyMember(labels) for _ in range(count)]


class DummyTrajectory:
    def __init__(self, compounds, forbidden_bonds=None):
        self.compounds = OrderedDict((compound.key, compound) for compound in compounds)
        self.forbidden_bonds = set(forbidden_bonds or set())


def make_water_na_cl_traj(water_count=10, na_count=1, cl_count=1, water_labels=None):
    water_labels = water_labels or ["O1", "H1", "H2"]
    compounds = [
        DummyCompound("Cl", ("Cl", tuple(), "hash_cl"), ["Cl1"], cl_count),
        DummyCompound("H2O", ("H2O", (("H", "O"), ("H", "O")), "hash_h2o"), water_labels, water_count),
        DummyCompound("Na", ("Na", tuple(), "hash_na"), ["Na1"], na_count),
    ]
    return DummyTrajectory(compounds, forbidden_bonds={(1, 3)})


class PreparedSetupTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        traj = make_water_na_cl_traj()
        prepared = build_prepared_setup(traj, "lowconc.lmp", "lammps", [0.0, 0.0, 0.0])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prepared_setup.json"
            save_prepared_setup(path, prepared)
            loaded = load_prepared_setup(path)

        self.assertEqual(loaded.recipe["trajectory_format"], "lammps")
        self.assertEqual(loaded.recipe["forbidden_bonds"], [[1, 3]])
        self.assertEqual([entry["rep"] for entry in loaded.compound_types], ["Cl", "H2O", "Na"])

    def test_validation_accepts_same_compound_types_with_different_counts(self):
        saved_traj = make_water_na_cl_traj(water_count=100, na_count=1, cl_count=1)
        loaded_traj = make_water_na_cl_traj(water_count=20, na_count=5, cl_count=5)
        prepared = build_prepared_setup(saved_traj, "lowconc.lmp", "lammps", [0.0, 0.0, 0.0])

        validate_prepared_setup(loaded_traj, prepared)

    def test_validation_rejects_label_mismatch(self):
        saved_traj = make_water_na_cl_traj()
        loaded_traj = make_water_na_cl_traj(water_labels=["O1", "H1", "H3"])
        prepared = build_prepared_setup(saved_traj, "lowconc.lmp", "lammps", [0.0, 0.0, 0.0])

        with self.assertRaises(PreparedSetupValidationError):
            validate_prepared_setup(loaded_traj, prepared)

    def test_apply_prepared_setup_recipe_restores_forbidden_bonds(self):
        source_traj = make_water_na_cl_traj()
        prepared = build_prepared_setup(source_traj, "lowconc.lmp", "lammps", [0.0, 0.0, 0.0])
        target_traj = make_water_na_cl_traj()
        target_traj.forbidden_bonds = set()

        apply_prepared_setup_recipe(target_traj, prepared)

        self.assertEqual(target_traj.forbidden_bonds, {(1, 3)})


if __name__ == "__main__":
    unittest.main()
