import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.topology import CompoundType, CompoundTypeRegistry, TopologyFrame
from workflow.prepared_setup import (
    PreparedSetupValidationError,
    apply_prepared_setup,
    build_prepared_setup,
    load_prepared_setup,
    save_prepared_setup,
    validate_prepared_setup,
)


class DummyTrajectory:
    def __init__(self, compound_specs, forbidden_bonds=None):
        self.forbidden_bonds = set(forbidden_bonds or set())
        compound_types = []
        molecule_atom_ids_by_key = {}
        atom_to_type_id_parts = []
        atom_to_molecule_index_parts = []
        atom_to_local_index_parts = []
        next_atom_id = 0

        for type_id, spec in enumerate(compound_specs):
            compound_type = CompoundType(
                type_id=type_id,
                key=spec["key"],
                formula=spec["formula"],
                canonical_labels=tuple(spec["labels"]),
                label_to_local_index={label: i for i, label in enumerate(spec["labels"])},
                local_bonds=tuple(spec.get("local_bonds", tuple())),
                local_elements=tuple(label.rstrip("0123456789") for label in spec["labels"]),
                atomic_masses=tuple(1.0 for _ in spec["labels"]),
            )
            compound_types.append(compound_type)

            n_members = spec["count"]
            n_local = len(spec["labels"])
            molecule_atom_ids = np.arange(next_atom_id, next_atom_id + n_members * n_local, dtype=np.int32).reshape(n_members, n_local)
            next_atom_id += n_members * n_local
            molecule_atom_ids_by_key[spec["key"]] = molecule_atom_ids

            atom_to_type_id_parts.append(np.full(n_members * n_local, type_id, dtype=np.int32))
            atom_to_molecule_index_parts.append(
                np.repeat(np.arange(n_members, dtype=np.int32), n_local)
            )
            atom_to_local_index_parts.append(
                np.tile(np.arange(n_local, dtype=np.int32), n_members)
            )

        self.topology_registry = CompoundTypeRegistry(compound_types)
        self.topology_frame = TopologyFrame(
            registry=self.topology_registry,
            molecule_atom_ids_by_key=molecule_atom_ids_by_key,
            atom_to_type_id=np.concatenate(atom_to_type_id_parts) if atom_to_type_id_parts else np.empty(0, dtype=np.int32),
            atom_to_molecule_index=np.concatenate(atom_to_molecule_index_parts) if atom_to_molecule_index_parts else np.empty(0, dtype=np.int32),
            atom_to_local_index=np.concatenate(atom_to_local_index_parts) if atom_to_local_index_parts else np.empty(0, dtype=np.int32),
        )


def make_water_na_cl_traj(water_count=10, na_count=1, cl_count=1, water_labels=None):
    water_labels = water_labels or ["O1", "H1", "H2"]
    compound_specs = [
        {"formula": "Cl", "key": ("Cl", tuple(), "hash_cl"), "labels": ["Cl1"], "count": cl_count},
        {
            "formula": "H2O",
            "key": ("H2O", (("H", "O"), ("H", "O")), "hash_h2o"),
            "labels": water_labels,
            "count": water_count,
            "local_bonds": ((0, 1), (0, 2)),
        },
        {"formula": "Na", "key": ("Na", tuple(), "hash_na"), "labels": ["Na1"], "count": na_count},
    ]
    return DummyTrajectory(compound_specs, forbidden_bonds={(1, 3)})


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
        self.assertEqual([entry["formula"] for entry in loaded.compound_types], ["Cl", "H2O", "Na"])

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

    def test_apply_prepared_setup_restores_forbidden_bonds(self):
        source_traj = make_water_na_cl_traj()
        prepared = build_prepared_setup(source_traj, "lowconc.lmp", "lammps", [0.0, 0.0, 0.0])
        target_traj = make_water_na_cl_traj()
        target_traj.forbidden_bonds = set()

        apply_prepared_setup(target_traj, prepared)

        self.assertEqual(target_traj.forbidden_bonds, {(1, 3)})


if __name__ == "__main__":
    unittest.main()
