from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.trajectory_loader import BOND_DISTANCE_SCALE, EXCLUDED_ELEMENTS, NEIGHBOR_SEARCH_SCALE


class PreparedSetupValidationError(RuntimeError):
    """Raised when a prepared setup does not match the current trajectory state."""

    pass


@dataclass(frozen=True)
class PreparedSetup:
    """Saved topology-review state used to reconstruct compatible runs."""

    payload: dict

    @property
    def recipe(self):
        return self.payload["recipe"]

    @property
    def compound_types(self):
        return self.payload["compound_types"]

    @property
    def metadata(self):
        return self.payload["metadata"]


def build_prepared_setup(traj, traj_file: str, traj_format: str, cell_vectors) -> PreparedSetup:
    """Build a reusable prepared-setup payload from the current trajectory state."""
    observed_counts = Counter()
    for compound_type in traj.topology_registry:
        observed_counts[compound_type.formula] += traj.topology_frame.get_molecule_count(compound_type)

    payload = {
        "format_version": 1,
        "recipe": {
            "trajectory_format": traj_format,
            "cell_vectors": [float(x) for x in cell_vectors],
            "excluded_elements": sorted(EXCLUDED_ELEMENTS),
            "neighbor_search_scale": float(NEIGHBOR_SEARCH_SCALE),
            "bond_distance_scale": float(BOND_DISTANCE_SCALE),
            "forbidden_bonds": [list(pair) for pair in sorted(traj.forbidden_bonds)],
        },
        "compound_types": _compound_type_entries(traj),
        "metadata": {
            "source_file": str(traj_file),
            "observed_counts": dict(sorted(observed_counts.items())),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return PreparedSetup(payload)


def save_prepared_setup(path: str | Path, prepared_setup: PreparedSetup):
    """Write a prepared setup JSON file."""
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(prepared_setup.payload, fout, indent=2, sort_keys=False)
        fout.write("\n")


def load_prepared_setup(path: str | Path) -> PreparedSetup:
    """Load a prepared setup JSON file."""
    with open(path, "r", encoding="utf-8") as fin:
        payload = json.load(fin)
    return PreparedSetup(payload)


def apply_prepared_setup(traj, prepared_setup: PreparedSetup):
    """Apply the prepared-setup recipe to a trajectory before rebuilding topology."""
    recipe = prepared_setup.recipe
    traj.forbidden_bonds = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in recipe.get("forbidden_bonds", [])
    }


def validate_prepared_setup(traj, prepared_setup: PreparedSetup):
    """Validate that reconstructed topology matches the prepared setup."""
    recipe = prepared_setup.recipe
    _validate_recipe_compatibility(recipe)

    expected = {_compound_signature(entry): entry for entry in prepared_setup.compound_types}
    observed = {_compound_signature(entry): entry for entry in _compound_type_entries(traj)}

    expected_keys = set(expected)
    observed_keys = set(observed)
    if expected_keys != observed_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        details = []
        if missing:
            details.append(f"missing expected compound types: {missing}")
        if extra:
            details.append(f"found unexpected compound types: {extra}")
        raise PreparedSetupValidationError("Prepared setup does not match reconstructed compound list: " + "; ".join(details))

    for signature in sorted(expected_keys):
        expected_entry = expected[signature]
        observed_entry = observed[signature]
        if expected_entry["labels"] != observed_entry["labels"]:
            raise PreparedSetupValidationError(
                f"Prepared setup labels for compound {expected_entry['formula']} do not match reconstructed labels: "
                f"expected {expected_entry['labels']}, got {observed_entry['labels']}"
            )


def _validate_recipe_compatibility(recipe):
    excluded_elements = sorted(EXCLUDED_ELEMENTS)
    if recipe.get("excluded_elements", []) != excluded_elements:
        raise PreparedSetupValidationError(
            f"Prepared setup excluded elements {recipe.get('excluded_elements', [])} do not match current configuration {excluded_elements}."
        )
    if float(recipe.get("neighbor_search_scale", NEIGHBOR_SEARCH_SCALE)) != float(NEIGHBOR_SEARCH_SCALE):
        raise PreparedSetupValidationError(
            f"Prepared setup neighbor_search_scale {recipe.get('neighbor_search_scale')} does not match current configuration {NEIGHBOR_SEARCH_SCALE}."
        )
    if float(recipe.get("bond_distance_scale", BOND_DISTANCE_SCALE)) != float(BOND_DISTANCE_SCALE):
        raise PreparedSetupValidationError(
            f"Prepared setup bond_distance_scale {recipe.get('bond_distance_scale')} does not match current configuration {BOND_DISTANCE_SCALE}."
        )


def _compound_type_entries(traj):
    entries = []
    for compound_type in traj.topology_registry:
        formula, bond_types, graph_hash = compound_type.key
        entries.append(
            {
                "formula": formula,
                "bond_types": [list(pair) for pair in bond_types],
                "graph_hash": graph_hash,
                "labels": list(compound_type.canonical_labels),
            }
        )
    entries.sort(key=_compound_signature)
    return entries


def _compound_signature(entry):
    return (
        entry["formula"],
        tuple(tuple(pair) for pair in entry["bond_types"]),
        entry["graph_hash"],
    )
