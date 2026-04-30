"""Common atom-selection helpers for analyses."""

from __future__ import annotations

from utils import label_matches


def collect_atom_indices(compound, labels):
    """Collect global atom indices in a compound matching one or more label patterns."""
    if isinstance(labels, str):
        labels = [labels]

    return [
        idx
        for mol in compound.members
        for label, idx in mol.label_to_global_id.items()
        if any(label_matches(user_label, label) for user_label in labels)
    ]


def build_atom_to_molecule(compound):
    """Return a mapping from global atom index to the molecule object that owns it."""
    atom_to_mol = {}
    for mol in compound.members:
        for idx in mol.label_to_global_id.values():
            atom_to_mol[idx] = mol
    return atom_to_mol


def collect_indices_for_compounds(compounds, labels_by_key, keys):
    """Collect atom indices across compounds using labels keyed by compound key."""
    indices = []
    for key, compound in zip(keys, compounds):
        indices.extend(collect_atom_indices(compound, labels_by_key[key]))
    return indices

