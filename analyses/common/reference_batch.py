"""Shared per-frame reference context for multichannel analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceBatch:
    """Per-frame context shared by channels for one reference compound."""

    ref_compound_key: tuple
    ref_compound_type: object
    molecule_atom_ids: np.ndarray
    coords: np.ndarray
    box: np.ndarray
    topology_frame: object

    def __post_init__(self):
        self.molecule_atom_ids = np.asarray(self.molecule_atom_ids)
        self.coords = np.asarray(self.coords)
        self.box = np.asarray(self.box)

        if self.molecule_atom_ids.ndim != 2:
            raise ValueError("molecule_atom_ids must be a 2D array.")
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("coords must be a (natoms, 3) array.")
        if self.box.shape != (3,):
            raise ValueError("box must be a length-3 array.")

    @property
    def n_references(self) -> int:
        return int(self.molecule_atom_ids.shape[0])
