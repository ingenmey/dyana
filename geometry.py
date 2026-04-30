"""Periodic-boundary geometry helpers used across analyses.

The current codebase assumes orthorhombic simulation boxes represented as
three box lengths. Keeping these operations in one module makes that assumption
visible and gives us one place to extend later if triclinic boxes are needed.
"""

from __future__ import annotations

import numpy as np


def as_box_array(box_size) -> np.ndarray:
    """Return box dimensions as a float NumPy array."""
    box = np.asarray(box_size, dtype=float)
    if box.shape != (3,):
        raise ValueError(f"Expected orthorhombic box with shape (3,), got {box.shape}.")
    return box


def wrap(coords, box_size):
    """Wrap coordinates into an orthorhombic periodic box."""
    box = as_box_array(box_size)
    return np.mod(coords, box)


def minimum_image(delta, box_size):
    """Apply the minimum-image convention to displacement vector(s)."""
    box = as_box_array(box_size)
    return delta - box * np.round(delta / box)


def distance_squared(a, b, box_size) -> float:
    """Squared minimum-image distance between two positions."""
    delta = minimum_image(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), box_size)
    return float(np.dot(delta, delta))


def unwrap_around_reference(coords, box_size):
    """Unwrap a cluster of wrapped coordinates around its first point."""
    points = np.asarray(coords, dtype=float)
    if len(points) == 0:
        return points.copy()

    reference = points[0]
    deltas = minimum_image(points - reference, box_size)
    return reference + deltas


def periodic_center(coords, box_size, weights=None):
    """Average wrapped coordinates after unwrapping around the first point."""
    points = np.asarray(coords, dtype=float)
    if len(points) == 0:
        return np.zeros(3, dtype=float)

    unwrapped = unwrap_around_reference(points, box_size)
    center = np.average(unwrapped, axis=0, weights=weights)
    return wrap(center, box_size)

