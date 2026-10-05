"""Small orientation utilities used by gravity and tidal models."""

from __future__ import annotations

import numpy as np


def inertial2fixed_matrix_from_pole(pole_vector) -> np.ndarray:
    """Build an inertial-to-fixed matrix whose fixed z-axis is ``pole_vector``."""
    pole = np.asarray(pole_vector, dtype=float)
    if pole.shape != (3,) or not np.all(np.isfinite(pole)):
        raise ValueError("pole_vector must be a finite three-vector")
    norm = np.linalg.norm(pole)
    if norm == 0.0:
        raise ValueError("pole_vector must be nonzero")
    z_axis = pole / norm
    reference = np.array((1.0, 0.0, 0.0))
    if abs(float(reference @ z_axis)) > 0.9:
        reference = np.array((0.0, 1.0, 0.0))
    x_axis = reference - (reference @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    matrix = np.ascontiguousarray(np.vstack((x_axis, y_axis, z_axis)))
    matrix.setflags(write=False)
    return matrix
