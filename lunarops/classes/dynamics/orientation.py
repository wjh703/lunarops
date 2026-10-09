"""Explicit scalar and batch orientation services for orbit dynamics."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from lunarops.classes.time import Epoch


def inertial2fixed_matrix_from_pole(pole_vector) -> np.ndarray:
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


@dataclass(frozen=True, slots=True)
class OrientationProvider:
    matrix: Callable[[Epoch], np.ndarray]
    matrices: Callable[[Sequence[Epoch]], np.ndarray]

    @classmethod
    def from_scalar(cls, matrix):
        return cls(matrix, lambda epochs: np.asarray([matrix(epoch) for epoch in epochs]))
