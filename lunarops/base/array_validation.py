"""Array validation helpers shared by physical-model modules."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def finite_array(
    value,
    *,
    shape: tuple[int, ...] | None = None,
    size: int | None = None,
    name: str,
    copy: bool = True,
    readonly: bool = False,
) -> np.ndarray:
    """Return a finite float array with one explicit shape/size contract.

    ``shape`` is preferred for matrix-like values.  ``size`` is useful when the
    accepted input may be any shape but must contain a fixed number of values.
    The returned array is always reshaped when either contract is supplied.
    """
    if shape is not None and size is not None:
        raise ValueError("finite_array() accepts either shape or size, not both.")
    array = np.array(value, dtype=float, copy=copy)
    if shape is not None:
        expected = int(np.prod(shape))
        if array.size != expected:
            raise ValueError(f"{name} must contain exactly {expected} values, got shape {array.shape}.")
        array = array.reshape(shape)
    elif size is not None:
        if array.size != int(size):
            raise ValueError(f"{name} must contain exactly {int(size)} values, got shape {array.shape}.")
        array = array.reshape(int(size))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if readonly:
        array.setflags(write=False)
    return array


def vector3(
    value: ArrayLike,
    *,
    name: str,
    copy: bool = True,
    readonly: bool = False,
) -> np.ndarray:
    return finite_array(value, size=3, name=name, copy=copy, readonly=readonly)


__all__ = ["finite_array", "vector3"]
