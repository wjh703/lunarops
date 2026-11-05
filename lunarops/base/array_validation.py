"""Validation of numeric array inputs: finite float values with a fixed shape or size."""

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
    """Return a value as a finite float array, optionally checked against a shape or a size.

    ``shape`` suits matrix-like values. ``size`` accepts input of any shape that holds the given number of
    values. When either is given, the result is reshaped to ``shape`` or to the 1-D shape ``(size,)``.

    Parameters
    ----------
    value : array_like
        Input values.
    shape : tuple of int, optional
        Required shape of the result; the input must hold exactly ``prod(shape)`` values.
    size : int, optional
        Required number of values; the result has shape ``(size,)``.
    name : str
        Name of the value used in error messages.
    copy : bool, optional
        Copy the input; passed to ``numpy.array``. Default True.
    readonly : bool, optional
        Mark the result as not writeable. Default False.

    Returns
    -------
    ndarray of float
        Validated array, shape ``shape`` or ``(size,)`` when given, otherwise the input shape.

    Raises
    ------
    ValueError
        If both ``shape`` and ``size`` are given, if the number of values does not match, or if any value is
        not finite.
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
    """Return a value as a finite float 3-vector.

    Parameters
    ----------
    value : array_like
        Input holding exactly three values.
    name : str
        Name of the value used in error messages.
    copy : bool, optional
        Copy the input. Default True.
    readonly : bool, optional
        Mark the result as not writeable. Default False.

    Returns
    -------
    ndarray, shape (3,)
        Validated vector, in the units and frame of the input.

    Raises
    ------
    ValueError
        If the input does not hold three values or a value is not finite.
    """
    return finite_array(value, size=3, name=name, copy=copy, readonly=readonly)


__all__ = ["finite_array", "vector3"]
