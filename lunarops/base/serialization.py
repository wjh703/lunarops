"""Conversion of LunarOps values to plain JSON/YAML-compatible data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np


def plain_data(value):
    """Convert a value recursively to plain data of dicts, lists, strings, numbers, booleans and None.

    Dataclass instances become dicts, NumPy arrays and tuples become lists, NumPy scalars become Python
    scalars, sets become lists sorted by ``repr``, mapping keys become strings, and paths, dates and
    date-times become strings.

    Parameters
    ----------
    value : object
        Value to convert.

    Returns
    -------
    object
        Plain-data equivalent of ``value``.

    Raises
    ------
    ValueError
        If a float is not finite, a mapping key is empty, or two keys collide after conversion to strings.
    TypeError
        If a value has a type that cannot be converted.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return plain_data(asdict(value))
    if isinstance(value, np.ndarray):
        if value.ndim and value.dtype.kind in "biuf":
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError("Structured LunarOps data rejects non-finite floats.")
            return value.tolist()
        return [plain_data(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return plain_data(value.item())
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if not normalized_key:
                raise ValueError("Structured LunarOps data rejects empty mapping keys.")
            if normalized_key in result:
                raise ValueError(f"Structured LunarOps data has colliding mapping key {normalized_key!r}.")
            result[normalized_key] = plain_data(item)
        return result
    if isinstance(value, set):
        return sorted((plain_data(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    if isinstance(value, (Path, date, datetime)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Structured LunarOps data rejects non-finite floats.")
        return value
    raise TypeError(f"Structured LunarOps data cannot encode {type(value).__name__} objects.")


__all__ = ["plain_data"]
