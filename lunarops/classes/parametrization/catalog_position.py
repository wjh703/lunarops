"""Helpers shared by the parametrizations of catalog positions (stations and reflectors)."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np

AXES = ("x", "y", "z")


def select_catalog_keys(
    requested: Sequence[str] | None,
    observed: Sequence[str],
    catalog: Collection[str],
    *,
    block: str,
    noun: str,
) -> list[str]:
    """Return the catalog keys to estimate: the requested keys, or every observed key present in the catalog.

    Parameters
    ----------
    requested : Sequence[str] or None
        Configured keys; ``None`` selects every observed key that is in the catalog.
    observed : Sequence[str]
        Sorted keys with observations.
    catalog : Collection[str]
        Keys of the catalog.
    block : str
        Parametrization type used in error messages, for example ``"stationPosition"``.
    noun : str
        Kind of key used in error messages, for example ``"station"``.

    Raises
    ------
    KeyError
        If a requested key is not in the catalog.
    ValueError
        If a requested key has no observations.
    """
    if requested is None:
        return [key for key in observed if key in catalog]
    missing = set(requested) - set(catalog)
    if missing:
        raise KeyError(f"{block}: unknown {noun} key(s) {sorted(missing)}")
    unobserved = set(requested) - set(observed)
    if unobserved:
        raise ValueError(f"{block} requested {noun}(s) have no observations: {sorted(unobserved)}")
    return list(requested)


def catalog_snapshot(catalog: Mapping[str, object], *attributes: str) -> dict[str, tuple[np.ndarray, ...]]:
    """Return a copy of the named vector attributes of every catalog record, as float arrays, keyed by record."""
    return {
        key: tuple(np.asarray(getattr(record, name), dtype=float).copy() for name in attributes)
        for key, record in catalog.items()
    }


def float_list(value) -> list[float]:
    """Return the values of a float vector as a list of Python floats."""
    return [float(item) for item in np.asarray(value, dtype=float)]


def largest_position_update(delta: np.ndarray, count: int, *, stride: int = 3) -> float:
    """Return the largest norm of the 3-D position corrections, in metres.

    Each of the ``count`` blocks of ``stride`` columns starts with the X, Y, Z position correction.
    """
    if not len(delta):
        return 0.0
    return max(float(np.linalg.norm(delta[stride * j : stride * j + 3])) for j in range(count))


__all__ = ["AXES", "catalog_snapshot", "float_list", "largest_position_update", "select_catalog_keys"]
