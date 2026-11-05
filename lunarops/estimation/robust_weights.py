"""Robust observation-weight models for iterative LLR adjustment."""

from __future__ import annotations

from typing import cast

import numpy as np

IGG3_MODEL = "igg3"
DIRECT_REJECTION_MODEL = "directRejection"
ROBUST_WEIGHT_MODELS = frozenset((IGG3_MODEL, DIRECT_REJECTION_MODEL))


def robust_factors(model: str, k0: float, k1: float | None, standardized: np.ndarray) -> np.ndarray:
    """Map standardized residuals to robust weight factors in [0, 1].

    With ``u = |standardized|``, ``directRejection`` gives 1 for ``u <= k0`` and 0 otherwise. ``igg3`` is the
    IGG-III model for iteratively reweighted least squares with thresholds ``k0 < k1``: 1 for ``u <= k0``,
    ``(k0 / u) * ((k1 - u) / (k1 - k0))**2`` for ``k0 < u <= k1`` and 0 for ``u > k1``.

    Parameters
    ----------
    model : str
        ``directRejection`` or ``igg3``; any other value is treated as ``igg3``.
    k0 : float
        Full-weight threshold.
    k1 : float or None
        Rejection threshold of ``igg3``; not used by ``directRejection``.
    standardized : ndarray, shape (n,)
        Standardized residuals.

    Returns
    -------
    ndarray, shape (n,)
        Weight factors.
    """
    if model not in ROBUST_WEIGHT_MODELS:
        raise ValueError(f"Unknown robust weight model {model!r}; expected one of {sorted(ROBUST_WEIGHT_MODELS)}.")
    magnitude = np.abs(standardized)
    if model == DIRECT_REJECTION_MODEL:
        return np.where(magnitude <= k0, 1.0, 0.0)
    k1 = cast(float, k1)
    result = np.zeros_like(magnitude)
    full = magnitude <= k0
    middle = (magnitude > k0) & (magnitude <= k1)
    result[full] = 1.0
    result[middle] = k0 / magnitude[middle] * ((k1 - magnitude[middle]) / (k1 - k0)) ** 2
    return result


__all__ = [
    "DIRECT_REJECTION_MODEL",
    "IGG3_MODEL",
    "ROBUST_WEIGHT_MODELS",
    "robust_factors",
]
