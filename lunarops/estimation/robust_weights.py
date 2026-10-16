"""Robust observation-weight models for iterative LLR adjustment."""

from __future__ import annotations

from typing import Optional

from typing import cast

import numpy as np

IGG3_MODEL = "igg3"
DIRECT_REJECTION_MODEL = "directRejection"
ROBUST_WEIGHT_MODELS = frozenset((IGG3_MODEL, DIRECT_REJECTION_MODEL))


def robust_factors(model: str, k0: float, k1: Optional[float], standardized: np.ndarray) -> np.ndarray:
    """Map standardized residuals to weight factors in [0, 1].

    ``directRejection`` keeps full weight through ``k0`` and rejects larger
    residuals; ``igg3`` is the IGGIII IRLS model with zero weight beyond ``k1``.
    """
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
