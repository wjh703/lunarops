"""Robust observation-weight models for iterative LLR adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Hashable, Mapping, Optional, Sequence

import numpy as np

ObsKey = Hashable

IGG3_MODEL = "igg3"
DIRECT_REJECTION_MODEL = "directRejection"
ROBUST_WEIGHT_MODELS = frozenset((IGG3_MODEL, DIRECT_REJECTION_MODEL))


class RobustWeightModel:
    """Map standardized residuals to observation weight factors."""

    def factor_values(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def target_factors(
        self,
        standardized_residuals: Mapping[ObsKey, float],
        keys: Sequence[ObsKey],
    ) -> dict[ObsKey, float]:
        if not isinstance(standardized_residuals, Mapping):
            raise TypeError("Standardized residuals must be a mapping.")
        if len(set(keys)) != len(keys):
            raise ValueError("Robust-weight observation keys must be unique.")
        values = np.asarray([standardized_residuals[key] for key in keys], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Standardized residuals must be finite.")
        factors = np.asarray(self.factor_values(values), dtype=float)
        if (
            factors.shape != values.shape
            or not np.all(np.isfinite(factors))
            or np.any((factors < 0.0) | (factors > 1.0))
        ):
            raise ValueError("Robust-weight model factors must be finite and in [0, 1].")
        return {key: float(value) for key, value in zip(keys, factors)}


@dataclass(frozen=True)
class Igg3WeightModel(RobustWeightModel):
    """IGGIII IRLS model with immediate zero-target rejection."""

    k0: float = 1.5
    k1: float = 6.0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in (self.k0, self.k1)
        ):
            raise TypeError("IGGIII thresholds must be real numbers.")
        k0 = float(self.k0)
        k1 = float(self.k1)
        if not np.isfinite(k0) or not np.isfinite(k1):
            raise ValueError("IGGIII thresholds must be finite.")
        if not 0.0 < k0 < k1:
            raise ValueError("IGGIII thresholds must satisfy 0 < k0 < k1.")
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "k1", k1)

    def factor_values(self, values: np.ndarray) -> np.ndarray:
        return igg3_factors(values, k0=self.k0, k1=self.k1)


@dataclass(frozen=True)
class DirectRejectionWeightModel(RobustWeightModel):
    """Keep full weight through k0 and reject larger residuals immediately."""

    k0: float = 3.0

    def __post_init__(self) -> None:
        if isinstance(self.k0, (bool, np.bool_)) or not isinstance(self.k0, Real):
            raise TypeError("Direct-rejection threshold must be a real number.")
        k0 = float(self.k0)
        if not np.isfinite(k0) or k0 <= 0.0:
            raise ValueError("Direct-rejection threshold k0 must be finite and positive.")
        object.__setattr__(self, "k0", k0)

    def factor_values(self, values: np.ndarray) -> np.ndarray:
        return direct_rejection_factors(values, k0=self.k0)


def igg3_factors(values: np.ndarray, *, k0: float, k1: float) -> np.ndarray:
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in (k0, k1)
    ):
        raise TypeError("IGGIII thresholds must be real numbers.")
    k0 = float(k0)
    k1 = float(k1)
    if not np.isfinite(k0) or not np.isfinite(k1) or not 0.0 < k0 < k1:
        raise ValueError("IGGIII thresholds must satisfy 0 < k0 < k1.")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Standardized residuals must be finite.")
    magnitude = np.abs(values)
    result = np.zeros_like(magnitude)
    full = magnitude <= k0
    middle = (magnitude > k0) & (magnitude <= k1)
    result[full] = 1.0
    result[middle] = k0 / magnitude[middle] * ((k1 - magnitude[middle]) / (k1 - k0)) ** 2
    return result


def direct_rejection_factors(values: np.ndarray, *, k0: float) -> np.ndarray:
    if isinstance(k0, (bool, np.bool_)) or not isinstance(k0, Real):
        raise TypeError("Direct-rejection threshold must be a real number.")
    k0 = float(k0)
    if not np.isfinite(k0) or k0 <= 0.0:
        raise ValueError("Direct-rejection threshold k0 must be finite and positive.")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Standardized residuals must be finite.")
    return np.where(np.abs(values) <= k0, 1.0, 0.0)


def create_robust_weight_model(
    *,
    model: str,
    k0: float,
    k1: Optional[float],
) -> RobustWeightModel:
    if model == IGG3_MODEL:
        if k1 is None:
            raise ValueError("IGGIII requires k1.")
        return Igg3WeightModel(k0=k0, k1=k1)
    if model == DIRECT_REJECTION_MODEL:
        if k1 is not None:
            raise ValueError("directRejection uses k0 only; omit k1.")
        return DirectRejectionWeightModel(k0=k0)
    raise ValueError(f"Robust model must be one of {sorted(ROBUST_WEIGHT_MODELS)}, got {model!r}.")


__all__ = [
    "DIRECT_REJECTION_MODEL",
    "IGG3_MODEL",
    "ROBUST_WEIGHT_MODELS",
    "DirectRejectionWeightModel",
    "Igg3WeightModel",
    "RobustWeightModel",
    "create_robust_weight_model",
    "direct_rejection_factors",
    "igg3_factors",
]
