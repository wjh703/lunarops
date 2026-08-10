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


@dataclass(frozen=True)
class RobustWeightUpdate:
    target_factors: dict[ObsKey, float]
    applied_factors: dict[ObsKey, float]
    target_change_quantile: float
    active_set_change_fraction: float
    maximum_applied_change: float


class RobustWeightModel:
    """Shared IRLS update mechanics for robust observation-weight models."""

    active_threshold: float
    convergence_floor: float
    change_quantile: float

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
        if factors.shape != values.shape or not np.all(np.isfinite(factors)) or np.any((factors < 0.0) | (factors > 1.0)):
            raise ValueError("Robust-weight model factors must be finite and in [0, 1].")
        return {key: float(value) for key, value in zip(keys, factors)}

    def update(
        self,
        standardized_residuals: Mapping[ObsKey, float],
        current_factors: Mapping[ObsKey, float],
        previous_target_factors: Mapping[ObsKey, float],
        keys: Sequence[ObsKey],
    ) -> RobustWeightUpdate:
        if len(set(keys)) != len(keys):
            raise ValueError("Robust-weight observation keys must be unique.")
        _validate_factor_mapping(current_factors, keys, "current factors")
        _validate_factor_mapping(previous_target_factors, (), "previous target factors")
        targets = self.target_factors(standardized_residuals, keys)
        applied = dict(current_factors)
        applied.update(targets)
        previous_targets = {key: previous_target_factors.get(key, current_factors.get(key, 1.0)) for key in keys}
        return RobustWeightUpdate(
            target_factors=targets,
            applied_factors=applied,
            target_change_quantile=robust_factor_change_quantile(
                current_factors,
                targets,
                keys,
                quantile=self.change_quantile,
                significance_floor=self.convergence_floor,
            ),
            active_set_change_fraction=active_set_change_fraction(
                previous_targets,
                targets,
                keys,
                active_threshold=self.active_threshold,
            ),
            maximum_applied_change=maximum_robust_factor_change(
                current_factors,
                applied,
                keys,
                significance_floor=self.convergence_floor,
            ),
        )


def _validate_common_options(
    active_threshold: float,
    convergence_floor: float,
    change_quantile: float,
) -> tuple[float, float, float]:
    values = (active_threshold, convergence_floor, change_quantile)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values):
        raise TypeError("Robust-weight thresholds must be real numbers.")
    active_threshold, convergence_floor, change_quantile = (float(value) for value in values)
    if not np.all(np.isfinite((active_threshold, convergence_floor, change_quantile))):
        raise ValueError("Robust-weight thresholds must be finite.")
    if not 0.0 < active_threshold < 1.0:
        raise ValueError("Active robust-factor threshold must be in (0, 1).")
    if not 0.0 <= convergence_floor <= 1.0:
        raise ValueError("Robust-factor convergence floor must be in [0, 1].")
    if not 0.0 < change_quantile <= 1.0:
        raise ValueError("Robust factor change quantile must be in (0, 1].")
    return active_threshold, convergence_floor, change_quantile


def _validate_factor_mapping(mapping: Mapping[ObsKey, float], keys: Sequence[ObsKey], label: str) -> None:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    for key in keys:
        if key not in mapping:
            raise KeyError(f"{label} are missing observation {key!r}.")
    raw_values = list(mapping.values())
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in raw_values):
        raise TypeError(f"{label} must contain real numbers.")
    values = np.asarray(raw_values, dtype=float)
    if values.size and (not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0))):
        raise ValueError(f"{label} must be finite and in [0, 1].")


@dataclass(frozen=True)
class Igg3WeightModel(RobustWeightModel):
    """IGGIII IRLS model with immediate zero-target rejection."""

    k0: float = 1.5
    k1: float = 6.0
    active_threshold: float = 1.0e-12
    convergence_floor: float = 1.0e-3
    change_quantile: float = 0.999

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
        active_threshold, convergence_floor, change_quantile = _validate_common_options(
            self.active_threshold,
            self.convergence_floor,
            self.change_quantile,
        )
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "active_threshold", active_threshold)
        object.__setattr__(self, "convergence_floor", convergence_floor)
        object.__setattr__(self, "change_quantile", change_quantile)

    def factor_values(self, values: np.ndarray) -> np.ndarray:
        return igg3_factors(values, k0=self.k0, k1=self.k1)


@dataclass(frozen=True)
class DirectRejectionWeightModel(RobustWeightModel):
    """Keep full weight through k0 and reject larger residuals immediately."""

    k0: float = 3.0
    active_threshold: float = 1.0e-12
    convergence_floor: float = 1.0e-3
    change_quantile: float = 0.999

    def __post_init__(self) -> None:
        if isinstance(self.k0, (bool, np.bool_)) or not isinstance(self.k0, Real):
            raise TypeError("Direct-rejection threshold must be a real number.")
        k0 = float(self.k0)
        if not np.isfinite(k0) or k0 <= 0.0:
            raise ValueError("Direct-rejection threshold k0 must be finite and positive.")
        active_threshold, convergence_floor, change_quantile = _validate_common_options(
            self.active_threshold,
            self.convergence_floor,
            self.change_quantile,
        )
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "active_threshold", active_threshold)
        object.__setattr__(self, "convergence_floor", convergence_floor)
        object.__setattr__(self, "change_quantile", change_quantile)

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
    active_threshold: float,
    convergence_floor: float,
    change_quantile: float,
) -> RobustWeightModel:
    common = {
        "active_threshold": active_threshold,
        "convergence_floor": convergence_floor,
        "change_quantile": change_quantile,
    }
    if model == IGG3_MODEL:
        if k1 is None:
            raise ValueError("IGGIII requires k1.")
        return Igg3WeightModel(k0=k0, k1=k1, **common)
    if model == DIRECT_REJECTION_MODEL:
        if k1 is not None:
            raise ValueError("directRejection uses k0 only; omit k1.")
        return DirectRejectionWeightModel(k0=k0, **common)
    raise ValueError(f"Robust model must be one of {sorted(ROBUST_WEIGHT_MODELS)}, got {model!r}.")


def maximum_robust_factor_change(
    old_factors,
    new_factors,
    keys,
    *,
    significance_floor=0.0,
):
    return max(
        (
            abs(new_factors[key] - old_factors[key])
            for key in keys
            if max(abs(old_factors[key]), abs(new_factors[key])) >= significance_floor
        ),
        default=0.0,
    )


def robust_factor_change_quantile(
    old_factors,
    target_factors,
    keys,
    *,
    quantile,
    significance_floor=0.0,
):
    changes = np.asarray(
        [
            abs(target_factors[key] - old_factors[key])
            for key in keys
            if max(abs(old_factors[key]), abs(target_factors[key])) >= significance_floor
        ],
        dtype=float,
    )
    return 0.0 if not len(changes) else float(np.quantile(changes, quantile, method="higher"))


def active_set_change_fraction(old_factors, new_factors, keys, *, active_threshold):
    if not keys:
        return 0.0
    changed = sum((old_factors[key] > active_threshold) != (new_factors[key] > active_threshold) for key in keys)
    return float(changed / len(keys))


__all__ = [
    "DIRECT_REJECTION_MODEL",
    "IGG3_MODEL",
    "ROBUST_WEIGHT_MODELS",
    "DirectRejectionWeightModel",
    "Igg3WeightModel",
    "RobustWeightModel",
    "RobustWeightUpdate",
    "active_set_change_fraction",
    "create_robust_weight_model",
    "direct_rejection_factors",
    "igg3_factors",
    "maximum_robust_factor_change",
    "robust_factor_change_quantile",
]
