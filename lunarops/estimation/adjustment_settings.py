"""Validated settings for the nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Optional

import numpy as np

from lunarops.estimation.robust_weights import IGG3_MODEL, ROBUST_WEIGHT_MODELS
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_real(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _optional_nonnegative_real(value: object, name: str) -> float | None:
    return None if value is None else _nonnegative_real(value, name)


@dataclass(frozen=True, slots=True)
class AdjustmentControlSettings:
    """Prefit rejection and nonlinear outer-iteration controls."""

    prefit_gross_threshold_m: Optional[float] = 20.0
    prefit_gross_threshold_by_station_m: Optional[Mapping[str, Optional[float]]] = None
    max_iteration_count: int = 3
    convergence_threshold_m: float = 1.0e-2
    convergence_threshold_by_parametrization_m: Optional[Mapping[str, float]] = None
    compute_residuals: bool = True
    adjust_sigma0: bool = True
    compute_weights: bool = True

    def __post_init__(self) -> None:
        _positive_integer(self.max_iteration_count, "max_iteration_count")
        prefit = _optional_nonnegative_real(self.prefit_gross_threshold_m, "prefit_gross_threshold_m")
        threshold = _nonnegative_real(self.convergence_threshold_m, "convergence_threshold_m")
        station_thresholds = {
            station: _optional_nonnegative_real(value, f"Prefit station threshold for {station!r}")
            for station, value in (self.prefit_gross_threshold_by_station_m or {}).items()
        }
        block_thresholds = {
            block: _nonnegative_real(value, f"Parametrization convergence threshold for {block!r}")
            for block, value in (self.convergence_threshold_by_parametrization_m or {}).items()
        }
        if not self.compute_residuals and (self.adjust_sigma0 or self.compute_weights):
            raise ValueError("adjust_sigma0 and compute_weights require compute_residuals=true.")

        object.__setattr__(self, "prefit_gross_threshold_m", prefit)
        object.__setattr__(self, "prefit_gross_threshold_by_station_m", station_thresholds or None)
        object.__setattr__(self, "convergence_threshold_m", threshold)
        object.__setattr__(self, "convergence_threshold_by_parametrization_m", block_thresholds or None)


@dataclass(frozen=True, slots=True)
class AccuracyScreeningSettings:
    """Reject implausibly small reported a-priori observation accuracies."""

    minimum_one_way_m: float = 1.0e-3
    minimum_fraction_of_group_median: float = 0.1

    def __post_init__(self) -> None:
        minimum = _finite_real(self.minimum_one_way_m, "minimum_one_way_m")
        fraction = _finite_real(
            self.minimum_fraction_of_group_median,
            "minimum_fraction_of_group_median",
        )
        if minimum < 0.0:
            raise ValueError("Minimum one-way accuracy must be non-negative.")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Minimum fraction of group median must be in [0, 1].")
        object.__setattr__(self, "minimum_one_way_m", minimum)
        object.__setattr__(self, "minimum_fraction_of_group_median", fraction)


@dataclass(frozen=True, slots=True)
class RobustWeightSettings:
    """Direct-rejection or IGG3 observation weights."""

    model: str = IGG3_MODEL
    k0: float = 1.5
    k1: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise TypeError("Robust-weight model must be a string.")
        model = self.model.strip()
        if model not in ROBUST_WEIGHT_MODELS:
            raise ValueError(f"Robust-weight model must be one of {sorted(ROBUST_WEIGHT_MODELS)}.")
        if model == IGG3_MODEL and self.k1 is None:
            object.__setattr__(self, "k1", 6.0)
        k0 = _finite_real(self.k0, "k0")
        k1 = None if self.k1 is None else _finite_real(self.k1, "k1")
        if model == IGG3_MODEL and (k1 is None or not 0.0 < k0 < k1):
            raise ValueError("IGGIII thresholds must satisfy 0 < k0 < k1.")
        if model != IGG3_MODEL and k1 is not None:
            raise ValueError("directRejection uses k0 only; omit k1.")
        if model != IGG3_MODEL and k0 <= 0.0:
            raise ValueError("Direct-rejection threshold k0 must be positive.")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "k1", k1)


@dataclass(frozen=True, slots=True)
class VarianceComponentSettings:
    """Station, equipment-era, and wavelength variance-component groups."""

    components: tuple[VarianceComponentDefinition, ...]

    def __post_init__(self) -> None:
        if isinstance(self.components, (str, bytes)) or not isinstance(self.components, Sequence):
            raise TypeError("Variance components must be a sequence.")
        components = tuple(self.components)
        if not components or not all(isinstance(item, VarianceComponentDefinition) for item in components):
            raise ValueError("At least one valid variance component is required.")
        if len({item.id for item in components}) != len(components):
            raise ValueError("Variance-component IDs must be unique.")
        object.__setattr__(self, "components", components)


@dataclass(frozen=True, slots=True)
class LlrAdjustmentSettings:
    """Complete resolved settings for one nonlinear adjustment stage."""

    variance_components: VarianceComponentSettings
    adjustment: AdjustmentControlSettings = field(default_factory=AdjustmentControlSettings)
    accuracy_screening: AccuracyScreeningSettings = field(default_factory=AccuracyScreeningSettings)
    robust_weights: RobustWeightSettings = field(default_factory=RobustWeightSettings)

    def to_report_settings(self) -> dict[str, object]:
        adjustment = self.adjustment
        return {
            "observationScreening": {
                "residual": {
                    "maximumAbsoluteM": adjustment.prefit_gross_threshold_m,
                    "maximumAbsoluteByStationM": adjustment.prefit_gross_threshold_by_station_m,
                },
                "reportedSigma": {
                    "minimumOneWayM": self.accuracy_screening.minimum_one_way_m,
                    "minimumFractionOfGroupMedian": self.accuracy_screening.minimum_fraction_of_group_median,
                },
            },
            "estimate": {
                "maxIterationCount": adjustment.max_iteration_count,
                "convergenceThresholdM": adjustment.convergence_threshold_m,
                "convergenceThresholdByParametrizationM": adjustment.convergence_threshold_by_parametrization_m,
                "computeResiduals": adjustment.compute_residuals,
                "estimateVarianceFactors": adjustment.adjust_sigma0,
                "estimateRobustWeights": adjustment.compute_weights,
            },
            "robustWeighting": {
                "model": self.robust_weights.model,
                "k0": self.robust_weights.k0,
                "k1": self.robust_weights.k1,
            },
        }


__all__ = [
    "AccuracyScreeningSettings",
    "AdjustmentControlSettings",
    "LlrAdjustmentSettings",
    "RobustWeightSettings",
    "VarianceComponentSettings",
]
