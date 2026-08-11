"""Validated settings for the nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Optional

import numpy as np

from lunarops.base.station_identity import canonical_station_id
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


def _optional_nonnegative_real(value: object, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


@dataclass(frozen=True, slots=True)
class AdjustmentControlSettings:
    """Prefit rejection and nonlinear outer-iteration controls."""

    prefit_gross_threshold_m: Optional[float] = 20.0
    prefit_gross_threshold_by_station_m: Optional[Mapping[str, Optional[float]]] = None
    max_iteration_count: int = 20
    parameter_update_factor: float = 1.0
    convergence_threshold_m: float = 1.0e-3
    convergence_threshold_by_block_m: Optional[Mapping[str, float]] = None
    required_consecutive_converged_iterations: int = 1

    def __post_init__(self) -> None:
        _positive_integer(self.max_iteration_count, "max_iteration_count")
        _positive_integer(
            self.required_consecutive_converged_iterations,
            "required_consecutive_converged_iterations",
        )
        prefit = _optional_nonnegative_real(self.prefit_gross_threshold_m, "prefit_gross_threshold_m")
        update_factor = _finite_real(self.parameter_update_factor, "parameter_update_factor")
        threshold = _finite_real(self.convergence_threshold_m, "convergence_threshold_m")
        if not 0.0 < update_factor <= 1.0:
            raise ValueError("Parameter update factor must be in (0, 1].")
        if threshold < 0.0:
            raise ValueError("Parameter convergence threshold must be non-negative.")

        station_thresholds = self.prefit_gross_threshold_by_station_m
        if station_thresholds is not None and not isinstance(station_thresholds, Mapping):
            raise TypeError("prefit_gross_threshold_by_station_m must be a mapping or null.")
        canonical_thresholds: dict[str, float | None] = {}
        for raw_station, raw_threshold in (station_thresholds or {}).items():
            if not isinstance(raw_station, str) or not raw_station.strip():
                raise ValueError("Prefit station threshold keys must be non-empty strings.")
            station = canonical_station_id(raw_station)
            if station in canonical_thresholds:
                raise ValueError("Prefit station thresholds must not duplicate canonical station identifiers.")
            canonical_thresholds[station] = _optional_nonnegative_real(
                raw_threshold, f"Prefit station threshold for {station!r}"
            )

        block_thresholds = self.convergence_threshold_by_block_m
        if block_thresholds is not None and not isinstance(block_thresholds, Mapping):
            raise TypeError("convergence_threshold_by_block_m must be a mapping or null.")
        normalized: dict[str, float] = {}
        for raw_block, raw_threshold in (block_thresholds or {}).items():
            if not isinstance(raw_block, str) or not raw_block.strip():
                raise ValueError("Block convergence-threshold keys must be non-empty strings.")
            block = raw_block.strip()
            value = _finite_real(raw_threshold, f"Block convergence threshold for {block!r}")
            if value < 0.0:
                raise ValueError("Block convergence thresholds must be non-negative.")
            normalized[block] = value

        object.__setattr__(self, "prefit_gross_threshold_m", prefit)
        object.__setattr__(self, "prefit_gross_threshold_by_station_m", canonical_thresholds or None)
        object.__setattr__(self, "parameter_update_factor", update_factor)
        object.__setattr__(self, "convergence_threshold_m", threshold)
        object.__setattr__(self, "convergence_threshold_by_block_m", normalized or None)


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
class InitializationSettings:
    """Deterministic bias initialization settings."""

    bias_weight_cap: float = 1.0e12
    bias_maximum_iterations: int = 30

    def __post_init__(self) -> None:
        _positive_integer(self.bias_maximum_iterations, "bias_maximum_iterations")
        cap = _finite_real(self.bias_weight_cap, "bias_weight_cap")
        if cap <= 0.0:
            raise ValueError("Bias weight cap must be positive.")
        object.__setattr__(self, "bias_weight_cap", cap)


@dataclass(frozen=True, slots=True)
class RobustWeightSettings:
    """Direct-rejection or IGG3 observation weights."""

    model: str = IGG3_MODEL
    k0: float = 1.5
    k1: Optional[float] = None
    active_weight_threshold: float = 1.0e-12

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
        active = _finite_real(self.active_weight_threshold, "active_weight_threshold")
        if model == IGG3_MODEL and (k1 is None or not 0.0 < k0 < k1):
            raise ValueError("IGGIII thresholds must satisfy 0 < k0 < k1.")
        if model != IGG3_MODEL and k1 is not None:
            raise ValueError("directRejection uses k0 only; omit k1.")
        if model != IGG3_MODEL and k0 <= 0.0:
            raise ValueError("Direct-rejection threshold k0 must be positive.")
        if not 0.0 < active < 1.0:
            raise ValueError("Active weight threshold must be in (0, 1).")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "active_weight_threshold", active)


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
    initialization: InitializationSettings = field(default_factory=InitializationSettings)
    robust_weights: RobustWeightSettings = field(default_factory=RobustWeightSettings)

    def __post_init__(self) -> None:
        expected = (
            (self.adjustment, AdjustmentControlSettings, "adjustment"),
            (self.accuracy_screening, AccuracyScreeningSettings, "accuracy_screening"),
            (self.initialization, InitializationSettings, "initialization"),
            (self.robust_weights, RobustWeightSettings, "robust_weights"),
            (self.variance_components, VarianceComponentSettings, "variance_components"),
        )
        for value, expected_type, name in expected:
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__} instance.")

    def to_report_settings(self) -> dict[str, object]:
        values = asdict(self)
        values["variance_components"].pop("components")
        return values


__all__ = [
    "AccuracyScreeningSettings",
    "AdjustmentControlSettings",
    "InitializationSettings",
    "LlrAdjustmentSettings",
    "RobustWeightSettings",
    "VarianceComponentSettings",
]
