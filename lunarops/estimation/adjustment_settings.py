"""Validated settings grouped by the scientific adjustment subsystem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Optional

import numpy as np

from lunarops.base.station_identity import canonical_station_id
from lunarops.estimation.robust_weights import (
    DIRECT_REJECTION_MODEL,
    IGG3_MODEL,
    ROBUST_WEIGHT_MODELS,
)
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
    """Controls for prefit quality control and nonlinear parameter updates."""

    prefit_gross_threshold_m: Optional[float] = 20.0
    prefit_gross_threshold_by_station_m: Optional[Mapping[str, Optional[float]]] = None
    maximum_linearizations: int = 20
    parameter_update_factor: float = 1.0
    uncertainty_floor_minimum_m: float = 0.0
    uncertainty_floor_group_median_fraction: float = 0.0
    update_tolerance_m: float = 1.0e-3
    update_tolerance_by_block_m: Optional[Mapping[str, float]] = None
    required_consecutive_converged_linearizations: int = 2

    def __post_init__(self) -> None:
        _positive_integer(self.maximum_linearizations, "maximum_linearizations")
        _positive_integer(
            self.required_consecutive_converged_linearizations,
            "required_consecutive_converged_linearizations",
        )
        prefit = _optional_nonnegative_real(self.prefit_gross_threshold_m, "prefit_gross_threshold_m")
        parameter_update_factor = _finite_real(self.parameter_update_factor, "parameter_update_factor")
        uncertainty_floor_minimum = _finite_real(
            self.uncertainty_floor_minimum_m,
            "uncertainty_floor_minimum_m",
        )
        uncertainty_floor_fraction = _finite_real(
            self.uncertainty_floor_group_median_fraction,
            "uncertainty_floor_group_median_fraction",
        )
        update_tolerance = _finite_real(self.update_tolerance_m, "update_tolerance_m")
        if not 0.0 < parameter_update_factor <= 1.0:
            raise ValueError("Parameter update factor must be in (0, 1].")
        if uncertainty_floor_minimum < 0.0:
            raise ValueError("Uncertainty floor minimum must be non-negative.")
        if not 0.0 <= uncertainty_floor_fraction <= 1.0:
            raise ValueError("Uncertainty floor group-median fraction must be in [0, 1].")
        if update_tolerance < 0.0:
            raise ValueError("Parameter update tolerance must be non-negative.")

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
                raw_threshold,
                f"Prefit station threshold for {station!r}",
            )

        block_tolerances = self.update_tolerance_by_block_m
        if block_tolerances is not None and not isinstance(block_tolerances, Mapping):
            raise TypeError("update_tolerance_by_block_m must be a mapping or null.")
        normalized_tolerances: dict[str, float] = {}
        for raw_block, raw_tolerance in (block_tolerances or {}).items():
            if not isinstance(raw_block, str) or not raw_block.strip():
                raise ValueError("Block tolerance keys must be non-empty strings.")
            block = raw_block.strip()
            if block in normalized_tolerances:
                raise ValueError("Block tolerance keys must be unique after trimming.")
            tolerance = _finite_real(raw_tolerance, f"Block parameter tolerance for {raw_block!r}")
            if tolerance < 0.0:
                raise ValueError("Block parameter tolerances must be finite and non-negative.")
            normalized_tolerances[block] = tolerance

        object.__setattr__(self, "prefit_gross_threshold_m", prefit)
        object.__setattr__(self, "prefit_gross_threshold_by_station_m", canonical_thresholds or None)
        object.__setattr__(self, "parameter_update_factor", parameter_update_factor)
        object.__setattr__(self, "uncertainty_floor_minimum_m", uncertainty_floor_minimum)
        object.__setattr__(self, "uncertainty_floor_group_median_fraction", uncertainty_floor_fraction)
        object.__setattr__(self, "update_tolerance_m", update_tolerance)
        object.__setattr__(self, "update_tolerance_by_block_m", normalized_tolerances or None)


@dataclass(frozen=True, slots=True)
class InitializationSettings:
    """Settings for deterministic bias and initial stochastic-scale estimates."""

    minimum_mad_count: int = 10
    minimum_initial_scale: float = 1.0
    bias_weight_cap: float = 1.0e12
    bias_maximum_iterations: int = 30

    def __post_init__(self) -> None:
        _positive_integer(self.minimum_mad_count, "minimum_mad_count")
        _positive_integer(self.bias_maximum_iterations, "bias_maximum_iterations")
        minimum_scale = _finite_real(self.minimum_initial_scale, "minimum_initial_scale")
        bias_weight_cap = _finite_real(self.bias_weight_cap, "bias_weight_cap")
        if minimum_scale <= 0.0:
            raise ValueError("Minimum initial scale must be positive.")
        if bias_weight_cap <= 0.0:
            raise ValueError("Bias weight cap must be positive.")
        object.__setattr__(self, "minimum_initial_scale", minimum_scale)
        object.__setattr__(self, "bias_weight_cap", bias_weight_cap)


@dataclass(frozen=True, slots=True)
class RobustEstimationSettings:
    """Robust residual-weight model and leverage settings."""

    model: str = IGG3_MODEL
    k0: float = 1.5
    k1: Optional[float] = None
    minimum_one_minus_leverage: float = 1.0e-8
    active_factor_threshold: float = 1.0e-12
    convergence_factor_floor: float = 1.0e-3
    change_quantile: float = 0.999

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise TypeError("robust model must be a string.")
        model = self.model.strip()
        if model not in ROBUST_WEIGHT_MODELS:
            raise ValueError(f"robust model must be one of {sorted(ROBUST_WEIGHT_MODELS)}, got {self.model!r}.")
        if model == IGG3_MODEL and self.k1 is None:
            object.__setattr__(self, "k1", 6.0)
        k0 = _finite_real(self.k0, "k0")
        k1 = None if self.k1 is None else _finite_real(self.k1, "k1")
        minimum_one_minus_leverage = _finite_real(
            self.minimum_one_minus_leverage,
            "minimum_one_minus_leverage",
        )
        active_factor_threshold = _finite_real(self.active_factor_threshold, "active_factor_threshold")
        convergence_factor_floor = _finite_real(
            self.convergence_factor_floor,
            "convergence_factor_floor",
        )
        change_quantile = _finite_real(self.change_quantile, "change_quantile")
        if model == IGG3_MODEL:
            if k1 is None or not 0.0 < k0 < k1:
                raise ValueError("IGGIII thresholds must satisfy 0 < k0 < k1.")
        elif k1 is not None:
            raise ValueError("directRejection uses k0 only; omit k1.")
        elif k0 <= 0.0:
            raise ValueError("Direct-rejection threshold k0 must be positive.")
        if not 0.0 < minimum_one_minus_leverage <= 1.0:
            raise ValueError("Minimum one-minus-leverage must be in (0, 1].")
        if not 0.0 < active_factor_threshold < 1.0:
            raise ValueError("Active robust-factor threshold must be in (0, 1).")
        if not 0.0 <= convergence_factor_floor <= 1.0:
            raise ValueError("Robust-factor convergence floor must be in [0, 1].")
        if not 0.0 < change_quantile <= 1.0:
            raise ValueError("Robust factor change quantile must be in (0, 1].")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "minimum_one_minus_leverage", minimum_one_minus_leverage)
        object.__setattr__(self, "active_factor_threshold", active_factor_threshold)
        object.__setattr__(self, "convergence_factor_floor", convergence_factor_floor)
        object.__setattr__(self, "change_quantile", change_quantile)


@dataclass(frozen=True, slots=True)
class VarianceComponentEstimationSettings:
    """Helmert VCE component definitions and stochastic convergence controls."""

    components: tuple[VarianceComponentDefinition, ...]
    maximum_stochastic_iterations: int = 8
    minimum_effective_redundancy: float = 20.0
    scale_log_tolerance: float = 2.5e-2
    robust_factor_change_tolerance: float = 2.0e-2
    active_set_change_tolerance: float = 1.0e-3
    minimum_variance_ratio_per_iteration: float = 0.25
    maximum_variance_ratio_per_iteration: float = 4.0

    def __post_init__(self) -> None:
        if isinstance(self.components, (str, bytes)) or not isinstance(self.components, Sequence):
            raise TypeError("Variance components must be a sequence of VarianceComponentDefinition instances.")
        components = tuple(self.components)
        if not components:
            raise ValueError("At least one variance component is required.")
        if not all(isinstance(component, VarianceComponentDefinition) for component in components):
            raise TypeError("All variance components must be VarianceComponentDefinition instances.")
        component_ids = [component.id for component in components]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("Variance-component IDs must be unique.")
        _positive_integer(self.maximum_stochastic_iterations, "maximum_stochastic_iterations")
        minimum_redundancy = _finite_real(self.minimum_effective_redundancy, "minimum_effective_redundancy")
        scale_log_tolerance = _finite_real(self.scale_log_tolerance, "scale_log_tolerance")
        factor_change_tolerance = _finite_real(
            self.robust_factor_change_tolerance,
            "robust_factor_change_tolerance",
        )
        active_set_tolerance = _finite_real(self.active_set_change_tolerance, "active_set_change_tolerance")
        minimum_ratio = _finite_real(
            self.minimum_variance_ratio_per_iteration,
            "minimum_variance_ratio_per_iteration",
        )
        maximum_ratio = _finite_real(
            self.maximum_variance_ratio_per_iteration,
            "maximum_variance_ratio_per_iteration",
        )
        if minimum_redundancy < 0.0:
            raise ValueError("Minimum effective redundancy must be non-negative.")
        if scale_log_tolerance < 0.0 or factor_change_tolerance < 0.0:
            raise ValueError("Stochastic convergence tolerances must be non-negative.")
        if not 0.0 <= active_set_tolerance <= 1.0:
            raise ValueError("Active-set change tolerance must be in [0, 1].")
        if not 0.0 < minimum_ratio <= maximum_ratio:
            raise ValueError("VCE variance-ratio limits must be positive and ordered.")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "minimum_effective_redundancy", minimum_redundancy)
        object.__setattr__(self, "scale_log_tolerance", scale_log_tolerance)
        object.__setattr__(self, "robust_factor_change_tolerance", factor_change_tolerance)
        object.__setattr__(self, "active_set_change_tolerance", active_set_tolerance)
        object.__setattr__(self, "minimum_variance_ratio_per_iteration", minimum_ratio)
        object.__setattr__(self, "maximum_variance_ratio_per_iteration", maximum_ratio)


@dataclass(frozen=True, slots=True)
class LlrAdjustmentSettings:
    """Complete resolved settings for one nonlinear LLR adjustment stage."""

    vce: VarianceComponentEstimationSettings
    adjustment: AdjustmentControlSettings = field(default_factory=AdjustmentControlSettings)
    initialization: InitializationSettings = field(default_factory=InitializationSettings)
    robust_estimation: RobustEstimationSettings = field(default_factory=RobustEstimationSettings)

    def __post_init__(self) -> None:
        if not isinstance(self.adjustment, AdjustmentControlSettings):
            raise TypeError("adjustment must be an AdjustmentControlSettings instance.")
        if not isinstance(self.initialization, InitializationSettings):
            raise TypeError("initialization must be an InitializationSettings instance.")
        if not isinstance(self.robust_estimation, RobustEstimationSettings):
            raise TypeError("robust_estimation must be a RobustEstimationSettings instance.")
        if not isinstance(self.vce, VarianceComponentEstimationSettings):
            raise TypeError("vce must be a VarianceComponentEstimationSettings instance.")

    def to_report_settings(self) -> dict[str, object]:
        """Serialize numerical settings without repeating component definitions."""
        values = asdict(self)
        values["vce"].pop("components")
        return values


__all__ = [
    "AdjustmentControlSettings",
    "InitializationSettings",
    "LlrAdjustmentSettings",
    "RobustEstimationSettings",
    "VarianceComponentEstimationSettings",
]
