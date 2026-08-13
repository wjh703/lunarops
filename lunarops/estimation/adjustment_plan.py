"""Typed GROOPS-style processing steps for nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real

from lunarops.estimation.adjustment_settings import LlrAdjustmentSettings, RobustWeightSettings


def _selectors(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings.")
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise TypeError(f"{name} must contain non-empty strings.")
    result = tuple(value.strip() for value in result)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique.")
    return result


@dataclass(frozen=True)
class SelectParametrizationsStep:
    """Select the complete parametrization set for subsequent estimates."""

    parametrizations: tuple[str, ...]

    def __post_init__(self) -> None:
        values = _selectors(self.parametrizations, "Selected parametrizations")
        if not values:
            raise ValueError("selectParametrizations must select at least one parametrization.")
        object.__setattr__(self, "parametrizations", values)

    def apply(self, available: Sequence[str]) -> tuple[str, ...]:
        available_ids = _selectors(available, "Available parametrizations")
        missing = set(self.parametrizations) - set(available_ids)
        if missing:
            raise KeyError(f"Unknown parametrization selector(s): {sorted(missing)}")
        selected = set(self.parametrizations)
        return tuple(block_id for block_id in available_ids if block_id in selected)


@dataclass(frozen=True)
class EstimateStep:
    """One nonlinear least-squares estimate using the current selection."""

    name: str
    max_iteration_count: int = 3
    convergence_threshold_m: float = 1.0e-2
    convergence_threshold_by_parametrization_m: Mapping[str, float] | None = None
    compute_residuals: bool = True
    adjust_sigma0: bool = True
    compute_weights: bool = True
    robust_weights: RobustWeightSettings | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Estimate step name must be a non-empty string.")
        if isinstance(self.max_iteration_count, bool) or not isinstance(self.max_iteration_count, int):
            raise TypeError("Estimate max_iteration_count must be an integer.")
        if self.max_iteration_count < 1:
            raise ValueError("Estimate max_iteration_count must be positive.")
        threshold = self.convergence_threshold_m
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise TypeError("Estimate convergence_threshold_m must be a real number.")
        threshold = float(threshold)
        if not isfinite(threshold) or threshold < 0.0:
            raise ValueError("Estimate convergence_threshold_m must be finite and non-negative.")
        raw_thresholds = self.convergence_threshold_by_parametrization_m
        if raw_thresholds is not None and not isinstance(raw_thresholds, Mapping):
            raise TypeError("Estimate parametrization convergence thresholds must be a mapping or null.")
        thresholds: dict[str, float] = {}
        for raw_name, raw_value in (raw_thresholds or {}).items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("Parametrization convergence-threshold keys must be non-empty strings.")
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError("Parametrization convergence thresholds must be real numbers.")
            value = float(raw_value)
            if not isfinite(value) or value < 0.0:
                raise ValueError("Parametrization convergence thresholds must be finite and non-negative.")
            thresholds[raw_name.strip()] = value
        for field_name in ("compute_residuals", "adjust_sigma0", "compute_weights"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"Estimate {field_name} must be a boolean.")
        if not self.compute_residuals and (self.adjust_sigma0 or self.compute_weights):
            raise ValueError("adjustSigma0 and computeWeights require computeResiduals=true.")
        if self.robust_weights is not None and not isinstance(self.robust_weights, RobustWeightSettings):
            raise TypeError("Estimate robust_weights must be RobustWeightSettings or null.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "convergence_threshold_m", threshold)
        object.__setattr__(self, "convergence_threshold_by_parametrization_m", thresholds or None)

    def apply(self, settings: LlrAdjustmentSettings) -> LlrAdjustmentSettings:
        if not isinstance(settings, LlrAdjustmentSettings):
            raise TypeError("Estimate application requires LlrAdjustmentSettings.")
        return replace(
            settings,
            adjustment=replace(
                settings.adjustment,
                max_iteration_count=self.max_iteration_count,
                convergence_threshold_m=self.convergence_threshold_m,
                convergence_threshold_by_parametrization_m=self.convergence_threshold_by_parametrization_m,
                compute_residuals=self.compute_residuals,
                adjust_sigma0=self.adjust_sigma0,
                compute_weights=self.compute_weights,
            ),
            robust_weights=self.robust_weights or settings.robust_weights,
        )


AdjustmentProcessingStep = SelectParametrizationsStep | EstimateStep


@dataclass(frozen=True)
class LlrAdjustmentPlan:
    settings: LlrAdjustmentSettings
    processing_steps: tuple[AdjustmentProcessingStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.settings, LlrAdjustmentSettings):
            raise TypeError("Adjustment plan settings must be LlrAdjustmentSettings.")
        steps = tuple(self.processing_steps)
        if not steps or not all(isinstance(step, (SelectParametrizationsStep, EstimateStep)) for step in steps):
            raise ValueError("Adjustment plan must contain valid processing steps.")
        estimates = [step for step in steps if isinstance(step, EstimateStep)]
        if not estimates:
            raise ValueError("Adjustment plan must contain at least one estimate step.")
        if len({step.name for step in estimates}) != len(estimates):
            raise ValueError("Estimate step names must be unique.")
        object.__setattr__(self, "processing_steps", steps)


__all__ = [
    "AdjustmentProcessingStep",
    "EstimateStep",
    "LlrAdjustmentPlan",
    "SelectParametrizationsStep",
]
