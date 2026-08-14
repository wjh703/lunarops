"""Typed GROOPS-style steps for the LLR processing program."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real

from lunarops.estimation.adjustment_settings import (
    AccuracyScreeningSettings,
    AdjustmentControlSettings,
    LlrAdjustmentSettings,
    RobustWeightSettings,
)


@dataclass(frozen=True)
class ScreenObservationsStep:
    """Permanently define the observation domain before estimation."""

    maximum_absolute_residual_m: float | None = 20.0
    maximum_absolute_residual_by_station_m: Mapping[str, float | None] | None = None
    minimum_reported_one_way_sigma_m: float = 1.0e-3
    minimum_reported_sigma_fraction_of_group_median: float = 0.1

    def screening_settings(self) -> tuple[AdjustmentControlSettings, AccuracyScreeningSettings]:
        return (
            AdjustmentControlSettings(
                prefit_gross_threshold_m=self.maximum_absolute_residual_m,
                prefit_gross_threshold_by_station_m=self.maximum_absolute_residual_by_station_m,
            ),
            AccuracyScreeningSettings(
                minimum_one_way_m=self.minimum_reported_one_way_sigma_m,
                minimum_fraction_of_group_median=self.minimum_reported_sigma_fraction_of_group_median,
            ),
        )

    def __post_init__(self) -> None:
        residual, reported_sigma = self.screening_settings()
        object.__setattr__(self, "maximum_absolute_residual_m", residual.prefit_gross_threshold_m)
        object.__setattr__(
            self,
            "maximum_absolute_residual_by_station_m",
            residual.prefit_gross_threshold_by_station_m,
        )
        object.__setattr__(self, "minimum_reported_one_way_sigma_m", reported_sigma.minimum_one_way_m)
        object.__setattr__(
            self,
            "minimum_reported_sigma_fraction_of_group_median",
            reported_sigma.minimum_fraction_of_group_median,
        )


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
    estimate_variance_factors: bool = True
    estimate_robust_weights: bool = True
    robust_weighting: RobustWeightSettings | None = None

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
        for field_name in ("compute_residuals", "estimate_variance_factors", "estimate_robust_weights"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"Estimate {field_name} must be a boolean.")
        if not self.compute_residuals and (self.estimate_variance_factors or self.estimate_robust_weights):
            raise ValueError("estimateVarianceFactors and estimateRobustWeights require computeResiduals=true.")
        if self.robust_weighting is not None and not isinstance(self.robust_weighting, RobustWeightSettings):
            raise TypeError("Estimate robust_weighting must be RobustWeightSettings or null.")
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
                adjust_sigma0=self.estimate_variance_factors,
                compute_weights=self.estimate_robust_weights,
            ),
            robust_weights=self.robust_weighting or settings.robust_weights,
        )


def _output_path(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    return value.strip()


@dataclass(frozen=True)
class WriteResidualsStep:
    output_file: str
    output_level: str = "standard"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_file", _output_path(self.output_file, "writeResiduals outputFile"))
        level = str(self.output_level).strip().lower()
        if level not in {"standard", "full"}:
            raise ValueError("writeResiduals outputLevel must be 'standard' or 'full'.")
        object.__setattr__(self, "output_level", level)


@dataclass(frozen=True)
class WriteNormalEquationsStep:
    output_file: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_file",
            _output_path(self.output_file, "writeNormalEquations outputFile"),
        )


@dataclass(frozen=True)
class WriteResultsStep:
    output_file_report: str | None = None
    output_file_state: str | None = None
    output_file_solution: str | None = None
    output_file_covariance: str | None = None
    output_file_reflector_catalog: str | None = None

    def __post_init__(self) -> None:
        names = (
            "output_file_report",
            "output_file_state",
            "output_file_solution",
            "output_file_covariance",
            "output_file_reflector_catalog",
        )
        if not any(getattr(self, name) is not None for name in names):
            raise ValueError("writeResults requires at least one output file.")
        for name in names:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _output_path(value, f"writeResults {name}"))


AdjustmentProcessingStep = (
    ScreenObservationsStep
    | SelectParametrizationsStep
    | EstimateStep
    | WriteResidualsStep
    | WriteNormalEquationsStep
    | WriteResultsStep
)


@dataclass(frozen=True)
class LlrAdjustmentPlan:
    settings: LlrAdjustmentSettings
    processing_steps: tuple[AdjustmentProcessingStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.settings, LlrAdjustmentSettings):
            raise TypeError("Adjustment plan settings must be LlrAdjustmentSettings.")
        steps = tuple(self.processing_steps)
        valid_types = (
            ScreenObservationsStep,
            SelectParametrizationsStep,
            EstimateStep,
            WriteResidualsStep,
            WriteNormalEquationsStep,
            WriteResultsStep,
        )
        if not steps or not all(isinstance(step, valid_types) for step in steps):
            raise ValueError("Adjustment plan must contain valid processing steps.")
        estimates = [step for step in steps if isinstance(step, EstimateStep)]
        screens = [step for step in steps if isinstance(step, ScreenObservationsStep)]
        if len(screens) != 1 or not isinstance(steps[0], ScreenObservationsStep):
            raise ValueError("Processing plan must start with exactly one screenObservations step.")
        if not estimates:
            raise ValueError("Adjustment plan must contain at least one estimate step.")
        if not any(
            isinstance(step, (WriteResidualsStep, WriteNormalEquationsStep, WriteResultsStep)) for step in steps
        ):
            raise ValueError("Processing plan must contain at least one output step.")
        if len({step.name for step in estimates}) != len(estimates):
            raise ValueError("Estimate step names must be unique.")
        first_writer = next(
            (
                index
                for index, step in enumerate(steps)
                if not isinstance(step, (ScreenObservationsStep, SelectParametrizationsStep, EstimateStep))
            ),
            len(steps),
        )
        if any(
            isinstance(step, (ScreenObservationsStep, SelectParametrizationsStep, EstimateStep))
            for step in steps[first_writer:]
        ):
            raise ValueError("Selection and estimate steps must precede all output steps.")
        object.__setattr__(self, "processing_steps", steps)


__all__ = [
    "AdjustmentProcessingStep",
    "EstimateStep",
    "LlrAdjustmentPlan",
    "ScreenObservationsStep",
    "SelectParametrizationsStep",
    "WriteNormalEquationsStep",
    "WriteResidualsStep",
    "WriteResultsStep",
]
