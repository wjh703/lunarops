"""Typed GROOPS-style steps for the LLR processing program."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from lunarops.estimation.adjustment_settings import (
    AccuracyScreeningSettings,
    AdjustmentControlSettings,
    LlrAdjustmentSettings,
    RobustWeightSettings,
)


@dataclass(frozen=True)
class ScreenObservationsStep:
    """Permanently define the observation domain before estimation."""

    adjustment: AdjustmentControlSettings = field(default_factory=AdjustmentControlSettings)
    accuracy: AccuracyScreeningSettings = field(default_factory=AccuracyScreeningSettings)


@dataclass(frozen=True)
class SelectParametrizationsStep:
    """Select the complete parametrization set for subsequent estimates."""

    parametrizations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.parametrizations:
            raise ValueError("selectParametrizations must select at least one parametrization.")
        if len(set(self.parametrizations)) != len(self.parametrizations):
            raise ValueError("Selected parametrizations must be unique.")

    def apply(self, available: Sequence[str]) -> tuple[str, ...]:
        selected = set(self.parametrizations)
        missing = selected - set(available)
        if missing:
            raise KeyError(f"Unknown parametrization selector(s): {sorted(missing)}")
        return tuple(block_id for block_id in available if block_id in selected)


@dataclass(frozen=True)
class EstimateStep:
    """One nonlinear least-squares estimate using the current selection."""

    name: str
    adjustment: AdjustmentControlSettings = field(default_factory=AdjustmentControlSettings)
    robust_weighting: RobustWeightSettings | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Estimate step name must be a non-empty string.")
        object.__setattr__(self, "name", self.name.strip())

    def apply(self, settings: LlrAdjustmentSettings) -> LlrAdjustmentSettings:
        return replace(
            settings,
            adjustment=replace(
                self.adjustment,
                prefit_gross_threshold_m=settings.adjustment.prefit_gross_threshold_m,
                prefit_gross_threshold_by_station_m=settings.adjustment.prefit_gross_threshold_by_station_m,
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
        steps = self.processing_steps
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
