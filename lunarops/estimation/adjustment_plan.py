"""Typed stage and plan models for nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real

from lunarops.estimation.adjustment_settings import LlrAdjustmentSettings


@dataclass(frozen=True)
class LlrAdjustmentStage:
    """One stage and its outer-loop option overrides."""

    name: str
    parametrizations: tuple[str, ...] = ()
    maximum_linearizations: int | None = None
    parameter_update_factor: float | None = None
    update_tolerance_m: float | None = None
    required_consecutive_converged_linearizations: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Adjustment stage name must be a non-empty string.")
        if isinstance(self.parametrizations, (str, bytes)) or not isinstance(self.parametrizations, Sequence):
            raise TypeError("Adjustment stage parametrization selectors must be a sequence of strings.")
        selectors = tuple(self.parametrizations)
        if any(not isinstance(selector, str) or not selector.strip() for selector in selectors):
            raise TypeError("Adjustment stage parametrization selectors must be non-empty strings.")
        selectors = tuple(selector.strip() for selector in selectors)
        if len(set(selectors)) != len(selectors):
            raise ValueError("Adjustment stage parametrization selectors must be unique.")
        if self.maximum_linearizations is not None and (
            isinstance(self.maximum_linearizations, bool)
            or not isinstance(self.maximum_linearizations, int)
            or self.maximum_linearizations < 1
        ):
            raise ValueError("Stage maximum_linearizations must be a positive integer.")
        if self.parameter_update_factor is not None and (
            isinstance(self.parameter_update_factor, bool)
            or not isinstance(self.parameter_update_factor, Real)
            or not isfinite(float(self.parameter_update_factor))
            or not 0.0 < self.parameter_update_factor <= 1.0
        ):
            raise ValueError("Stage parameter update factor must be finite and in (0, 1].")
        if self.update_tolerance_m is not None and (
            isinstance(self.update_tolerance_m, bool)
            or not isinstance(self.update_tolerance_m, Real)
            or not isfinite(float(self.update_tolerance_m))
            or self.update_tolerance_m < 0.0
        ):
            raise ValueError("Stage update_tolerance_m must be finite and non-negative.")
        if self.required_consecutive_converged_linearizations is not None and (
            isinstance(self.required_consecutive_converged_linearizations, bool)
            or not isinstance(self.required_consecutive_converged_linearizations, int)
            or self.required_consecutive_converged_linearizations < 1
        ):
            raise ValueError("Stage required_consecutive_converged_linearizations must be positive.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "parametrizations", selectors)

    def apply(self, settings: LlrAdjustmentSettings) -> LlrAdjustmentSettings:
        if not isinstance(settings, LlrAdjustmentSettings):
            raise TypeError("Adjustment stage application requires LlrAdjustmentSettings.")
        adjustment = settings.adjustment
        return replace(
            settings,
            adjustment=replace(
                adjustment,
                maximum_linearizations=(
                    adjustment.maximum_linearizations
                    if self.maximum_linearizations is None
                    else self.maximum_linearizations
                ),
                parameter_update_factor=(
                    adjustment.parameter_update_factor
                    if self.parameter_update_factor is None
                    else self.parameter_update_factor
                ),
                update_tolerance_m=(
                    adjustment.update_tolerance_m
                    if self.update_tolerance_m is None
                    else self.update_tolerance_m
                ),
                required_consecutive_converged_linearizations=(
                    adjustment.required_consecutive_converged_linearizations
                    if self.required_consecutive_converged_linearizations is None
                    else self.required_consecutive_converged_linearizations
                ),
            ),
        )

    def validate(self, settings: LlrAdjustmentSettings) -> None:
        if not isinstance(settings, LlrAdjustmentSettings):
            raise TypeError("Adjustment stage validation requires LlrAdjustmentSettings.")
        self.apply(settings)


@dataclass(frozen=True)
class LlrAdjustmentPlan:
    settings: LlrAdjustmentSettings
    stages: tuple[LlrAdjustmentStage, ...]
    warm_start_stochastic_model_across_stages: bool

    def __post_init__(self) -> None:
        if not isinstance(self.settings, LlrAdjustmentSettings):
            raise TypeError("Adjustment plan settings must be LlrAdjustmentSettings.")
        stages = tuple(self.stages)
        if not stages or not all(isinstance(stage, LlrAdjustmentStage) for stage in stages):
            raise ValueError("Adjustment plan must contain at least one valid stage.")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("Adjustment plan stage names must be unique.")
        if not isinstance(self.warm_start_stochastic_model_across_stages, bool):
            raise TypeError("Adjustment plan warm-start flag must be a boolean.")
        object.__setattr__(self, "stages", stages)


__all__ = ["LlrAdjustmentPlan", "LlrAdjustmentStage"]
