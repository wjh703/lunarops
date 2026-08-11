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
    max_iteration_count: int | None = None
    convergence_threshold_m: float | None = None

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
        if self.max_iteration_count is not None and (
            isinstance(self.max_iteration_count, bool)
            or not isinstance(self.max_iteration_count, int)
            or self.max_iteration_count < 1
        ):
            raise ValueError("Stage max_iteration_count must be a positive integer.")
        if self.convergence_threshold_m is not None and (
            isinstance(self.convergence_threshold_m, bool)
            or not isinstance(self.convergence_threshold_m, Real)
            or not isfinite(float(self.convergence_threshold_m))
            or self.convergence_threshold_m < 0.0
        ):
            raise ValueError("Stage convergence_threshold_m must be finite and non-negative.")
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
                max_iteration_count=(
                    adjustment.max_iteration_count
                    if self.max_iteration_count is None
                    else self.max_iteration_count
                ),
                convergence_threshold_m=(
                    adjustment.convergence_threshold_m
                    if self.convergence_threshold_m is None
                    else self.convergence_threshold_m
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
    warm_start_sigma_and_weights_across_stages: bool

    def __post_init__(self) -> None:
        if not isinstance(self.settings, LlrAdjustmentSettings):
            raise TypeError("Adjustment plan settings must be LlrAdjustmentSettings.")
        stages = tuple(self.stages)
        if not stages or not all(isinstance(stage, LlrAdjustmentStage) for stage in stages):
            raise ValueError("Adjustment plan must contain at least one valid stage.")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("Adjustment plan stage names must be unique.")
        if not isinstance(self.warm_start_sigma_and_weights_across_stages, bool):
            raise TypeError("Adjustment plan warm-start flag must be a boolean.")
        object.__setattr__(self, "stages", stages)


__all__ = ["LlrAdjustmentPlan", "LlrAdjustmentStage"]
