"""Strict configuration schema for the nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from lunarops.base.station_identity import canonical_station_id
from lunarops.estimation.adjustment_plan import LlrAdjustmentPlan, LlrAdjustmentStage
from lunarops.estimation.adjustment_settings import (
    AccuracyScreeningSettings,
    AdjustmentControlSettings,
    InitializationSettings,
    LlrAdjustmentSettings,
    RobustWeightSettings,
    VarianceComponentSettings,
)
from lunarops.estimation.robust_weights import DIRECT_REJECTION_MODEL
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings.")
    return value


def _reject_unknown(value: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}.")


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite.")
    return result


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string.")
    return value.strip()


def _number_mapping(value: object, path: str, *, allow_none: bool = False) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for raw_key, raw_value in _mapping(value, path).items():
        key = _string(raw_key, f"{path} key")
        result[key] = None if raw_value is None and allow_none else _number(raw_value, f"{path}.{key}")
    return result


_ADJUSTMENT_KEYS = {
    "convergenceThreshold",
    "convergenceThresholdByBlock",
    "maxIterationCount",
    "prefitGrossThresholdByStationM",
    "prefitGrossThresholdM",
    "stages",
}
_ACCURACY_KEYS = {"minimumFractionOfGroupMedian", "minimumOneWayM"}
_INITIALIZATION_KEYS = {"biasMaximumIterations", "biasWeightCap"}
_ROBUST_KEYS = {"activeWeightThreshold", "k0", "k1", "model"}
_VARIANCE_COMPONENT_KEYS = {"components"}
_STAGE_KEYS = {
    "convergenceThreshold",
    "maxIterationCount",
    "name",
    "parametrizations",
}


def _parse_stages(value: object) -> tuple[LlrAdjustmentStage, ...]:
    if value is None:
        return (LlrAdjustmentStage(name="joint"),)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("adjustment.stages must be a sequence.")
    stages: list[LlrAdjustmentStage] = []
    for index, raw in enumerate(value):
        path = f"adjustment.stages[{index}]"
        stage = _mapping(raw, path)
        _reject_unknown(stage, _STAGE_KEYS, path)
        raw_selectors = stage.get("parametrizations", ())
        if isinstance(raw_selectors, (str, bytes)) or not isinstance(raw_selectors, Sequence):
            raise TypeError(f"{path}.parametrizations must be a sequence.")
        stages.append(
            LlrAdjustmentStage(
                name=_string(stage.get("name"), f"{path}.name"),
                parametrizations=tuple(
                    _string(item, f"{path}.parametrizations[{i}]") for i, item in enumerate(raw_selectors)
                ),
                max_iteration_count=(
                    None
                    if "maxIterationCount" not in stage
                    else _integer(stage["maxIterationCount"], f"{path}.maxIterationCount")
                ),
                convergence_threshold_m=(
                    None
                    if "convergenceThreshold" not in stage
                    else _number(stage["convergenceThreshold"], f"{path}.convergenceThreshold")
                ),
            )
        )
    if not stages:
        raise ValueError("adjustment.stages must contain at least one stage.")
    if len({item.name for item in stages}) != len(stages):
        raise ValueError("adjustment.stages names must be unique.")
    return tuple(stages)


def parse_adjustment_plan(config: Mapping[str, object]) -> LlrAdjustmentPlan:
    """Parse the canonical schema; obsolete names are intentionally rejected."""

    obsolete_sections = {"robustEstimation", "robust_estimation", "vce"} & set(config)
    if obsolete_sections:
        raise ValueError(f"Obsolete adjustment section(s): {sorted(obsolete_sections)}.")
    adjustment = _mapping(config.get("adjustment"), "adjustment")
    accuracy = _mapping(config.get("accuracyScreening"), "accuracyScreening")
    initialization = _mapping(config.get("initialization"), "initialization")
    robust = _mapping(config.get("robustWeights"), "robustWeights")
    variance = _mapping(config.get("varianceComponents"), "varianceComponents")
    _reject_unknown(adjustment, _ADJUSTMENT_KEYS, "adjustment")
    _reject_unknown(accuracy, _ACCURACY_KEYS, "accuracyScreening")
    _reject_unknown(initialization, _INITIALIZATION_KEYS, "initialization")
    _reject_unknown(robust, _ROBUST_KEYS, "robustWeights")
    _reject_unknown(variance, _VARIANCE_COMPONENT_KEYS, "varianceComponents")

    raw_components = variance.get("components")
    if isinstance(raw_components, (str, bytes)) or not isinstance(raw_components, Sequence):
        raise TypeError("varianceComponents.components must be a sequence.")
    components = tuple(
        VarianceComponentDefinition.from_config(_mapping(item, f"varianceComponents.components[{index}]"))
        for index, item in enumerate(raw_components)
    )
    defaults = LlrAdjustmentSettings(variance_components=VarianceComponentSettings(components))
    station_thresholds = _number_mapping(
        adjustment.get("prefitGrossThresholdByStationM"),
        "adjustment.prefitGrossThresholdByStationM",
        allow_none=True,
    )
    canonical_thresholds = {canonical_station_id(key): value for key, value in station_thresholds.items()}
    if len(canonical_thresholds) != len(station_thresholds):
        raise ValueError("Duplicate canonical station IDs in prefit thresholds.")
    block_thresholds = _number_mapping(
        adjustment.get("convergenceThresholdByBlock"),
        "adjustment.convergenceThresholdByBlock",
    )
    model = _string(robust.get("model", defaults.robust_weights.model), "robustWeights.model")
    k1 = (
        None
        if model == DIRECT_REJECTION_MODEL and "k1" not in robust
        else _number(robust.get("k1", defaults.robust_weights.k1), "robustWeights.k1")
    )
    prefit = adjustment.get("prefitGrossThresholdM", defaults.adjustment.prefit_gross_threshold_m)
    settings = LlrAdjustmentSettings(
        variance_components=VarianceComponentSettings(components),
        adjustment=AdjustmentControlSettings(
            prefit_gross_threshold_m=None if prefit is None else _number(prefit, "adjustment.prefitGrossThresholdM"),
            prefit_gross_threshold_by_station_m=canonical_thresholds or None,
            max_iteration_count=_integer(
                adjustment.get("maxIterationCount", defaults.adjustment.max_iteration_count),
                "adjustment.maxIterationCount",
            ),
            convergence_threshold_m=_number(
                adjustment.get("convergenceThreshold", defaults.adjustment.convergence_threshold_m),
                "adjustment.convergenceThreshold",
            ),
            convergence_threshold_by_block_m={
                key: float(value) for key, value in block_thresholds.items() if value is not None
            },
        ),
        accuracy_screening=AccuracyScreeningSettings(
            minimum_one_way_m=_number(
                accuracy.get("minimumOneWayM", defaults.accuracy_screening.minimum_one_way_m),
                "accuracyScreening.minimumOneWayM",
            ),
            minimum_fraction_of_group_median=_number(
                accuracy.get(
                    "minimumFractionOfGroupMedian",
                    defaults.accuracy_screening.minimum_fraction_of_group_median,
                ),
                "accuracyScreening.minimumFractionOfGroupMedian",
            ),
        ),
        initialization=InitializationSettings(
            bias_weight_cap=_number(
                initialization.get("biasWeightCap", defaults.initialization.bias_weight_cap),
                "initialization.biasWeightCap",
            ),
            bias_maximum_iterations=_integer(
                initialization.get(
                    "biasMaximumIterations",
                    defaults.initialization.bias_maximum_iterations,
                ),
                "initialization.biasMaximumIterations",
            ),
        ),
        robust_weights=RobustWeightSettings(
            model=model,
            k0=_number(robust.get("k0", defaults.robust_weights.k0), "robustWeights.k0"),
            k1=k1,
            active_weight_threshold=_number(
                robust.get("activeWeightThreshold", defaults.robust_weights.active_weight_threshold),
                "robustWeights.activeWeightThreshold",
            ),
        ),
    )
    stages = _parse_stages(adjustment.get("stages"))
    for stage in stages:
        stage.validate(settings)
    return LlrAdjustmentPlan(settings=settings, stages=stages)


__all__ = ["parse_adjustment_plan"]
