"""Strict configuration schema for the nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from lunarops.base.station_identity import canonical_station_id
from lunarops.estimation.adjustment_plan import EstimateStep, LlrAdjustmentPlan, SelectParametrizationsStep
from lunarops.estimation.adjustment_settings import (
    AccuracyScreeningSettings,
    AdjustmentControlSettings,
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
    "prefitGrossThresholdByStationM",
    "prefitGrossThresholdM",
    "processingSteps",
}
_ACCURACY_KEYS = {"minimumFractionOfGroupMedian", "minimumOneWayM"}
_ROBUST_KEYS = {"k0", "k1", "model"}
_VARIANCE_COMPONENT_KEYS = {"components"}
_PROCESSING_STEP_KEYS = {
    "adjustSigma0",
    "computeResiduals",
    "computeWeights",
    "convergenceThreshold",
    "convergenceThresholdByParametrizations",
    "maxIterationCount",
    "name",
    "parametrizations",
    "robustWeights",
    "type",
}


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean.")
    return value


def _string_sequence(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence.")
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))


def _parse_robust_weights(value: object, path: str, defaults: RobustWeightSettings) -> RobustWeightSettings:
    robust = _mapping(value, path)
    _reject_unknown(robust, _ROBUST_KEYS, path)
    model = _string(robust.get("model", defaults.model), f"{path}.model")
    if model == DIRECT_REJECTION_MODEL:
        k1 = None if "k1" not in robust else _number(robust["k1"], f"{path}.k1")
    else:
        default_k1 = defaults.k1 if defaults.model == model else 6.0
        k1 = _number(robust.get("k1", default_k1), f"{path}.k1")
    return RobustWeightSettings(
        model=model,
        k0=_number(robust.get("k0", defaults.k0), f"{path}.k0"),
        k1=k1,
    )


def _parse_processing_steps(
    value: object,
    defaults: LlrAdjustmentSettings,
) -> tuple[SelectParametrizationsStep | EstimateStep, ...]:
    if value is None:
        return (
            EstimateStep(
                name="joint",
                robust_weights=defaults.robust_weights,
            ),
        )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("adjustment.processingSteps must be a sequence.")
    steps: list[SelectParametrizationsStep | EstimateStep] = []
    for index, raw in enumerate(value):
        path = f"adjustment.processingSteps[{index}]"
        step = _mapping(raw, path)
        _reject_unknown(step, _PROCESSING_STEP_KEYS, path)
        step_type = _string(step.get("type"), f"{path}.type")
        if step_type == "selectParametrizations":
            invalid = set(step) - {"parametrizations", "type"}
            if invalid:
                raise ValueError(f"{path}: key(s) {sorted(invalid)} are not valid for selectParametrizations.")
            steps.append(
                SelectParametrizationsStep(
                    parametrizations=_string_sequence(
                        step.get("parametrizations", ()),
                        f"{path}.parametrizations",
                    ),
                )
            )
            continue
        if step_type != "estimate":
            raise ValueError(f"{path}.type must be 'selectParametrizations' or 'estimate'.")
        invalid = set(step) - {
            "adjustSigma0",
            "computeResiduals",
            "computeWeights",
            "convergenceThreshold",
            "convergenceThresholdByParametrizations",
            "maxIterationCount",
            "name",
            "robustWeights",
            "type",
        }
        if invalid:
            raise ValueError(f"{path}: key(s) {sorted(invalid)} are not valid for estimate.")
        thresholds = _number_mapping(
            step.get("convergenceThresholdByParametrizations"),
            f"{path}.convergenceThresholdByParametrizations",
        )
        steps.append(
            EstimateStep(
                name=_string(step.get("name"), f"{path}.name"),
                max_iteration_count=_integer(step.get("maxIterationCount", 3), f"{path}.maxIterationCount"),
                convergence_threshold_m=_number(
                    step.get("convergenceThreshold", 1.0e-2),
                    f"{path}.convergenceThreshold",
                ),
                convergence_threshold_by_parametrization_m={
                    key: float(item) for key, item in thresholds.items() if item is not None
                },
                compute_residuals=_boolean(step.get("computeResiduals", True), f"{path}.computeResiduals"),
                adjust_sigma0=_boolean(step.get("adjustSigma0", True), f"{path}.adjustSigma0"),
                compute_weights=_boolean(step.get("computeWeights", True), f"{path}.computeWeights"),
                robust_weights=(
                    defaults.robust_weights
                    if "robustWeights" not in step
                    else _parse_robust_weights(step["robustWeights"], f"{path}.robustWeights", defaults.robust_weights)
                ),
            )
        )
    if not steps:
        raise ValueError("adjustment.processingSteps must contain at least one step.")
    return tuple(steps)


def parse_adjustment_plan(config: Mapping[str, object]) -> LlrAdjustmentPlan:
    """Parse the canonical schema; obsolete names are intentionally rejected."""

    obsolete_sections = {"robustEstimation", "robust_estimation", "vce"} & set(config)
    if obsolete_sections:
        raise ValueError(f"Obsolete adjustment section(s): {sorted(obsolete_sections)}.")
    adjustment = _mapping(config.get("adjustment"), "adjustment")
    accuracy = _mapping(config.get("accuracyScreening"), "accuracyScreening")
    if "initialization" in config:
        raise ValueError("Obsolete adjustment section 'initialization'; bias parameters are estimated normally.")
    robust = _mapping(config.get("robustWeights"), "robustWeights")
    variance = _mapping(config.get("varianceComponents"), "varianceComponents")
    _reject_unknown(adjustment, _ADJUSTMENT_KEYS, "adjustment")
    _reject_unknown(accuracy, _ACCURACY_KEYS, "accuracyScreening")
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
    robust_settings = _parse_robust_weights(robust, "robustWeights", defaults.robust_weights)
    prefit = adjustment.get("prefitGrossThresholdM", defaults.adjustment.prefit_gross_threshold_m)
    settings = LlrAdjustmentSettings(
        variance_components=VarianceComponentSettings(components),
        adjustment=AdjustmentControlSettings(
            prefit_gross_threshold_m=None if prefit is None else _number(prefit, "adjustment.prefitGrossThresholdM"),
            prefit_gross_threshold_by_station_m=canonical_thresholds or None,
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
        robust_weights=robust_settings,
    )
    steps = _parse_processing_steps(adjustment.get("processingSteps"), settings)
    return LlrAdjustmentPlan(settings=settings, processing_steps=steps)


__all__ = ["parse_adjustment_plan"]
