"""Strict configuration schema for the nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from lunarops.estimation.adjustment_plan import (
    EstimateStep,
    LlrAdjustmentPlan,
    ScreenObservationsStep,
    SelectParametrizationsStep,
    WriteNormalEquationsStep,
    WriteResidualsStep,
    WriteResultsStep,
)
from lunarops.estimation.adjustment_settings import (
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


_ROBUST_KEYS = {"k0", "k1", "model"}
_RESIDUAL_SCREENING_KEYS = {"maximumAbsoluteByStationM", "maximumAbsoluteM"}
_REPORTED_SIGMA_SCREENING_KEYS = {"minimumFractionOfGroupMedian", "minimumOneWayM"}
_PROCESSING_STEP_KEYS = {
    "computeResiduals",
    "convergenceThreshold",
    "convergenceThresholdByParametrizations",
    "estimateRobustWeights",
    "estimateVarianceFactors",
    "maxIterationCount",
    "name",
    "parametrizations",
    "reportedSigma",
    "residual",
    "robustWeighting",
    "type",
    "outputFile",
    "outputFileCovariance",
    "outputFileReflectorCatalog",
    "outputFileReport",
    "outputFileSolution",
    "outputFileState",
    "outputLevel",
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
) -> tuple[
    ScreenObservationsStep
    | SelectParametrizationsStep
    | EstimateStep
    | WriteResidualsStep
    | WriteNormalEquationsStep
    | WriteResultsStep,
    ...,
]:
    if value is None:
        raise ValueError("processingSteps is required.")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("processingSteps must be a sequence.")
    steps: list[
        ScreenObservationsStep
        | SelectParametrizationsStep
        | EstimateStep
        | WriteResidualsStep
        | WriteNormalEquationsStep
        | WriteResultsStep
    ] = []
    for index, raw in enumerate(value):
        path = f"processingSteps[{index}]"
        step = _mapping(raw, path)
        _reject_unknown(step, _PROCESSING_STEP_KEYS, path)
        step_type = _string(step.get("type"), f"{path}.type")
        if step_type == "screenObservations":
            invalid = set(step) - {"reportedSigma", "residual", "type"}
            if invalid:
                raise ValueError(f"{path}: key(s) {sorted(invalid)} are not valid for screenObservations.")
            residual = _mapping(step.get("residual"), f"{path}.residual")
            reported_sigma = _mapping(step.get("reportedSigma"), f"{path}.reportedSigma")
            _reject_unknown(residual, _RESIDUAL_SCREENING_KEYS, f"{path}.residual")
            _reject_unknown(reported_sigma, _REPORTED_SIGMA_SCREENING_KEYS, f"{path}.reportedSigma")
            by_station = _number_mapping(
                residual.get("maximumAbsoluteByStationM"),
                f"{path}.residual.maximumAbsoluteByStationM",
                allow_none=True,
            )
            steps.append(
                ScreenObservationsStep(
                    maximum_absolute_residual_m=(
                        None
                        if residual.get("maximumAbsoluteM", defaults.adjustment.prefit_gross_threshold_m) is None
                        else _number(
                            residual.get("maximumAbsoluteM", defaults.adjustment.prefit_gross_threshold_m),
                            f"{path}.residual.maximumAbsoluteM",
                        )
                    ),
                    maximum_absolute_residual_by_station_m=by_station or None,
                    minimum_reported_one_way_sigma_m=_number(
                        reported_sigma.get("minimumOneWayM", defaults.accuracy_screening.minimum_one_way_m),
                        f"{path}.reportedSigma.minimumOneWayM",
                    ),
                    minimum_reported_sigma_fraction_of_group_median=_number(
                        reported_sigma.get(
                            "minimumFractionOfGroupMedian",
                            defaults.accuracy_screening.minimum_fraction_of_group_median,
                        ),
                        f"{path}.reportedSigma.minimumFractionOfGroupMedian",
                    ),
                )
            )
            continue
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
        if step_type == "writeResiduals":
            invalid = set(step) - {"outputFile", "outputLevel", "type"}
            if invalid:
                raise ValueError(f"{path}: key(s) {sorted(invalid)} are not valid for writeResiduals.")
            steps.append(
                WriteResidualsStep(
                    output_file=_string(step.get("outputFile"), f"{path}.outputFile"),
                    output_level=_string(step.get("outputLevel", "standard"), f"{path}.outputLevel"),
                )
            )
            continue
        if step_type == "writeNormalEquations":
            invalid = set(step) - {"outputFile", "type"}
            if invalid:
                raise ValueError(f"{path}: key(s) {sorted(invalid)} are not valid for writeNormalEquations.")
            steps.append(
                WriteNormalEquationsStep(output_file=_string(step.get("outputFile"), f"{path}.outputFile"))
            )
            continue
        if step_type == "writeResults":
            keys = {
                "outputFileReport",
                "outputFileState",
                "outputFileSolution",
                "outputFileCovariance",
                "outputFileReflectorCatalog",
            }
            invalid = set(step) - keys - {"type"}
            if invalid:
                raise ValueError(f"{path}: key(s) {sorted(invalid)} are not valid for writeResults.")
            steps.append(
                WriteResultsStep(
                    output_file_report=step.get("outputFileReport"),
                    output_file_state=step.get("outputFileState"),
                    output_file_solution=step.get("outputFileSolution"),
                    output_file_covariance=step.get("outputFileCovariance"),
                    output_file_reflector_catalog=step.get("outputFileReflectorCatalog"),
                )
            )
            continue
        if step_type != "estimate":
            raise ValueError(
                f"{path}.type must be screenObservations, selectParametrizations, estimate, writeResiduals, "
                "writeNormalEquations, or writeResults."
            )
        invalid = set(step) - {
            "computeResiduals",
            "convergenceThreshold",
            "convergenceThresholdByParametrizations",
            "estimateRobustWeights",
            "estimateVarianceFactors",
            "maxIterationCount",
            "name",
            "robustWeighting",
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
                estimate_variance_factors=_boolean(
                    step.get("estimateVarianceFactors", True),
                    f"{path}.estimateVarianceFactors",
                ),
                estimate_robust_weights=_boolean(
                    step.get("estimateRobustWeights", True),
                    f"{path}.estimateRobustWeights",
                ),
                robust_weighting=(
                    defaults.robust_weights
                    if "robustWeighting" not in step
                    else _parse_robust_weights(
                        step["robustWeighting"],
                        f"{path}.robustWeighting",
                        defaults.robust_weights,
                    )
                ),
            )
        )
    if not steps:
        raise ValueError("processingSteps must contain at least one step.")
    return tuple(steps)


def parse_adjustment_plan(config: Mapping[str, object]) -> LlrAdjustmentPlan:
    """Parse the canonical schema; obsolete names are intentionally rejected."""

    obsolete_sections = {
        "accuracyScreening",
        "adjustment",
        "initialization",
        "robustEstimation",
        "robustWeights",
        "robust_estimation",
        "vce",
    } & set(config)
    if obsolete_sections:
        raise ValueError(f"Obsolete adjustment section(s): {sorted(obsolete_sections)}.")
    raw_components = config.get("varianceComponents")
    if isinstance(raw_components, (str, bytes)) or not isinstance(raw_components, Sequence):
        raise TypeError("varianceComponents must be a sequence.")
    components = tuple(
        VarianceComponentDefinition.from_config(_mapping(item, f"varianceComponents[{index}]"))
        for index, item in enumerate(raw_components)
    )
    defaults = LlrAdjustmentSettings(variance_components=VarianceComponentSettings(components))
    steps = _parse_processing_steps(config.get("processingSteps"), defaults)
    screen = next((step for step in steps if isinstance(step, ScreenObservationsStep)), None)
    settings = defaults
    if screen is not None:
        adjustment, accuracy = screen.screening_settings()
        settings = LlrAdjustmentSettings(
            variance_components=defaults.variance_components,
            adjustment=adjustment,
            accuracy_screening=accuracy,
            robust_weights=defaults.robust_weights,
        )
    return LlrAdjustmentPlan(settings=settings, processing_steps=steps)


__all__ = ["parse_adjustment_plan"]
