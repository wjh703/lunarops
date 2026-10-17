"""Validated configuration and execution plan for LLR adjustment."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Optional, cast
import math

import numpy as np

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


def _nonnegative_real(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _optional_nonnegative_real(value: object, name: str) -> float | None:
    return None if value is None else _nonnegative_real(value, name)


@dataclass(frozen=True, slots=True)
class ResidualScreening:
    maximumAbsoluteM: float | None = 20.0
    maximumAbsoluteByStationM: Mapping[str, float | None] | None = None

    def __post_init__(self):
        _optional_nonnegative_real(self.maximumAbsoluteM, "maximumAbsoluteM")
        for station, value in (self.maximumAbsoluteByStationM or {}).items():
            _optional_nonnegative_real(value, f"maximumAbsoluteByStationM.{station}")


@dataclass(frozen=True, slots=True)
class EstimateControl:
    maxIterationCount: int = 3
    convergenceThreshold: float = 1.0e-2
    convergenceThresholdByParametrizations: Mapping[str, float] | None = None
    computeResiduals: bool = True
    estimateVarianceFactors: bool = True
    estimateRobustWeights: bool = True

    def __post_init__(self):
        _positive_integer(self.maxIterationCount, "maxIterationCount")
        _nonnegative_real(self.convergenceThreshold, "convergenceThreshold")
        for block, value in (self.convergenceThresholdByParametrizations or {}).items():
            _nonnegative_real(value, f"convergenceThresholdByParametrizations.{block}")
        if not self.computeResiduals and (self.estimateVarianceFactors or self.estimateRobustWeights):
            raise ValueError("estimateVarianceFactors and estimateRobustWeights require computeResiduals=true.")


@dataclass(frozen=True, slots=True)
class AccuracyScreeningSettings:
    """Reject implausibly small reported a-priori observation accuracies."""

    minimumOneWayM: float = 1.0e-3
    minimumFractionOfGroupMedian: float = 0.1

    def __post_init__(self) -> None:
        minimum = _finite_real(self.minimumOneWayM, "minimumOneWayM")
        fraction = _finite_real(
            self.minimumFractionOfGroupMedian,
            "minimumFractionOfGroupMedian",
        )
        if minimum < 0.0:
            raise ValueError("Minimum one-way accuracy must be non-negative.")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Minimum fraction of group median must be in [0, 1].")
        object.__setattr__(self, "minimumOneWayM", minimum)
        object.__setattr__(self, "minimumFractionOfGroupMedian", fraction)


@dataclass(frozen=True, slots=True)
class RobustWeightSettings:
    """Direct-rejection or IGG3 observation weights."""

    model: str = IGG3_MODEL
    k0: float = 1.5
    k1: Optional[float] = None

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
        if model == IGG3_MODEL and (k1 is None or not 0.0 < k0 < k1):
            raise ValueError("IGGIII thresholds must satisfy 0 < k0 < k1.")
        if model != IGG3_MODEL and k1 is not None:
            raise ValueError("directRejection uses k0 only; omit k1.")
        if model != IGG3_MODEL and k0 <= 0.0:
            raise ValueError("Direct-rejection threshold k0 must be positive.")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "k1", k1)


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
    adjustment: EstimateControl = field(default_factory=EstimateControl)
    residual: ResidualScreening = field(default_factory=ResidualScreening)
    accuracy_screening: AccuracyScreeningSettings = field(default_factory=AccuracyScreeningSettings)
    robust_weights: RobustWeightSettings = field(default_factory=RobustWeightSettings)


@dataclass(frozen=True)
class ScreenObservationsStep:
    """Permanently define the observation domain before estimation."""

    residual: ResidualScreening = field(default_factory=ResidualScreening)
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
    adjustment: EstimateControl = field(default_factory=EstimateControl)
    robust_weighting: RobustWeightSettings | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Estimate step name must be a non-empty string.")
        object.__setattr__(self, "name", self.name.strip())

    def apply(self, plan: LlrAdjustmentPlan) -> LlrAdjustmentSettings:
        screen = cast(ScreenObservationsStep, plan.processing_steps[0])
        return LlrAdjustmentSettings(
            variance_components=plan.variance_components,
            residual=screen.residual,
            accuracy_screening=screen.accuracy,
            adjustment=self.adjustment,
            robust_weights=self.robust_weighting or RobustWeightSettings(),
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
    variance_components: VarianceComponentSettings
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


def _optional_string(value: object, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _number_mapping(value: object, path: str, *, allow_none: bool = False) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for raw_key, raw_value in _mapping(value, path).items():
        key = _string(raw_key, f"{path} key")
        result[key] = None if raw_value is None and allow_none else _number(raw_value, f"{path}.{key}")
    return result


_ROBUST_KEYS = {"k0", "k1", "model"}
_RESIDUAL_SCREENING_KEYS = {"maximumAbsoluteByStationM", "maximumAbsoluteM"}
_REPORTED_SIGMA_SCREENING_KEYS = {"minimumFractionOfGroupMedian", "minimumOneWayM"}
_ESTIMATE_KEYS = {
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
_WRITE_RESULTS_KEYS = {
    "outputFileCovariance",
    "outputFileReflectorCatalog",
    "outputFileReport",
    "outputFileSolution",
    "outputFileState",
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
    return RobustWeightSettings(
        model=_string(robust.get("model", defaults.model), f"{path}.model"),
        k0=_number(robust.get("k0", defaults.k0), f"{path}.k0"),
        k1=None if "k1" not in robust else _number(robust["k1"], f"{path}.k1"),
    )


def _parse_processing_steps(
    value: object,
    defaults: LlrAdjustmentSettings,
) -> tuple[AdjustmentProcessingStep, ...]:
    if value is None:
        raise ValueError("processingSteps is required.")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("processingSteps must be a sequence.")
    steps: list[AdjustmentProcessingStep] = []
    for index, raw in enumerate(value):
        path = f"processingSteps[{index}]"
        step = _mapping(raw, path)
        step_type = _string(step.get("type"), f"{path}.type")
        if step_type == "screenObservations":
            _reject_unknown(step, {"reportedSigma", "residual", "type"}, path)
            residual = _mapping(step.get("residual"), f"{path}.residual")
            reported_sigma = _mapping(step.get("reportedSigma"), f"{path}.reportedSigma")
            _reject_unknown(residual, _RESIDUAL_SCREENING_KEYS, f"{path}.residual")
            _reject_unknown(reported_sigma, _REPORTED_SIGMA_SCREENING_KEYS, f"{path}.reportedSigma")
            by_station = _number_mapping(
                residual.get("maximumAbsoluteByStationM"),
                f"{path}.residual.maximumAbsoluteByStationM",
                allow_none=True,
            )
            maximum = residual.get("maximumAbsoluteM", defaults.residual.maximumAbsoluteM)
            steps.append(
                ScreenObservationsStep(
                    residual=ResidualScreening(
                        maximumAbsoluteM=(
                            None if maximum is None else _number(maximum, f"{path}.residual.maximumAbsoluteM")
                        ),
                        maximumAbsoluteByStationM=by_station or None,
                    ),
                    accuracy=AccuracyScreeningSettings(
                        minimumOneWayM=_number(
                            reported_sigma.get("minimumOneWayM", defaults.accuracy_screening.minimumOneWayM),
                            f"{path}.reportedSigma.minimumOneWayM",
                        ),
                        minimumFractionOfGroupMedian=_number(
                            reported_sigma.get(
                                "minimumFractionOfGroupMedian",
                                defaults.accuracy_screening.minimumFractionOfGroupMedian,
                            ),
                            f"{path}.reportedSigma.minimumFractionOfGroupMedian",
                        ),
                    ),
                )
            )
            continue
        if step_type == "selectParametrizations":
            _reject_unknown(step, {"parametrizations", "type"}, path)
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
            _reject_unknown(step, {"outputFile", "outputLevel", "type"}, path)
            steps.append(
                WriteResidualsStep(
                    output_file=_string(step.get("outputFile"), f"{path}.outputFile"),
                    output_level=_string(step.get("outputLevel", "standard"), f"{path}.outputLevel"),
                )
            )
            continue
        if step_type == "writeNormalEquations":
            _reject_unknown(step, {"outputFile", "type"}, path)
            steps.append(WriteNormalEquationsStep(output_file=_string(step.get("outputFile"), f"{path}.outputFile")))
            continue
        if step_type == "writeResults":
            _reject_unknown(step, _WRITE_RESULTS_KEYS, path)
            steps.append(
                WriteResultsStep(
                    output_file_report=_optional_string(step.get("outputFileReport"), f"{path}.outputFileReport"),
                    output_file_state=_optional_string(step.get("outputFileState"), f"{path}.outputFileState"),
                    output_file_solution=_optional_string(step.get("outputFileSolution"), f"{path}.outputFileSolution"),
                    output_file_covariance=_optional_string(
                        step.get("outputFileCovariance"), f"{path}.outputFileCovariance"
                    ),
                    output_file_reflector_catalog=_optional_string(
                        step.get("outputFileReflectorCatalog"), f"{path}.outputFileReflectorCatalog"
                    ),
                )
            )
            continue
        if step_type != "estimate":
            raise ValueError(
                f"{path}.type must be screenObservations, selectParametrizations, estimate, writeResiduals, "
                "writeNormalEquations, or writeResults."
            )
        _reject_unknown(step, _ESTIMATE_KEYS, path)
        thresholds = _number_mapping(
            step.get("convergenceThresholdByParametrizations"),
            f"{path}.convergenceThresholdByParametrizations",
        )
        steps.append(
            EstimateStep(
                name=_string(step.get("name"), f"{path}.name"),
                adjustment=EstimateControl(
                    maxIterationCount=_integer(step.get("maxIterationCount", 3), f"{path}.maxIterationCount"),
                    convergenceThreshold=_number(
                        step.get("convergenceThreshold", 1.0e-2),
                        f"{path}.convergenceThreshold",
                    ),
                    convergenceThresholdByParametrizations=cast(dict[str, float], thresholds),
                    computeResiduals=_boolean(step.get("computeResiduals", True), f"{path}.computeResiduals"),
                    estimateVarianceFactors=_boolean(
                        step.get("estimateVarianceFactors", True),
                        f"{path}.estimateVarianceFactors",
                    ),
                    estimateRobustWeights=_boolean(
                        step.get("estimateRobustWeights", True),
                        f"{path}.estimateRobustWeights",
                    ),
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
    """Parse the canonical processing schema into a typed plan."""
    raw_components = config.get("varianceComponents")
    if isinstance(raw_components, (str, bytes)) or not isinstance(raw_components, Sequence):
        raise TypeError("varianceComponents must be a sequence.")
    components = tuple(
        VarianceComponentDefinition.from_config(_mapping(item, f"varianceComponents[{index}]"))
        for index, item in enumerate(raw_components)
    )
    defaults = LlrAdjustmentSettings(variance_components=VarianceComponentSettings(components))
    steps = _parse_processing_steps(config.get("processingSteps"), defaults)
    return LlrAdjustmentPlan(variance_components=defaults.variance_components, processing_steps=steps)
