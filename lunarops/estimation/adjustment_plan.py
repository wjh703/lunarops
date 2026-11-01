from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from lunarops.base.station_interval import StationInterval
from lunarops.estimation.robust_weights import IGG3_MODEL
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition


@dataclass(frozen=True, slots=True)
class ResidualScreening:
    maximumAbsoluteM: float | None = 20.0
    maximumAbsoluteByStationM: Mapping[str, float | None] | None = None


@dataclass(frozen=True, slots=True)
class EstimateControl:
    maxIterationCount: int = 3
    convergenceThreshold: float = 1.0e-2
    convergenceThresholdByParametrizations: Mapping[str, float] | None = None
    computeResiduals: bool = True
    estimateVarianceFactors: bool = True
    estimateRobustWeights: bool = True


@dataclass(frozen=True, slots=True)
class AccuracyScreeningSettings:
    """Reject implausibly small reported a-priori observation accuracies."""

    minimumOneWayM: float = 1.0e-3
    minimumFractionOfGroupMedian: float = 0.1


@dataclass(frozen=True, slots=True)
class RobustWeightSettings:
    """Direct-rejection or IGG3 observation weights."""

    model: str = IGG3_MODEL
    k0: float = 1.5
    k1: float | None = 6.0


@dataclass(frozen=True, slots=True)
class VarianceComponentSettings:
    """Station, equipment-era, and wavelength variance-component groups."""

    components: tuple[VarianceComponentDefinition, ...]


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

    def apply(self, plan: LlrAdjustmentPlan) -> LlrAdjustmentSettings:
        screen = cast(ScreenObservationsStep, plan.processing_steps[0])
        return LlrAdjustmentSettings(
            variance_components=plan.variance_components,
            residual=screen.residual,
            accuracy_screening=screen.accuracy,
            adjustment=self.adjustment,
            robust_weights=self.robust_weighting or RobustWeightSettings(),
        )


@dataclass(frozen=True)
class WriteResidualsStep:
    output_file: str
    output_level: str = "standard"


@dataclass(frozen=True)
class WriteNormalEquationsStep:
    output_file: str


@dataclass(frozen=True)
class WriteResultsStep:
    output_file_report: str | None = None
    output_file_state: str | None = None
    output_file_solution: str | None = None
    output_file_covariance: str | None = None
    output_file_reflector_catalog: str | None = None
    output_file_station_catalog: str | None = None


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


def parse_adjustment_plan(config: Mapping[str, object]) -> LlrAdjustmentPlan:
    components = tuple(
        VarianceComponentDefinition(
            id=item["id"],
            interval=StationInterval(item["station"], StationInterval.parse(item["start"]),
                                     None if item.get("endExclusive") is None else StationInterval.parse(item["endExclusive"])),
            wavelength_min_nm=item.get("wavelengthMinNm"),
            wavelength_max_exclusive_nm=item.get("wavelengthMaxExclusiveNm"),
        )
        for item in cast(Sequence[dict], config["varianceComponents"])
    )
    steps: list[AdjustmentProcessingStep] = []
    for step in cast(Sequence[dict], config["processingSteps"]):
        typ = step["type"]
        if typ == "screenObservations":
            residual = step["residual"]
            steps.append(
                ScreenObservationsStep(
                    residual=ResidualScreening(
                        float(residual["maximumAbsoluteM"]) if residual["maximumAbsoluteM"] is not None else None,
                        residual["maximumAbsoluteByStationM"] or None,
                    ),
                    accuracy=AccuracyScreeningSettings(
                        **{key: float(value) for key, value in step["reportedSigma"].items()}
                    ),
                )
            )
        elif typ == "selectParametrizations":
            steps.append(SelectParametrizationsStep(tuple(step["parametrizations"])))
        elif typ == "estimate":
            control = {key: step[key] for key in EstimateControl.__dataclass_fields__}
            control["convergenceThreshold"] = float(control["convergenceThreshold"])
            steps.append(
                EstimateStep(
                    step["name"],
                    EstimateControl(**control),
                    RobustWeightSettings(
                        model=step["robustWeighting"]["model"],
                        k0=step["robustWeighting"]["k0"],
                        k1=step["robustWeighting"].get("k1"),
                    ),
                )
            )
        elif typ == "writeResiduals":
            steps.append(WriteResidualsStep(step["outputFile"], step["outputLevel"]))
        elif typ == "writeNormalEquations":
            steps.append(WriteNormalEquationsStep(step["outputFile"]))
        elif typ == "writeResults":
            steps.append(
                WriteResultsStep(**{_OUTPUT_NAMES[key]: value for key, value in step.items() if key != "type"})
            )
    return LlrAdjustmentPlan(VarianceComponentSettings(components), tuple(steps))


_OUTPUT_NAMES = {
    "outputFileReport": "output_file_report",
    "outputFileState": "output_file_state",
    "outputFileSolution": "output_file_solution",
    "outputFileCovariance": "output_file_covariance",
    "outputFileReflectorCatalog": "output_file_reflector_catalog",
    "outputFileStationCatalog": "output_file_station_catalog",
}
