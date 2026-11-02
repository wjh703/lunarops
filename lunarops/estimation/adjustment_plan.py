from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from lunarops.base.station_interval import StationInterval
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition


@dataclass(frozen=True, slots=True)
class ResidualScreening:
    maximumAbsoluteM: float | None
    maximumAbsoluteByStationM: Mapping[str, float | None] | None = None


@dataclass(frozen=True, slots=True)
class EstimateControl:
    maxIterationCount: int
    convergenceThreshold: float
    computeResiduals: bool
    estimateVarianceFactors: bool
    estimateRobustWeights: bool
    convergenceThresholdByParametrizations: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class AccuracyScreeningSettings:
    """Reject implausibly small reported a-priori observation accuracies."""

    minimumOneWayM: float
    minimumFractionOfGroupMedian: float


@dataclass(frozen=True, slots=True)
class RobustWeightSettings:
    """Direct-rejection or IGG3 observation weights."""

    model: str
    k0: float
    k1: float | None = 6.0


@dataclass(frozen=True, slots=True)
class VarianceComponentSettings:
    """Station, equipment-era, and wavelength variance-component groups."""

    components: tuple[VarianceComponentDefinition, ...]


@dataclass(frozen=True, slots=True)
class LlrAdjustmentSettings:
    """Complete resolved settings for one nonlinear adjustment stage."""

    variance_components: VarianceComponentSettings
    adjustment: EstimateControl
    residual: ResidualScreening
    accuracy_screening: AccuracyScreeningSettings
    robust_weights: RobustWeightSettings


@dataclass(frozen=True)
class ScreenObservationsStep:
    """Permanently define the observation domain before estimation."""

    residual: ResidualScreening
    accuracy: AccuracyScreeningSettings


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
    adjustment: EstimateControl
    robust_weighting: RobustWeightSettings | None = None

    def apply(self, plan: LlrAdjustmentPlan) -> LlrAdjustmentSettings:
        screen = cast(ScreenObservationsStep, plan.processing_steps[0])
        return LlrAdjustmentSettings(
            variance_components=plan.variance_components,
            residual=screen.residual,
            accuracy_screening=screen.accuracy,
            adjustment=self.adjustment,
            robust_weights=cast(RobustWeightSettings, self.robust_weighting),
        )


@dataclass(frozen=True)
class WriteResidualsStep:
    output_file: str
    output_level: str


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
                        residual["maximumAbsoluteM"],
                        residual["maximumAbsoluteByStationM"] or None,
                    ),
                    accuracy=AccuracyScreeningSettings(
                        **step["reportedSigma"]
                    ),
                )
            )
        elif typ == "selectParametrizations":
            steps.append(SelectParametrizationsStep(tuple(step["parametrizations"])))
        elif typ == "estimate":
            control = {key: step[key] for key in EstimateControl.__dataclass_fields__}
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
        else:
            raise ValueError(f"Unknown processing step type {typ!r}")
    return LlrAdjustmentPlan(VarianceComponentSettings(components), tuple(steps))


_OUTPUT_NAMES = {
    "outputFileReport": "output_file_report",
    "outputFileState": "output_file_state",
    "outputFileSolution": "output_file_solution",
    "outputFileCovariance": "output_file_covariance",
    "outputFileReflectorCatalog": "output_file_reflector_catalog",
    "outputFileStationCatalog": "output_file_station_catalog",
}
