"""Settings and processing steps of an LLR adjustment plan.

A plan consists of variance-component definitions and an ordered list of processing steps: observation
screening, parametrization selection, nonlinear estimates and output steps. ``parse_adjustment_plan`` builds
it from a validated configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from lunarops.base.station_interval import StationInterval
from lunarops.estimation.constraints import AprioriSigmaConstraint, MinimalConstraint, ParameterConstraints
from lunarops.estimation.robust_weights import IGG3_MODEL, ROBUST_WEIGHT_MODELS
from lunarops.estimation.stochastic_model import VARIANCE_MODELS
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition

VARIANCE_ESTIMATION_SCHEMES = ("consistent", "legacy")
#: The empty constraint set (immutable), the default of plans and settings.
NO_CONSTRAINTS = ParameterConstraints()

@dataclass(frozen=True, slots=True)
class ResidualScreening:
    """Pre-fit gross-error thresholds on the absolute reduced one-way O-C residual.

    Attributes
    ----------
    maximumAbsoluteM : float or None
        Default threshold, in metres; ``None`` disables the screening.
    maximumAbsoluteByStationM : Mapping[str, float or None] or None
        Thresholds per station key, in metres, overriding the default; ``None`` disables a station's screening.
    """

    maximumAbsoluteM: float | None
    maximumAbsoluteByStationM: Mapping[str, float | None] | None = None


@dataclass(frozen=True, slots=True)
class EstimateControl:
    """Iteration control of one nonlinear estimate.

    Attributes
    ----------
    maxIterationCount : int
        Maximum number of linearizations (outer iterations).
    convergenceThreshold : float
        Default convergence threshold on each block's update norm, in the block's update unit (metres for
        position and bias blocks).
    computeResiduals : bool
        Compute redundancies of the post-fit residuals; required for variance-factor and robust-weight
        estimation.
    estimateVarianceFactors : bool
        Estimate the sigma factor of every variance component.
    estimateRobustWeights : bool
        Estimate robust observation weight factors.
    convergenceThresholdByParametrizations : Mapping[str, float] or None
        Thresholds per parametrization type, overriding ``convergenceThreshold``.
    varianceEstimation : str
        ``consistent``: at every linearization, iterate solve, consistent robust variance factors and robust
        weights (residuals standardized by their variance under the current weights) to convergence. ``legacy``: one solve per
        linearization followed by ten alternating variance/weight updates on its frozen residuals.
    sessionCorrelation : bool
        Model a random offset per night session and variance component (``consistent`` only).
    sessionMaximumGapHours : float
        Largest gap between consecutive normal points of one session, in hours.
    sessionInitialSigmaM : float
        Starting sigma of the session offsets, in metres.
    varianceModel : str
        Variance function of the observations (see :mod:`lunarops.estimation.stochastic_model`): ``scaled``
        (``sigma = s sigma_rep``) or ``floorAndScaled`` (``sigma^2 = a^2 + (s sigma_rep)^2``, ``consistent`` only).
    earthOrientationCorrection : bool
        Estimate corrections of UT1 and the polar motion per station night as random effects with the tabulated
        uncertainties as prior sigmas (``consistent`` only); reported as UT0 and the variation of latitude.
    earthOrientationSigmaScale : float
        Factor applied to the tabulated Earth-orientation uncertainties.
    """

    maxIterationCount: int
    convergenceThreshold: float
    computeResiduals: bool
    estimateVarianceFactors: bool
    estimateRobustWeights: bool
    convergenceThresholdByParametrizations: Mapping[str, float] | None = None
    varianceEstimation: str = "consistent"
    sessionCorrelation: bool = False
    sessionMaximumGapHours: float = 4.0
    sessionInitialSigmaM: float = 0.005
    varianceModel: str = "scaled"
    earthOrientationCorrection: bool = False
    earthOrientationSigmaScale: float = 1.0

    def __post_init__(self) -> None:
        """Check the options and the rules between them; the configuration validator applies the same checks."""
        if not self.computeResiduals and (self.estimateVarianceFactors or self.estimateRobustWeights):
            raise ValueError("estimateVarianceFactors and estimateRobustWeights require computeResiduals=true.")
        if self.varianceEstimation not in VARIANCE_ESTIMATION_SCHEMES:
            raise ValueError(f"Unknown varianceEstimation {self.varianceEstimation!r}.")
        if self.varianceModel not in VARIANCE_MODELS:
            raise ValueError(f"Unknown varianceModel {self.varianceModel!r}.")
        if self.sessionCorrelation and self.varianceEstimation != "consistent":
            raise ValueError("sessionCorrelation requires varianceEstimation 'consistent'.")
        if self.varianceModel != "scaled" and self.varianceEstimation != "consistent":
            raise ValueError("varianceModel 'floorAndScaled' requires varianceEstimation 'consistent'.")
        if self.earthOrientationCorrection and self.varianceEstimation != "consistent":
            raise ValueError("earthOrientationCorrection requires varianceEstimation 'consistent'.")
        if not self.earthOrientationSigmaScale > 0.0:
            raise ValueError("earthOrientationSigmaScale must be positive.")


@dataclass(frozen=True, slots=True)
class AccuracyScreeningSettings:
    """Limits that reject implausibly small reported a-priori observation sigmas.

    An observation is rejected when its one-way a-priori sigma is below
    ``max(minimumOneWayM, minimumFractionOfGroupMedian * median)``, the median taken over its variance component.

    Attributes
    ----------
    minimumOneWayM : float
        Absolute lower limit of the one-way sigma, in metres.
    minimumFractionOfGroupMedian : float
        Lower limit as a fraction of the component's median one-way sigma.
    """

    minimumOneWayM: float
    minimumFractionOfGroupMedian: float


@dataclass(frozen=True, slots=True)
class RobustWeightSettings:
    """Robust weight model applied to standardized residuals: direct rejection or IGG-III.

    Attributes
    ----------
    model : str
        ``directRejection`` or ``igg3``.
    k0 : float
        Standardized-residual threshold up to which observations keep full weight.
    k1 : float or None
        IGG-III threshold (``k0 < k1``) beyond which observations get zero weight; not used by
        ``directRejection``.
    """

    model: str
    k0: float
    k1: float | None = 6.0

    def __post_init__(self) -> None:
        """Check the model name and its thresholds."""
        if self.model not in ROBUST_WEIGHT_MODELS:
            raise ValueError(f"Unknown robust weight model {self.model!r}; expected one of {sorted(ROBUST_WEIGHT_MODELS)}.")
        if not self.k0 > 0:
            raise ValueError("Robust weighting requires k0 > 0.")
        if self.model == IGG3_MODEL and (self.k1 is None or not self.k0 < self.k1):
            raise ValueError("IGG-III robust weighting requires 0 < k0 < k1.")


@dataclass(frozen=True, slots=True)
class VarianceComponentSettings:
    """Variance-component groups, defined by station, date interval and wavelength range.

    Attributes
    ----------
    components : tuple[VarianceComponentDefinition, ...]
        Component definitions; every observation must match exactly one.
    """

    components: tuple[VarianceComponentDefinition, ...]


@dataclass(frozen=True, slots=True)
class LlrAdjustmentSettings:
    """Resolved settings of one nonlinear estimate.

    Attributes
    ----------
    variance_components : VarianceComponentSettings
        Variance-component groups.
    adjustment : EstimateControl
        Iteration control.
    residual : ResidualScreening
        Gross-error screening settings of the plan.
    accuracy_screening : AccuracyScreeningSettings
        A-priori sigma screening settings of the plan.
    robust_weights : RobustWeightSettings
        Robust weight model.
    constraints : ParameterConstraints
        Parameter constraints, applied as pseudo-observations to the parameters of the estimate.
    """

    variance_components: VarianceComponentSettings
    adjustment: EstimateControl
    residual: ResidualScreening
    accuracy_screening: AccuracyScreeningSettings
    robust_weights: RobustWeightSettings
    constraints: ParameterConstraints = NO_CONSTRAINTS


@dataclass(frozen=True)
class ScreenObservationsStep:
    """Processing step that fixes the observation domain used by all later estimates.

    Attributes
    ----------
    residual : ResidualScreening
        Gross-error thresholds.
    accuracy : AccuracyScreeningSettings
        A-priori sigma limits.
    """

    residual: ResidualScreening
    accuracy: AccuracyScreeningSettings


@dataclass(frozen=True)
class SelectParametrizationsStep:
    """Processing step that selects the parametrizations estimated by subsequent estimates.

    Attributes
    ----------
    parametrizations : tuple[str, ...]
        Parametrization block identifiers (``TYPE`` values) to estimate.
    """

    parametrizations: tuple[str, ...]

    def apply(self, available: Sequence[str]) -> tuple[str, ...]:
        """Return the selected block identifiers in the order of ``available``.

        Raises
        ------
        KeyError
            If a selected identifier is not in ``available``.
        """
        selected = set(self.parametrizations)
        missing = selected - set(available)
        if missing:
            raise KeyError(f"Unknown parametrization selector(s): {sorted(missing)}")
        return tuple(block_id for block_id in available if block_id in selected)


@dataclass(frozen=True)
class EstimateStep:
    """Processing step running one nonlinear least-squares estimate with the current parametrization selection.

    Attributes
    ----------
    name : str
        Step name used in reports and progress output.
    adjustment : EstimateControl
        Iteration control.
    robust_weighting : RobustWeightSettings or None
        Robust weight model.
    """

    name: str
    adjustment: EstimateControl
    robust_weighting: RobustWeightSettings | None = None

    def apply(self, plan: LlrAdjustmentPlan) -> LlrAdjustmentSettings:
        """Return the estimate settings, taking the screening settings from the plan's first processing step.

        Raises
        ------
        ValueError
            If the plan does not start with a screenObservations step.
        """
        screen = plan.processing_steps[0] if plan.processing_steps else None
        if not isinstance(screen, ScreenObservationsStep):
            raise ValueError("The processing plan must start with a screenObservations step.")  # noqa: TRY004 -- invalid plan
        return LlrAdjustmentSettings(
            variance_components=plan.variance_components,
            residual=screen.residual,
            accuracy_screening=screen.accuracy,
            adjustment=self.adjustment,
            robust_weights=cast(RobustWeightSettings, self.robust_weighting),
            constraints=plan.constraints,
        )


@dataclass(frozen=True)
class WriteResidualsStep:
    """Processing step that writes per-observation residuals of the final estimate.

    Attributes
    ----------
    output_file : str
        Output path.
    output_level : str
        Detail level of the written rows.
    """

    output_file: str
    output_level: str


@dataclass(frozen=True)
class WriteNormalEquationsStep:
    """Processing step that writes the normal equations of the final estimate.

    Attributes
    ----------
    output_file : str
        Output path.
    """

    output_file: str


@dataclass(frozen=True)
class WriteResultsStep:
    """Processing step that writes the products of the final estimate; each output is skipped when ``None``.

    Attributes
    ----------
    outputFileReport : str or None
        Processing report.
    outputFileState : str or None
        Processing state (parametrization values, sigma and weight factors) for warm starts.
    outputFileSolution : str or None
        Parameter vector with uncertainties.
    outputFileCovariance : str or None
        Parameter covariance (or cofactor) matrix.
    outputFileReflectorCatalog : str or None
        Reflector catalog with the estimated positions.
    outputFileStationCatalog : str or None
        Station catalog with the estimated positions and velocities.
    """

    outputFileReport: str | None = None
    outputFileState: str | None = None
    outputFileSolution: str | None = None
    outputFileCovariance: str | None = None
    outputFileReflectorCatalog: str | None = None
    outputFileStationCatalog: str | None = None


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
    """Variance components and ordered processing steps of an LLR adjustment.

    Attributes
    ----------
    variance_components : VarianceComponentSettings
        Variance-component groups.
    processing_steps : tuple[AdjustmentProcessingStep, ...]
        Processing steps in execution order.
    constraints : ParameterConstraints
        Parameter constraints of every estimate.
    """

    variance_components: VarianceComponentSettings
    processing_steps: tuple[AdjustmentProcessingStep, ...]
    constraints: ParameterConstraints = NO_CONSTRAINTS


def parse_adjustment_plan(config: Mapping[str, object]) -> LlrAdjustmentPlan:
    """Build an adjustment plan from a validated configuration.

    Parameters
    ----------
    config : Mapping[str, object]
        Configuration with ``varianceComponents`` (``id``, ``station``, ``start``, optional ``endExclusive``
        as UTC dates, optional ``wavelengthMinNm`` and ``wavelengthMaxExclusiveNm``) and ``processingSteps``
        of type ``screenObservations``, ``selectParametrizations``, ``estimate``, ``writeResiduals``,
        ``writeNormalEquations`` or ``writeResults``.

    Returns
    -------
    LlrAdjustmentPlan
        The plan.

    Raises
    ------
    ValueError
        If a processing step has an unknown type.
    """
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
                WriteResultsStep(**{key: value for key, value in step.items() if key != "type"})
            )
        else:
            raise ValueError(f"Unknown processing step type {typ!r}")
    constraints: list[AprioriSigmaConstraint | MinimalConstraint] = []
    for item in cast(Sequence[dict], config.get("constraints") or []):
        if item["type"] == "aprioriSigma":
            constraints.append(AprioriSigmaConstraint(tuple(item["parameters"]), float(item["sigma"])))
        elif item["type"] == "minimalConstraints":
            stations = item.get("stations")
            constraints.append(MinimalConstraint(None if stations is None else tuple(stations),
                                                 tuple(item["conditions"]), float(item["sigmaM"])))
        else:
            raise ValueError(f"Unknown constraint type {item['type']!r}")
    return LlrAdjustmentPlan(VarianceComponentSettings(components), tuple(steps), ParameterConstraints(tuple(constraints)))
