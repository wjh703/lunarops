"""Report records of an LLR adjustment: parameters, residual statistics, variance components, observations.

Residuals are one-way and in metres; standardized residuals and weight factors are dimensionless.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch, diagnostic_values
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.classes.time import Epoch, format_time_with_utc_offset
from lunarops.estimation.adjustment.plan import LlrAdjustmentSettings
from lunarops.estimation.adjustment.results import AdjustmentState, LlrAdjustmentResult
from lunarops.estimation.lsq.normal_equations import NormalEquations
from lunarops.estimation.lsq.products import PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER
from lunarops.estimation.stochastic.components import ObsKey, VarianceComponentDefinition

_EMPTY_ACCURACY_SCREENING_GROUP = {
    "medianReportedSigmaM": None,
    "observationCount": 0,
    "rejectedCount": 0,
    "retainedCount": 0,
}


def weight_factor_summary(
    equations: EquationBatch,
    factors: Mapping[ObsKey, float],
) -> dict[str, object]:
    """Summarize the robust weight factors of the observations in ``equations``.

    Parameters
    ----------
    equations : EquationBatch
        Observations to summarize.
    factors : Mapping[ObsKey, float]
        Weight factor in [0, 1] per observation identity.

    Returns
    -------
    dict[str, object]
        Counts of all, full-weight (1), down-weighted (between 0 and 1) and rejected (0) observations, and,
        when there are observations, the minimum, 5 % quantile, median, 95 % quantile and maximum.
    """
    values = np.asarray([factors[key] for key in equations.observation_id], dtype=float)
    if not len(values):
        return {
            "observationCount": 0,
            "fullWeightCount": 0,
            "downweightedCount": 0,
            "rejectedCount": 0,
        }
    active = values > 0.0
    full = values == 1.0
    return {
        "observationCount": len(values),
        "fullWeightCount": int(np.count_nonzero(full)),
        "downweightedCount": int(np.count_nonzero(active & ~full)),
        "rejectedCount": int(np.count_nonzero(~active)),
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def distribution_summary(values: np.ndarray) -> dict[str, object]:
    """Summarize a sample distribution.

    Parameters
    ----------
    values : ndarray, shape (n,)
        Sample values.

    Returns
    -------
    dict[str, object]
        Count, RMS, median, MAD scaled by 1.4826, the 50/90/95/99 % quantiles and the maximum of the absolute
        values, in the unit of ``values``; only the count for an empty sample.
    """
    if not len(values):
        return {"count": 0}
    median = float(np.median(values))
    absolute = np.abs(values)
    return {
        "count": len(values),
        "rms": float(np.sqrt(np.mean(values**2))),
        "median": median,
        "mad": 1.4826 * float(np.median(np.abs(values - median))),
        "absoluteP50": float(np.quantile(absolute, 0.50)),
        "absoluteP90": float(np.quantile(absolute, 0.90)),
        "absoluteP95": float(np.quantile(absolute, 0.95)),
        "absoluteP99": float(np.quantile(absolute, 0.99)),
        "absoluteMaximum": float(np.max(absolute)),
    }


def residual_summary(
    equations: EquationBatch,
    residual_by_identity: Mapping[ObsKey, float],
    standardized: Mapping[ObsKey, float],
    weights: np.ndarray,
    factors: Mapping[ObsKey, float],
) -> dict[str, object]:
    """Summarize the residuals of all observations.

    Parameters
    ----------
    equations : EquationBatch
        Observations in report order.
    residual_by_identity : Mapping[ObsKey, float]
        One-way residual per observation, in metres.
    standardized : Mapping[ObsKey, float]
        Standardized residual per observation.
    weights : ndarray, shape (n,)
        Observation weights, in 1/m^2, aligned with ``equations``.
    factors : Mapping[ObsKey, float]
        Robust weight factor per observation.

    Returns
    -------
    dict[str, object]
        Residual and standardized-residual distributions, the weighted RMS residual in metres
        (``sqrt(sum(p v^2) / sum(p))``) and the weight-factor summary.
    """
    residuals = np.asarray([residual_by_identity[key] for key in equations.observation_id], dtype=float)
    standards = np.asarray([standardized[key] for key in equations.observation_id], dtype=float)
    weight_sum = float(np.sum(weights))
    return {
        "residualM": distribution_summary(residuals),
        "standardizedResidual": distribution_summary(standards),
        "equivalentWeightedRmsM": (
            None if weight_sum <= 0.0 else float(np.sqrt(np.sum(weights * residuals**2) / weight_sum))
        ),
        "weightFactors": weight_factor_summary(equations, factors),
    }


def parameter_records(
    normals: NormalEquations,
    correction: np.ndarray,
    names: Sequence[ParameterName],
    covariance: np.ndarray,
    sigma0_post: float | None,
    reference_values: np.ndarray,
) -> list[dict[str, object]]:
    """Build one report record per parameter.

    Uncertainties are ``PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER`` times the one-sigma values from the cofactor
    diagonal (``cofactorUncertaintyM``) and scaled by ``sigma0_post`` (``formalUncertaintyM``). The
    ``M``-suffixed keys hold values in the parameter's own unit (``unit``), which is metres for position and
    bias parameters.

    Parameters
    ----------
    normals : NormalEquations
        Final normal equations; ``x0`` holds the final parameter values.
    correction : ndarray, shape (m,)
        Remaining correction solved at the final state and not applied.
    names : Sequence[ParameterName]
        Parameter names in column order.
    covariance : ndarray, shape (m, m)
        Cofactor matrix ``N^-1``.
    sigma0_post : float or None
        A-posteriori unit-weight sigma.
    reference_values : ndarray, shape (m,)
        Parameter values at the start of the estimate.

    Returns
    -------
    list[dict[str, object]]
        Name, type, remaining correction, uncertainties, largest absolute correlation and its parameter,
        reference value, applied update, final value and unit of each parameter.
    """
    diagonal = np.maximum(np.diag(covariance), 0.0)
    cofactor_one_sigma = np.sqrt(diagonal)
    denominator = np.outer(cofactor_one_sigma, cofactor_one_sigma)
    correlations = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    records: list[dict[str, object]] = []
    for index, name in enumerate(names):
        candidates = np.abs(correlations[index]).copy()
        candidates[index] = -1.0
        correlated_index = int(np.argmax(candidates)) if len(candidates) > 1 else None
        records.append(
            {
                "name": str(name),
                "type": name.parameter_type,
                "remainingLinearizedCorrectionM": float(correction[index]),
                "cofactorUncertaintyM": float(PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER * cofactor_one_sigma[index]),
                "formalUncertaintyM": (
                    None
                    if sigma0_post is None
                    else float(PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER * sigma0_post * cofactor_one_sigma[index])
                ),
                "maximumAbsoluteCorrelation": (
                    None if correlated_index is None else float(abs(correlations[index, correlated_index]))
                ),
                "maximumCorrelatedParameter": (None if correlated_index is None else str(names[correlated_index])),
                "referenceValue": float(reference_values[index]),
                "appliedUpdate": float(normals.x0[index] - reference_values[index]),
                "finalValue": float(normals.x0[index]),
                "unit": normals.parameter_units[index],
            }
        )
    return records


def variance_component_records(
    equations: EquationBatch,
    residuals: Mapping[ObsKey, float],
    standardized: Mapping[ObsKey, float],
    sigma_factors: Mapping[str, float],
    initial_sigma_factors: Mapping[str, float],
    weight_factors: Mapping[ObsKey, float],
    proposed_weight_factors: Mapping[ObsKey, float],
    *,
    assignments: Mapping[ObsKey, str],
    components: Sequence[VarianceComponentDefinition],
    diagnostics: Mapping[str, Mapping[str, object]],
    floor_sigmas: Mapping[str, float] | None = None,
    initial_floor_sigmas: Mapping[str, float] | None = None,
    accuracy_screening_groups: Mapping[str, Mapping[str, object]],
    utc_offset_hours: float = 0.0,
) -> list[dict[str, object]]:
    """Build one report record per variance component.

    Parameters
    ----------
    equations : EquationBatch
        Final observations.
    residuals : Mapping[ObsKey, float]
        Linearized post-fit one-way residual per observation, in metres.
    standardized : Mapping[ObsKey, float]
        Standardized residual per observation.
    sigma_factors : Mapping[str, float]
        Final sigma factor per component.
    initial_sigma_factors : Mapping[str, float]
        Sigma factor per component at the start of the estimate.
    weight_factors : Mapping[ObsKey, float]
        Applied robust weight factor per observation.
    proposed_weight_factors : Mapping[ObsKey, float]
        Weight factor proposed from the final residuals (not applied).
    assignments : Mapping[ObsKey, str]
        Component identifier per retained observation.
    components : Sequence[VarianceComponentDefinition]
        Component definitions, in report order.
    diagnostics : Mapping[str, Mapping[str, object]]
        Sigma-factor estimator diagnostics per component, copied into the record.
    floor_sigmas : Mapping[str, float], optional
        Final floor sigma per component, in metres; zero when omitted.
    initial_floor_sigmas : Mapping[str, float], optional
        Floor sigma per component at the start of the estimate, in metres; zero when omitted.
    accuracy_screening_groups : Mapping[str, Mapping[str, object]]
        A-priori sigma screening summary per component.
    utc_offset_hours : float, default 0.0
        Offset of local time from UTC, in hours.

    Returns
    -------
    list[dict[str, object]]
        Per component: configured interval, first and last transmit epochs (UTC and local), counts, initial and
        final sigma factors, floor sigmas and variance factor, residual statistics in metres (RMS, weighted RMS with
        weights ``f / (a^2 + s^2 sigma^2)``), standardized-residual statistics and weight-factor summaries.
    """
    offset = float(utc_offset_hours)
    floors = floor_sigmas or {}
    initial_floors = initial_floor_sigmas or {}
    records: list[dict[str, object]] = []
    component_ids = np.asarray([assignments[key] for key in equations.observation_id])
    for component in components:
        selected = equations[component_ids == component.id]
        identities = selected.observation_id.tolist()
        residual_values = np.asarray([residuals[key] for key in identities], dtype=float)
        standardized_values = np.asarray([standardized[key] for key in identities], dtype=float)
        factor_summary = weight_factor_summary(selected, weight_factors)
        floor = float(floors.get(component.id, 0.0))
        weights = np.asarray(
            [
                weight_factors[key] / (floor**2 + sigma_factors[component.id] ** 2 * sigma**2)
                for key, sigma in zip(identities, selected.sigma)
            ],
            dtype=float,
        )
        epochs = [selected.epoch(index) for index in range(len(selected))]
        weight_sum = float(np.sum(weights))
        item = dict(diagnostics.get(component.id, {}))
        item.update(
            {
                "componentId": component.id,
                "configuredStart": component.interval.start.isoformat(),
                "configuredEnd": component.interval.end_exclusive.isoformat()
                if component.interval.end_exclusive
                else "present",
                "actualStartEpochUtc": (min(epochs).isot(precision=9) if selected else None),
                "actualStartEpochLocal": (
                    format_time_with_utc_offset(
                        min(epochs),
                        utc_offset_hours=offset,
                        precision=9,
                    )
                    if selected
                    else None
                ),
                "actualEndEpochUtc": (max(epochs).isot(precision=9) if selected else None),
                "actualEndEpochLocal": (
                    format_time_with_utc_offset(
                        max(epochs),
                        utc_offset_hours=offset,
                        precision=9,
                    )
                    if selected
                    else None
                ),
                "proposedSigmaFactorApplied": False,
                "observationCount": len(selected),
                "retainedObservationCount": sum(value == component.id for value in assignments.values()),
                "initialSigmaFactor": float(initial_sigma_factors[component.id]),
                "finalSigmaFactor": float(sigma_factors[component.id]),
                "varianceFactor": float(sigma_factors[component.id] ** 2),
                "initialFloorSigmaM": float(initial_floors.get(component.id, 0.0)),
                "finalFloorSigmaM": floor,
                # A configured component can have no observations in the processed time window.
                "reportedSigmaScreening": dict(
                    accuracy_screening_groups.get(component.id, _EMPTY_ACCURACY_SCREENING_GROUP)
                ),
                "residualRmsM": (float(np.sqrt(np.mean(residual_values**2))) if len(residual_values) else None),
                "standardizedRms": (
                    float(np.sqrt(np.mean(standardized_values**2))) if len(standardized_values) else None
                ),
                "medianStandardizedResidual": (
                    float(np.median(standardized_values)) if len(standardized_values) else None
                ),
                "madStandardizedResidual": distribution_summary(standardized_values).get("mad"),
                "residualWrmsM": (
                    None if weight_sum <= 0.0 else float(np.sqrt(np.sum(weights * residual_values**2) / weight_sum))
                ),
                "residualSummaryM": distribution_summary(residual_values),
                "standardizedResidualSummary": distribution_summary(standardized_values),
                "weightFactorSummary": factor_summary,
                "finalStateProposedWeightFactorSummary": weight_factor_summary(
                    selected,
                    proposed_weight_factors,
                ),
                "fullWeightCount": factor_summary["fullWeightCount"],
                "downweightedCount": factor_summary["downweightedCount"],
                "rejectedCount": factor_summary["rejectedCount"],
            }
        )
        records.append(item)
    return records


@dataclass(frozen=True, slots=True)
class ObservationReportValues:
    """Values of one observation from which ``OBSERVATION_REPORT_FIELDS`` are evaluated.

    Attributes
    ----------
    identity : object
        Observation identity.
    converged : bool
        Whether the light-time iteration converged.
    clamped : bool
        Whether a troposphere elevation was clamped.
    epoch : Epoch
        Transmit epoch, UTC.
    station : str
        Station key of the observation.
    sigma : float
        A-priori one-way sigma, in metres.
    factor : float
        Applied robust weight factor.
    proposed_factor : float
        Weight factor proposed from the final residuals.
    component_id : str
        Variance-component identifier.
    component_station : str
        Station of the variance component definition.
    screening : Mapping[str, object]
        A-priori sigma screening record.
    sigma_factor : float
        Sigma factor of the component.
    floor_sigma : float
        Floor sigma of the component, in metres.
    base_variance : float
        ``floor_sigma**2 + sigma_factor**2 * sigma**2``, in m^2.
    current_residual : float
        Reduced one-way O-C at the final state, in metres.
    residual : float
        Linearized post-fit one-way residual, in metres.
    residual_sigma : float
        Sigma of the post-fit residual, in metres.
    standardized : float
        Standardized residual.
    status : str
        ``FULL_WEIGHT``, ``DOWNWEIGHTED`` or ``REJECTED`` for the applied factor.
    proposed_status : str
        The same classification for the proposed factor.
    matched_parameters : list[str]
        Names of the parameters with a non-zero design entry for the observation.
    offset : float
        Offset of local time from UTC, in hours.
    """

    identity: object
    converged: bool
    clamped: bool
    epoch: Epoch
    station: str
    sigma: float
    factor: float
    proposed_factor: float
    component_id: str
    component_station: str
    screening: Mapping[str, object]
    sigma_factor: float
    floor_sigma: float
    base_variance: float
    current_residual: float
    residual: float
    residual_sigma: float
    standardized: float
    status: str
    proposed_status: str
    matched_parameters: list[str]
    offset: float


# (report name, unit, getter); residual names follow the same field declaration.
OBSERVATION_REPORT_FIELDS: tuple[tuple[str, str, Callable[[ObservationReportValues], object]], ...] = (
    ("observationId", "1", lambda r: str(r.identity)),
    ("lightTimeConverged", "1", lambda r: r.converged),
    ("troposphereClamped", "1", lambda r: r.clamped),
    ("epochUtc", "UTC", lambda r: r.epoch.isot(precision=9)),
    ("epochLocal", "local", lambda r: format_time_with_utc_offset(r.epoch, utc_offset_hours=r.offset, precision=9)),
    ("stationId", "1", lambda r: r.station),
    ("station", "1", lambda r: r.component_station),
    ("varianceComponentId", "1", lambda r: r.component_id),
    ("aprioriSigmaM", "m", lambda r: r.screening["reportedSigmaM"]),
    ("reportedSigmaScreeningStatus", "1", lambda r: r.screening["status"]),
    ("reportedSigmaScreeningReason", "1", lambda r: r.screening["reason"]),
    ("minimumValidSigmaM", "m", lambda r: r.screening["minimumValidSigmaM"]),
    ("sigmaFactor", "1", lambda r: r.sigma_factor),
    ("floorSigmaM", "m", lambda r: r.floor_sigma),
    ("sigma0M", "m", lambda r: float(np.hypot(r.floor_sigma, r.sigma_factor * r.sigma))),
    ("baseVarianceM2", "m2", lambda r: float(r.base_variance)),
    ("baseWeightPerM2", "1/m2", lambda r: float(1.0 / r.base_variance)),
    ("currentStateResidualM", "m", lambda r: r.current_residual),
    ("linearizedPostfitResidualM", "m", lambda r: r.residual),
    ("residualSigmaM", "m", lambda r: r.residual_sigma),
    ("leverage", "1", lambda r: float(1.0 - r.residual_sigma**2 / r.base_variance)),
    ("standardizedResidual", "1", lambda r: r.standardized),
    ("appliedWeightFactor", "1", lambda r: r.factor),
    ("finalStateProposedWeightFactor", "1", lambda r: r.proposed_factor),
    ("proposedWeightFactorApplied", "1", lambda r: False),
    ("equivalentWeightPerM2", "1/m2", lambda r: float(r.factor / r.base_variance)),
    ("appliedWeightStatus", "1", lambda r: r.status),
    ("finalStateProposedWeightStatus", "1", lambda r: r.proposed_status),
    ("matchedParameterNames", "1", lambda r: r.matched_parameters),
)


def observation_records(
    equations: EquationBatch,
    current_state_residuals: Mapping[ObsKey, float],
    linearized_postfit_residuals: Mapping[ObsKey, float],
    residual_sigmas: Mapping[ObsKey, float],
    standardized: Mapping[ObsKey, float],
    sigma_factors: Mapping[str, float],
    weight_factors: Mapping[ObsKey, float],
    proposed_weight_factors: Mapping[ObsKey, float],
    *,
    assignments: Mapping[ObsKey, str],
    parametrization: ParametrizationList,
    components: Sequence[VarianceComponentDefinition],
    accuracy_screening_records: Mapping[ObsKey, Mapping[str, object]],
    utc_offset_hours: float = 0.0,
    floor_sigmas: Mapping[str, float] | None = None,
) -> list[dict[str, object]]:
    """Build one report record per observation with the fields of ``OBSERVATION_REPORT_FIELDS``.

    Parameters
    ----------
    equations : EquationBatch
        Final observations.
    current_state_residuals : Mapping[ObsKey, float]
        Reduced one-way O-C at the final state, in metres.
    linearized_postfit_residuals : Mapping[ObsKey, float]
        Linearized post-fit one-way residual, in metres.
    residual_sigmas : Mapping[ObsKey, float]
        Sigma of the post-fit residual, in metres.
    standardized : Mapping[ObsKey, float]
        Standardized residual.
    sigma_factors : Mapping[str, float]
        Sigma factor per variance component.
    weight_factors : Mapping[ObsKey, float]
        Applied robust weight factor.
    proposed_weight_factors : Mapping[ObsKey, float]
        Weight factor proposed from the final residuals.
    assignments : Mapping[ObsKey, str]
        Variance-component identifier per observation.
    parametrization : ParametrizationList
        Estimated parametrization, used for the matched parameter names.
    components : Sequence[VarianceComponentDefinition]
        Component definitions.
    accuracy_screening_records : Mapping[ObsKey, Mapping[str, object]]
        A-priori sigma screening record per observation.
    utc_offset_hours : float, default 0.0
        Offset of local time from UTC, in hours.
    floor_sigmas : Mapping[str, float], optional
        Floor sigma per variance component, in metres; zero when omitted.

    Returns
    -------
    list[dict[str, object]]
        Records in batch order.
    """
    offset = float(utc_offset_hours)
    floors = floor_sigmas or {}
    stations = {component.id: component.interval.station for component in components}
    batch = equations
    design = parametrization.design(batch)
    parameter_names = [str(name) for name in parametrization.names]
    columns = zip(
        batch.observation_id.tolist(),
        batch.converged.tolist(),
        batch.clamped.tolist(),
        batch.station.tolist(),
        batch.sigma.tolist(),
        strict=True,
    )
    records: list[dict[str, object]] = []
    for index, (identity, converged, clamped, station, sigma) in enumerate(columns):
        epoch = batch.epoch(index)
        factor = float(weight_factors[identity])
        proposed_factor = float(proposed_weight_factors[identity])
        component_id = assignments[identity]
        matched_parameters = [parameter_names[j] for j in np.flatnonzero(design[index])]
        status = "REJECTED" if factor == 0.0 else ("FULL_WEIGHT" if factor == 1.0 else "DOWNWEIGHTED")
        proposed_status = (
            "REJECTED" if proposed_factor == 0.0 else ("FULL_WEIGHT" if proposed_factor == 1.0 else "DOWNWEIGHTED")
        )
        screening = accuracy_screening_records[identity]
        sigma_factor = float(sigma_factors[component_id])
        floor_sigma = float(floors.get(component_id, 0.0))
        base_variance = floor_sigma**2 + sigma_factor**2 * sigma**2
        values = ObservationReportValues(
            identity,
            converged,
            clamped,
            epoch,
            station,
            sigma,
            factor,
            proposed_factor,
            component_id,
            stations[component_id],
            screening,
            sigma_factor,
            floor_sigma,
            base_variance,
            float(current_state_residuals[identity]),
            float(linearized_postfit_residuals[identity]),
            float(residual_sigmas[identity]),
            float(standardized[identity]),
            status,
            proposed_status,
            matched_parameters,
            offset,
        )
        records.append(diagnostic_values(OBSERVATION_REPORT_FIELDS, values))
    return records


__all__ = [
    "build_report",
    "distribution_summary",
    "observation_records",
    "parameter_records",
    "residual_summary",
    "variance_component_records",
    "weight_factor_summary",
]


def build_report(state: AdjustmentState, plan: LlrAdjustmentSettings) -> LlrAdjustmentResult:
    """Build the adjustment result from a finished solver state.

    Parameter, global-residual, variance-component and observation records are built only when
    ``state.finalize`` is true.

    Parameters
    ----------
    state : AdjustmentState
        Solver state.
    plan : LlrAdjustmentSettings
        Settings of the estimate.

    Returns
    -------
    LlrAdjustmentResult
        Report and products.
    """
    accuracy_rejected = {
        str(key): value
        for key, value in state.observation_domain.accuracy_records.items()
        if value["status"] == "REJECTED"
    }
    summary = {
        "converged": state.converged,
        "terminationReason": state.termination_reason,
        "sourceObservationCount": state.equation_evaluations[0]["sourceObservationCount"],
        "initialLightTimeConvergedCount": state.equation_evaluations[0]["lightTimeConvergedCount"],
        "grossRejectedCount": len(state.observation_domain.gross_rejected),
        "accuracyRejectedCount": len(accuracy_rejected),
        "retainedObservationCount": len(state.observation_domain.retained_keys),
        "lastSolvedEquationCount": len(state.last_solution.equations),
        "equationEvaluationCount": len(state.equation_evaluations),
        "adjustmentIterationCount": len(state.adjustment_iterations),
        "sigmaWeightIterationCount": len(state.iterations),
        # False when the solve/reweight passes of the last linearization stopped at their limit.
        "stochasticModelConverged": bool(
            state.adjustment_iterations[-1].get("stochasticModelConverged", True)
            if state.adjustment_iterations
            else True
        ),
        "lastNormalMatrixRank": state.last_solution.rank,
        "lastNormalMatrixCondition": state.last_solution.condition,
        "performanceSeconds": dict(state.performance_seconds),
    }
    final_parameter_records = []
    global_residuals = {}
    component_records = []
    observations = []
    final_solution = state.solution
    if state.finalize:
        final_equations = state.equations
        keys = final_equations.observation_id.tolist()
        final_residuals = dict(zip(keys, map(float, final_solution.residual_vector), strict=True))
        final_standardized = dict(zip(keys, map(float, state.standardized), strict=True))
        final_residual_sigmas = dict(zip(keys, map(float, state.residual_sigmas), strict=True))
        final_proposed_weight_factors = state.proposed_weight_factors
        current_state_residuals = {
            identity: float(value)
            for identity, value in zip(final_equations.observation_id, state.parametrization.reduce(final_equations))
        }
        final_parameter_records = parameter_records(
            final_solution.normals,
            final_solution.delta,
            final_solution.normals.parameter_names,
            final_solution.covariance,
            final_solution.sigma0_post,
            state.initial_reference_values,
        )
        global_residuals = residual_summary(
            final_equations,
            final_residuals,
            final_standardized,
            final_solution.observation_weights,
            state.weight_factors,
        )
        summary.update(
            {
                "finalEquationCount": len(final_equations),
                "equationEvaluationCount": len(state.equation_evaluations),
                "performanceSeconds": dict(state.performance_seconds),
                "observationCount": int(final_solution.normals.obs_count),
                "parameterCount": len(final_solution.normals.parameter_names),
                "rank": final_solution.rank,
                "conditionNumber": final_solution.condition,
                "sigma0Post": final_solution.sigma0_post,
                "parameterUncertaintySigmaMultiplier": PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER,
                "pseudoObservationCount": len(state.constraints),
                "constraints": list(state.constraints),
            }
        )
        component_records = variance_component_records(
            final_equations,
            final_residuals,
            final_standardized,
            state.sigma_factors,
            state.initial_sigma_factors,
            state.weight_factors,
            final_proposed_weight_factors,
            assignments=state.observation_domain.assignments,
            components=plan.variance_components.components,
            diagnostics=state.diagnostics,
            floor_sigmas=state.floor_sigmas,
            initial_floor_sigmas=state.initial_floor_sigmas,
            accuracy_screening_groups=state.observation_domain.accuracy_groups,
            utc_offset_hours=state.utc_offset_hours,
        )
        observations = observation_records(
            final_equations,
            current_state_residuals,
            final_residuals,
            final_residual_sigmas,
            final_standardized,
            state.sigma_factors,
            state.weight_factors,
            final_proposed_weight_factors,
            assignments=state.observation_domain.assignments,
            parametrization=state.parametrization,
            components=plan.variance_components.components,
            accuracy_screening_records=state.observation_domain.accuracy_records,
            utc_offset_hours=state.utc_offset_hours,
            floor_sigmas=state.floor_sigmas,
        )
    return LlrAdjustmentResult(
        converged=state.converged,
        termination_reason=state.termination_reason,
        settings={
            "observationScreening": {
                "residual": asdict(plan.residual),
                "reportedSigma": asdict(plan.accuracy_screening),
            },
            "estimate": asdict(plan.adjustment),
            "robustWeighting": asdict(plan.robust_weights),
            "warmStartedSigmaFactorCount": state.warm_sigma_count,
            "warmStartedWeightFactorCount": state.warm_weight_count,
        },
        equation_evaluations=list(state.equation_evaluations),
        state=state.parametrization.state(),
        gross_rejected=dict(state.observation_domain.gross_rejected),
        accuracy_screening={
            "action": "reject",
            "minimumOneWayM": plan.accuracy_screening.minimumOneWayM,
            "minimumFractionOfGroupMedian": plan.accuracy_screening.minimumFractionOfGroupMedian,
            "rejectedCount": len(accuracy_rejected),
            "rejectedObservations": accuracy_rejected,
            "groups": dict(state.observation_domain.accuracy_groups),
        },
        sigma_factors=dict(state.sigma_factors),
        floor_sigmas=dict(state.floor_sigmas),
        weight_factors=dict(state.weight_factors),
        sigma_weight_iterations=state.iterations,
        adjustment_iterations=state.adjustment_iterations,
        summary=summary,
        observation_domain=state.observation_domain,
        parameters=final_parameter_records,
        global_residuals=global_residuals,
        variance_components=component_records,
        observations=observations,
        normals=final_solution.normals,
        remaining_correction=final_solution.delta,
        cofactor=final_solution.covariance,
        sigma0_post=final_solution.sigma0_post,
        earth_orientation=list(state.earth_orientation),
        session_sigmas=dict(state.session_sigmas),
    )
