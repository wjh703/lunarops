"""Pure report and diagnostic assembly for nonlinear LLR adjustment."""

from __future__ import annotations

from typing import Hashable, Mapping, Optional, Sequence, cast

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.classes.time import format_time_with_utc_offset, validate_utc_offset_hours
from lunarops.estimation.normal_equations import NormalEquations
from lunarops.estimation.parameter_products import PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition

ObsKey = Hashable
_EMPTY_ACCURACY_SCREENING_GROUP = {
    "medianReportedSigmaM": None,
    "observationCount": 0,
    "rejectedCount": 0,
    "retainedCount": 0,
}


def report_state(state: Mapping[str, object]) -> dict[str, object]:
    """Return parametrization state with report field names.

    The processing-state file keeps the station-bias interval key
    ``end_exclusive``; reports name it ``endExclusive``.
    """
    result = dict(state)
    bias = cast(dict, result.get("stationRangeBias"))
    if bias is not None:
        result["stationRangeBias"] = {
            **bias,
            "intervals": [
                {("endExclusive" if key == "end_exclusive" else key): value for key, value in interval.items()}
                for interval in bias["intervals"]
            ],
        }
    return result


def weight_factor_summary(
    equations: Sequence[ObservationEquation],
    factors: Mapping[ObsKey, float],
) -> dict[str, object]:
    values = np.asarray([factors[eq.observation_id] for eq in equations], dtype=float)
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
        "observationCount": int(len(values)),
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
    if not len(values):
        return {"count": 0}
    median = float(np.median(values))
    absolute = np.abs(values)
    return {
        "count": int(len(values)),
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
    equations: Sequence[ObservationEquation],
    residual_by_identity: Mapping[ObsKey, float],
    standardized: Mapping[ObsKey, float],
    weights: np.ndarray,
    factors: Mapping[ObsKey, float],
) -> dict[str, object]:
    residuals = np.asarray([residual_by_identity[eq.observation_id] for eq in equations], dtype=float)
    standards = np.asarray([standardized[eq.observation_id] for eq in equations], dtype=float)
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
    sigma0_post: Optional[float],
    reference_values: np.ndarray,
) -> list[dict[str, object]]:
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
    equations: Sequence[ObservationEquation],
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
    accuracy_screening_groups: Mapping[str, Mapping[str, object]],
    utc_offset_hours: object = 0.0,
) -> list[dict[str, object]]:
    offset = validate_utc_offset_hours(utc_offset_hours)
    records: list[dict[str, object]] = []
    for component in components:
        selected = [equation for equation in equations if assignments[equation.observation_id] == component.id]
        residual_values = np.asarray([residuals[equation.observation_id] for equation in selected], dtype=float)
        standardized_values = np.asarray([standardized[equation.observation_id] for equation in selected], dtype=float)
        factor_summary = weight_factor_summary(selected, weight_factors)
        weights = np.asarray(
            [
                weight_factors[equation.observation_id]
                / (sigma_factors[component.id] ** 2 * equation.sigma_one_way_m**2)
                for equation in selected
            ],
            dtype=float,
        )
        weight_sum = float(np.sum(weights))
        item = dict(diagnostics.get(component.id, {}))
        item.update(
            {
                "componentId": component.id,
                "configuredStart": component.start,
                "configuredEnd": component.end_exclusive or "present",
                "actualStartEpochUtc": (
                    min(equation.transmit_epoch_utc for equation in selected).isot(precision=9) if selected else None
                ),
                "actualStartEpochLocal": (
                    format_time_with_utc_offset(
                        min(equation.transmit_epoch_utc for equation in selected),
                        utc_offset_hours=offset,
                        precision=9,
                    )
                    if selected
                    else None
                ),
                "actualEndEpochUtc": (
                    max(equation.transmit_epoch_utc for equation in selected).isot(precision=9) if selected else None
                ),
                "actualEndEpochLocal": (
                    format_time_with_utc_offset(
                        max(equation.transmit_epoch_utc for equation in selected),
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


def observation_records(
    equations: Sequence[ObservationEquation],
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
    utc_offset_hours: object = 0.0,
) -> list[dict[str, object]]:
    offset = validate_utc_offset_hours(utc_offset_hours)
    stations = {component.id: component.station for component in components}
    records: list[dict[str, object]] = []
    for equation in equations:
        factor = float(weight_factors[equation.observation_id])
        proposed_factor = float(proposed_weight_factors[equation.observation_id])
        component_id = assignments[equation.observation_id]
        matched_parameters = parametrization.matched_parameter_names(equation)
        status = "REJECTED" if factor == 0.0 else ("FULL_WEIGHT" if factor == 1.0 else "DOWNWEIGHTED")
        proposed_status = (
            "REJECTED" if proposed_factor == 0.0 else ("FULL_WEIGHT" if proposed_factor == 1.0 else "DOWNWEIGHTED")
        )
        screening = accuracy_screening_records[equation.observation_id]
        sigma_factor = float(sigma_factors[component_id])
        base_variance = sigma_factor**2 * equation.sigma_one_way_m**2
        records.append(
            {
                "observation_id": str(equation.observation_id),
                "light_time_converged": equation.light_time_converged,
                "troposphere_clamped": equation.troposphere_clamped,
                "epoch_utc": equation.transmit_epoch_utc.isot(precision=9),
                "epoch_local": format_time_with_utc_offset(
                    equation.transmit_epoch_utc,
                    utc_offset_hours=offset,
                    precision=9,
                ),
                "station_id": equation.station_key,
                "station": stations[component_id],
                "variance_component_id": component_id,
                "apriori_sigma_m": cast(float, screening["reportedSigmaM"]),
                "reported_sigma_screening_status": screening["status"],
                "reported_sigma_screening_reason": screening["reason"],
                "minimum_valid_sigma_m": cast(float, screening["minimumValidSigmaM"]),
                "sigma_factor": sigma_factor,
                "sigma0_m": float(sigma_factor * equation.sigma_one_way_m),
                "base_variance_m2": float(base_variance),
                "base_weight_per_m2": float(1.0 / base_variance),
                "current_state_residual_m": float(current_state_residuals[equation.observation_id]),
                "linearized_postfit_residual_m": float(linearized_postfit_residuals[equation.observation_id]),
                "residual_sigma_m": float(residual_sigmas[equation.observation_id]),
                "leverage": float(1.0 - residual_sigmas[equation.observation_id] ** 2 / base_variance),
                "standardized_residual": float(standardized[equation.observation_id]),
                "applied_weight_factor": factor,
                "final_state_proposed_weight_factor": proposed_factor,
                "proposed_weight_factor_applied": False,
                "equivalent_weight_per_m2": float(factor / base_variance),
                "applied_weight_status": status,
                "final_state_proposed_weight_status": proposed_status,
                "matched_parameter_names": matched_parameters,
            }
        )
    return records


__all__ = [
    "distribution_summary",
    "observation_records",
    "parameter_records",
    "report_state",
    "residual_summary",
    "weight_factor_summary",
    "variance_component_records",
]
