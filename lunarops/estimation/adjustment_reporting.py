"""Pure report and diagnostic assembly for nonlinear LLR adjustment."""

from __future__ import annotations

from typing import Hashable, Mapping, Optional, Sequence, cast

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.normal_equations import NormalEquations
from lunarops.estimation.normal_equation_solver import normal_matrix_condition, normal_matrix_rank
from lunarops.estimation.uncertainty_conventions import PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition

ObsKey = Hashable


def weight_factor_summary(
    equations: Sequence[ObservationEquation],
    factors: Mapping[ObsKey, float],
    *,
    active_threshold: float,
) -> dict[str, object]:
    values = np.asarray([factors[eq.observation_id] for eq in equations], dtype=float)
    if not len(values):
        return {
            "observation_count": 0,
            "full_weight_count": 0,
            "downweighted_count": 0,
            "rejected_count": 0,
        }
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Weight factors must be finite and in [0, 1].")
    active = values > active_threshold
    full = values == 1.0
    return {
        "observation_count": int(len(values)),
        "full_weight_count": int(np.count_nonzero(full)),
        "downweighted_count": int(np.count_nonzero(active & ~full)),
        "rejected_count": int(np.count_nonzero(~active)),
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def distribution_summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    if not np.all(np.isfinite(values)):
        raise ValueError("Reported distributions require finite values.")
    median = float(np.median(values))
    absolute = np.abs(values)
    return {
        "count": int(len(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "median": median,
        "mad": 1.4826 * float(np.median(np.abs(values - median))),
        "absolute_p50": float(np.quantile(absolute, 0.50)),
        "absolute_p90": float(np.quantile(absolute, 0.90)),
        "absolute_p95": float(np.quantile(absolute, 0.95)),
        "absolute_p99": float(np.quantile(absolute, 0.99)),
        "absolute_maximum": float(np.max(absolute)),
    }


def residual_summary(
    equations: Sequence[ObservationEquation],
    residual_by_identity: Mapping[ObsKey, float],
    standardized: Mapping[ObsKey, float],
    weights: np.ndarray,
    factors: Mapping[ObsKey, float],
    *,
    active_threshold: float,
) -> dict[str, object]:
    residuals = np.asarray([residual_by_identity[eq.observation_id] for eq in equations], dtype=float)
    standards = np.asarray([standardized[eq.observation_id] for eq in equations], dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != residuals.shape:
        raise ValueError("Residual weights must match the equation count.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Residual weights must be finite and non-negative.")
    weight_sum = float(np.sum(weights))
    return {
        "residual_m": distribution_summary(residuals),
        "standardized_residual": distribution_summary(standards),
        "equivalent_weighted_rms_m": (
            None if weight_sum <= 0.0 else float(np.sqrt(np.sum(weights * residuals**2) / weight_sum))
        ),
        "weight_factors": weight_factor_summary(equations, factors, active_threshold=active_threshold),
    }


def parameter_records(
    normals: NormalEquations,
    delta: np.ndarray,
    names: Sequence[ParameterName],
    covariance: np.ndarray,
    sigma0_post: Optional[float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (len(names),):
        raise ValueError("Parameter correction length does not match parameter names.")
    if not np.all(np.isfinite(delta)):
        raise ValueError("Parameter corrections must be finite.")
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (len(names), len(names)):
        raise ValueError("Parameter covariance shape does not match parameter names.")
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
                "remaining_linearized_correction_m": float(delta[index]),
                "cofactor_uncertainty_m": float(PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER * cofactor_one_sigma[index]),
                "formal_uncertainty_m": (
                    None
                    if sigma0_post is None
                    else float(PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER * sigma0_post * cofactor_one_sigma[index])
                ),
                "maximum_absolute_correlation": (
                    None if correlated_index is None else float(abs(correlations[index, correlated_index]))
                ),
                "maximum_correlated_parameter": (None if correlated_index is None else str(names[correlated_index])),
            }
        )
    return records, {
        "observation_count": int(normals.obs_count),
        "parameter_count": len(names),
        "rank": normal_matrix_rank(normals),
        "condition_number": normal_matrix_condition(normals),
        "sigma0_post": sigma0_post,
        "parameter_uncertainty_sigma_multiplier": (PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER),
    }


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
    active_threshold: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for component in components:
        selected = [equation for equation in equations if assignments[equation.observation_id] == component.id]
        residual_values = np.asarray([residuals[equation.observation_id] for equation in selected], dtype=float)
        standardized_values = np.asarray([standardized[equation.observation_id] for equation in selected], dtype=float)
        factor_summary = weight_factor_summary(selected, weight_factors, active_threshold=active_threshold)
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
                "component_id": component.id,
                "configured_start": component.start,
                "configured_end": component.end_exclusive or "present",
                "actual_start_epoch": (
                    min(equation.transmit_epoch_utc for equation in selected).isot() if selected else None
                ),
                "actual_end_epoch": (
                    max(equation.transmit_epoch_utc for equation in selected).isot() if selected else None
                ),
                "proposed_sigma_factor_applied": False,
                "observation_count": len(selected),
                "retained_observation_count": sum(value == component.id for value in assignments.values()),
                "initial_sigma_factor": float(initial_sigma_factors[component.id]),
                "final_sigma_factor": float(sigma_factors[component.id]),
                "variance_factor": float(sigma_factors[component.id] ** 2),
                "accuracy_screening": dict(accuracy_screening_groups[component.id]),
                "residual_rms_m": (float(np.sqrt(np.mean(residual_values**2))) if len(residual_values) else None),
                "standardized_rms": (
                    float(np.sqrt(np.mean(standardized_values**2))) if len(standardized_values) else None
                ),
                "median_standardized_residual": (
                    float(np.median(standardized_values)) if len(standardized_values) else None
                ),
                "mad_standardized_residual": distribution_summary(standardized_values).get("mad"),
                "residual_wrms_m": (
                    None if weight_sum <= 0.0 else float(np.sqrt(np.sum(weights * residual_values**2) / weight_sum))
                ),
                "residual_summary_m": distribution_summary(residual_values),
                "standardized_residual_summary": distribution_summary(standardized_values),
                "weight_factor_summary": factor_summary,
                "final_state_proposed_weight_factor_summary": weight_factor_summary(
                    selected,
                    proposed_weight_factors,
                    active_threshold=active_threshold,
                ),
                "full_weight_count": factor_summary["full_weight_count"],
                "downweighted_count": factor_summary["downweighted_count"],
                "rejected_count": factor_summary["rejected_count"],
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
    active_threshold: float,
) -> list[dict[str, object]]:
    stations = {component.id: component.station for component in components}
    records: list[dict[str, object]] = []
    for equation in equations:
        factor = float(weight_factors[equation.observation_id])
        proposed_factor = float(proposed_weight_factors[equation.observation_id])
        component_id = assignments[equation.observation_id]
        matched_parameters = parametrization.matched_parameter_names(equation)
        status = "REJECTED" if factor <= active_threshold else ("FULL_WEIGHT" if factor == 1.0 else "DOWNWEIGHTED")
        proposed_status = (
            "REJECTED"
            if proposed_factor <= active_threshold
            else ("FULL_WEIGHT" if proposed_factor == 1.0 else "DOWNWEIGHTED")
        )
        screening = accuracy_screening_records[equation.observation_id]
        sigma_factor = float(sigma_factors[component_id])
        base_variance = sigma_factor**2 * equation.sigma_one_way_m**2
        records.append(
            {
                "observation_id": str(equation.observation_id),
                "epoch": equation.transmit_epoch_utc.isot(),
                "station_id": equation.station_key,
                "station": stations[component_id],
                "variance_component_id": component_id,
                "apriori_sigma_m": cast(float, screening["reported_sigma_m"]),
                "accuracy_screening_status": screening["status"],
                "accuracy_screening_reason": screening["reason"],
                "minimum_valid_sigma_m": cast(float, screening["minimum_valid_sigma_m"]),
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
    "residual_summary",
    "weight_factor_summary",
    "variance_component_records",
]
