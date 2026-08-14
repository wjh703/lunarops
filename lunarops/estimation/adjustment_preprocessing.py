"""Observation preprocessing for nonlinear LLR adjustment."""

from __future__ import annotations

from typing import Hashable, Mapping, Optional, Sequence, cast

import numpy as np

from lunarops.base.station_identity import canonical_station_id
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.adjustment_result_models import LlrAdjustmentObservationDomain
from lunarops.estimation.adjustment_settings import (
    AccuracyScreeningSettings,
    AdjustmentControlSettings,
    VarianceComponentSettings,
)
from lunarops.estimation.variance_component_groups import assign_variance_components

ObsKey = Hashable


def prefit_gross_threshold(
    equation: ObservationEquation,
    default: Optional[float],
    by_station: Optional[Mapping[str, Optional[float]]],
) -> Optional[float]:
    overrides = by_station or {}
    station = canonical_station_id(equation.station_key)
    if station in overrides:
        value = overrides[station]
        return None if value is None else float(value)
    return None if default is None else float(default)


def prefit_gross_rejections(
    equations: Sequence[ObservationEquation],
    parametrization: ParametrizationList,
    *,
    threshold_m: Optional[float],
    threshold_by_station_m: Optional[Mapping[str, Optional[float]]],
) -> dict[ObsKey, float]:
    rejected: dict[ObsKey, float] = {}
    for equation in equations:
        threshold = prefit_gross_threshold(equation, threshold_m, threshold_by_station_m)
        if threshold is None:
            continue
        residual = float(parametrization.reduced_observation(equation))
        if abs(residual) > threshold:
            rejected[equation.observation_id] = residual
    return rejected


def reject_implausible_apriori_accuracies(
    equations: Sequence[ObservationEquation],
    assignments: Mapping[ObsKey, str],
    *,
    minimum_one_way_m: float,
    minimum_group_median_fraction: float,
) -> tuple[
    list[ObservationEquation],
    dict[ObsKey, dict[str, object]],
    dict[str, dict[str, object]],
]:
    """Permanently reject reported accuracies below their group validity limit."""
    grouped_sigmas: dict[str, list[float]] = {}
    for equation in equations:
        component_id = assignments[equation.observation_id]
        grouped_sigmas.setdefault(component_id, []).append(float(equation.sigma_one_way_m))

    group_diagnostics: dict[str, dict[str, object]] = {}
    for component_id, values in grouped_sigmas.items():
        median = float(np.median(np.asarray(values, dtype=float)))
        limit = max(
            float(minimum_one_way_m),
            float(minimum_group_median_fraction) * median,
        )
        group_diagnostics[component_id] = {
            "median_reported_sigma_m": median,
            "minimum_valid_sigma_m": limit,
        }

    retained: list[ObservationEquation] = []
    records: dict[ObsKey, dict[str, object]] = {}
    for equation in equations:
        component_id = assignments[equation.observation_id]
        reported = float(equation.sigma_one_way_m)
        limit = cast(float, group_diagnostics[component_id]["minimum_valid_sigma_m"])
        rejected = reported < limit
        qc = {
            "component_id": component_id,
            "reported_sigma_m": reported,
            "minimum_valid_sigma_m": limit,
            "status": "REJECTED" if rejected else "RETAINED",
            "reason": "APRIORI_SIGMA_BELOW_VALIDITY_LIMIT" if rejected else None,
        }
        records[equation.observation_id] = qc
        if not rejected:
            retained.append(equation)

    for component_id, diagnostics in group_diagnostics.items():
        component_records = [item for item in records.values() if item["component_id"] == component_id]
        diagnostics["observation_count"] = len(component_records)
        diagnostics["rejected_count"] = sum(item["status"] == "REJECTED" for item in component_records)
        diagnostics["retained_count"] = sum(item["status"] == "RETAINED" for item in component_records)
    return retained, records, group_diagnostics


def screen_observations(
    equations: Sequence[ObservationEquation],
    parametrization: ParametrizationList,
    *,
    model_state: object,
    residual: AdjustmentControlSettings,
    reported_sigma: AccuracyScreeningSettings,
    variance_components: VarianceComponentSettings,
) -> LlrAdjustmentObservationDomain:
    """Create the permanent observation domain used by all estimate steps."""
    if not isinstance(parametrization, ParametrizationList):
        raise TypeError("screenObservations parametrization must be a ParametrizationList.")
    if not isinstance(residual, AdjustmentControlSettings):
        raise TypeError("screenObservations residual settings must be AdjustmentControlSettings.")
    if not isinstance(reported_sigma, AccuracyScreeningSettings):
        raise TypeError("screenObservations reportedSigma settings must be AccuracyScreeningSettings.")
    if not isinstance(variance_components, VarianceComponentSettings):
        raise TypeError("screenObservations varianceComponents must be VarianceComponentSettings.")
    values = list(equations)
    if not all(isinstance(item, ObservationEquation) for item in values):
        raise TypeError("screenObservations requires ObservationEquation objects.")
    identities = [item.observation_id for item in values]
    if len(set(identities)) != len(identities):
        raise ValueError("Observation identities must be unique.")
    converged = [item for item in values if item.light_time_converged]
    if not converged:
        raise ValueError("screenObservations has no light-time-converged observations.")

    parametrization.setup(converged, model_state)
    gross_rejected = prefit_gross_rejections(
        converged,
        parametrization,
        threshold_m=residual.prefit_gross_threshold_m,
        threshold_by_station_m=residual.prefit_gross_threshold_by_station_m,
    )
    gross_retained = [item for item in converged if item.observation_id not in gross_rejected]
    assignments = assign_variance_components(gross_retained, variance_components.components)
    retained, sigma_records, sigma_groups = reject_implausible_apriori_accuracies(
        gross_retained,
        assignments,
        minimum_one_way_m=reported_sigma.minimum_one_way_m,
        minimum_group_median_fraction=reported_sigma.minimum_fraction_of_group_median,
    )
    if not retained:
        raise ValueError("screenObservations rejected every observation.")
    retained_keys = {item.observation_id for item in retained}
    return LlrAdjustmentObservationDomain(
        gross_rejected=gross_rejected,
        retained_keys=retained_keys,
        assignments={key: value for key, value in assignments.items() if key in retained_keys},
        accuracy_records=sigma_records,
        accuracy_groups=sigma_groups,
        observation_signatures={
            item.observation_id: (
                item.station_key,
                item.reflector_key,
                item.transmit_epoch_utc,
                item.wavelength_nm,
            )
            for item in retained
        },
    )


__all__ = [
    "reject_implausible_apriori_accuracies",
    "screen_observations",
    "prefit_gross_rejections",
    "prefit_gross_threshold",
]
