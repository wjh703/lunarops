"""Observation preprocessing for nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import cast

import numpy as np

from lunarops.base.station_identity import validate_station_id
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.adjustment_plan import (
    AccuracyScreeningSettings,
    ResidualScreening,
    VarianceComponentSettings,
)
from lunarops.estimation.adjustment_result_models import LlrAdjustmentObservationDomain
from lunarops.estimation.variance_component_groups import assign_variance_components

ObsKey = Hashable


def prefit_gross_threshold(
    equation: ObservationEquation,
    default: float | None,
    by_station: Mapping[str, float | None] | None,
) -> float | None:
    overrides = by_station or {}
    station = validate_station_id(equation.station_key)
    if station in overrides:
        value = overrides[station]
        return None if value is None else float(value)
    return None if default is None else float(default)


def prefit_gross_rejections(
    equations: Sequence[ObservationEquation],
    parametrization: ParametrizationList,
    *,
    threshold_m: float | None,
    threshold_by_station_m: Mapping[str, float | None] | None,
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
    minimumOneWayM: float,
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
            float(minimumOneWayM),
            float(minimum_group_median_fraction) * median,
        )
        group_diagnostics[component_id] = {
            "medianReportedSigmaM": median,
            "minimumValidSigmaM": limit,
        }

    retained: list[ObservationEquation] = []
    records: dict[ObsKey, dict[str, object]] = {}
    for equation in equations:
        component_id = assignments[equation.observation_id]
        reported = float(equation.sigma_one_way_m)
        limit = cast(float, group_diagnostics[component_id]["minimumValidSigmaM"])
        rejected = reported < limit
        qc = {
            "componentId": component_id,
            "reportedSigmaM": reported,
            "minimumValidSigmaM": limit,
            "status": "REJECTED" if rejected else "RETAINED",
            "reason": "APRIORI_SIGMA_BELOW_VALIDITY_LIMIT" if rejected else None,
        }
        records[equation.observation_id] = qc
        if not rejected:
            retained.append(equation)

    for component_id, diagnostics in group_diagnostics.items():
        component_records = [item for item in records.values() if item["componentId"] == component_id]
        diagnostics["observationCount"] = len(component_records)
        diagnostics["rejectedCount"] = sum(item["status"] == "REJECTED" for item in component_records)
        diagnostics["retainedCount"] = sum(item["status"] == "RETAINED" for item in component_records)
    return retained, records, group_diagnostics


def screen_observations(
    equations: Sequence[ObservationEquation],
    parametrization: ParametrizationList,
    *,
    model_state: object,
    residual: ResidualScreening,
    reported_sigma: AccuracyScreeningSettings,
    variance_components: VarianceComponentSettings,
) -> LlrAdjustmentObservationDomain:
    """Create the permanent observation domain used by all estimate steps."""
    converged = [item for item in equations if item.light_time_converged]
    if not converged:
        raise ValueError("screenObservations has no light-time-converged observations.")

    parametrization.setup(converged, model_state)
    gross_rejected = prefit_gross_rejections(
        converged,
        parametrization,
        threshold_m=residual.maximumAbsoluteM,
        threshold_by_station_m=residual.maximumAbsoluteByStationM,
    )
    gross_retained = [item for item in converged if item.observation_id not in gross_rejected]
    assignments = assign_variance_components(gross_retained, variance_components.components)
    retained, sigma_records, sigma_groups = reject_implausible_apriori_accuracies(
        gross_retained,
        assignments,
        minimumOneWayM=reported_sigma.minimumOneWayM,
        minimum_group_median_fraction=reported_sigma.minimumFractionOfGroupMedian,
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
    "prefit_gross_rejections",
    "prefit_gross_threshold",
    "reject_implausible_apriori_accuracies",
    "screen_observations",
]
