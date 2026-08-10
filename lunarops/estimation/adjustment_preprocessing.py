"""Observation preprocessing for nonlinear LLR adjustment."""

from __future__ import annotations

from dataclasses import replace
from typing import Hashable, Mapping, Optional, Sequence, cast

import numpy as np

from lunarops.base.station_identity import canonical_station_id
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition

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


def floor_prefit_uncertainties(
    equations: Sequence[ObservationEquation],
    assignments: Mapping[ObsKey, str],
    *,
    minimum_sigma_m: float,
    minimum_group_median_fraction: float,
) -> tuple[
    list[ObservationEquation],
    dict[ObsKey, dict[str, object]],
    dict[str, dict[str, object]],
]:
    """Apply a fixed prefit sigma floor within each variance-component group."""
    grouped_sigmas: dict[str, list[float]] = {}
    for equation in equations:
        component_id = assignments[equation.observation_id]
        grouped_sigmas.setdefault(component_id, []).append(float(equation.sigma_one_way_m))

    group_diagnostics: dict[str, dict[str, object]] = {}
    for component_id, values in grouped_sigmas.items():
        median = float(np.median(np.asarray(values, dtype=float)))
        floor = max(
            float(minimum_sigma_m),
            float(minimum_group_median_fraction) * median,
        )
        group_diagnostics[component_id] = {
            "median_reported_sigma_m": median,
            "sigma_floor_m": floor,
        }

    adjusted: list[ObservationEquation] = []
    records: dict[ObsKey, dict[str, object]] = {}
    for equation in equations:
        component_id = assignments[equation.observation_id]
        reported = float(equation.sigma_one_way_m)
        floor = cast(float, group_diagnostics[component_id]["sigma_floor_m"])
        effective = max(reported, floor)
        floored = effective > reported
        qc = {
            "component_id": component_id,
            "reported_sigma_m": reported,
            "effective_sigma_m": effective,
            "sigma_floor_m": floor,
            "status": "FLOORED" if floored else "UNCHANGED",
            "reason": "BELOW_PREFIT_UNCERTAINTY_FLOOR" if floored else None,
        }
        records[equation.observation_id] = qc
        adjusted.append(replace(equation, sigma_one_way_m=effective))

    for component_id, diagnostics in group_diagnostics.items():
        component_records = [item for item in records.values() if item["component_id"] == component_id]
        diagnostics["observation_count"] = len(component_records)
        diagnostics["floored_count"] = sum(item["status"] == "FLOORED" for item in component_records)
    return adjusted, records, group_diagnostics


def initialize_mad_scales(
    equations: Sequence[ObservationEquation],
    parametrization: ParametrizationList,
    assignments: Mapping[ObsKey, str],
    components: Sequence[VarianceComponentDefinition],
    *,
    minimum_count: int,
    minimum_scale: float,
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for component in components:
        values = np.asarray(
            [
                parametrization.reduced_observation(eq) / eq.sigma_one_way_m
                for eq in equations
                if assignments[eq.observation_id] == component.id
            ],
            dtype=float,
        )
        if len(values) < minimum_count:
            scales[component.id] = minimum_scale
            continue
        median = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - median)))
        scales[component.id] = minimum_scale if not np.isfinite(scale) or scale <= 0.0 else max(minimum_scale, scale)
    return scales


__all__ = [
    "floor_prefit_uncertainties",
    "initialize_mad_scales",
    "prefit_gross_rejections",
    "prefit_gross_threshold",
]
