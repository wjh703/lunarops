"""Variance-component groups and observation assignment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lunarops.base.station_interval import StationInterval
from lunarops.classes.observation.equations import EquationBatch
from lunarops.estimation.adjustment_result_models import ObsKey


@dataclass(frozen=True)
class VarianceComponentDefinition:
    id: str
    interval: StationInterval
    wavelength_min_nm: float | None = None
    wavelength_max_exclusive_nm: float | None = None



def assign_variance_components(
    equations: EquationBatch, components: Sequence[VarianceComponentDefinition]
) -> dict[ObsKey, str]:
    identities = equations.observation_id.tolist()
    if len(set(identities)) != len(identities):
        raise ValueError("Observation identities are not unique.")
    days = equations.dates
    assigned = np.empty(len(equations), dtype=object)
    matches: list[list[str]] = [[] for _ in identities]
    for component in components:
        mask = component.interval.contains(equations.station, days)
        if component.wavelength_min_nm is not None or component.wavelength_max_exclusive_nm is not None:
            available = equations.wavelength_nm != None
            mask &= available
            wavelengths = np.asarray(equations.wavelength_nm[available], dtype=float)
            if component.wavelength_min_nm is not None:
                mask[available] &= wavelengths >= component.wavelength_min_nm
            if component.wavelength_max_exclusive_nm is not None:
                mask[available] &= wavelengths < component.wavelength_max_exclusive_nm
        assigned[mask] = component.id
        for index in np.flatnonzero(mask):
            matches[index].append(component.id)
    for identity, day, found in zip(identities, days, matches):
        if len(found) != 1:
            detail = "no matching component" if not found else f"multiple matching components {found!r}"
            raise ValueError(f"Observation {identity!r} at {day} has {detail}.")
    return dict(zip(identities, assigned.tolist()))


__all__ = ["VarianceComponentDefinition", "assign_variance_components"]
