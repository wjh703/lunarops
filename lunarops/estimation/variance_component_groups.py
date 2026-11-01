"""Variance-component groups and observation assignment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from lunarops.base.station_interval import StationInterval
from lunarops.classes.observation.equations import EquationBatch
from lunarops.estimation.adjustment_result_models import ObsKey


@dataclass(frozen=True)
class VarianceComponentDefinition:
    id: str
    interval: StationInterval
    wavelength_min_nm: float | None = None
    wavelength_max_exclusive_nm: float | None = None

    def matches(self, equation: EquationBatch, day: date) -> bool:
        if not self.interval.contains(equation.station[0], day):
            return False
        wavelength = equation.wavelength_nm[0]
        if self.wavelength_min_nm is not None or self.wavelength_max_exclusive_nm is not None:
            if wavelength is None:
                return False
            wavelength = float(wavelength)
            if self.wavelength_min_nm is not None and wavelength < self.wavelength_min_nm:
                return False
            if self.wavelength_max_exclusive_nm is not None and wavelength >= self.wavelength_max_exclusive_nm:
                return False
        return True


def assign_variance_components(
    equations: Sequence[EquationBatch], components: Sequence[VarianceComponentDefinition]
) -> dict[ObsKey, str]:
    by_station: dict[str, list[VarianceComponentDefinition]] = {}
    for component in components:
        by_station.setdefault(component.interval.station, []).append(component)
    assignments: dict[ObsKey, str] = {}
    for equation in equations:
        if equation.observation_id[0] in assignments:
            raise ValueError(f"Observation identity {equation.observation_id[0]!r} is not unique.")
        day = StationInterval.parse(equation.epoch(0).date_iso())
        matches = [
            component.id for component in by_station.get(equation.station[0], ()) if component.matches(equation, day)
        ]
        if len(matches) != 1:
            detail = "no matching component" if not matches else f"multiple matching components {matches!r}"
            raise ValueError(f"Observation {equation.observation_id[0]!r} at {day} has {detail}.")
        assignments[equation.observation_id[0]] = matches[0]
    return assignments


__all__ = ["VarianceComponentDefinition", "assign_variance_components"]
