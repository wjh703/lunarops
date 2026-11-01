"""Variance-component groups and observation assignment."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from datetime import date

from lunarops.base.station_interval import StationInterval
from lunarops.classes.observation.equations import ObservationEquation

ObsKey = Hashable


@dataclass(frozen=True)
class VarianceComponentDefinition:
    id: str
    interval: StationInterval
    wavelength_min_nm: float | None = None
    wavelength_max_exclusive_nm: float | None = None

    def matches(self, equation: ObservationEquation, day: date) -> bool:
        if not self.interval.contains(equation.station_key, day):
            return False
        wavelength = equation.wavelength_nm
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
    equations: Sequence[ObservationEquation], components: Sequence[VarianceComponentDefinition]
) -> dict[ObsKey, str]:
    by_station: dict[str, list[VarianceComponentDefinition]] = {}
    for component in components:
        by_station.setdefault(component.interval.station, []).append(component)
    assignments: dict[ObsKey, str] = {}
    for equation in equations:
        if equation.observation_id in assignments:
            raise ValueError(f"Observation identity {equation.observation_id!r} is not unique.")
        day = StationInterval.parse(equation.transmit_epoch_utc.date_iso())
        matches = [
            component.id for component in by_station.get(equation.station_key, ()) if component.matches(equation, day)
        ]
        if len(matches) != 1:
            detail = "no matching component" if not matches else f"multiple matching components {matches!r}"
            raise ValueError(f"Observation {equation.observation_id!r} at {day} has {detail}.")
        assignments[equation.observation_id] = matches[0]
    return assignments


__all__ = ["VarianceComponentDefinition", "assign_variance_components"]
