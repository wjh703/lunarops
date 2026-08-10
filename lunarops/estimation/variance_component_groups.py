"""Variance-component groups and observation assignment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Hashable, Mapping, Optional, Sequence

from lunarops.base.station_identity import canonical_station_id
from lunarops.classes.observation.equations import ObservationEquation

ObsKey = Hashable


_COMPONENT_CONFIG_KEYS = {
    "endExclusive",
    "id",
    "start",
    "station",
    "wavelengthMaxExclusiveNm",
    "wavelengthMinNm",
}


def _date_text(value: object, field: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Variance component {field} must be YYYY-MM-DD.") from exc


def _optional_wavelength(value: object, field: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Variance component {field} must be a number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"Variance component {field} must be finite and positive.")
    return result


@dataclass(frozen=True)
class VarianceComponentDefinition:
    id: str
    station: str
    start: str
    end_exclusive: Optional[str]
    wavelength_min_nm: Optional[float] = None
    wavelength_max_exclusive_nm: Optional[float] = None

    def __post_init__(self) -> None:
        component_id = self.id.strip() if isinstance(self.id, str) else None
        if not component_id:
            raise ValueError("Variance component id must be a non-empty string.")
        if not isinstance(self.station, str):
            raise TypeError("Variance component station must be a string.")
        station = canonical_station_id(self.station)
        start = _date_text(self.start, "start")
        end = None if self.end_exclusive is None else _date_text(self.end_exclusive, "endExclusive")
        if end is not None and end <= start:
            raise ValueError(f"Variance component {component_id!r} endExclusive must be after start.")
        wavelength_min = _optional_wavelength(self.wavelength_min_nm, "wavelengthMinNm")
        wavelength_max = _optional_wavelength(self.wavelength_max_exclusive_nm, "wavelengthMaxExclusiveNm")
        if wavelength_min is not None and wavelength_max is not None and wavelength_min >= wavelength_max:
            raise ValueError(f"Variance component {component_id!r} wavelength range is empty.")
        object.__setattr__(self, "id", component_id)
        object.__setattr__(self, "station", station)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end_exclusive", end)
        object.__setattr__(self, "wavelength_min_nm", wavelength_min)
        object.__setattr__(self, "wavelength_max_exclusive_nm", wavelength_max)

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> "VarianceComponentDefinition":
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise TypeError("Variance-component keys must be strings.")
        unknown = set(value) - _COMPONENT_CONFIG_KEYS
        if unknown:
            raise ValueError(f"Variance component: unknown key(s) {sorted(unknown)}.")
        component_id_value = value.get("id")
        station_value = value.get("station")
        if not isinstance(component_id_value, str):
            raise TypeError("Variance component id must be a string.")
        if not isinstance(station_value, str):
            raise TypeError("Variance component station must be a string.")
        component_id = component_id_value.strip()
        station = canonical_station_id(station_value)
        start_value = value.get("start")
        start = "" if start_value is None else _date_text(start_value, "start")
        if not component_id or not start:
            raise ValueError("Each variance component requires id, station, and start.")
        end_value = value.get("endExclusive")
        end = None if end_value is None else _date_text(end_value, "endExclusive")
        if end is not None and end <= start:
            raise ValueError(f"Variance component {component_id!r} endExclusive must be after start.")
        wavelength_min = _optional_wavelength(value.get("wavelengthMinNm"), "wavelengthMinNm")
        wavelength_max = _optional_wavelength(value.get("wavelengthMaxExclusiveNm"), "wavelengthMaxExclusiveNm")
        if wavelength_min is not None and wavelength_max is not None and wavelength_min >= wavelength_max:
            raise ValueError(f"Variance component {component_id!r} wavelength range is empty.")
        component = cls(
            id=component_id,
            station=station,
            start=start,
            end_exclusive=end,
            wavelength_min_nm=wavelength_min,
            wavelength_max_exclusive_nm=wavelength_max,
        )
        return component

    def matches(self, equation: ObservationEquation) -> bool:
        date = equation.transmit_epoch_utc.date_iso()
        if date < self.start or (self.end_exclusive is not None and date >= self.end_exclusive):
            return False
        if canonical_station_id(equation.station_key) != self.station:
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
    if not components:
        raise ValueError("At least one variance component is required.")
    if not all(isinstance(component, VarianceComponentDefinition) for component in components):
        raise TypeError("Variance components must be VarianceComponentDefinition instances.")
    component_ids = [component.id for component in components]
    if len(set(component_ids)) != len(component_ids):
        raise ValueError("Variance-component IDs must be unique.")
    assignments: dict[ObsKey, str] = {}
    for equation in equations:
        if not isinstance(equation, ObservationEquation):
            raise TypeError("Variance-component assignment requires ObservationEquation objects.")
        if equation.observation_id in assignments:
            raise ValueError(f"Observation identity {equation.observation_id!r} is not unique.")
        matches = [component.id for component in components if component.matches(equation)]
        if len(matches) != 1:
            detail = "no matching component" if not matches else f"multiple matching components {matches!r}"
            raise ValueError(
                f"Observation {equation.observation_id!r} at {equation.transmit_epoch_utc.date_iso()} has {detail}."
            )
        assignments[equation.observation_id] = matches[0]
    return assignments


__all__ = ["VarianceComponentDefinition", "assign_variance_components"]
