"""Minimal linearized observation equations used by estimation."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from lunarops.classes.time import Epoch, TimeScale


class ObservationResultDetail(str, Enum):
    STANDARD = "standard"
    FULL = "full"

    @classmethod
    def parse(cls, value: object) -> ObservationResultDetail:
        if isinstance(value, cls):
            return value
        candidate = cls.STANDARD.value if value is None else str(value).strip().lower()
        try:
            return cls(candidate)
        except ValueError as exc:
            raise ValueError(f"Unknown observation result detail {value!r}.") from exc


STANDARD_OUTPUT_FIELDS = (
    "obs_time_utc",
    "normal_point_index",
    "station_id",
    "station_name",
    "reflector_id",
    "reflector_name",
    "observed_rtt_s",
    "computed_rtt_s",
    "oc_one_way_m",
    "observation_sigma_one_way_m",
    "elevation_up_deg",
    "light_time_converged",
    "status",
)

REFLECTOR_DESIGN_OUTPUT_FIELDS = (
    "design_reflector_pa_x",
    "design_reflector_pa_y",
    "design_reflector_pa_z",
)


@dataclass(frozen=True, slots=True, eq=False)
class ObservationEquation:
    observed_minus_computed_one_way_m: float
    sigma_one_way_m: float
    design_partials: Mapping[str, np.ndarray]
    observation_id: Hashable
    station_key: str
    reflector_key: str
    transmit_epoch_utc: Epoch
    light_time_converged: bool = True
    wavelength_nm: float | None = None

    def __post_init__(self) -> None:
        residual = float(self.observed_minus_computed_one_way_m)
        sigma = float(self.sigma_one_way_m)
        if not np.isfinite(residual):
            raise ValueError("observed_minus_computed_one_way_m must be finite.")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_one_way_m must be positive and finite.")
        if not isinstance(self.transmit_epoch_utc, Epoch):
            raise TypeError("transmit_epoch_utc must be an Epoch.")
        self.transmit_epoch_utc.require_scale(TimeScale.UTC, name="transmit_epoch_utc")
        try:
            hash(self.observation_id)
        except TypeError as exc:
            raise TypeError("observation_id must be hashable.") from exc
        if not isinstance(self.station_key, str) or not self.station_key:
            raise TypeError("station_key must be a non-empty string.")
        if not isinstance(self.reflector_key, str) or not self.reflector_key:
            raise TypeError("reflector_key must be a non-empty string.")
        if not isinstance(self.light_time_converged, bool):
            raise TypeError("light_time_converged must be a bool.")
        normalized: dict[str, np.ndarray] = {}
        for name, values in self.design_partials.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Design-partial block names must be non-empty strings.")
            array = np.array(values, dtype=float, copy=True).reshape(-1)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Design-partial block {name!r} contains non-finite values.")
            array.setflags(write=False)
            normalized[name] = array
        wavelength = self.wavelength_nm
        if wavelength is not None:
            wavelength = float(wavelength)
            if not np.isfinite(wavelength) or wavelength <= 0.0:
                raise ValueError("wavelength_nm must be positive and finite.")
        object.__setattr__(self, "observed_minus_computed_one_way_m", residual)
        object.__setattr__(self, "sigma_one_way_m", sigma)
        object.__setattr__(self, "design_partials", normalized)
        object.__setattr__(self, "wavelength_nm", wavelength)


__all__ = [
    "REFLECTOR_DESIGN_OUTPUT_FIELDS",
    "STANDARD_OUTPUT_FIELDS",
    "ObservationEquation",
    "ObservationResultDetail",
]
