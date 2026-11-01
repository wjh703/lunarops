"""Minimal linearized observation equations used by estimation."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from lunarops.classes.time import Epoch


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
    "obs_time_local",
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
    troposphere_clamped: bool = False

__all__ = [
    "REFLECTOR_DESIGN_OUTPUT_FIELDS",
    "STANDARD_OUTPUT_FIELDS",
    "ObservationEquation",
    "ObservationResultDetail",
]
