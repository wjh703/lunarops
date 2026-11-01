"""Minimal linearized observation equations used by estimation."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from lunarops.classes.time import Epoch

from .resolver import ObservationCatalogSelection


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


@dataclass(frozen=True, slots=True)
class ObservationOptions:
    station_identifier: str | None = None
    reflector_identifier: str | None = None
    min_elevation_deg: float = 0.0
    low_elevation_policy: str = "rejectClamped"
    utc_offset_hours: float = 0.0
    include_reflector_position_partials: bool = False
    include_station_position_partials: bool = False
    include_lunar_orientation_partials: bool = False
    show_progress: bool = False
    progress_description: str | None = None

    def __post_init__(self) -> None:
        if self.low_elevation_policy not in {"rejectClamped", "allowFlagged"}:
            raise ValueError("low_elevation_policy must be rejectClamped or allowFlagged")
        min_elevation = float(self.min_elevation_deg)
        if not np.isfinite(min_elevation):
            raise ValueError("min_elevation_deg must be finite.")
        object.__setattr__(self, "min_elevation_deg", min_elevation)

    @property
    def catalog_selection(self) -> ObservationCatalogSelection:
        return ObservationCatalogSelection(self.station_identifier, self.reflector_identifier)

    def with_progress(
        self,
        description: str | None,
        *,
        enabled: bool | None = None,
    ) -> ObservationOptions:
        return replace(
            self,
            progress_description=description,
            show_progress=self.show_progress if enabled is None else bool(enabled),
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
