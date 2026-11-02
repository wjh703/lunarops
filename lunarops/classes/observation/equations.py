"""Minimal linearized observation equations used by estimation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property

import erfa
import numpy as np

from lunarops.classes.time import Epoch, TimeScale

from .resolver import ObservationCatalogSelection


def diagnostic_values[T](fields: Sequence[tuple[str, str, Callable[[T], object]]], values: T) -> dict[str, object]:
    return {name: getter(values) for name, _, getter in fields}


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


@dataclass(frozen=True, eq=False)
class EquationBatch:
    oc: np.ndarray
    sigma: np.ndarray
    observation_id: np.ndarray
    station_idx: np.ndarray
    reflector_idx: np.ndarray
    epoch_jd1: np.ndarray
    epoch_jd2: np.ndarray
    receive_jd1: np.ndarray
    receive_jd2: np.ndarray
    converged: np.ndarray
    wavelength_nm: np.ndarray
    clamped: np.ndarray
    reflector_position_pa: np.ndarray
    station_position_itrf: np.ndarray
    lunar_orientation_pa: np.ndarray
    lunar_ecliptic_libration: np.ndarray
    stations: np.ndarray
    reflectors: np.ndarray

    _COLUMNS = (
        "oc", "sigma", "observation_id", "epoch_jd1", "epoch_jd2", "receive_jd1", "receive_jd2",
        "converged", "wavelength_nm", "clamped", "reflector_position_pa", "station_position_itrf",
        "lunar_orientation_pa", "lunar_ecliptic_libration",
    )

    @cached_property
    def fundamental_arguments(self) -> tuple[np.ndarray, ...]:
        t = ((self.epoch_jd1 + self.epoch_jd2) - 2_451_545.0) / 36_525.0
        return tuple(function(t) for function in (erfa.fal03, erfa.falp03, erfa.faf03, erfa.fad03, erfa.faom03))

    def __len__(self) -> int:
        return len(self.oc)

    def __getitem__(self, rows):
        return EquationBatch(
            **{name: getattr(self, name)[rows] for name in self._COLUMNS},
            station_idx=self.station_idx[rows], reflector_idx=self.reflector_idx[rows],
            stations=self.stations, reflectors=self.reflectors,
        )

    @property
    def station(self) -> np.ndarray:
        return self.stations[self.station_idx]

    @property
    def reflector(self) -> np.ndarray:
        return self.reflectors[self.reflector_idx]

    def epoch(self, index: int) -> Epoch:
        return Epoch(self.epoch_jd1[index], self.epoch_jd2[index], TimeScale.UTC)

    @classmethod
    def single(
        cls, *, oc, sigma, observation_id, station, reflector, epoch, receive_epoch,
        reflector_position_pa=(), station_position_itrf=(), lunar_orientation_pa=(), lunar_ecliptic_libration=(),
        converged=True, wavelength_nm=None, clamped=False,
    ) -> EquationBatch:
        identities = np.empty(1, dtype=object)
        identities[0] = observation_id
        return cls(
            oc=np.asarray([oc], dtype=float), sigma=np.asarray([sigma], dtype=float), observation_id=identities,
            station_idx=np.zeros(1, dtype=int), reflector_idx=np.zeros(1, dtype=int),
            epoch_jd1=np.asarray([epoch.jd1]), epoch_jd2=np.asarray([epoch.jd2]),
            receive_jd1=np.asarray([receive_epoch.jd1]), receive_jd2=np.asarray([receive_epoch.jd2]),
            converged=np.asarray([converged], dtype=bool), wavelength_nm=np.asarray([wavelength_nm], dtype=object),
            clamped=np.asarray([clamped], dtype=bool),
            reflector_position_pa=np.asarray(reflector_position_pa, dtype=float).reshape(1, -1),
            station_position_itrf=np.asarray(station_position_itrf, dtype=float).reshape(1, -1),
            lunar_orientation_pa=np.asarray(lunar_orientation_pa, dtype=float).reshape(1, -1),
            lunar_ecliptic_libration=np.asarray(lunar_ecliptic_libration, dtype=float).reshape(1, -1),
            stations=np.asarray([station]), reflectors=np.asarray([reflector]),
        )

    @classmethod
    def empty(cls) -> EquationBatch:
        return cls(
            oc=np.empty(0), sigma=np.empty(0), observation_id=np.empty(0, dtype=object),
            station_idx=np.empty(0, dtype=int), reflector_idx=np.empty(0, dtype=int),
            epoch_jd1=np.empty(0), epoch_jd2=np.empty(0), receive_jd1=np.empty(0), receive_jd2=np.empty(0),
            converged=np.empty(0, dtype=bool), wavelength_nm=np.empty(0, dtype=object), clamped=np.empty(0, dtype=bool),
            reflector_position_pa=np.empty((0, 0)), station_position_itrf=np.empty((0, 0)),
            lunar_orientation_pa=np.empty((0, 0)), lunar_ecliptic_libration=np.empty((0, 0)),
            stations=np.empty(0, dtype=str), reflectors=np.empty(0, dtype=str),
        )

    @classmethod
    def concatenate(cls, chunks) -> EquationBatch:
        chunks = [chunk for chunk in chunks if len(chunk)]
        if not chunks:
            return cls.empty()
        stations = np.unique(np.concatenate([chunk.stations for chunk in chunks]))
        reflectors = np.unique(np.concatenate([chunk.reflectors for chunk in chunks]))
        return cls(
            **{name: np.concatenate([getattr(chunk, name) for chunk in chunks]) for name in cls._COLUMNS},
            station_idx=np.concatenate([np.searchsorted(stations, chunk.station) for chunk in chunks]),
            reflector_idx=np.concatenate([np.searchsorted(reflectors, chunk.reflector) for chunk in chunks]),
            stations=stations, reflectors=reflectors,
        )

__all__ = [
    "REFLECTOR_DESIGN_OUTPUT_FIELDS",
    "STANDARD_OUTPUT_FIELDS",
    "EquationBatch",
    "ObservationResultDetail",
]
