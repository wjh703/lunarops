"""Domain models and identity resolution for station and reflector catalogs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.array_validation import catalog_vector3
from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.displacement.terrestrial_geometry import GeodeticPosition, itrf2geodetic


@dataclass(eq=False, repr=False)
class StationRecord:
    """Reference station coordinates and a linear secular velocity model."""

    name: str
    itrf_xyz_m: ArrayLike
    aliases: Sequence[str] = field(default_factory=tuple)
    itrf_velocity_m_per_year: ArrayLike = (0.0, 0.0, 0.0)
    position_epoch_utc: str = "2010-01-01T00:00:00"

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("Station catalog names must not be empty.")
        self.itrf_xyz_m = catalog_vector3(self.itrf_xyz_m, name="station.itrf_xyz_m")
        self.itrf_velocity_m_per_year = catalog_vector3(
            self.itrf_velocity_m_per_year,
            name="station.itrf_velocity_m_per_year",
        )
        self.aliases = tuple(str(alias).strip() for alias in self.aliases)
        if any(not alias for alias in self.aliases) or len(set(self.aliases)) != len(self.aliases):
            raise ValueError("Station aliases must be non-empty and unique.")
        self.position_epoch_utc = str(self.position_epoch_utc).strip()
        Epoch.from_isot(self.position_epoch_utc, scale=TimeScale.UTC)

    def _position_epoch(self) -> Epoch:
        cached = getattr(self, "_position_epoch_cache", None)
        if cached is None:
            cached = Epoch.from_isot(self.position_epoch_utc, scale=TimeScale.UTC)
            self._position_epoch_cache = cached
        return cached

    @staticmethod
    def _utc_epoch(value: Epoch) -> Epoch:
        if not isinstance(value, Epoch):
            raise TypeError("Station catalog queries require an Epoch.")
        return value.require_scale(TimeScale.UTC, name="obstime_utc")

    def itrf_xyz_at(self, obstime_utc: Epoch) -> np.ndarray:
        """Return XYZ(t) = XYZ0 + V * (t - epoch)."""
        epoch = self._position_epoch()
        time = self._utc_epoch(obstime_utc)
        years = epoch.seconds_until(time) / (365.25 * SECONDS_PER_DAY)
        return np.asarray(self.itrf_xyz_m, dtype=float) + years * np.asarray(self.itrf_velocity_m_per_year, dtype=float)

    def geodetic_at(self, obstime_utc: Epoch) -> GeodeticPosition:
        return itrf2geodetic(self.itrf_xyz_at(obstime_utc))

    @property
    def geodetic(self) -> GeodeticPosition:
        return itrf2geodetic(self.itrf_xyz_m)

    @property
    def latitude_rad(self) -> float:
        return self.geodetic.latitude_rad

    @property
    def height_m(self) -> float:
        return self.geodetic.ellipsoidal_height_m

    def latitude_rad_at(self, obstime_utc: Epoch) -> float:
        return self.geodetic_at(obstime_utc).latitude_rad

    def height_m_at(self, obstime_utc: Epoch) -> float:
        return self.geodetic_at(obstime_utc).ellipsoidal_height_m


@dataclass(eq=False, repr=False)
class ReflectorRecord:
    """Moon-fixed reflector coordinates and aliases."""

    name: str
    moon_fixed_xyz_m: ArrayLike
    aliases: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("Reflector catalog names must not be empty.")
        self.moon_fixed_xyz_m = catalog_vector3(
            self.moon_fixed_xyz_m,
            name="reflector.moon_fixed_xyz_m",
        )
        self.aliases = tuple(str(alias).strip() for alias in self.aliases)
        if any(not alias for alias in self.aliases) or len(set(self.aliases)) != len(self.aliases):
            raise ValueError("Reflector aliases must be non-empty and unique.")


def _canonical_catalog_token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def resolve_catalog_key(value: object, catalog: Mapping[str, object], label: str) -> str:
    """Resolve exact, case-insensitive, alias, and compact catalog identifiers."""
    raw = str(value or "").strip()
    if not raw:
        raise KeyError(f"{label} name is empty and cannot be resolved.")
    if raw in catalog:
        return raw
    raw_upper = raw.upper()
    for key in catalog:
        if key.upper() == raw_upper:
            return key
    target = _canonical_catalog_token(raw)
    for key, record in catalog.items():
        tokens = {
            _canonical_catalog_token(key),
            _canonical_catalog_token(getattr(record, "name", "")),
        }
        tokens.update(_canonical_catalog_token(alias) for alias in getattr(record, "aliases", ()))
        if target in tokens:
            return key
    raise KeyError(f"{label} '{raw}' not found in catalog.")


def first_resolvable_key(candidates: Sequence[object], catalog: Mapping[str, object], label: str) -> str:
    last_error = None
    for candidate in candidates:
        if candidate is None or str(candidate).strip() == "":
            continue
        try:
            return resolve_catalog_key(candidate, catalog, label)
        except KeyError as exc:
            last_error = exc
    raise last_error or KeyError(f"{label} could not be resolved.")


__all__ = [
    "ReflectorRecord",
    "StationRecord",
    "first_resolvable_key",
    "resolve_catalog_key",
]
