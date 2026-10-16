"""Domain models and identity resolution for station and reflector catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.array_validation import vector3
from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.classes.time import Epoch, TimeScale


@dataclass(eq=False, repr=False)
class StationRecord:
    """Reference station coordinates and a linear secular velocity model."""

    name: str
    itrf_xyz_m: ArrayLike
    itrf_velocity_m_per_year: ArrayLike = (0.0, 0.0, 0.0)
    position_epoch_utc: str = "2010-01-01T00:00:00"

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("Station catalog names must not be empty.")
        self.itrf_xyz_m = vector3(self.itrf_xyz_m, name="station.itrf_xyz_m")
        self.itrf_velocity_m_per_year = vector3(
            self.itrf_velocity_m_per_year,
            name="station.itrf_velocity_m_per_year",
        )
        self.position_epoch_utc = str(self.position_epoch_utc).strip()
        self._position_epoch = Epoch.from_isot(self.position_epoch_utc, scale=TimeScale.UTC)

    def itrf_xyz_at(self, obstime_utc: Epoch) -> np.ndarray:
        """Return XYZ(t) = XYZ0 + V * (t - epoch)."""
        years = self._position_epoch.seconds_until(obstime_utc) / (365.25 * SECONDS_PER_DAY)
        return np.asarray(self.itrf_xyz_m, dtype=float) + years * np.asarray(self.itrf_velocity_m_per_year, dtype=float)


@dataclass(eq=False, repr=False)
class ReflectorRecord:
    """Moon-fixed reflector coordinates."""

    name: str
    moon_fixed_xyz_m: ArrayLike

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("Reflector catalog names must not be empty.")
        self.moon_fixed_xyz_m = vector3(
            self.moon_fixed_xyz_m,
            name="reflector.moon_fixed_xyz_m",
        )


def resolve_catalog_key(value: str, catalog: Mapping[str, object], label: str) -> str:
    """Resolve the one exact catalog key supplied by the caller."""
    if not isinstance(value, str) or not value or value not in catalog:
        raise KeyError(f"{label} {value!r} not found in catalog.")
    return value


__all__ = [
    "ReflectorRecord",
    "StationRecord",
    "resolve_catalog_key",
]
