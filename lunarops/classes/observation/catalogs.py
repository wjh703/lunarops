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
    """Reference station coordinates with a linear secular velocity model.

    Attributes
    ----------
    name : str
        Station name; surrounding whitespace is stripped and it must not be empty.
    itrf_xyz_m : ndarray, shape (3,)
        ITRF reference position (X, Y, Z) at ``position_epoch_utc``, in metres.
    itrf_velocity_m_per_year : ndarray, shape (3,)
        ITRF velocity (X, Y, Z), in metres per Julian year. Defaults to zero.
    position_epoch_utc : str
        Reference epoch of ``itrf_xyz_m`` as an ISO timestamp in UTC.

    Raises
    ------
    ValueError
        If ``name`` is empty or a coordinate vector does not have three finite components.
    """

    name: str
    itrf_xyz_m: ArrayLike
    itrf_velocity_m_per_year: ArrayLike = (0.0, 0.0, 0.0)
    position_epoch_utc: str = "2010-01-01T00:00:00"

    def __post_init__(self) -> None:
        """Normalize the name, coordinate vectors and reference epoch."""
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

    def years_since_position_epoch(self, obstime_utc: Epoch) -> float:
        """Return the Julian years elapsed from the reference position epoch to ``obstime_utc``.

        Parameters
        ----------
        obstime_utc : Epoch
            Epoch in UTC.

        Returns
        -------
        float
            Elapsed time in Julian years of 365.25 days of 86400 s, negative before the reference epoch.
        """
        return self._position_epoch.seconds_until(obstime_utc) / (365.25 * SECONDS_PER_DAY)

    def itrf_xyz_at(self, obstime_utc: Epoch) -> np.ndarray:
        """Return the ITRF reference position propagated linearly to ``obstime_utc``.

        The position is ``XYZ(t) = XYZ0 + V * (t - t0)`` with ``t - t0`` in Julian years. Station
        displacements (tides, loading) are not included.

        Parameters
        ----------
        obstime_utc : Epoch
            Epoch in UTC.

        Returns
        -------
        ndarray, shape (3,)
            ITRF position (X, Y, Z) in metres.
        """
        years = self.years_since_position_epoch(obstime_utc)
        return np.asarray(self.itrf_xyz_m, dtype=float) + years * np.asarray(self.itrf_velocity_m_per_year, dtype=float)


@dataclass(eq=False, repr=False)
class ReflectorRecord:
    """Reflector coordinates in the Moon principal-axis (PA) frame.

    Attributes
    ----------
    name : str
        Reflector name; surrounding whitespace is stripped and it must not be empty.
    moon_fixed_xyz_m : ndarray, shape (3,)
        Reference reflector position (x, y, z) in the Moon PA frame, in metres.

    Raises
    ------
    ValueError
        If ``name`` is empty or the position does not have three finite components.
    """

    name: str
    moon_fixed_xyz_m: ArrayLike

    def __post_init__(self) -> None:
        """Normalize the name and position vector."""
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("Reflector catalog names must not be empty.")
        self.moon_fixed_xyz_m = vector3(
            self.moon_fixed_xyz_m,
            name="reflector.moon_fixed_xyz_m",
        )


def resolve_catalog_key(value: str, catalog: Mapping[str, object], label: str) -> str:
    """Return ``value`` after checking that it is a key of ``catalog``.

    Parameters
    ----------
    value : str
        Catalog key to look up; it is matched exactly.
    catalog : Mapping[str, object]
        Catalog mapping keys to records.
    label : str
        Name of the catalog kind used in the error message, e.g. ``"Station"``.

    Returns
    -------
    str
        The unchanged key.

    Raises
    ------
    KeyError
        If ``value`` is not a non-empty string present in ``catalog``.
    """
    if not isinstance(value, str) or not value or value not in catalog:
        raise KeyError(f"{label} {value!r} not found in catalog.")
    return value


__all__ = [
    "ReflectorRecord",
    "StationRecord",
    "resolve_catalog_key",
]
