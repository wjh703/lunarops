"""WGS84 geodetic and local east/north/up coordinate helpers for ITRF positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from lunarops.classes.frames.constants import (
    WGS84_FIRST_ECCENTRICITY_SQUARED,
    WGS84_FLATTENING,
    WGS84_SEMI_MAJOR_AXIS_M,
)


@dataclass(frozen=True, slots=True)
class GeodeticPosition:
    """WGS84 geodetic coordinates of a point.

    Attributes
    ----------
    latitude_rad : float
        Geodetic latitude, north positive, in radians.
    longitude_rad : float
        Longitude, east positive, in radians.
    ellipsoidal_height_m : float
        Height above the WGS84 ellipsoid, in metres.
    """

    latitude_rad: float
    longitude_rad: float
    ellipsoidal_height_m: float

    @property
    def latitude_deg(self) -> float:
        """Geodetic latitude, north positive, in degrees."""
        return float(np.rad2deg(self.latitude_rad))

    @property
    def longitude_deg(self) -> float:
        """Longitude, east positive, in degrees."""
        return float(np.rad2deg(self.longitude_rad))


def enu2itrf(
    enu_m: ArrayLike,
    *,
    latitude_rad: float,
    longitude_rad: float,
) -> np.ndarray:
    """Rotate a local east/north/up vector into ITRF (X, Y, Z) components.

    Parameters
    ----------
    enu_m : array_like, shape (3,)
        Vector components (east, north, up); the output has the same unit.
    latitude_rad : float
        Latitude defining the local vertical, in radians, in [-pi/2, pi/2].
    longitude_rad : float
        Longitude, east positive, in radians.

    Returns
    -------
    ndarray, shape (3,)
        ITRF components (X, Y, Z).

    Raises
    ------
    ValueError
        If an angle is not finite or the latitude is out of range.
    """
    east_m, north_m, up_m = np.asarray(enu_m, float)
    lat = float(latitude_rad)
    lon = float(longitude_rad)
    if not np.isfinite(lat):
        raise ValueError("latitude_rad must be finite.")
    if not np.isfinite(lon):
        raise ValueError("longitude_rad must be finite.")
    if not -0.5 * np.pi <= lat <= 0.5 * np.pi:
        raise ValueError("latitude_rad must be in [-pi/2, pi/2].")

    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)

    return np.array(
        [
            -sin_lon * east_m - sin_lat * cos_lon * north_m + cos_lat * cos_lon * up_m,
            cos_lon * east_m - sin_lat * sin_lon * north_m + cos_lat * sin_lon * up_m,
            cos_lat * north_m + sin_lat * up_m,
        ],
        dtype=float,
    )


def itrf2enu(
    itrf_m: ArrayLike,
    *,
    latitude_rad: float,
    longitude_rad: float,
) -> np.ndarray:
    """Rotate an ITRF vector into local east/north/up components.

    Parameters
    ----------
    itrf_m : array_like, shape (3,)
        ITRF components (X, Y, Z); the output has the same unit.
    latitude_rad : float
        Latitude defining the local vertical, in radians, in [-pi/2, pi/2].
    longitude_rad : float
        Longitude, east positive, in radians.

    Returns
    -------
    ndarray, shape (3,)
        Local components (east, north, up).

    Raises
    ------
    ValueError
        If an angle is not finite or the latitude is out of range.
    """
    x_m, y_m, z_m = np.asarray(itrf_m, float)
    lat = float(latitude_rad)
    lon = float(longitude_rad)
    if not np.isfinite(lat) or not np.isfinite(lon):
        raise ValueError("latitude_rad and longitude_rad must be finite.")
    if not -0.5 * np.pi <= lat <= 0.5 * np.pi:
        raise ValueError("latitude_rad must be in [-pi/2, pi/2].")

    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    return np.array(
        [
            -sin_lon * x_m + cos_lon * y_m,
            -sin_lat * cos_lon * x_m - sin_lat * sin_lon * y_m + cos_lat * z_m,
            cos_lat * cos_lon * x_m + cos_lat * sin_lon * y_m + sin_lat * z_m,
        ],
        dtype=float,
    )


def itrf2geodetic(station_itrf_m: ArrayLike) -> GeodeticPosition:
    """Convert ITRF Cartesian coordinates to WGS84 geodetic coordinates.

    The latitude is found by fixed-point iteration (at most eight steps, tolerance 1e-15 rad); points on
    the polar axis are handled directly.

    Parameters
    ----------
    station_itrf_m : array_like, shape (3,)
        ITRF position (X, Y, Z), in metres.

    Returns
    -------
    GeodeticPosition
        Geodetic latitude, east longitude and ellipsoidal height.
    """
    x_m, y_m, z_m = np.asarray(station_itrf_m, float)
    lon = float(np.arctan2(y_m, x_m))
    p = float(np.hypot(x_m, y_m))
    if p == 0.0:
        lat = np.pi / 2.0 if z_m >= 0.0 else -np.pi / 2.0
        height = abs(z_m) - WGS84_SEMI_MAJOR_AXIS_M * (1.0 - WGS84_FLATTENING)
        return GeodeticPosition(float(lat), lon, float(height))
    lat = float(np.arctan2(z_m, p * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED)))
    height = 0.0
    for _ in range(8):
        sin_lat = np.sin(lat)
        n = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_lat * sin_lat)
        height = p / np.cos(lat) - n
        updated = float(
            np.arctan2(
                z_m,
                p * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * n / (n + height)),
            )
        )
        if abs(updated - lat) < 1.0e-15:
            lat = updated
            break
        lat = updated
    return GeodeticPosition(
        latitude_rad=float(lat),
        longitude_rad=lon,
        ellipsoidal_height_m=float(height),
    )


def geodetic2itrf(
    *,
    latitude_rad: float,
    longitude_rad: float,
    ellipsoidal_height_m: float,
) -> np.ndarray:
    """Convert WGS84 geodetic coordinates to ITRF Cartesian coordinates.

    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude, north positive, in radians, in [-pi/2, pi/2].
    longitude_rad : float
        Longitude, east positive, in radians.
    ellipsoidal_height_m : float
        Height above the WGS84 ellipsoid, in metres.

    Returns
    -------
    ndarray, shape (3,)
        ITRF position (X, Y, Z), in metres.

    Raises
    ------
    ValueError
        If an input is not finite or the latitude is out of range.
    """
    lat = float(latitude_rad)
    lon = float(longitude_rad)
    height = float(ellipsoidal_height_m)
    if not np.isfinite(lat):
        raise ValueError("latitude_rad must be finite.")
    if not -0.5 * np.pi <= lat <= 0.5 * np.pi:
        raise ValueError("latitude_rad must be in [-pi/2, pi/2].")
    if not np.isfinite(lon):
        raise ValueError("longitude_rad must be finite.")
    if not np.isfinite(height):
        raise ValueError("ellipsoidal_height_m must be finite.")

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    prime_vertical_radius_m = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_lat * sin_lat
    )
    return np.array(
        [
            (prime_vertical_radius_m + height) * cos_lat * np.cos(lon),
            (prime_vertical_radius_m + height) * cos_lat * np.sin(lon),
            (
                prime_vertical_radius_m * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED)
                + height
            )
            * sin_lat,
        ],
        dtype=float,
    )


def itrf2geocentric(station_itrf_m: ArrayLike) -> tuple[float, float]:
    """Return the geocentric (latitude, east longitude), in radians, of an ITRF position (X, Y, Z) in metres."""
    x_m, y_m, z_m = np.asarray(station_itrf_m, float)
    latitude_rad = float(np.arctan2(z_m, np.hypot(x_m, y_m)))
    longitude_rad = float(np.arctan2(y_m, x_m))
    return latitude_rad, longitude_rad


def local_up_unit_itrf(station_itrf_m: ArrayLike) -> np.ndarray:
    """Return the ITRF unit vector, shape (3,), along the WGS84 ellipsoidal normal at an ITRF position in metres."""
    site = itrf2geodetic(station_itrf_m)
    return enu2itrf(
        [0.0, 0.0, 1.0],
        latitude_rad=site.latitude_rad,
        longitude_rad=site.longitude_rad,
    )


__all__ = [
    "GeodeticPosition",
    "enu2itrf",
    "geodetic2itrf",
    "itrf2enu",
    "itrf2geocentric",
    "itrf2geodetic",
    "local_up_unit_itrf",
]
