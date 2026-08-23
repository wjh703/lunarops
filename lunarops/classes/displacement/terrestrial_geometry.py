"""Small coordinate helpers used by displacement and topocentric models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.array_validation import vector3
from lunarops.classes.frames.constants import (
    WGS84_FIRST_ECCENTRICITY_SQUARED,
    WGS84_FLATTENING,
    WGS84_SEMI_MAJOR_AXIS_M,
)


@dataclass(frozen=True, slots=True)
class GeodeticPosition:
    latitude_rad: float
    longitude_rad: float
    ellipsoidal_height_m: float

    @property
    def latitude_deg(self) -> float:
        return float(np.rad2deg(self.latitude_rad))

    @property
    def longitude_deg(self) -> float:
        return float(np.rad2deg(self.longitude_rad))


def enu2itrf(
    enu_m: ArrayLike,
    *,
    latitude_rad: float,
    longitude_rad: float,
) -> np.ndarray:
    """Rotate a local east/north/up displacement into ITRF XYZ."""
    east_m, north_m, up_m = vector3(enu_m, name="enu_m")
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
    """Rotate an ITRF vector into local east/north/up components."""
    x_m, y_m, z_m = vector3(itrf_m, name="itrf_m")
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
    """Convert ITRF XYZ metres to WGS84 geodetic coordinates.

    The iteration is Bowring-style and converges well for terrestrial stations;
    it avoids the previous Astropy/EarthLocation dependency in the hot path.
    """
    x_m, y_m, z_m = vector3(station_itrf_m, name="station_itrf_m")
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
    """Convert WGS84 geodetic latitude, longitude, and height to ITRF XYZ."""
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
    """Return geocentric latitude and east longitude in radians."""
    x_m, y_m, z_m = vector3(station_itrf_m, name="station_itrf_m")
    latitude_rad = float(np.arctan2(z_m, np.hypot(x_m, y_m)))
    longitude_rad = float(np.arctan2(y_m, x_m))
    return latitude_rad, longitude_rad


def local_up_unit_itrf(station_itrf_m: ArrayLike) -> np.ndarray:
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
