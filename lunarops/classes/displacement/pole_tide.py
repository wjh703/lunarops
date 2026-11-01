"""IERS 2010 solid-Earth pole-tide displacement."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from lunarops.classes.frames.earth_orientation import EarthOrientationProvider
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.schema import ConfigSchema

from .base import StationDisplacementInput
from .terrestrial_geometry import enu2itrf, itrf2geocentric


def secular_pole_2018_arcsec(epoch_utc: Epoch) -> tuple[float, float]:
    """IERS Chapter 7 2018 update, Eq. (21), in arcseconds."""
    epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
    years_since_j2000 = (epoch_utc.jd - 2451545.0) / 365.25
    return (
        (55.0 + 1.677 * years_since_j2000) / 1000.0,
        (320.5 + 3.460 * years_since_j2000) / 1000.0,
    )


def polar_wobble(
    epoch_utc: Epoch,
    earth_orientation_provider: EarthOrientationProvider,
) -> np.ndarray:
    """Read polar motion and form the IERS wobble variables."""
    pole = earth_orientation_provider.polar_motion(epoch_utc)
    xp_arcsec = pole.xp_arcsec
    yp_arcsec = pole.yp_arcsec
    secular_x, secular_y = secular_pole_2018_arcsec(epoch_utc)
    return np.array([xp_arcsec - secular_x, -(yp_arcsec - secular_y)])


class Iers2010SolidEarthPoleTide:
    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(fields=(), type_name="iers2010PoleTide")

    @classmethod
    def from_config(cls, cfg, ctx, *, earth_orientation_provider):
        return cls(earth_orientation_provider=earth_orientation_provider)

    def __init__(self, earth_orientation_provider: EarthOrientationProvider) -> None:
        self.earth_orientation_provider = earth_orientation_provider
        self._site_cache: dict[tuple, tuple[float, float]] = {}

    def evaluate(self, data: StationDisplacementInput) -> np.ndarray:
        key = tuple(data.reference_position_itrf_m)
        if key not in self._site_cache:
            self._site_cache[key] = itrf2geocentric(data.reference_position_itrf_m)
        latitude_rad, longitude_rad = self._site_cache[key]
        theta = 0.5 * np.pi - latitude_rad
        wobble = polar_wobble(data.epoch_utc, self.earth_orientation_provider)

        sin_lon = np.sin(longitude_rad)
        cos_lon = np.cos(longitude_rad)
        common = wobble[0] * cos_lon + wobble[1] * sin_lon

        south_mm = -9.0 * np.cos(2.0 * theta) * common
        east_mm = 9.0 * np.cos(theta) * (wobble[0] * sin_lon - wobble[1] * cos_lon)
        up_mm = -33.0 * np.sin(2.0 * theta) * common

        enu_m = np.array([east_mm, -south_mm, up_mm], dtype=float) * 1.0e-3
        itrf_m = enu2itrf(
            enu_m,
            latitude_rad=latitude_rad,
            longitude_rad=longitude_rad,
        )
        return itrf_m

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        return self.evaluate(data)
