"""IERS 2010 solid-Earth pole-tide displacement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.base.array_validation import readonly_vector3
from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.frames.earth_orientation import EarthOrientationProvider

from .base import StationDisplacementInput
from .terrestrial_geometry import enu2itrf, itrf2geocentric


@dataclass(frozen=True, slots=True)
class PolarWobble:
    """Observed pole, secular pole, and resulting wobble components."""

    xp_arcsec: float
    yp_arcsec: float
    secular_x_arcsec: float
    secular_y_arcsec: float
    m1_arcsec: float
    m2_arcsec: float

    @property
    def m1_rad(self) -> float:
        return float(np.deg2rad(self.m1_arcsec / 3600.0))

    @property
    def m2_rad(self) -> float:
        return float(np.deg2rad(self.m2_arcsec / 3600.0))


@dataclass(frozen=True, slots=True, eq=False)
class PoleTideResult:
    """Typed solid-Earth pole-tide result and diagnostics."""

    displacement_itrf_m: np.ndarray
    displacement_enu_m: np.ndarray
    wobble: PolarWobble
    geocentric_latitude_rad: float
    longitude_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "displacement_itrf_m",
            readonly_vector3(self.displacement_itrf_m, name="displacement_itrf_m"),
        )
        object.__setattr__(
            self,
            "displacement_enu_m",
            readonly_vector3(self.displacement_enu_m, name="displacement_enu_m"),
        )


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
) -> PolarWobble:
    """Read polar motion and form the IERS wobble variables."""
    pole = earth_orientation_provider.polar_motion(epoch_utc)
    xp_arcsec = pole.xp_arcsec
    yp_arcsec = pole.yp_arcsec
    secular_x, secular_y = secular_pole_2018_arcsec(epoch_utc)
    return PolarWobble(
        xp_arcsec=xp_arcsec,
        yp_arcsec=yp_arcsec,
        secular_x_arcsec=secular_x,
        secular_y_arcsec=secular_y,
        m1_arcsec=xp_arcsec - secular_x,
        m2_arcsec=-(yp_arcsec - secular_y),
    )


class Iers2010SolidEarthPoleTide:
    """Solid-Earth pole tide using an explicitly supplied IERS table."""

    def __init__(self, earth_orientation_provider: EarthOrientationProvider) -> None:
        if not isinstance(earth_orientation_provider, EarthOrientationProvider):
            raise TypeError("earth_orientation_provider must be an EarthOrientationProvider instance.")
        self.earth_orientation_provider = earth_orientation_provider

    def evaluate(self, data: StationDisplacementInput) -> PoleTideResult:
        latitude_rad, longitude_rad = itrf2geocentric(data.reference_position_itrf_m)
        theta = 0.5 * np.pi - latitude_rad
        wobble = polar_wobble(data.epoch_utc, self.earth_orientation_provider)

        sin_lon = np.sin(longitude_rad)
        cos_lon = np.cos(longitude_rad)
        common = wobble.m1_arcsec * cos_lon + wobble.m2_arcsec * sin_lon

        south_mm = -9.0 * np.cos(2.0 * theta) * common
        east_mm = 9.0 * np.cos(theta) * (wobble.m1_arcsec * sin_lon - wobble.m2_arcsec * cos_lon)
        up_mm = -33.0 * np.sin(2.0 * theta) * common

        enu_m = np.array([east_mm, -south_mm, up_mm], dtype=float) * 1.0e-3
        itrf_m = enu2itrf(
            enu_m,
            latitude_rad=latitude_rad,
            longitude_rad=longitude_rad,
        )
        return PoleTideResult(
            displacement_itrf_m=itrf_m,
            displacement_enu_m=enu_m,
            wobble=wobble,
            geocentric_latitude_rad=latitude_rad,
            longitude_rad=longitude_rad,
        )

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        return np.array(self.evaluate(data).displacement_itrf_m, copy=True)
