from __future__ import annotations

import numpy as np

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.classes.delays.base import TroposphereDelay, TroposphereInput

_ELEVATION_FLOOR_DEG = 3.0


class Iers2010MendesPavlisTroposphere(TroposphereDelay):
    """Optical troposphere model from IERS Conventions 2010 S9.1.

    The model consumes a :class:`TroposphereInput`, keeping the atmospheric,
    station, wavelength, and line-of-sight inputs together as one immutable
    value object.
    """

    @property
    def elevation_floor_rad(self) -> float:
        return float(np.deg2rad(_ELEVATION_FLOOR_DEG))

    @staticmethod
    def _water_vapor_pressure_hpa(
        temperature_k: float,
        relative_humidity_percent: float,
    ) -> float:
        temperature_k = float(temperature_k)
        relative_humidity_percent = float(relative_humidity_percent)
        if not np.isfinite(temperature_k) or not np.isfinite(relative_humidity_percent):
            raise ValueError("temperature_k and relative_humidity_percent must be finite.")
        if temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive.")
        if not 0.0 <= relative_humidity_percent <= 100.0:
            raise ValueError("relative_humidity_percent must be in [0, 100].")
        t_c = temperature_k - 273.15
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
        result = float((relative_humidity_percent / 100.0) * e_s)
        if not np.isfinite(result):
            raise ValueError("temperature_k is outside the water-vapour conversion domain.")
        return result

    def slant_delay_m(self, data: TroposphereInput) -> float:
        if not isinstance(data, TroposphereInput):
            raise TypeError("data must be a TroposphereInput.")
        elevation_deg = max(
            float(np.rad2deg(data.elevation_rad)),
            _ELEVATION_FLOOR_DEG,
        )
        latitude_deg = float(np.rad2deg(data.latitude_rad))
        wvp_hpa = self._water_vapor_pressure_hpa(
            data.temperature_k,
            data.relative_humidity_percent,
        )
        ellipsoidal_height_m = data.height_m
        mean_sea_level_height_m = data.height_m
        ztd, _, _ = _iers2010.fculzd_hpa(
            latitude_deg,
            ellipsoidal_height_m,
            data.pressure_hpa,
            wvp_hpa,
            data.wavelength_um,
        )
        mapping = _iers2010.fcul_a(
            latitude_deg,
            mean_sea_level_height_m,
            data.temperature_k,
            elevation_deg,
        )
        return float(ztd * mapping)
