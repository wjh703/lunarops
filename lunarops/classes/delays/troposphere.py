"""Mendes-Pavlis optical tropospheric delay for laser ranging."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.classes.delays.base import TroposphereInput
from lunarops.config.schema import ConfigSchema

_ELEVATION_FLOOR_DEG = 3.0


class Iers2010MendesPavlisTroposphere:
    """IERS 2010 Mendes-Pavlis optical tropospheric slant delay at a laser-ranging station.

    The slant delay is the Mendes-Pavlis zenith total delay times the Mendes-Pavlis mean mapping
    function. The water-vapour pressure is derived from temperature and relative humidity with the
    saturation vapour pressure over water. Callers clamp the leg elevation to ``elevation_floor_rad``.

    References
    ----------
    .. [1] Petit, G., & Luzum, B. (Eds.) (2010). IERS Conventions (2010), IERS Technical Note 36, Sec. 9.1.
    .. [2] Mendes, V. B., & Pavlis, E. C. (2004). High-accuracy zenith delay prediction at optical
       wavelengths. Geophysical Research Letters, 31, L14602.
    .. [3] Mendes, V. B., Prates, G., Pavlis, E. C., Pavlis, D. E., & Langley, R. B. (2002). Improved
       mapping functions for atmospheric refraction correction in SLR. Geophysical Research Letters,
       29(10), 1414.
    """

    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(fields=(), type_name="mendesPavlis")

    @classmethod
    def from_config(cls, cfg, ctx):
        """Create the model; it has no configuration fields."""
        return cls()

    @property
    def elevation_floor_rad(self) -> float:
        """Lowest elevation at which the model is evaluated: 3 degrees, in radians."""
        return float(np.deg2rad(_ELEVATION_FLOOR_DEG))

    @staticmethod
    def _water_vapor_pressure_hpa(
        temperature_k: float,
        relative_humidity_percent: float,
    ) -> float:
        t_c = temperature_k - 273.15
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
        result = float((relative_humidity_percent / 100.0) * e_s)
        if not np.isfinite(result):
            raise ValueError("temperature_k is outside the water-vapour conversion domain.")
        return result

    def slant_delay_m(self, data: TroposphereInput, *, elevation_rad: float) -> float:
        """Return the one-way tropospheric slant delay.

        Parameters
        ----------
        data : TroposphereInput
            Station meteorology, position and laser wavelength.
        elevation_rad : float
            Elevation of the target above the station horizon, in radians.

        Returns
        -------
        float
            One-way slant delay (excess path length), in metres.

        Raises
        ------
        ValueError
            If ``data.temperature_k`` is outside the water-vapour conversion domain.
        """
        elevation_deg = float(np.rad2deg(elevation_rad))
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
