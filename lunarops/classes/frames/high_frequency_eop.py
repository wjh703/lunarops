"""IERS 2010 high-frequency Earth-orientation corrections."""

from __future__ import annotations

import math
from dataclasses import dataclass

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.classes.time import Epoch, TimeScale, utc2tt

_MICROARCSECOND_TO_ARCSECOND = 1.0e-6
_MICROSECOND_TO_SECOND = 1.0e-6


@dataclass(frozen=True, slots=True)
class HighFrequencyEopCorrection:
    ocean_delta_xp_arcsec: float = 0.0
    ocean_delta_yp_arcsec: float = 0.0
    ocean_delta_ut1_s: float = 0.0
    libration_delta_xp_arcsec: float = 0.0
    libration_delta_yp_arcsec: float = 0.0
    libration_delta_ut1_s: float = 0.0
    libration_delta_lod_s_per_day: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "ocean_delta_xp_arcsec",
            "ocean_delta_yp_arcsec",
            "ocean_delta_ut1_s",
            "libration_delta_xp_arcsec",
            "libration_delta_yp_arcsec",
            "libration_delta_ut1_s",
            "libration_delta_lod_s_per_day",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a real scalar.")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a real scalar.") from exc
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, normalized)

    @property
    def delta_xp_arcsec(self) -> float:
        return self.ocean_delta_xp_arcsec + self.libration_delta_xp_arcsec

    @property
    def delta_yp_arcsec(self) -> float:
        return self.ocean_delta_yp_arcsec + self.libration_delta_yp_arcsec

    @property
    def delta_ut1_s(self) -> float:
        return self.ocean_delta_ut1_s + self.libration_delta_ut1_s


def _require_utc_epoch(epoch_utc: Epoch) -> Epoch:
    if not isinstance(epoch_utc, Epoch):
        raise TypeError("High-frequency EOP requires an Epoch.")
    return epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")


def _compute_ut1_mjd(epoch_utc: Epoch, background_ut1_minus_utc_s: float) -> float:
    epoch = _require_utc_epoch(epoch_utc)
    if isinstance(background_ut1_minus_utc_s, bool):
        raise TypeError("background_ut1_minus_utc_s must be a real scalar.")
    try:
        value = float(background_ut1_minus_utc_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("background_ut1_minus_utc_s must be a real scalar.") from exc
    if not math.isfinite(value):
        raise ValueError("background_ut1_minus_utc_s must be finite.")
    return epoch.mjd + value / 86_400.0


def _dynamical_time_mjd(epoch_tt_or_tdb: Epoch) -> float:
    """Return the MJD expected by the IERS TT/TDB-compatible routines."""
    if not isinstance(epoch_tt_or_tdb, Epoch):
        raise TypeError("Earth-rotation libration EOP requires an Epoch.")
    if epoch_tt_or_tdb.scale not in (TimeScale.TT, TimeScale.TDB):
        raise ValueError("libration EOP requires a TT or TDB Epoch.")
    return float(epoch_tt_or_tdb.mjd)


def ocean_tide_eop_correction(
    epoch_utc: Epoch,
    *,
    background_ut1_minus_utc_s: float,
) -> HighFrequencyEopCorrection:
    """Return ORTHO_EOP corrections at the corresponding UT1 epoch."""
    delta_xp, delta_yp, delta_ut1 = _iers2010.ortho_eop(_compute_ut1_mjd(epoch_utc, background_ut1_minus_utc_s))
    return HighFrequencyEopCorrection(
        ocean_delta_xp_arcsec=delta_xp * _MICROARCSECOND_TO_ARCSECOND,
        ocean_delta_yp_arcsec=delta_yp * _MICROARCSECOND_TO_ARCSECOND,
        ocean_delta_ut1_s=delta_ut1 * _MICROSECOND_TO_SECOND,
    )


def earth_rotation_libration_eop_correction(
    epoch_tt_or_tdb: Epoch,
) -> HighFrequencyEopCorrection:
    """Return PMSDNUT2 and UTLIBR corrections at a TT or TDB epoch."""
    mjd = _dynamical_time_mjd(epoch_tt_or_tdb)
    delta_xp, delta_yp = _iers2010.pmsdnut2(mjd)
    delta_ut1, delta_lod = _iers2010.utlibr(mjd)
    return HighFrequencyEopCorrection(
        libration_delta_xp_arcsec=delta_xp * _MICROARCSECOND_TO_ARCSECOND,
        libration_delta_yp_arcsec=delta_yp * _MICROARCSECOND_TO_ARCSECOND,
        libration_delta_ut1_s=delta_ut1 * _MICROSECOND_TO_SECOND,
        libration_delta_lod_s_per_day=delta_lod * _MICROSECOND_TO_SECOND,
    )


def high_frequency_eop_correction(
    epoch_utc: Epoch,
    *,
    background_ut1_minus_utc_s: float,
) -> HighFrequencyEopCorrection:
    """Return combined EOP corrections for an explicit UTC observation epoch."""
    epoch_utc = _require_utc_epoch(epoch_utc)
    ocean = ocean_tide_eop_correction(
        epoch_utc,
        background_ut1_minus_utc_s=background_ut1_minus_utc_s,
    )
    libration = earth_rotation_libration_eop_correction(utc2tt(epoch_utc))
    return HighFrequencyEopCorrection(
        ocean_delta_xp_arcsec=ocean.ocean_delta_xp_arcsec,
        ocean_delta_yp_arcsec=ocean.ocean_delta_yp_arcsec,
        ocean_delta_ut1_s=ocean.ocean_delta_ut1_s,
        libration_delta_xp_arcsec=libration.libration_delta_xp_arcsec,
        libration_delta_yp_arcsec=libration.libration_delta_yp_arcsec,
        libration_delta_ut1_s=libration.libration_delta_ut1_s,
        libration_delta_lod_s_per_day=libration.libration_delta_lod_s_per_day,
    )


__all__ = [
    "HighFrequencyEopCorrection",
    "earth_rotation_libration_eop_correction",
    "high_frequency_eop_correction",
    "ocean_tide_eop_correction",
]
