"""IERS 2010 high-frequency Earth-orientation corrections."""

from __future__ import annotations

from dataclasses import dataclass

from lunarops import _iers2010  # pyright: ignore[reportMissingModuleSource]
from lunarops.base.constants import SECONDS_PER_DAY
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

    @property
    def delta_xp_arcsec(self) -> float:
        return self.ocean_delta_xp_arcsec + self.libration_delta_xp_arcsec

    @property
    def delta_yp_arcsec(self) -> float:
        return self.ocean_delta_yp_arcsec + self.libration_delta_yp_arcsec

    @property
    def delta_ut1_s(self) -> float:
        return self.ocean_delta_ut1_s + self.libration_delta_ut1_s


def require_utc_epoch(epoch_utc: Epoch) -> Epoch:
    if not isinstance(epoch_utc, Epoch):
        raise TypeError("EOP evaluation requires an Epoch.")
    return epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")


def _ocean_tide_terms(epoch_utc: Epoch, background_ut1_minus_utc_s: float) -> dict[str, float]:
    ut1_mjd = require_utc_epoch(epoch_utc).mjd + background_ut1_minus_utc_s / SECONDS_PER_DAY
    delta_xp, delta_yp, delta_ut1 = map(float, _iers2010.ortho_eop(ut1_mjd))
    return {
        "ocean_delta_xp_arcsec": delta_xp * _MICROARCSECOND_TO_ARCSECOND,
        "ocean_delta_yp_arcsec": delta_yp * _MICROARCSECOND_TO_ARCSECOND,
        "ocean_delta_ut1_s": delta_ut1 * _MICROSECOND_TO_SECOND,
    }


def _libration_terms(epoch_tt_or_tdb: Epoch) -> dict[str, float]:
    if epoch_tt_or_tdb.scale not in (TimeScale.TT, TimeScale.TDB):
        raise ValueError("libration EOP requires a TT or TDB Epoch.")
    mjd = float(epoch_tt_or_tdb.mjd)
    delta_xp, delta_yp = map(float, _iers2010.pmsdnut2(mjd))
    delta_ut1, delta_lod = map(float, _iers2010.utlibr(mjd))
    return {
        "libration_delta_xp_arcsec": delta_xp * _MICROARCSECOND_TO_ARCSECOND,
        "libration_delta_yp_arcsec": delta_yp * _MICROARCSECOND_TO_ARCSECOND,
        "libration_delta_ut1_s": delta_ut1 * _MICROSECOND_TO_SECOND,
        "libration_delta_lod_s_per_day": delta_lod * _MICROSECOND_TO_SECOND,
    }


def ocean_tide_eop_correction(
    epoch_utc: Epoch,
    *,
    background_ut1_minus_utc_s: float,
) -> HighFrequencyEopCorrection:
    """Return ORTHO_EOP corrections at the corresponding UT1 epoch."""
    return HighFrequencyEopCorrection(**_ocean_tide_terms(epoch_utc, background_ut1_minus_utc_s))


def earth_rotation_libration_eop_correction(
    epoch_tt_or_tdb: Epoch,
) -> HighFrequencyEopCorrection:
    """Return PMSDNUT2 and UTLIBR corrections at a TT or TDB epoch."""
    return HighFrequencyEopCorrection(**_libration_terms(epoch_tt_or_tdb))


def high_frequency_eop_correction(
    epoch_utc: Epoch,
    *,
    background_ut1_minus_utc_s: float,
) -> HighFrequencyEopCorrection:
    """Return combined EOP corrections for an explicit UTC observation epoch."""
    ocean = _ocean_tide_terms(epoch_utc, background_ut1_minus_utc_s)
    return HighFrequencyEopCorrection(**ocean, **_libration_terms(utc2tt(epoch_utc)))


__all__ = [
    "HighFrequencyEopCorrection",
    "earth_rotation_libration_eop_correction",
    "high_frequency_eop_correction",
    "ocean_tide_eop_correction",
    "require_utc_epoch",
]
