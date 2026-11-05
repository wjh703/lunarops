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
    """Subdiurnal corrections to polar motion and UT1 from ocean tides and libration.

    Attributes
    ----------
    ocean_delta_xp_arcsec, ocean_delta_yp_arcsec : float
        Diurnal and semidiurnal ocean-tide polar-motion corrections, in arcseconds.
    ocean_delta_ut1_s : float
        Diurnal and semidiurnal ocean-tide UT1 correction, in seconds.
    libration_delta_xp_arcsec, libration_delta_yp_arcsec : float
        Polar-motion libration corrections with periods below two days, in arcseconds.
    libration_delta_ut1_s : float
        Subdiurnal libration UT1 correction, in seconds.
    libration_delta_lod_s_per_day : float
        Subdiurnal libration length-of-day correction, in seconds per day.

    References
    ----------
    .. [1] Petit, G., & Luzum, B. (Eds.) (2010). IERS Conventions (2010), IERS Technical Note 36,
       Sec. 8.2 (ocean tides), Tables 5.1a and 5.1b (libration).
    """

    ocean_delta_xp_arcsec: float = 0.0
    ocean_delta_yp_arcsec: float = 0.0
    ocean_delta_ut1_s: float = 0.0
    libration_delta_xp_arcsec: float = 0.0
    libration_delta_yp_arcsec: float = 0.0
    libration_delta_ut1_s: float = 0.0
    libration_delta_lod_s_per_day: float = 0.0

    @property
    def delta_xp_arcsec(self) -> float:
        """Total x_p correction (ocean tide plus libration), in arcseconds."""
        return self.ocean_delta_xp_arcsec + self.libration_delta_xp_arcsec

    @property
    def delta_yp_arcsec(self) -> float:
        """Total y_p correction (ocean tide plus libration), in arcseconds."""
        return self.ocean_delta_yp_arcsec + self.libration_delta_yp_arcsec

    @property
    def delta_ut1_s(self) -> float:
        """Total UT1 correction (ocean tide plus libration), in seconds."""
        return self.ocean_delta_ut1_s + self.libration_delta_ut1_s


def require_utc_epoch(epoch_utc: Epoch) -> Epoch:
    """Return ``epoch_utc`` after checking that it is on the UTC scale; raise ``ValueError`` otherwise."""
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
    """Return the ocean-tide polar-motion and UT1 corrections (IERS Conventions (2010), Sec. 8.2).

    Parameters
    ----------
    epoch_utc : Epoch
        Epoch on the UTC scale.
    background_ut1_minus_utc_s : float
        Interpolated UT1 - UTC, in seconds, used to evaluate the model at the UT1 epoch.

    Returns
    -------
    HighFrequencyEopCorrection
        Correction with only the ocean-tide terms set.
    """
    return HighFrequencyEopCorrection(**_ocean_tide_terms(epoch_utc, background_ut1_minus_utc_s))


def earth_rotation_libration_eop_correction(
    epoch_tt_or_tdb: Epoch,
) -> HighFrequencyEopCorrection:
    """Return the libration polar-motion, UT1 and LOD corrections (IERS Conventions (2010), Tables 5.1a/b).

    Parameters
    ----------
    epoch_tt_or_tdb : Epoch
        Epoch on the TT or TDB scale; its MJD is passed to the libration models.

    Returns
    -------
    HighFrequencyEopCorrection
        Correction with only the libration terms set.

    Raises
    ------
    ValueError
        If the epoch is not on TT or TDB.
    """
    return HighFrequencyEopCorrection(**_libration_terms(epoch_tt_or_tdb))


def high_frequency_eop_correction(
    epoch_utc: Epoch,
    *,
    background_ut1_minus_utc_s: float,
) -> HighFrequencyEopCorrection:
    """Return the combined ocean-tide and libration EOP corrections at a UTC epoch.

    The ocean-tide terms are evaluated at the UT1 epoch and the libration terms at the TT epoch
    corresponding to ``epoch_utc``.

    Parameters
    ----------
    epoch_utc : Epoch
        Epoch on the UTC scale.
    background_ut1_minus_utc_s : float
        Interpolated UT1 - UTC, in seconds.

    Returns
    -------
    HighFrequencyEopCorrection
        Ocean-tide and libration corrections.
    """
    ocean = _ocean_tide_terms(epoch_utc, background_ut1_minus_utc_s)
    return HighFrequencyEopCorrection(**ocean, **_libration_terms(utc2tt(epoch_utc)))


__all__ = [
    "HighFrequencyEopCorrection",
    "earth_rotation_libration_eop_correction",
    "high_frequency_eop_correction",
    "ocean_tide_eop_correction",
    "require_utc_epoch",
]
