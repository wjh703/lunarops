"""IERS Conventions (2010) routines on top of the compiled numerical core.

The numerical work is done by the Cython extension
``lunarops._iers2010_core``, which implements the algorithms of the IERS
Conventions (2010) software. This module adds the calendar and time-scale
handling, done with ERFA: UTC calendar dates are converted to TT Julian
centuries and the fundamental arguments are evaluated with ERFA.

Functions
---------
fcul_a
    Mendes-Pavlis mean mapping function for optical ranging.
fculzd_hpa
    Mendes-Pavlis zenith delays for optical ranging.
ortho_eop
    Diurnal and semidiurnal ocean-tide EOP variations.
pmsdnut2
    Polar-motion libration (subdiurnal nutation).
utlibr
    UT1 and LOD libration.
fundarg
    Delaunay fundamental arguments.
dehanttideinel
    Solid Earth tide station displacement.
hardisp
    Ocean-loading station displacement series.

Attributes
----------
HARDISP_MIN_UTC : tuple of int
    Earliest supported UTC start ``(year, month, day, hour, minute,
    second)`` of :func:`hardisp`.
HARDISP_VALID_UNTIL_UTC_EXCLUSIVE : tuple of int
    Exclusive upper bound of the UTC epochs of a :func:`hardisp` series.

References
----------
IERS Conventions (2010), Petit & Luzum (eds.), IERS Technical Note 36,
Chs. 5, 7, 8 and 9.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta

import erfa
import numpy as np
from numpy.typing import ArrayLike, NDArray

from lunarops import _iers2010_core as _core
from lunarops.base.constants import SECONDS_PER_DAY

_J2000 = 2_451_545.0
_JULIAN_CENTURY_DAYS = 36_525.0
HARDISP_MIN_UTC = (1960, 1, 1, 0, 0, 0)
HARDISP_VALID_UNTIL_UTC_EXCLUSIVE = (2027, 7, 1, 0, 0, 0)
_HARDISP_VALID_UNTIL = datetime(*HARDISP_VALID_UNTIL_UTC_EXCLUSIVE, tzinfo=UTC)
_UTC_OFFSET_TRANSITIONS = tuple(
    datetime(int(item["year"]), int(item["month"]), 1, tzinfo=UTC)
    for item in erfa.leap_seconds.get()
    if (int(item["year"]), int(item["month"]), 1) > HARDISP_MIN_UTC[:3]
)


fcul_a = _core.lunarops_fcul_a
fculzd_hpa = _core.lunarops_fculzd_hpa
ortho_eop = _core.lunarops_ortho_eop


def fundarg(t: float) -> tuple[float, float, float, float, float]:
    """Return the Delaunay fundamental arguments of lunisolar nutation.

    Parameters
    ----------
    t : float
        TT Julian centuries since J2000.0.

    Returns
    -------
    tuple of float
        ``(l, l', F, D, Ω)`` in rad: mean anomalies of the Moon and the Sun,
        mean argument of latitude of the Moon, mean elongation of the Moon
        from the Sun, and mean longitude of the Moon's ascending node.

    References
    ----------
    IERS Conventions (2010), Ch. 5, Eq. 5.43.
    """
    return (
        float(erfa.fal03(t)),
        float(erfa.falp03(t)),
        float(erfa.faf03(t)),
        float(erfa.fad03(t)),
        float(erfa.faom03(t)),
    )


def pmsdnut2(rmjd: float) -> NDArray[np.float64]:
    """Return the polar-motion libration with periods below two days.

    Parameters
    ----------
    rmjd : float
        Epoch as a Modified Julian Date. It is used as TT for the
        fundamental arguments and as UT1 for GMST, as in the IERS routine.

    Returns
    -------
    numpy.ndarray
        Polar-motion corrections ``(Δx_p, Δy_p)`` in µas, shape ``(2,)``.

    References
    ----------
    IERS Conventions (2010), Ch. 5, Table 5.1a.
    """
    return _core.lunarops_pmsdnut2(rmjd, *fundarg((rmjd - 51_544.5) / _JULIAN_CENTURY_DAYS))


def utlibr(rmjd: float) -> tuple[float, float]:
    """Return the subdiurnal libration in UT1 and LOD.

    Parameters
    ----------
    rmjd : float
        Epoch as a Modified Julian Date. It is used as TT for the
        fundamental arguments and as UT1 for GMST, as in the IERS routine.

    Returns
    -------
    dut1 : float
        UT1 correction in µs.
    dlod : float
        Length-of-day correction in µs per day.

    References
    ----------
    IERS Conventions (2010), Ch. 5, Table 5.1b.
    """
    return _core.lunarops_utlibr(rmjd, *fundarg((rmjd - 51_544.5) / _JULIAN_CENTURY_DAYS))


def _utc_epoch_parts(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: float,
) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", erfa.ErfaWarning)
        utc1, utc2 = erfa.dtf2d("UTC", year, month, day, hour, minute, second)
        tai1, tai2 = erfa.utctai(utc1, utc2)
        tt1, tt2 = erfa.taitt(tai1, tai2)
    tt_centuries = ((float(tt1) - _J2000) + float(tt2)) / _JULIAN_CENTURY_DAYS
    utc_day_fraction = (hour * 3600.0 + minute * 60.0 + second) / SECONDS_PER_DAY
    return tt_centuries, utc_day_fraction


def dehanttideinel(
    xsta: ArrayLike,
    yr: int,
    month: int,
    day: int,
    fhr: float,
    xsun: ArrayLike,
    xmon: ArrayLike,
) -> NDArray[np.float64]:
    """Return the solid Earth tide displacement of a station.

    Parameters
    ----------
    xsta : array_like
        Geocentric station position in the ITRF, m, shape ``(3,)``.
    yr, month, day : int
        UTC calendar date.
    fhr : float
        UTC hour of day in ``[0, 24)``.
    xsun : array_like
        Geocentric position of the Sun in the ITRF, m, shape ``(3,)``.
    xmon : array_like
        Geocentric position of the Moon in the ITRF, m, shape ``(3,)``.

    Returns
    -------
    numpy.ndarray
        Station displacement in the ITRF, m, shape ``(3,)``.

    Raises
    ------
    ValueError
        If ``fhr`` is not finite or outside ``[0, 24)``.
    erfa.ErfaWarning
        If ERFA flags the UTC date as dubious or invalid.

    References
    ----------
    IERS Conventions (2010), Ch. 7, Sec. 7.1.1.
    """
    station = np.asarray(xsta, float)
    sun = np.asarray(xsun, float)
    moon = np.asarray(xmon, float)
    if not np.isfinite(fhr) or not 0.0 <= fhr < 24.0:
        raise ValueError("fhr must be finite and in [0, 24).")
    hour = int(fhr)
    minute_value = (float(fhr) - hour) * 60.0
    minute = int(minute_value)
    second = (minute_value - minute) * 60.0
    tt_centuries, _ = _utc_epoch_parts(yr, month, day, hour, minute, second)
    return _core.lunarops_dehanttideinel(station, fhr, sun, moon, tt_centuries)


def hardisp(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    n: int,
    sample: float,
    blq_amp: ArrayLike,
    blq_phase: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return an ocean-loading displacement series at regular UTC epochs.

    Parameters
    ----------
    year, month, day, hour, minute, second : int
        UTC start epoch; ``second`` must not be a leap second (60).
    n : int
        Number of samples, positive.
    sample : float
        Sampling interval in s, positive.
    blq_amp : array_like
        BLQ amplitudes in m of the 11 main tidal constituents, shape
        ``(3, 11)`` with rows Up, West, South.
    blq_phase : array_like
        BLQ phases in degrees (Greenwich lags), shape ``(3, 11)``.

    Returns
    -------
    up, west, south : numpy.ndarray
        Displacements in m in BLQ axis order, each of shape ``(n,)``.

    Raises
    ------
    ValueError
        If the start epoch is a leap second or outside
        ``[HARDISP_MIN_UTC, HARDISP_VALID_UNTIL_UTC_EXCLUSIVE)``, the series
        extends beyond that interval or crosses a change of TAI-UTC, ``n``
        or ``sample`` is not positive, or the BLQ arrays have the wrong
        shape or non-finite values.
    erfa.ErfaWarning
        If ERFA flags the UTC date as dubious or invalid.

    References
    ----------
    IERS Conventions (2010), Ch. 7, Sec. 7.1.2.
    """
    if second == 60:
        raise ValueError("HARDISP cannot represent an exact UTC leap-second label.")
    calendar = (year, month, day, hour, minute, second)
    if calendar < HARDISP_MIN_UTC or calendar >= HARDISP_VALID_UNTIL_UTC_EXCLUSIVE:
        raise ValueError("HARDISP supports UTC epochs only from 1960-01-01T00:00:00 through 2027-06-30T23:59:59.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if not np.isfinite(sample) or sample <= 0.0:
        raise ValueError("sample must be finite and positive.")
    start = datetime(*calendar, tzinfo=UTC)
    duration_seconds = (n - 1) * float(sample)
    if not np.isfinite(duration_seconds):
        raise ValueError("the HARDISP series duration must be finite.")
    try:
        end = start + timedelta(seconds=duration_seconds)
    except OverflowError as error:
        raise ValueError("the HARDISP series end is outside the supported calendar range.") from error
    if end >= _HARDISP_VALID_UNTIL:
        raise ValueError("the HARDISP series extends beyond the supported UTC interval.")
    if any(start < transition <= end for transition in _UTC_OFFSET_TRANSITIONS):
        raise ValueError("a regular HARDISP series must not cross a UTC offset transition.")
    amplitudes = np.asarray(blq_amp, dtype=np.float64)
    phases = np.asarray(blq_phase, dtype=np.float64)
    if amplitudes.shape != (3, 11) or phases.shape != (3, 11):
        raise ValueError(
            f"blq_amp and blq_phase must each have shape (3, 11); got {amplitudes.shape} and {phases.shape}."
        )
    if not np.all(np.isfinite(amplitudes)) or not np.all(np.isfinite(phases)):
        raise ValueError("BLQ amplitudes and phases must contain only finite values.")
    tt_centuries, utc_day_fraction = _utc_epoch_parts(year, month, day, hour, minute, float(second))
    return _core.lunarops_hardisp(
        tt_centuries,
        utc_day_fraction,
        *fundarg(tt_centuries),
        n,
        sample,
        amplitudes,
        phases,
    )


__all__ = [
    "HARDISP_MIN_UTC",
    "HARDISP_VALID_UNTIL_UTC_EXCLUSIVE",
    "dehanttideinel",
    "fcul_a",
    "fculzd_hpa",
    "fundarg",
    "hardisp",
    "ortho_eop",
    "pmsdnut2",
    "utlibr",
]
