"""ERFA facade for the private Cython IERS 2010 numerical core."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import erfa
import numpy as np
from numpy.typing import ArrayLike, NDArray

from lunarops import _iers2010_core as _core

_MJD0 = 2_400_000.5
_J2000 = 2_451_545.0
_JULIAN_CENTURY_DAYS = 36_525.0
HARDISP_MIN_UTC = (1960, 1, 1, 0, 0, 0)
HARDISP_VALID_UNTIL_UTC_EXCLUSIVE = (2027, 7, 1, 0, 0, 0)
_HARDISP_VALID_UNTIL = datetime(*HARDISP_VALID_UNTIL_UTC_EXCLUSIVE)
_UTC_OFFSET_TRANSITIONS = tuple(
    datetime(int(item["year"]), int(item["month"]), 1)
    for item in erfa.leap_seconds.get()
    if (int(item["year"]), int(item["month"]), 1) > HARDISP_MIN_UTC[:3]
)


def _vector3(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,); got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(result)


def _fundamental_arguments(t: float) -> tuple[float, float, float, float, float]:
    return (
        float(erfa.fal03(t)),
        float(erfa.falp03(t)),
        float(erfa.faf03(t)),
        float(erfa.fad03(t)),
        float(erfa.faom03(t)),
    )


def fcul_a(latitude: float, height_m: float, t_k: float, elev_deg: float) -> float:
    return float(_core.lunarops_fcul_a(latitude, height_m, t_k, elev_deg))


def fculzd_hpa(
    latitude: float,
    ellip_ht: float,
    pressure: float,
    wvp: float,
    lambda_um: float,
) -> tuple[float, float, float]:
    return _core.lunarops_fculzd_hpa(latitude, ellip_ht, pressure, wvp, lambda_um)


def ortho_eop(time: float) -> NDArray[np.float64]:
    return _core.lunarops_ortho_eop(time)


def pmsdnut2(rmjd: float) -> NDArray[np.float64]:
    return _core.lunarops_pmsdnut2(
        rmjd,
        *_fundamental_arguments((rmjd - 51_544.5) / _JULIAN_CENTURY_DAYS),
    )


def utlibr(rmjd: float) -> tuple[float, float]:
    return _core.lunarops_utlibr(
        rmjd,
        *_fundamental_arguments((rmjd - 51_544.5) / _JULIAN_CENTURY_DAYS),
    )


def fundarg(t: float) -> tuple[float, float, float, float, float]:
    return _fundamental_arguments(float(t))


def _utc_epoch_parts(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: float,
) -> tuple[float, float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", erfa.ErfaWarning)
        utc1, utc2 = erfa.dtf2d("UTC", year, month, day, hour, minute, second)
        tai1, tai2 = erfa.utctai(utc1, utc2)
        tt1, tt2 = erfa.taitt(tai1, tai2)
    tt_centuries = ((float(tt1) - _J2000) + float(tt2)) / _JULIAN_CENTURY_DAYS
    utc_day_fraction = (hour * 3600.0 + minute * 60.0 + second) / 86_400.0
    return tt_centuries, utc_day_fraction, float(utc1 + utc2 - _MJD0)


def dehanttideinel(
    xsta: ArrayLike,
    yr: int,
    month: int,
    day: int,
    fhr: float,
    xsun: ArrayLike,
    xmon: ArrayLike,
) -> NDArray[np.float64]:
    station = _vector3(xsta, name="xsta")
    sun = _vector3(xsun, name="xsun")
    moon = _vector3(xmon, name="xmon")
    if not np.isfinite(fhr) or not 0.0 <= fhr < 24.0:
        raise ValueError("fhr must be finite and in [0, 24).")
    hour = int(fhr)
    minute_value = (float(fhr) - hour) * 60.0
    minute = int(minute_value)
    second = (minute_value - minute) * 60.0
    tt_centuries, _, _ = _utc_epoch_parts(yr, month, day, hour, minute, second)
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
    """Return displacement arrays in BLQ order: Up, West, South (metres)."""
    if second == 60:
        raise ValueError("HARDISP cannot represent an exact UTC leap-second label.")
    calendar = (year, month, day, hour, minute, second)
    if calendar < HARDISP_MIN_UTC or calendar >= HARDISP_VALID_UNTIL_UTC_EXCLUSIVE:
        raise ValueError(
            "HARDISP supports UTC epochs only from 1960-01-01T00:00:00 "
            "through 2027-06-30T23:59:59."
        )
    if n <= 0:
        raise ValueError("n must be positive.")
    if not np.isfinite(sample) or sample <= 0.0:
        raise ValueError("sample must be finite and positive.")
    start = datetime(*calendar)
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
            "blq_amp and blq_phase must each have shape (3, 11); "
            f"got {amplitudes.shape} and {phases.shape}."
        )
    if not np.all(np.isfinite(amplitudes)) or not np.all(np.isfinite(phases)):
        raise ValueError("BLQ amplitudes and phases must contain only finite values.")
    tt_centuries, utc_day_fraction, _ = _utc_epoch_parts(year, month, day, hour, minute, float(second))
    return _core.lunarops_hardisp(
        tt_centuries,
        utc_day_fraction,
        *_fundamental_arguments(tt_centuries),
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
