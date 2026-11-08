"""Scalar epochs on an explicit UTC, TT or TDB time scale.

``Epoch`` is the time value used throughout LunarOps. It stores a two-part Julian Date and its scale.
ERFA performs civil-date parsing and formatting, UTC<->TAI<->TT conversion, leap-second handling and
UTC elapsed-time arithmetic. TT<->TDB conversion is provided by
:class:`lunarops.classes.time.converter.TimeScaleConverter`, because it can include a station-dependent
term.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from functools import total_ordering
from typing import cast

import erfa
import numpy as np

from lunarops.base.constants import SECONDS_PER_DAY


class TimeScale(StrEnum):
    """Time scale of an ``Epoch``: UTC, TT or TDB."""

    UTC = "utc"
    TT = "tt"
    TDB = "tdb"

    @classmethod
    def parse(cls, value: object) -> TimeScale:
        """Return the time scale named by ``value``.

        Parameters
        ----------
        value : object
            A ``TimeScale`` or a case-insensitive name (``"utc"``, ``"tt"``, ``"tdb"``).

        Returns
        -------
        TimeScale
            The parsed scale.

        Raises
        ------
        ValueError
            If ``value`` names no supported scale.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(scale.value for scale in cls)
            raise ValueError(f"Unsupported time scale {value!r}; expected one of {allowed}.") from exc


_ISO_RE = re.compile(
    r"^\s*(?P<date>\d{4}-\d{2}-\d{2}|\d{8})"
    r"(?:[T\s](?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}(?:\.\d*)?))?"
    r"(?:(?P<utc_marker>Z)|(?P<offset_sign>[+-])(?P<offset_hour>\d{2}):?(?P<offset_minute>\d{2}))?\s*$"
)


def _normalized_jd_pair(jd1: float, jd2: float) -> tuple[float, float]:
    first = float(jd1)
    second = float(jd2)
    if not np.isfinite(first) or not np.isfinite(second):
        raise ValueError("Epoch Julian-date parts must be finite.")
    carry = math.floor(second + 0.5)
    return first + carry, second - carry


def _erfa_call(function, *args):
    try:
        return function(*args)
    except erfa.ErfaWarning as exc:
        raise ValueError(f"ERFA {function.__name__} rejected the epoch: {exc}") from exc
    except erfa.ErfaError as exc:
        raise ValueError(f"ERFA {function.__name__} failed: {exc}") from exc


def _civil_to_jd(
    scale: TimeScale,
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> tuple[float, float]:
    if scale is TimeScale.TDB:
        raise ValueError("TDB has no civil constructor; use Epoch(jd1, jd2, 'tdb') or TimeScaleConverter.")
    if not math.isfinite(second):
        raise ValueError("second must be finite.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", erfa.ErfaWarning)
        jd1, jd2 = _erfa_call(
            erfa.dtf2d,
            scale.value.upper(),
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            float(second),
        )
    return float(jd1), float(jd2)


def _civil_fields(epoch: Epoch, precision: int) -> tuple[int, int, int, int, int, int, int]:
    year, month, day, fields = _erfa_call(erfa.d2dtf, epoch.scale.value.upper(), int(precision), epoch.jd1, epoch.jd2)
    return (int(year), int(month), int(day), int(fields["h"]), int(fields["m"]), int(fields["s"]), int(fields["f"]))


def _parse_isot(value: str) -> tuple[int, int, int, int, int, float, float | None]:
    text = str(value).strip()
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    match = _ISO_RE.match(text)
    if match is None:
        raise ValueError(f"Unsupported ISO epoch text {value!r}.")
    raw_date = match.group("date")
    if len(raw_date) == 8:
        year, month, day = int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:8])
    else:
        year, month, day = map(int, raw_date.split("-"))
    hour = int(match.group("hour") or 0)
    minute = int(match.group("minute") or 0)
    second = float(match.group("second") or 0.0)
    offset_hours: float | None = None
    if match.group("utc_marker") is not None:
        offset_hours = 0.0
    elif match.group("offset_sign") is not None:
        offset_hour = int(match.group("offset_hour"))
        offset_minute = int(match.group("offset_minute"))
        if offset_minute >= 60 or offset_hour > 24 or (offset_hour == 24 and offset_minute != 0):
            raise ValueError("UTC offset must be in [-24:00, 24:00].")
        offset_hours = offset_hour + offset_minute / 60.0
        if match.group("offset_sign") == "-":
            offset_hours = -offset_hours
    return year, month, day, hour, minute, second, offset_hours


def utc2tt(epoch: Epoch) -> Epoch:
    """Convert a UTC epoch to TT through TAI with ERFA ``utctai`` and ``taitt``.

    Parameters
    ----------
    epoch : Epoch
        Epoch on the UTC scale.

    Returns
    -------
    Epoch
        The same instant on the TT scale.

    Raises
    ------
    ValueError
        If ``epoch`` is not on the UTC scale or ERFA rejects it.
    """
    epoch.require_scale(TimeScale.UTC)
    tai1, tai2 = _erfa_call(erfa.utctai, epoch.jd1, epoch.jd2)
    tt1, tt2 = _erfa_call(erfa.taitt, tai1, tai2)
    return Epoch(tt1, tt2, TimeScale.TT)


def tt2utc(epoch: Epoch) -> Epoch:
    """Convert a TT epoch to UTC through TAI with ERFA ``tttai`` and ``taiutc``.

    Parameters
    ----------
    epoch : Epoch
        Epoch on the TT scale.

    Returns
    -------
    Epoch
        The same instant on the UTC scale.

    Raises
    ------
    ValueError
        If ``epoch`` is not on the TT scale or ERFA rejects it.
    """
    epoch.require_scale(TimeScale.TT)
    tai1, tai2 = _erfa_call(erfa.tttai, epoch.jd1, epoch.jd2)
    utc1, utc2 = _erfa_call(erfa.taiutc, tai1, tai2)
    return Epoch(utc1, utc2, TimeScale.UTC)


@total_ordering
@dataclass(frozen=True, slots=True, eq=False)
class Epoch:
    """Scalar epoch stored as a two-part Julian Date on an explicit time scale.

    The Julian Date is ``jd1 + jd2`` days. On construction the parts are normalised so that ``jd2`` lies in
    [-0.5, 0.5) and ``jd1`` carries the remaining whole days. UTC epochs use the ERFA quasi-Julian-Date
    convention, in which a day containing a leap second is 86401 s long. Epochs compare and order only
    against epochs on the same scale; comparing different scales raises ``ValueError``.

    Attributes
    ----------
    jd1 : float
        First part of the Julian Date, in days.
    jd2 : float
        Second part of the Julian Date, in days, in [-0.5, 0.5).
    scale : TimeScale
        Time scale of the Julian Date (UTC, TT or TDB).

    Raises
    ------
    ValueError
        If a Julian-Date part is not finite or ``scale`` is not a supported time scale.
    """

    jd1: float
    jd2: float
    scale: TimeScale

    def __post_init__(self) -> None:
        """Normalise the Julian-Date parts and parse the scale."""
        jd1, jd2 = _normalized_jd_pair(self.jd1, self.jd2)
        object.__setattr__(self, "jd1", jd1)
        object.__setattr__(self, "jd2", jd2)
        object.__setattr__(self, "scale", TimeScale.parse(self.scale))

    @classmethod
    def _from_normalized(cls, scale: TimeScale, jd1: float, jd2: float) -> Epoch:
        first, second = float(jd1), float(jd2)
        carry = math.floor(second + 0.5)
        epoch = object.__new__(cls)
        object.__setattr__(epoch, "jd1", first + carry)
        object.__setattr__(epoch, "jd2", second - carry)
        object.__setattr__(epoch, "scale", scale)
        return epoch

    @classmethod
    def from_isot(
        cls,
        value: str,
        *,
        scale: TimeScale | str = TimeScale.UTC,
    ) -> Epoch:
        """Create an epoch from ISO 8601 date or date-time text.

        Accepted forms are ``YYYY-MM-DD`` or ``YYYYMMDD``, optionally followed by ``T`` or a space and
        ``hh:mm:ss[.fff]``, optionally followed by ``Z`` or a UTC offset ``±hh:mm``/``±hhmm``. An offset
        is converted to UTC by subtracting it from the civil time.

        Parameters
        ----------
        value : str
            ISO 8601 text.
        scale : TimeScale or str, optional
            Scale of the civil time, UTC (default) or TT. TDB has no civil representation.

        Returns
        -------
        Epoch
            Epoch on ``scale``.

        Raises
        ------
        ValueError
            If the text is malformed, a UTC offset is given for a non-UTC scale, or ``scale`` is TDB.
        """
        parsed_scale = TimeScale.parse(scale)
        year, month, day, hour, minute, second, embedded_offset = _parse_isot(value)
        if embedded_offset is not None and parsed_scale is not TimeScale.UTC:
            raise ValueError("ISO UTC offsets are only valid when constructing a UTC epoch.")
        jd1, jd2 = _civil_to_jd(parsed_scale, year, month, day, hour, minute, second)
        epoch = cls(jd1, jd2, parsed_scale)
        if embedded_offset is not None and embedded_offset != 0.0:
            epoch = epoch.shifted(-embedded_offset * 3600.0)
        return epoch

    @classmethod
    def from_calendar(
        cls,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: float = 0.0,
        *,
        scale: TimeScale | str = TimeScale.UTC,
    ) -> Epoch:
        """Create an epoch from calendar date and time-of-day fields.

        Parameters
        ----------
        year, month, day : int
            Gregorian calendar date.
        hour, minute : int, optional
            Time of day; default 0.
        second : float, optional
            Seconds of the minute; up to 61 on a UTC leap-second day. Default 0.
        scale : TimeScale or str, optional
            Scale of the civil time, UTC (default) or TT.

        Returns
        -------
        Epoch
            Epoch on ``scale``.

        Raises
        ------
        ValueError
            If ``scale`` is TDB, ``second`` is not finite, or ERFA ``dtf2d`` rejects the fields.
        """
        parsed_scale = TimeScale.parse(scale)
        second_value = float(second)
        jd1, jd2 = _civil_to_jd(parsed_scale, int(year), int(month), int(day), int(hour), int(minute), second_value)
        return cls(jd1, jd2, parsed_scale)

    @classmethod
    def from_date_seconds(
        cls,
        value: date | str,
        seconds_of_day: float,
        *,
        scale: TimeScale | str = TimeScale.UTC,
    ) -> Epoch:
        """Create an epoch from a calendar date and seconds of that day.

        Parameters
        ----------
        value : date or str
            Calendar date as a ``datetime.date``, ``YYYY-MM-DD``, ``YYYY/MM/DD`` or ``YYYYMMDD``.
        seconds_of_day : float
            Seconds since the start of the day, in [0, 86400), or in [0, 86401) on the UTC scale so that
            a leap second can be addressed.
        scale : TimeScale or str, optional
            Scale of the civil time, UTC (default) or TT.

        Returns
        -------
        Epoch
            Epoch on ``scale``.

        Raises
        ------
        ValueError
            If the date is malformed, ``seconds_of_day`` is out of range, or ``scale`` is TDB.
        """
        if isinstance(value, date):
            year, month, day = value.year, value.month, value.day
        else:
            text = str(value).strip().replace("/", "-")
            if len(text) == 8 and text.isdigit():
                year, month, day = int(text[:4]), int(text[4:6]), int(text[6:8])
            else:
                parts = text.split("-")
                if len(parts) != 3:
                    raise ValueError(f"Unsupported calendar date {value!r}.")
                year, month, day = map(int, parts)
        parsed_scale = TimeScale.parse(scale)
        seconds = float(seconds_of_day)
        day_length_limit = SECONDS_PER_DAY + 1.0 if parsed_scale is TimeScale.UTC else SECONDS_PER_DAY
        if not np.isfinite(seconds) or seconds < 0.0 or seconds >= day_length_limit:
            raise ValueError("seconds_of_day must be finite and within one UTC day.")
        if seconds < SECONDS_PER_DAY:
            hour = int(seconds // 3600.0)
            remainder = seconds - hour * 3600.0
            minute = int(remainder // 60.0)
            second = remainder - minute * 60.0
        else:
            hour, minute, second = 23, 59, 60.0 + seconds - SECONDS_PER_DAY
        jd1, jd2 = _civil_to_jd(parsed_scale, year, month, day, hour, minute, second)
        return cls(jd1, jd2, parsed_scale)

    def require_scale(self, scale: TimeScale | str, *, name: str = "epoch") -> Epoch:
        """Return this epoch after checking that it is on ``scale``.

        Parameters
        ----------
        scale : TimeScale or str
            Required time scale.
        name : str, optional
            Name of the epoch used in the error message.

        Returns
        -------
        Epoch
            ``self``.

        Raises
        ------
        ValueError
            If the epoch is on a different scale.
        """
        expected = TimeScale.parse(scale)
        if self.scale is not expected:
            raise ValueError(f"{name} must use the {expected.value.upper()} scale, got {self.scale.value.upper()}.")
        return self

    @property
    def jd(self) -> float:
        """Julian Date ``jd1 + jd2`` on the epoch's scale, in days."""
        return self.jd1 + self.jd2

    @property
    def mjd(self) -> float:
        """Modified Julian Date (JD - 2400000.5) on the epoch's scale, in days."""
        return self.jd - 2_400_000.5

    def shifted(self, seconds: float) -> Epoch:
        """Return the epoch shifted by an elapsed time.

        For UTC epochs the shift is applied on TAI, so it counts SI seconds across leap seconds. For TT
        and TDB it is added to the Julian Date.

        Parameters
        ----------
        seconds : float
            Elapsed time to add, in seconds; negative values shift backwards.

        Returns
        -------
        Epoch
            Shifted epoch on the same scale.

        Raises
        ------
        ValueError
            If ``seconds`` is not finite.
        """
        value = float(seconds)
        if not np.isfinite(value):
            raise ValueError("Epoch shift must be finite.")
        if self.scale is TimeScale.UTC:
            tai1, tai2 = _erfa_call(erfa.utctai, self.jd1, self.jd2)
            tai2 = tai2 + value / SECONDS_PER_DAY
            utc1, utc2 = _erfa_call(erfa.taiutc, tai1, tai2)
            return Epoch._from_normalized(TimeScale.UTC, utc1, utc2)
        return Epoch._from_normalized(self.scale, self.jd1, self.jd2 + value / SECONDS_PER_DAY)

    def seconds_until(self, other: Epoch) -> float:
        """Return the elapsed time from this epoch to ``other``.

        For UTC epochs the difference is taken on TAI, so leap seconds are counted.

        Parameters
        ----------
        other : Epoch
            Epoch on the same scale.

        Returns
        -------
        float
            ``other - self`` in seconds; positive when ``other`` is later.

        Raises
        ------
        ValueError
            If the two epochs are on different scales.
        """
        if self.scale is not other.scale:
            raise ValueError("Epoch differences require matching time scales.")
        if self.scale is TimeScale.UTC:
            self_tai = _erfa_call(erfa.utctai, self.jd1, self.jd2)
            other_tai = _erfa_call(erfa.utctai, other.jd1, other.jd2)
            return float(((other_tai[0] - self_tai[0]) + (other_tai[1] - self_tai[1])) * SECONDS_PER_DAY)
        return float(((other.jd1 - self.jd1) + (other.jd2 - self.jd2)) * SECONDS_PER_DAY)

    def date_iso(self) -> str:
        """Return the UTC calendar date as ``YYYY-MM-DD``.

        Raises
        ------
        ValueError
            If the epoch is not on the UTC scale.
        """
        self.require_scale(TimeScale.UTC, name="epoch")
        year, month, day, *_ = _civil_fields(self, 9)
        return f"{year:04d}-{month:02d}-{day:02d}"

    def isot(self, *, scale: TimeScale | str = TimeScale.UTC, precision: int = 9) -> str:
        """Format the epoch as ISO 8601 ``YYYY-MM-DDThh:mm:ss[.fff]`` text without a zone suffix.

        UTC and TT epochs can be formatted on either UTC or TT; TDB epochs cannot be formatted.

        Parameters
        ----------
        scale : TimeScale or str, optional
            Scale of the output text, UTC (default) or TT.
        precision : int, optional
            Number of decimal digits of the seconds field; default 9.

        Returns
        -------
        str
            ISO 8601 date-time on ``scale``.

        Raises
        ------
        ValueError
            If ``scale`` is TDB, the epoch is on TDB, or ``precision`` is negative.
        """
        target = TimeScale.parse(scale)
        if target is TimeScale.TDB:
            raise ValueError(
                "ISOT output is limited to UTC or TT. Format TDB as jd1/jd2, or convert it through the ephemeris first."
            )
        if self.scale is target:
            epoch = self
        elif self.scale is TimeScale.UTC and target is TimeScale.TT:
            epoch = utc2tt(self)
        elif self.scale is TimeScale.TT and target is TimeScale.UTC:
            epoch = tt2utc(self)
        else:
            raise ValueError("TDB epochs must be converted with TimeScaleConverter before ISOT formatting.")
        prec = int(precision)
        if prec < 0:
            raise ValueError("precision must be non-negative.")
        year, month, day, hour, minute, second, fraction = _civil_fields(epoch, prec)
        second_text = f"{second:02d}"
        if prec:
            second_text += f".{fraction:0{prec}d}"
        return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second_text}"

    def _comparison_key(self, other: object) -> tuple[float, float]:
        if not isinstance(other, Epoch):
            return NotImplemented
        if self.scale is not other.scale:
            raise ValueError("Epoch comparisons require matching time scales.")
        return (other.jd1, other.jd2)

    def __eq__(self, other: object) -> bool:
        """Return whether two epochs on the same scale have equal normalised Julian-Date parts."""
        key = self._comparison_key(other)
        if key is NotImplemented:
            return NotImplemented
        return (self.jd1, self.jd2) == key

    def __lt__(self, other: object) -> bool:
        """Return whether this epoch is earlier than ``other`` on the same scale."""
        key = self._comparison_key(other)
        if key is NotImplemented:
            return NotImplemented
        return (self.jd1, self.jd2) < key


def parse_time_with_utc_offset(
    value: object,
    *,
    utc_offset_hours: object = 0.0,
    name: str = "time",
) -> Epoch:
    """Parse a civil date or date-time and return it as a UTC epoch.

    The configured ``utc_offset_hours`` applies to text without a zone suffix: UTC = local time - offset.
    Text with an explicit ``Z`` or ISO offset uses that offset instead; when the configured offset is
    non-zero, the two must agree.

    Parameters
    ----------
    value : object
        ISO 8601 date or date-time text (see ``Epoch.from_isot`` for the accepted forms).
    utc_offset_hours : object, optional
        Offset of the local civil time from UTC, in hours, east positive (for example ``8`` for UTC+8);
        the value of the ``utcOffsetHours`` configuration key. Default 0.
    name : str, optional
        Name of the value used in error messages.

    Returns
    -------
    Epoch
        Epoch on the UTC scale.

    Raises
    ------
    ValueError
        If the text is not a valid date or date-time, or its explicit offset disagrees with a non-zero
        ``utc_offset_hours``.
    """
    offset = float(cast(float, utc_offset_hours))
    text = str(value).strip()
    try:
        year, month, day, hour, minute, second, embedded = _parse_isot(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO date/time.") from exc
    if embedded is not None:
        if abs(embedded - offset) > 1.0e-12 and offset != 0.0:
            raise ValueError(f"{name} has ISO offset {embedded:+g} h, which disagrees with utcOffsetHours={offset:g}.")
        epoch = Epoch.from_calendar(year, month, day, hour, minute, second)
        return epoch.shifted(-embedded * 3600.0) if embedded != 0.0 else epoch
    try:
        epoch = Epoch.from_calendar(year, month, day, hour, minute, second)
        return epoch.shifted(-offset * 3600.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid civil UTC/local ISO date or timestamp.") from exc


def format_time_with_utc_offset(
    epoch: Epoch,
    *,
    utc_offset_hours: object = 0.0,
    precision: int = 9,
) -> str:
    """Format a UTC epoch as local civil time followed by its ISO offset.

    Parameters
    ----------
    epoch : Epoch
        Epoch on the UTC scale.
    utc_offset_hours : object, optional
        Offset of the local civil time from UTC, in hours, east positive. The suffix ``±hh:mm`` is
        rounded to whole minutes and omitted when it rounds to zero. Default 0.
    precision : int, optional
        Number of decimal digits of the seconds field; default 9.

    Returns
    -------
    str
        Local ISO 8601 date-time, for example ``2025-05-01T20:00:00.000000000+08:00``.

    Raises
    ------
    ValueError
        If ``epoch`` is not on the UTC scale.
    """
    epoch.require_scale(TimeScale.UTC, name="epoch")
    offset = float(cast(float, utc_offset_hours))
    offset_minutes = offset * 60.0
    rounded_minutes = round(offset_minutes)
    local = epoch.shifted(offset * 3600.0)
    value = local.isot(precision=precision)
    if rounded_minutes == 0:
        return value
    sign = "+" if rounded_minutes >= 0 else "-"
    total = abs(rounded_minutes)
    return f"{value}{sign}{total // 60:02d}:{total % 60:02d}"


def require_tdb_epoch(epoch: Epoch, *, name: str = "epoch") -> Epoch:
    """Return ``epoch`` after checking that it is on the TDB scale; raise ``ValueError`` otherwise."""
    return epoch.require_scale(TimeScale.TDB, name=name)


__all__ = [
    "Epoch",
    "TimeScale",
    "format_time_with_utc_offset",
    "parse_time_with_utc_offset",
    "require_tdb_epoch",
    "tt2utc",
    "utc2tt",
]
