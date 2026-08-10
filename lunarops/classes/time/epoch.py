"""Unified scalar epochs with explicit UTC/TT/TDB scale contracts.

``Epoch`` is the only time value used by LunarOps.  It stores a two-part Julian
Date plus an explicit scale.  ERFA owns civil parsing/formatting, UTC<->TAI,
TAI<->TT, leap-second handling, and UTC elapsed-time arithmetic. TT<->TDB
conversion lives in :class:`lunarops.classes.time.converter.TimeScaleConverter`,
because that step depends on the configured ephemeris.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
import math
import re
import warnings
import erfa
import numpy as np

from lunarops.base.constants import SECONDS_PER_DAY


class TimeScale(StrEnum):
    UTC = "utc"
    TT = "tt"
    TDB = "tdb"

    @classmethod
    def parse(cls, value: object) -> "TimeScale":
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
    r"(?:Z)?\s*$"
)


def _normalized_jd_pair(jd1: float, jd2: float) -> tuple[float, float]:
    first = float(jd1)
    second = float(jd2)
    if not np.isfinite(first) or not np.isfinite(second):
        raise ValueError("Epoch Julian-date parts must be finite.")
    carry = math.floor(second + 0.5)
    return first + carry, second - carry


def _erfa_call(name: str, *args):
    function = getattr(erfa, name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", erfa.ErfaWarning)
            return function(*args)
    except erfa.ErfaWarning as exc:
        raise ValueError(f"ERFA {name} rejected the epoch: {exc}") from exc
    except erfa.ErfaError as exc:
        raise ValueError(f"ERFA {name} failed: {exc}") from exc


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
        raise ValueError("TDB has no direct civil constructor in LunarOps.")
    jd1, jd2 = _erfa_call(
        "dtf2d",
        scale.value.upper(),
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
        float(second),
    )
    return float(jd1), float(jd2)


def _civil_fields(
    epoch: "Epoch",
    precision: int,
) -> tuple[int, int, int, int, int, int, int]:
    year, month, day, fields = _erfa_call(
        "d2dtf",
        epoch.scale.value.upper(),
        int(precision),
        epoch.jd1,
        epoch.jd2,
    )
    return (
        int(year),
        int(month),
        int(day),
        int(fields["h"]),
        int(fields["m"]),
        int(fields["s"]),
        int(fields["f"]),
    )


def _parse_isot(value: str) -> tuple[int, int, int, int, int, float]:
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
    return year, month, day, hour, minute, second


def _utc2tt_epoch(epoch: "Epoch") -> "Epoch":
    epoch.require_scale(TimeScale.UTC)
    tai1, tai2 = _erfa_call("utctai", epoch.jd1, epoch.jd2)
    tt1, tt2 = _erfa_call("taitt", tai1, tai2)
    return Epoch(tt1, tt2, TimeScale.TT)


def _tt2utc_epoch(epoch: "Epoch") -> "Epoch":
    epoch.require_scale(TimeScale.TT)
    tai1, tai2 = _erfa_call("tttai", epoch.jd1, epoch.jd2)
    utc1, utc2 = _erfa_call("taiutc", tai1, tai2)
    return Epoch(utc1, utc2, TimeScale.UTC)


@dataclass(frozen=True, slots=True, eq=False)
class Epoch:
    """Scalar epoch represented by two-part Julian date and explicit scale."""

    jd1: float
    jd2: float
    scale: TimeScale

    def __post_init__(self) -> None:
        jd1, jd2 = _normalized_jd_pair(self.jd1, self.jd2)
        object.__setattr__(self, "jd1", jd1)
        object.__setattr__(self, "jd2", jd2)
        object.__setattr__(self, "scale", TimeScale.parse(self.scale))

    @classmethod
    def from_isot(
        cls,
        value: str,
        *,
        scale: TimeScale | str = TimeScale.UTC,
    ) -> "Epoch":
        parsed_scale = TimeScale.parse(scale)
        if parsed_scale is TimeScale.TDB:
            raise ValueError(
                "TDB has no direct civil/ISOT constructor in LunarOps. "
                "Use Epoch(jd1, jd2, scale='tdb') or convert from TT with "
                "TimeScaleConverter."
            )
        year, month, day, hour, minute, second = _parse_isot(value)
        if not np.isfinite(second) or second < 0.0 or second >= 61.0:
            raise ValueError("second must be finite and in [0, 61).")
        jd1, jd2 = _civil_to_jd(
            parsed_scale,
            year,
            month,
            day,
            hour,
            minute,
            second,
        )
        return cls(jd1, jd2, parsed_scale)

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
    ) -> "Epoch":
        parsed_scale = TimeScale.parse(scale)
        if parsed_scale is TimeScale.TDB:
            raise ValueError("TDB has no direct civil/calendar constructor in LunarOps.")
        second_value = float(second)
        if not np.isfinite(second_value) or second_value < 0.0 or second_value >= 61.0:
            raise ValueError("second must be finite and in [0, 61).")
        jd1, jd2 = _civil_to_jd(
            parsed_scale,
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            second_value,
        )
        return cls(jd1, jd2, parsed_scale)

    @classmethod
    def from_date_seconds(
        cls,
        value: date | str,
        seconds_of_day: float,
        *,
        scale: TimeScale | str = TimeScale.UTC,
    ) -> "Epoch":
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
        if parsed_scale is TimeScale.TDB:
            raise ValueError("TDB has no direct civil date/seconds constructor in LunarOps.")
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
        jd1, jd2 = _civil_to_jd(
            parsed_scale,
            year,
            month,
            day,
            hour,
            minute,
            second,
        )
        return cls(jd1, jd2, parsed_scale)

    def require_scale(self, scale: TimeScale | str, *, name: str = "epoch") -> "Epoch":
        expected = TimeScale.parse(scale)
        if self.scale is not expected:
            raise ValueError(f"{name} must use the {expected.value.upper()} scale, got {self.scale.value.upper()}.")
        return self

    @property
    def jd(self) -> float:
        return self.jd1 + self.jd2

    @property
    def mjd(self) -> float:
        return self.jd - 2_400_000.5

    def shifted(self, seconds: float) -> "Epoch":
        value = float(seconds)
        if not np.isfinite(value):
            raise ValueError("Epoch shift must be finite.")
        if self.scale is TimeScale.UTC:
            tai1, tai2 = _erfa_call("utctai", self.jd1, self.jd2)
            tai2 = tai2 + value / SECONDS_PER_DAY
            utc1, utc2 = _erfa_call("taiutc", tai1, tai2)
            return Epoch(utc1, utc2, TimeScale.UTC)
        return Epoch(self.jd1, self.jd2 + value / SECONDS_PER_DAY, self.scale)

    def seconds_until(self, other: "Epoch") -> float:
        if not isinstance(other, Epoch):
            raise TypeError("other must be an Epoch.")
        if self.scale is not other.scale:
            raise ValueError("Epoch differences require matching time scales.")
        if self.scale is TimeScale.UTC:
            self_tai = _erfa_call("utctai", self.jd1, self.jd2)
            other_tai = _erfa_call("utctai", other.jd1, other.jd2)
            return float(((other_tai[0] - self_tai[0]) + (other_tai[1] - self_tai[1])) * SECONDS_PER_DAY)
        return float(((other.jd1 - self.jd1) + (other.jd2 - self.jd2)) * SECONDS_PER_DAY)

    def date_iso(self) -> str:
        self.require_scale(TimeScale.UTC, name="epoch")
        year, month, day, *_ = _civil_fields(self, 9)
        return f"{year:04d}-{month:02d}-{day:02d}"

    def to_datetime(self) -> datetime:
        """Return a naive ``datetime`` representation for UTC/TT civil I/O."""
        if self.scale is TimeScale.TDB:
            raise ValueError("TDB Epoch cannot be formatted as a civil datetime directly.")
        year, month, day, hour, minute, second, microsecond = _civil_fields(self, 6)
        return datetime(
            year,
            month,
            day,
            hour,
            minute,
            min(second, 59),
            microsecond,
        ) + timedelta(seconds=max(0, second - 59))

    def isot(
        self,
        converter=None,
        *,
        scale: TimeScale | str = TimeScale.UTC,
        precision: int = 9,
    ) -> str:
        target = TimeScale.parse(scale)
        if target is TimeScale.TDB:
            raise ValueError(
                "ISOT output is limited to UTC or TT. Format TDB as jd1/jd2, or convert it through the ephemeris first."
            )
        if self.scale is target:
            epoch = self
        elif self.scale is TimeScale.UTC and target is TimeScale.TT:
            epoch = _utc2tt_epoch(self)
        elif self.scale is TimeScale.TT and target is TimeScale.UTC:
            epoch = _tt2utc_epoch(self)
        else:
            if converter is None:
                raise RuntimeError("TDB formatting requires a TimeScaleConverter.")
            epoch = converter.convert(self, target)
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
        key = self._comparison_key(other)
        if key is NotImplemented:
            return NotImplemented
        return (self.jd1, self.jd2) == key

    def __lt__(self, other: object) -> bool:
        key = self._comparison_key(other)
        if key is NotImplemented:
            return NotImplemented
        return (self.jd1, self.jd2) < key

    def __le__(self, other: object) -> bool:
        key = self._comparison_key(other)
        if key is NotImplemented:
            return NotImplemented
        return (self.jd1, self.jd2) <= key

    def __gt__(self, other: object) -> bool:
        key = self._comparison_key(other)
        if key is NotImplemented:
            return NotImplemented
        return (self.jd1, self.jd2) > key

    def __ge__(self, other: object) -> bool:
        key = self._comparison_key(other)
        if key is NotImplemented:
            return NotImplemented
        return (self.jd1, self.jd2) >= key


def utc2tt(epoch: Epoch) -> Epoch:
    return _utc2tt_epoch(epoch)


def tt2utc(epoch: Epoch) -> Epoch:
    return _tt2utc_epoch(epoch)


__all__ = ["Epoch", "TimeScale", "utc2tt", "tt2utc"]
