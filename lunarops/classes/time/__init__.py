"""Epoch values and explicit time-scale conversion services."""

from .converter import (
    TdbTopocentricArguments,
    TdbTopocentricArgumentsProvider,
    TimeScaleConverter,
)
from .epoch import (
    Epoch,
    TimeScale,
    format_time_with_utc_offset,
    parse_time_with_utc_offset,
    tt2utc,
    utc2tt,
    validate_utc_offset_hours,
)

__all__ = [
    "Epoch",
    "format_time_with_utc_offset",
    "parse_time_with_utc_offset",
    "TdbTopocentricArguments",
    "TdbTopocentricArgumentsProvider",
    "TimeScale",
    "TimeScaleConverter",
    "tt2utc",
    "utc2tt",
    "validate_utc_offset_hours",
]
