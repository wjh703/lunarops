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
    require_tdb_epoch,
    tt2utc,
    utc2tt,
    validate_utc_offset_hours,
)

__all__ = [
    "Epoch",
    "TdbTopocentricArguments",
    "TdbTopocentricArgumentsProvider",
    "TimeScale",
    "TimeScaleConverter",
    "format_time_with_utc_offset",
    "parse_time_with_utc_offset",
    "require_tdb_epoch",
    "tt2utc",
    "utc2tt",
    "validate_utc_offset_hours",
]
