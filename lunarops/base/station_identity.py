"""Exact public station IDs and explicit ILRS source-code mappings."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _StationIdentity:
    ilrs_code: str
    display_name: str


_STATIONS = {
    "APOLLO": _StationIdentity("70610", "Apache Point Observatory"),
    "GRASSE": _StationIdentity("01910", "Grasse"),
    "HALEAKALA": _StationIdentity("56610", "Haleakala"),
    "MATERA": _StationIdentity("07941", "Matera"),
    "MCDONALD": _StationIdentity("71110", "McDonald 2.70"),
    "MLRS1": _StationIdentity("71111", "McDonald MLRS1"),
    "MLRS2": _StationIdentity("71112", "McDonald MLRS2"),
    "WETTZELL": _StationIdentity("08834", "Wettzell"),
}
_ILRS_STATIONS = {record.ilrs_code: key for key, record in _STATIONS.items()}


def canonical_station_id(value: str) -> str:
    """Validate an exact catalog ID without changing its spelling."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Station identifier must be a non-empty exact string.")
    return value


def registered_station_id(value: str) -> str:
    station = canonical_station_id(value)
    if station not in _STATIONS:
        raise ValueError(f"Unknown registered station {value!r}.")
    return station


def station_id_from_ilrs_code(code: str) -> str:
    """Resolve the five-digit station code at the MINI input boundary."""
    try:
        return _ILRS_STATIONS[code]
    except KeyError:
        raise ValueError(f"Unknown ILRS station code {code!r}.") from None


def station_ilrs_code(value: str) -> str:
    return _STATIONS[registered_station_id(value)].ilrs_code


def station_display_name(value: str) -> str:
    return _STATIONS[registered_station_id(value)].display_name


__all__ = [
    "canonical_station_id",
    "registered_station_id",
    "station_id_from_ilrs_code",
    "station_ilrs_code",
    "station_display_name",
]
