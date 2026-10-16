"""Exact public station IDs and explicit ILRS source-code mappings."""

_STATION_ILRS_CODES = {
    "APOLLO": "70610",
    "GRASSE": "01910",
    "HALEAKALA": "56610",
    "MATERA": "07941",
    "MCDONALD": "71110",
    "MLRS1": "71111",
    "MLRS2": "71112",
    "WETTZELL": "08834",
}
_ILRS_STATIONS = {code: key for key, code in _STATION_ILRS_CODES.items()}


def canonical_station_id(value: str) -> str:
    """Validate an exact catalog ID without changing its spelling."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Station identifier must be a non-empty exact string.")
    return value


def station_id_from_ilrs_code(code: str) -> str:
    """Resolve the five-digit station code at the MINI input boundary."""
    try:
        return _ILRS_STATIONS[code]
    except KeyError:
        raise ValueError(f"Unknown ILRS station code {code!r}.") from None


def station_ilrs_code(value: str) -> str:
    station = canonical_station_id(value)
    if station not in _STATION_ILRS_CODES:
        raise ValueError(f"Unknown registered station {value!r}.")
    return _STATION_ILRS_CODES[station]


__all__ = [
    "canonical_station_id",
    "station_id_from_ilrs_code",
    "station_ilrs_code",
]
