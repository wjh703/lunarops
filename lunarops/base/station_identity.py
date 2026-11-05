"""Station identifiers and their mapping to ILRS five-digit station codes.

Station IDs are exact catalog keys such as ``GRASSE`` or ``APOLLO``; they are compared without changing case
or whitespace.
"""

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


def validate_station_id(value: str) -> str:
    """Validate a station catalog ID and return it unchanged.

    Parameters
    ----------
    value : str
        Station ID.

    Returns
    -------
    str
        The same ID.

    Raises
    ------
    ValueError
        If the value is not a string, is empty, or has leading or trailing whitespace.
    """
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Station identifier must be a non-empty exact string.")
    return value


def station_id_from_ilrs_code(code: str) -> str:
    """Return the station ID of an ILRS five-digit station code, as read from MINI normal-point files.

    Parameters
    ----------
    code : str
        ILRS station code, such as ``"01910"``.

    Returns
    -------
    str
        Station ID, such as ``"GRASSE"``.

    Raises
    ------
    ValueError
        If the code is not registered.
    """
    try:
        return _ILRS_STATIONS[code]
    except KeyError:
        raise ValueError(f"Unknown ILRS station code {code!r}.") from None


def station_ilrs_code(value: str) -> str:
    """Return the ILRS five-digit station code of a station ID.

    Parameters
    ----------
    value : str
        Station ID, such as ``"GRASSE"``.

    Returns
    -------
    str
        ILRS station code, such as ``"01910"``.

    Raises
    ------
    ValueError
        If the ID is invalid or not registered.
    """
    station = validate_station_id(value)
    if station not in _STATION_ILRS_CODES:
        raise ValueError(f"Unknown registered station {value!r}.")
    return _STATION_ILRS_CODES[station]


__all__ = [
    "station_id_from_ilrs_code",
    "station_ilrs_code",
    "validate_station_id",
]
