"""Read YAML coordinate source files used to create native station and reflector catalogs.

Catalog creation programs accept either the inline coordinate sequence declared by their program schema or a
YAML coordinate source file read by this module. Station entries give ITRF coordinates and reflector entries
give Moon PA coordinates, as defined by the coordinate schemas in :mod:`lunarops.fileio.catalog_coordinates`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from lunarops.config.schema import FieldSpec
from lunarops.fileio.catalog_coordinates import (
    REFLECTOR_COORDINATE_SCHEMA,
    STATION_COORDINATE_SCHEMA,
    validate_coordinate_keys,
)

_STATION_ENTRIES = FieldSpec(
    name="stationCoordinates",
    kind="sequence",
    item_kind="mapping",
    item_nested=STATION_COORDINATE_SCHEMA,
    min_items=1,
)
_REFLECTOR_ENTRIES = FieldSpec(
    name="reflectorCoordinates",
    kind="sequence",
    item_kind="mapping",
    item_nested=REFLECTOR_COORDINATE_SCHEMA,
    min_items=0,
)


def _yaml_coordinate_entries(path: Path, *, key: str) -> object:
    """Return the coordinate sequence of a YAML file: the whole document or its ``key`` entry if a mapping."""
    import yaml

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid coordinate source YAML {path}: {exc}") from exc
    if isinstance(payload, Mapping):
        if key not in payload:
            raise ValueError(f"Coordinate source {path} must contain {key!r}.")
        payload = payload[key]
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise ValueError(f"Coordinate source {path} must contain a coordinate sequence.")  # noqa: TRY004 -- invalid file content
    return payload


def load_station_coordinate_source(path: str | Path):
    """Read and validate a YAML station coordinate source file.

    The file holds either a sequence of station entries or a mapping whose ``stationCoordinates`` key holds
    that sequence. Each entry is validated against the station coordinate schema (``key`` plus an ITRF
    position or geodetic coordinates); at least one entry is required and keys must be unique
    (case-insensitive).

    Parameters
    ----------
    path : str or Path
        YAML file path; ``~`` is expanded.

    Returns
    -------
    list of mapping
        Validated station entries in file order.

    Raises
    ------
    ValueError
        If the YAML is invalid, the coordinate sequence is missing, an entry is invalid or keys repeat.
    TypeError
        If an entry value has the wrong type.
    """
    source = Path(path).expanduser()
    location = f"{source}:stationCoordinates"
    entries = _STATION_ENTRIES.validate(_yaml_coordinate_entries(source, key="stationCoordinates"), location)
    return validate_coordinate_keys(entries, location)


def load_reflector_coordinate_source(path: str | Path):
    """Read and validate a YAML reflector coordinate source file.

    The file holds either a sequence of reflector entries or a mapping whose ``reflectorCoordinates`` key
    holds that sequence. Each entry is validated against the reflector coordinate schema (``key`` and
    ``xyzM``, the Moon PA position in metres); the sequence may be empty and keys must be unique
    (case-insensitive).

    Parameters
    ----------
    path : str or Path
        YAML file path; ``~`` is expanded.

    Returns
    -------
    list of mapping
        Validated reflector entries in file order.

    Raises
    ------
    ValueError
        If the YAML is invalid, the coordinate sequence is missing, an entry is invalid or keys repeat.
    TypeError
        If an entry value has the wrong type.
    """
    source = Path(path).expanduser()
    location = f"{source}:reflectorCoordinates"
    entries = _REFLECTOR_ENTRIES.validate(_yaml_coordinate_entries(source, key="reflectorCoordinates"), location)
    return validate_coordinate_keys(entries, location)


__all__ = [
    "load_reflector_coordinate_source",
    "load_station_coordinate_source",
]
