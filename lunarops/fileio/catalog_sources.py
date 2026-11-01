"""Read coordinate sources used to create native observation catalogs.

Creation programs accept either the compact inline sequence declared by their
program schema or a YAML coordinate source file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from lunarops.config.coordinates import REFLECTOR_COORDINATE_SCHEMA, STATION_COORDINATE_SCHEMA, validate_coordinate_keys
from lunarops.config.schema import FieldSpec

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
    source = Path(path).expanduser()
    location = f"{source}:stationCoordinates"
    entries = _STATION_ENTRIES.validate(_yaml_coordinate_entries(source, key="stationCoordinates"), location)
    return validate_coordinate_keys(entries, location)


def load_reflector_coordinate_source(path: str | Path):
    source = Path(path).expanduser()
    location = f"{source}:reflectorCoordinates"
    entries = _REFLECTOR_ENTRIES.validate(_yaml_coordinate_entries(source, key="reflectorCoordinates"), location)
    return validate_coordinate_keys(entries, location)


__all__ = [
    "load_reflector_coordinate_source",
    "load_station_coordinate_source",
]
