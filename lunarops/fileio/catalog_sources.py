"""Read coordinate sources used to create native observation catalogs.

Creation programs accept either the compact inline sequence declared by their
program schema or a YAML coordinate source file.  A native catalog file is
also accepted as a source, which makes catalog regeneration and format
version upgrades explicit without exposing coordinate details to observation
programs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord
from lunarops.config.schema import FieldSpec
from lunarops.programs.specs import REFLECTOR_COORDINATE_SCHEMA, STATION_COORDINATE_SCHEMA

from .archive import is_text_path, open_text_reader
from .catalogs import read_reflector_catalog, read_station_catalog


_STATION_ENTRIES = FieldSpec(
    name="stationCoordinates",
    kind="sequence",
    item_kind="mapping",
    item_nested=STATION_COORDINATE_SCHEMA,
    min_items=1,
    allow_none=False,
)
_REFLECTOR_ENTRIES = FieldSpec(
    name="reflectorCoordinates",
    kind="sequence",
    item_kind="mapping",
    item_nested=REFLECTOR_COORDINATE_SCHEMA,
    min_items=0,
    allow_none=False,
)


def _coordinate_entries(value: object, field: FieldSpec, *, path: str) -> list[dict]:
    resolved = field.validate(value, path)
    if not isinstance(resolved, list):
        raise TypeError(f"{path} must resolve to a list of coordinate records.")
    return resolved


def station_catalog_from_coordinates(value: object, *, path: str = "stationCoordinates"):
    entries = _coordinate_entries(value, _STATION_ENTRIES, path=path)
    catalog = {}
    seen: set[str] = set()
    for entry in entries:
        key = str(entry["key"]).strip()
        folded = key.casefold()
        if folded in seen:
            raise ValueError(f"{path} contains duplicate key {key!r}.")
        seen.add(folded)
        catalog[key] = StationRecord(
            name=key,
            itrf_xyz_m=entry["xyzM"],
            itrf_velocity_m_per_year=entry["velocityMPerYear"],
            position_epoch_utc=entry["positionEpochUtc"],
        )
    return catalog


def reflector_catalog_from_coordinates(value: object, *, path: str = "reflectorCoordinates"):
    entries = _coordinate_entries(value, _REFLECTOR_ENTRIES, path=path)
    catalog = {}
    seen: set[str] = set()
    for entry in entries:
        key = str(entry["key"]).strip()
        folded = key.casefold()
        if folded in seen:
            raise ValueError(f"{path} contains duplicate key {key!r}.")
        seen.add(folded)
        catalog[key] = ReflectorRecord(name=key, moon_fixed_xyz_m=entry["xyzM"])
    return catalog


def _yaml_coordinate_entries(path: Path, *, key: str) -> object:
    import yaml

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid coordinate source YAML {path}: {exc}") from exc
    if isinstance(payload, Mapping):
        if key in payload:
            payload = payload[key]
        elif "coordinates" in payload:
            payload = payload["coordinates"]
        else:
            raise ValueError(f"Coordinate source {path} must contain {key!r} or 'coordinates'.")
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise ValueError(f"Coordinate source {path} must contain a coordinate sequence.")
    return payload


def _has_native_header(path: Path, artifact_type: str) -> bool:
    reader = open_text_reader(path) if is_text_path(path) else path.open("r", encoding="utf-8")
    with reader as stream:
        for raw in stream:
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            return text.startswith(f"lunarops {artifact_type} ")
    return False


def load_station_coordinate_source(path: str | Path):
    source = Path(path).expanduser()
    if _has_native_header(source, "stationCatalog"):
        return read_station_catalog(source)
    return station_catalog_from_coordinates(
        _yaml_coordinate_entries(source, key="stationCoordinates"),
        path=f"{source}:stationCoordinates",
    )


def load_reflector_coordinate_source(path: str | Path):
    source = Path(path).expanduser()
    if _has_native_header(source, "reflectorCatalog"):
        return read_reflector_catalog(source)
    return reflector_catalog_from_coordinates(
        _yaml_coordinate_entries(source, key="reflectorCoordinates"),
        path=f"{source}:reflectorCoordinates",
    )


__all__ = [
    "load_reflector_coordinate_source",
    "load_station_coordinate_source",
    "reflector_catalog_from_coordinates",
    "station_catalog_from_coordinates",
]
