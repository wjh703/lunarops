"""Native station and reflector catalogs with explicit reference frames."""

from pathlib import Path

from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord

from .result_table import read_records, write_records

_POSITION_FIELDS = tuple((axis + "_m", "float") for axis in "xyz")
_STATION_FIELDS = (
    ("key", "text"), *_POSITION_FIELDS,
    *((axis + "_m_per_year", "float") for axis in ("vx", "vy", "vz")), ("epoch", "text"),
)
_REFLECTOR_FIELDS = (("key", "text"), *_POSITION_FIELDS)


def _write_catalog(catalog, path, *, station):
    rows = []
    fields = _STATION_FIELDS if station else _REFLECTOR_FIELDS
    for key, record in sorted(catalog.items()):
        if station:
            values = (key, *record.itrf_xyz_m, *record.itrf_velocity_m_per_year, record.position_epoch_utc)
        else:
            values = (key, *record.moon_fixed_xyz_m)
        rows.append(dict(zip((name for name, _ in fields), values)))
    return write_records(path, "stationCatalog" if station else "reflectorCatalog",
                         {"frame": "ITRF" if station else "MOON_PA"}, fields, rows)


def _read_catalog(path, *, station):
    header, rows = read_records(path, "stationCatalog" if station else "reflectorCatalog",
                                expected_fields=_STATION_FIELDS if station else _REFLECTOR_FIELDS)
    if header["frame"] != ("ITRF" if station else "MOON_PA"):
        raise ValueError("Unexpected catalog reference frame.")
    catalog: dict[str, StationRecord | ReflectorRecord] = {}
    keys = set()
    for row in rows:
        key = row["key"]
        if not key or key.casefold() in keys:
            raise ValueError(f"Invalid or duplicate catalog key {key!r}.")
        keys.add(key.casefold())
        position = [row[axis + "_m"] for axis in "xyz"]
        if station:
            velocity = [row[axis + "_m_per_year"] for axis in ("vx", "vy", "vz")]
            catalog[key] = StationRecord(key, position, velocity, row["epoch"])
        else:
            catalog[key] = ReflectorRecord(key, position)
    return catalog


def write_station_catalog(catalog: dict[str, StationRecord], path: str | Path) -> Path:
    return _write_catalog(catalog, path, station=True)


def read_station_catalog(path: str | Path) -> dict[str, StationRecord]:
    return _read_catalog(path, station=True)


def write_reflector_catalog(catalog: dict[str, ReflectorRecord], path: str | Path) -> Path:
    return _write_catalog(catalog, path, station=False)


def read_reflector_catalog(path: str | Path) -> dict[str, ReflectorRecord]:
    return _read_catalog(path, station=False)
