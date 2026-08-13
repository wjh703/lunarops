"""Read, write, and load native station/reflector catalog artifacts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np

from lunarops.classes.observation.catalogs import (
    ReflectorRecord as _ReflectorRecord,
    StationRecord as _StationRecord,
)

from .archive import (
    atomic_text_writer,
    data_lines,
    decode_token,
    encode_token,
    format_float,
    open_text_reader,
    parse_float,
    parse_header,
)

FORMAT_VERSION = 1


def write_station_catalog(catalog: Dict[str, _StationRecord], path: str | Path) -> Path:
    if any(not str(key).strip() for key in catalog):
        raise ValueError("Station catalog keys must not be empty.")
    if not all(isinstance(record, _StationRecord) for record in catalog.values()):
        raise TypeError("Station catalogs must contain StationRecord values.")
    target = Path(path).expanduser()
    with atomic_text_writer(target, "stationCatalog", version=FORMAT_VERSION) as stream:
        stream.write("frame ITRF\n")
        stream.write(f"recordCount {len(catalog)}\n")
        stream.write(
            "# key name x_m y_m z_m vx_m_per_year vy_m_per_year "
            "vz_m_per_year position_epoch_utc alias_count aliases...\n"
        )
        stream.write("data\n")
        for key, record in sorted(catalog.items()):
            position = np.asarray(record.itrf_xyz_m, dtype=float)
            velocity = np.asarray(record.itrf_velocity_m_per_year, dtype=float)
            fields = [
                encode_token(key),
                encode_token(record.name),
                *(format_float(value) for value in position),
                *(format_float(value) for value in velocity),
                encode_token(record.position_epoch_utc),
                str(len(record.aliases)),
                *(encode_token(alias) for alias in record.aliases),
            ]
            stream.write(" ".join(fields) + "\n")
    return target


def read_station_catalog(path: str | Path) -> Dict[str, _StationRecord]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, "stationCatalog", expected_version=FORMAT_VERSION)
        lines = iter(data_lines(stream))
        try:
            frame = next(lines).split()
            count_line = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated station catalog header in {source}.") from exc
        if frame != ["frame", "ITRF"] or len(count_line) != 2 or count_line[0] != "recordCount" or marker != "data":
            raise ValueError(f"Malformed station catalog header in {source}.")
        count = int(count_line[1])
        if count < 0:
            raise ValueError("Station catalog record count must be non-negative.")
        catalog: Dict[str, _StationRecord] = {}
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                raise ValueError(f"Malformed station catalog row in {source}: {line!r}")
            alias_count = int(fields[9])
            if alias_count < 0 or len(fields) != 10 + alias_count:
                raise ValueError(f"Station alias count mismatch in {source}: {line!r}")
            key = decode_token(fields[0])
            if not key:
                raise ValueError("Station catalog keys must not be empty.")
            if key in catalog:
                raise ValueError(f"Duplicate station catalog key {key!r}.")
            catalog[key] = _StationRecord(
                name=decode_token(fields[1]),
                itrf_xyz_m=[parse_float(value, field="station position") for value in fields[2:5]],
                itrf_velocity_m_per_year=[parse_float(value, field="station velocity") for value in fields[5:8]],
                position_epoch_utc=decode_token(fields[8]),
                aliases=tuple(decode_token(value) for value in fields[10:]),
            )
    if len(catalog) != count:
        raise ValueError(f"Station catalog declares {count} records, found {len(catalog)}.")
    return catalog


def write_reflector_catalog(catalog: Dict[str, _ReflectorRecord], path: str | Path) -> Path:
    if any(not str(key).strip() for key in catalog):
        raise ValueError("Reflector catalog keys must not be empty.")
    if not all(isinstance(record, _ReflectorRecord) for record in catalog.values()):
        raise TypeError("Reflector catalogs must contain ReflectorRecord values.")
    target = Path(path).expanduser()
    with atomic_text_writer(target, "reflectorCatalog", version=FORMAT_VERSION) as stream:
        stream.write("frame MOON_PA\n")
        stream.write(f"recordCount {len(catalog)}\n")
        stream.write("# key name x_m y_m z_m alias_count aliases...\n")
        stream.write("data\n")
        for key, record in sorted(catalog.items()):
            position = np.asarray(record.moon_fixed_xyz_m, dtype=float)
            fields = [
                encode_token(key),
                encode_token(record.name),
                *(format_float(value) for value in position),
                str(len(record.aliases)),
                *(encode_token(alias) for alias in record.aliases),
            ]
            stream.write(" ".join(fields) + "\n")
    return target


def read_reflector_catalog(path: str | Path) -> Dict[str, _ReflectorRecord]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, "reflectorCatalog", expected_version=FORMAT_VERSION)
        lines = iter(data_lines(stream))
        try:
            frame = next(lines).split()
            count_line = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated reflector catalog header in {source}.") from exc
        if frame != ["frame", "MOON_PA"] or len(count_line) != 2 or count_line[0] != "recordCount" or marker != "data":
            raise ValueError(f"Malformed reflector catalog header in {source}.")
        count = int(count_line[1])
        if count < 0:
            raise ValueError("Reflector catalog record count must be non-negative.")
        catalog: Dict[str, _ReflectorRecord] = {}
        for line in lines:
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"Malformed reflector catalog row in {source}: {line!r}")
            alias_count = int(fields[5])
            if alias_count < 0 or len(fields) != 6 + alias_count:
                raise ValueError(f"Reflector alias count mismatch in {source}: {line!r}")
            key = decode_token(fields[0])
            if not key:
                raise ValueError("Reflector catalog keys must not be empty.")
            if key in catalog:
                raise ValueError(f"Duplicate reflector catalog key {key!r}.")
            catalog[key] = _ReflectorRecord(
                name=decode_token(fields[1]),
                moon_fixed_xyz_m=[parse_float(value, field="reflector position") for value in fields[2:5]],
                aliases=tuple(decode_token(value) for value in fields[6:]),
            )
    if len(catalog) != count:
        raise ValueError(f"Reflector catalog declares {count} records, found {len(catalog)}.")
    return catalog


def load_station_catalog(source: object) -> Dict[str, _StationRecord]:
    """Load a builtin, native-file, or already constructed station catalog."""
    if isinstance(source, dict) and all(isinstance(v, _StationRecord) for v in source.values()):
        return source
    if source in (None, "builtin"):
        from lunarops.classes.observation.builtin_catalogs import STATIONS

        return copy.deepcopy(STATIONS)
    if not isinstance(source, (str, Path)):
        raise TypeError("Station catalog source must be a path, 'builtin', or a station mapping.")
    return read_station_catalog(source)


def load_reflector_catalog(source: object) -> Dict[str, _ReflectorRecord]:
    """Load a builtin, native-file, or already constructed reflector catalog."""
    if isinstance(source, dict) and all(isinstance(v, _ReflectorRecord) for v in source.values()):
        return source
    if source in (None, "builtin"):
        from lunarops.classes.observation.builtin_catalogs import REFLECTORS

        return copy.deepcopy(REFLECTORS)
    if not isinstance(source, (str, Path)):
        raise TypeError("Reflector catalog source must be a path, 'builtin', or a reflector mapping.")
    return read_reflector_catalog(source)


__all__ = [
    "load_reflector_catalog",
    "load_station_catalog",
    "read_reflector_catalog",
    "read_station_catalog",
    "write_reflector_catalog",
    "write_station_catalog",
]
