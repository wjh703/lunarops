"""Read and write native station/reflector catalog artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from lunarops.classes.observation.catalogs import (
    ReflectorRecord as _ReflectorRecord,
)
from lunarops.classes.observation.catalogs import (
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

FORMAT_VERSION = 2


def write_station_catalog(catalog: Dict[str, _StationRecord], path: str | Path) -> Path:
    target = Path(path).expanduser()
    with atomic_text_writer(target, "stationCatalog", version=FORMAT_VERSION) as stream:
        stream.write("frame ITRF\n")
        stream.write(f"recordCount {len(catalog)}\n")
        stream.write(
            "# key x_m y_m z_m vx_m_per_year vy_m_per_year "
            "vz_m_per_year position_epoch_utc\n"
        )
        stream.write("data\n")
        for key, record in sorted(catalog.items()):
            position = np.asarray(record.itrf_xyz_m, dtype=float)
            velocity = np.asarray(record.itrf_velocity_m_per_year, dtype=float)
            fields = [
                encode_token(key),
                *(format_float(value) for value in position),
                *(format_float(value) for value in velocity),
                encode_token(record.position_epoch_utc),
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
        folded_keys: set[str] = set()
        for line in lines:
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(f"Malformed station catalog row in {source}: {line!r}")
            key = decode_token(fields[0])
            if not key:
                raise ValueError("Station catalog keys must not be empty.")
            if key in catalog or key.casefold() in folded_keys:
                raise ValueError(f"Duplicate station catalog key {key!r}.")
            folded_keys.add(key.casefold())
            catalog[key] = _StationRecord(
                name=key,
                itrf_xyz_m=[parse_float(value, field="station position") for value in fields[1:4]],
                itrf_velocity_m_per_year=[parse_float(value, field="station velocity") for value in fields[4:7]],
                position_epoch_utc=decode_token(fields[7]),
            )
    if len(catalog) != count:
        raise ValueError(f"Station catalog declares {count} records, found {len(catalog)}.")
    return catalog


def write_reflector_catalog(catalog: Dict[str, _ReflectorRecord], path: str | Path) -> Path:
    target = Path(path).expanduser()
    with atomic_text_writer(target, "reflectorCatalog", version=FORMAT_VERSION) as stream:
        stream.write("frame MOON_PA\n")
        stream.write(f"recordCount {len(catalog)}\n")
        stream.write("# key x_m y_m z_m\n")
        stream.write("data\n")
        for key, record in sorted(catalog.items()):
            position = np.asarray(record.moon_fixed_xyz_m, dtype=float)
            fields = [
                encode_token(key),
                *(format_float(value) for value in position),
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
        folded_keys: set[str] = set()
        for line in lines:
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"Malformed reflector catalog row in {source}: {line!r}")
            key = decode_token(fields[0])
            if not key:
                raise ValueError("Reflector catalog keys must not be empty.")
            if key in catalog or key.casefold() in folded_keys:
                raise ValueError(f"Duplicate reflector catalog key {key!r}.")
            folded_keys.add(key.casefold())
            catalog[key] = _ReflectorRecord(
                name=key,
                moon_fixed_xyz_m=[parse_float(value, field="reflector position") for value in fields[1:4]],
            )
    if len(catalog) != count:
        raise ValueError(f"Reflector catalog declares {count} records, found {len(catalog)}.")
    return catalog


__all__ = [
    "read_reflector_catalog",
    "read_station_catalog",
    "write_reflector_catalog",
    "write_station_catalog",
]
