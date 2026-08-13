"""Versioned ASCII observation-result tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

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


def _unit_for_field(name: str) -> str:
    lowered = name.casefold()
    for suffix, unit in (
        ("_rtt_s", "s"),
        ("_correction_s", "s"),
        ("_two_way_s", "s"),
        ("_m", "m"),
        ("_cm", "cm"),
        ("_deg", "deg"),
        ("_rad", "rad"),
        ("_hpa", "hPa"),
        ("_c", "degC"),
        ("_k", "K"),
        ("_nm", "nm"),
        ("_percent", "%"),
    ):
        if lowered.endswith(suffix):
            return unit
    return "1"


def _field_type(values: Sequence[object]) -> str:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return "text"
    if all(isinstance(value, bool) for value in non_null):
        return "bool"
    if all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) for value in non_null):
        return "int"
    if all(isinstance(value, (int, float, np.number)) and not isinstance(value, bool) for value in non_null):
        numbers = np.asarray(non_null, dtype=float)
        if not np.all(np.isfinite(numbers)):
            raise ValueError("Observation-result numeric fields must be finite.")
        return "float"
    if all(isinstance(value, str) for value in non_null):
        return "text"
    raise TypeError("Observation-result fields must have one consistent scalar type.")


def _format_value(value: object, field_type: str) -> str:
    if value is None:
        return "~"
    if field_type == "bool":
        return "1" if bool(value) else "0"
    if field_type == "int":
        return str(int(cast(Any, value)))
    if field_type == "float":
        return format_float(value)
    if str(value) == "":
        raise ValueError("Observation-result text fields must not be empty.")
    return encode_token(value)


def _parse_value(value: str, field_type: str, field_name: str):
    if value == "~":
        return None
    if field_type == "bool":
        if value not in {"0", "1"}:
            raise ValueError(f"Invalid boolean {value!r} for {field_name}.")
        return value == "1"
    if field_type == "int":
        return int(value)
    if field_type == "float":
        return parse_float(value, field=field_name)
    if field_type == "text":
        return decode_token(value)
    raise ValueError(f"Unknown observation-result field type {field_type!r}.")


def write_observation_results(
    results_by_source: Mapping[str, Sequence[Mapping[str, object]]],
    path: str | Path,
) -> Path:
    rows: list[dict[str, object]] = []
    for source, source_rows in results_by_source.items():
        if not str(source):
            raise ValueError("Observation-result source names must not be empty.")
        for row in source_rows:
            if not isinstance(row, Mapping):
                raise TypeError("Observation-result rows must be mappings.")
            if "source" in row:
                raise ValueError("Observation-result rows must not define reserved field 'source'.")
            item = {"source": str(source), **dict(row)}
            rows.append(item)
    if not rows:
        raise ValueError("No observation results to write.")

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if not isinstance(name, str) or not name:
                raise ValueError("Observation-result field names must be non-empty strings.")
            if name not in seen:
                fields.append(name)
                seen.add(name)
    types = {name: _field_type([row.get(name) for row in rows]) for name in fields}
    target = Path(path).expanduser()
    with atomic_text_writer(target, "observationResult", version=FORMAT_VERSION) as stream:
        stream.write(f"fieldCount {len(fields)}\n")
        for name in fields:
            stream.write(f"field {encode_token(name)} {types[name]} {encode_token(_unit_for_field(name))}\n")
        stream.write(f"recordCount {len(rows)}\n")
        stream.write("data\n")
        for row in rows:
            stream.write(" ".join(_format_value(row.get(name), types[name]) for name in fields) + "\n")
    return target


def read_observation_results(path: str | Path) -> list[dict[str, object]]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, "observationResult", expected_version=FORMAT_VERSION)
        lines = iter(data_lines(stream))
        try:
            count_parts = next(lines).split()
        except StopIteration as exc:
            raise ValueError(f"Truncated observation-result file {source}.") from exc
        if len(count_parts) != 2 or count_parts[0] != "fieldCount":
            raise ValueError(f"Malformed observation-result field count in {source}.")
        field_count = int(count_parts[1])
        if field_count <= 0:
            raise ValueError("Observation-result field count must be positive.")
        fields: list[tuple[str, str, str]] = []
        for _ in range(field_count):
            try:
                parts = next(lines).split()
            except StopIteration as exc:
                raise ValueError(f"Truncated observation-result schema in {source}.") from exc
            if len(parts) != 4 or parts[0] != "field":
                raise ValueError(f"Malformed observation-result field row in {source}.")
            name = decode_token(parts[1])
            field_type = parts[2]
            unit = decode_token(parts[3])
            if not name or name in {item[0] for item in fields}:
                raise ValueError(f"Invalid or duplicate observation-result field {name!r}.")
            if field_type not in {"bool", "int", "float", "text"}:
                raise ValueError(f"Unknown observation-result field type {field_type!r}.")
            if unit != _unit_for_field(name):
                raise ValueError(
                    f"Observation-result field {name!r} has unit {unit!r}; expected {_unit_for_field(name)!r}."
                )
            fields.append((name, field_type, unit))
        if not fields or fields[0][0] != "source" or fields[0][1] != "text":
            raise ValueError("Observation-result schema must begin with a text 'source' field.")
        try:
            record_parts = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated observation-result header in {source}.") from exc
        if len(record_parts) != 2 or record_parts[0] != "recordCount" or marker != "data":
            raise ValueError(f"Malformed observation-result header in {source}.")
        record_count = int(record_parts[1])
        if record_count < 0:
            raise ValueError("Observation-result record count must be non-negative.")
        rows: list[dict[str, object]] = []
        for row_number, line in enumerate(lines, start=1):
            values = line.split()
            if len(values) != field_count:
                raise ValueError(
                    f"Observation-result row {row_number} has {len(values)} fields; expected {field_count}."
                )
            rows.append(
                {
                    name: _parse_value(value, field_type, name)
                    for (name, field_type, _unit), value in zip(fields, values)
                }
            )
    if len(rows) != record_count:
        raise ValueError(f"Observation-result file declares {record_count} rows, found {len(rows)}.")
    return rows


__all__ = ["read_observation_results", "write_observation_results"]
