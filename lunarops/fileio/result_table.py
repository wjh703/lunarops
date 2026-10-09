"""Schema and rows for scalar ASCII result tables."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

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


def _unit_for_field(name: str) -> str:
    lowered = name.casefold()
    for suffix, unit in (
        ("_rtt_s", "s"),
        ("_correction_s", "s"),
        ("_two_way_s", "s"),
        ("_s", "s"),
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
            raise ValueError("Table numeric fields must be finite.")
        return "float"
    if all(isinstance(value, str) for value in non_null):
        return "text"
    raise TypeError("Table fields must have one consistent scalar type.")


def _format_value(value: object, field_type: str, precision: str | None = None) -> str:
    if value is None:
        return "~"
    if field_type == "bool":
        return "1" if bool(value) else "0"
    if field_type == "int":
        return str(int(cast(Any, value)))
    if field_type == "float":
        return (
            format_float(value)
            if precision is None
            else format(parse_float(str(value), field="table value"), precision)
        )
    if str(value) == "":
        raise ValueError("Table text fields must not be empty.")
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
    raise ValueError(f"Unknown table field type {field_type!r}.")


def infer_fields(rows):
    names = list(dict.fromkeys(name for row in rows for name in row))
    return tuple((name, _field_type([row.get(name) for row in rows])) for name in names)


def write_table(rows, path, *, artifact_type, version, fields, precision=None):
    precisions = precision or {}
    target = Path(path).expanduser()
    with atomic_text_writer(target, artifact_type, version=version) as stream:
        stream.write(f"fieldCount {len(fields)}\n")
        for name, kind in fields:
            stream.write(f"field {encode_token(name)} {kind} {encode_token(_unit_for_field(name))}\n")
        stream.write(f"recordCount {len(rows)}\n")
        stream.write("data\n")
        for row in rows:
            stream.write(" ".join(_format_value(row[name], kind, precisions.get(name)) for name, kind in fields) + "\n")
    return target


def read_table(path, *, artifact_type, version, expected_fields=None):
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, artifact_type, expected_version=version)
        lines = iter(data_lines(stream))
        try:
            count_parts = next(lines).split()
        except StopIteration as exc:
            raise ValueError(f"Truncated table file {source}.") from exc
        if len(count_parts) != 2 or count_parts[0] != "fieldCount":
            raise ValueError(f"Malformed table field count in {source}.")
        field_count = int(count_parts[1])
        if field_count <= 0:
            raise ValueError("Table field count must be positive.")
        fields: list[tuple[str, str, str]] = []
        for _ in range(field_count):
            try:
                parts = next(lines).split()
            except StopIteration as exc:
                raise ValueError(f"Truncated table schema in {source}.") from exc
            if len(parts) != 4 or parts[0] != "field":
                raise ValueError(f"Malformed table field row in {source}.")
            name = decode_token(parts[1])
            field_type = parts[2]
            unit = decode_token(parts[3])
            if not name or name in {item[0] for item in fields}:
                raise ValueError(f"Invalid or duplicate table field {name!r}.")
            if field_type not in {"bool", "int", "float", "text"}:
                raise ValueError(f"Unknown table field type {field_type!r}.")
            if unit != _unit_for_field(name):
                raise ValueError(f"Table field {name!r} has unit {unit!r}; expected {_unit_for_field(name)!r}.")
            fields.append((name, field_type, unit))
        if expected_fields is not None and tuple((name, kind) for name, kind, _ in fields) != expected_fields:
            raise ValueError(f"{artifact_type} schema does not match its declared fields.")
        try:
            record_parts = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated table header in {source}.") from exc
        if len(record_parts) != 2 or record_parts[0] != "recordCount" or marker != "data":
            raise ValueError(f"Malformed table header in {source}.")
        record_count = int(record_parts[1])
        if record_count < 0:
            raise ValueError("Table record count must be non-negative.")
        rows: list[dict[str, object]] = []
        for row_number, line in enumerate(lines, start=1):
            values = line.split()
            if len(values) != field_count:
                raise ValueError(f"Table row {row_number} has {len(values)} fields; expected {field_count}.")
            rows.append(
                {
                    name: _parse_value(value, field_type, name)
                    for (name, field_type, _unit), value in zip(fields, values)
                }
            )
    if len(rows) != record_count:
        raise ValueError(f"Table file declares {record_count} rows, found {len(rows)}.")
    return rows


__all__ = ["read_table", "write_table"]
