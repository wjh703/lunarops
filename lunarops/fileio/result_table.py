"""Read and write records tables, the scalar ASCII layout shared by native text artifacts.

A records file is a native text archive (``.txt`` or ``.txt.gz``) with this layout::

    lunarops <artifactType>
    header <JSON object>
    fieldCount <n>
    field <name> <kind> <unit>        (n lines)
    recordCount <m>
    data
    <token> <token> ...               (m lines, one token per field)

``kind`` is ``bool``, ``int``, ``float`` or ``text``. Values are whitespace-separated tokens: booleans as
``0`` or ``1``, integers in decimal, floats in ``.17e`` notation unless a per-field precision is given, text
percent-encoded by ``encode_token``, and a missing value of any kind as ``~``. Field names and units are
encoded the same way. Text from ``#`` to the end of a line is a comment, and blank lines are ignored.

Each field's unit comes from the artifact's unit map when it lists the field, otherwise from the name suffix
(case-insensitive): ``_s`` s, ``_m`` m, ``_cm`` cm, ``_deg`` deg, ``_rad`` rad, ``_hpa`` hPa, ``_c`` degC,
``_k`` K, ``_nm`` nm, ``_percent`` %, and ``1`` for any other name. The reader rejects a declared unit that
differs from this expectation.
"""

from __future__ import annotations

import json
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
        return format_float(value, precision or ".17e")
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
    """Infer a records schema from row mappings.

    Field order is the order of first appearance across the rows. A field's kind comes from its non-None
    values: ``bool`` if all are booleans, ``int`` if all are integers, ``float`` if all are real numbers, and
    ``text`` if all are strings or every value is None or missing.

    Parameters
    ----------
    rows : sequence of mapping
        Rows mapping field names to scalar values.

    Returns
    -------
    tuple of (str, str)
        ``(name, kind)`` pairs.

    Raises
    ------
    ValueError
        If a numeric field holds a NaN or infinite value.
    TypeError
        If a field mixes value types.
    """
    names = list(dict.fromkeys(name for row in rows for name in row))
    return tuple((name, _field_type([row.get(name) for row in rows])) for name in names)


def write_records(path, artifact_type, header: dict, fields, rows, *, precision=None, field_units=None):
    """Write a records table atomically.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.
    artifact_type : str
        Artifact type written in the ``lunarops <artifactType>`` line.
    header : dict
        Artifact header, converted to plain data and written as one JSON line with sorted keys.
    fields : sequence of (str, str)
        ``(name, kind)`` pairs in column order; ``kind`` is ``bool``, ``int``, ``float`` or ``text``.
    rows : sequence of mapping
        Rows holding a value (or None for missing) for every field name; other keys are ignored.
    precision : mapping of str to str, optional
        Float format specification per field name; ``.17e`` for unlisted fields.
    field_units : mapping of str to str, optional
        Unit per field name, overriding the unit derived from the name suffix.

    Returns
    -------
    Path
        The written path with ``~`` expanded.

    Raises
    ------
    ValueError
        If a float is NaN or infinite, a text value is empty, the header holds NaN or infinity, or the
        artifact type or suffix is invalid.
    """
    precisions = precision or {}
    units = field_units or {}
    target = Path(path).expanduser()
    with atomic_text_writer(target, artifact_type) as stream:
        from lunarops.base.serialization import plain_data

        stream.write("header " + json.dumps(plain_data(header), sort_keys=True, allow_nan=False) + "\n")
        stream.write(f"fieldCount {len(fields)}\n")
        for name, kind in fields:
            stream.write(f"field {encode_token(name)} {kind} {encode_token(units[name] if name in units else _unit_for_field(name))}\n")
        stream.write(f"recordCount {len(rows)}\n")
        stream.write("data\n")
        for row in rows:
            stream.write(" ".join(_format_value(row[name], kind, precisions.get(name)) for name, kind in fields) + "\n")
    return target


def read_records(path, artifact_type, *, expected_fields=None, field_units=None):
    """Read and validate a records table.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.
    artifact_type : str
        Artifact type the header line must declare.
    expected_fields : tuple of (str, str), optional
        ``(name, kind)`` pairs the declared schema must equal, in order. Any schema is accepted when None.
    field_units : mapping of str to str, optional
        Expected unit per field name, overriding the unit derived from the name suffix.

    Returns
    -------
    header : dict
        Decoded JSON header.
    rows : list of dict
        One mapping per record, from field name to value (bool, int, float, str, or None for ``~``), in field
        order.

    Raises
    ------
    ValueError
        If the artifact type differs, a structural line is missing or malformed, the field count is not
        positive, a field name is empty or repeated, a kind is unknown, a unit differs from the expected unit,
        the schema differs from ``expected_fields``, the record count is negative or does not match the data
        rows, a row has the wrong number of tokens, or a value cannot be parsed.
    FileNotFoundError
        If the file does not exist.
    """
    units = field_units or {}
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, artifact_type)
        lines = iter(data_lines(stream))
        try:
            header_line = next(lines)
            if not header_line.startswith("header "):
                raise ValueError(f"Missing records header in {source}.")
            header = json.loads(header_line[7:])
            if not isinstance(header, dict):
                raise ValueError(f"Records header must be a mapping in {source}.")  # noqa: TRY004
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
            expected_unit = units[name] if name in units else _unit_for_field(name)
            if unit != expected_unit:
                raise ValueError(f"Table field {name!r} has unit {unit!r}; expected {expected_unit!r}.")
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
    return header, rows


__all__ = ["read_records", "write_records"]
