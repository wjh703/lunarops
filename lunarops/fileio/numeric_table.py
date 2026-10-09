"""Compact native binary tables for regular scientific time series."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .archive import atomic_binary_writer, open_binary_reader

_MAGIC = b"LUNARTBL"
_PREFIX = struct.Struct("<8sIIQI")
_VERSION = 1


def write_numeric_table(
    path: str | Path,
    artifact_type: str,
    columns: tuple[str, ...],
    values,
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    target = Path(path).expanduser()
    data = np.asarray(values, dtype="<f8")
    if data.ndim != 2 or data.shape[1] != len(columns) or not np.all(np.isfinite(data)):
        raise ValueError("Numeric table values must be a finite matrix matching its columns")
    if not artifact_type or not columns or any(not name for name in columns) or len(set(columns)) != len(columns):
        raise ValueError("Numeric table type and column names must be unique and non-empty")
    header = json.dumps(
        {"artifactType": artifact_type, "columns": columns, "metadata": metadata or {}},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    with atomic_binary_writer(target) as stream:
        stream.write(_PREFIX.pack(_MAGIC, _VERSION, len(header), len(data), data.shape[1]))
        stream.write(header)
        stream.write(np.ascontiguousarray(data).tobytes())
    return target


def read_numeric_table(path: str | Path, expected_type: str | None = None):
    with open_binary_reader(path) as stream:
        prefix = stream.read(_PREFIX.size)
        if len(prefix) != _PREFIX.size:
            raise ValueError("Truncated LunarOps numeric table header")
        magic, version, header_size, rows, column_count = _PREFIX.unpack(prefix)
        if magic != _MAGIC or version != _VERSION:
            raise ValueError("Invalid or unsupported LunarOps numeric table")
        raw_header = stream.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError("Truncated LunarOps numeric table metadata")
        header = json.loads(raw_header)
        artifact_type = header.get("artifactType")
        columns = header.get("columns")
        if expected_type is not None and artifact_type != expected_type:
            raise ValueError(f"Expected {expected_type!r}, found {artifact_type!r}")
        if not isinstance(artifact_type, str) or not isinstance(columns, list) or len(columns) != column_count:
            raise ValueError("Invalid LunarOps numeric table metadata")
        payload = stream.read()
    expected_bytes = rows * column_count * 8
    if len(payload) != expected_bytes:
        raise ValueError("LunarOps numeric table payload size mismatch")
    values = np.frombuffer(payload, dtype="<f8").reshape(rows, column_count).copy()
    return artifact_type, tuple(columns), values, header.get("metadata", {})


def read_numeric_table_type(path: str | Path) -> str:
    with open_binary_reader(path) as stream:
        prefix = stream.read(_PREFIX.size)
        if len(prefix) != _PREFIX.size:
            raise ValueError("Truncated LunarOps numeric table header")
        magic, version, header_size, _, _ = _PREFIX.unpack(prefix)
        if magic != _MAGIC or version != _VERSION:
            raise ValueError("Invalid or unsupported LunarOps numeric table")
        header = json.loads(stream.read(header_size))
    artifact_type = header.get("artifactType")
    if not isinstance(artifact_type, str):
        raise TypeError("Invalid LunarOps numeric table artifact type")
    return artifact_type


__all__ = ["read_numeric_table", "read_numeric_table_type", "write_numeric_table"]
