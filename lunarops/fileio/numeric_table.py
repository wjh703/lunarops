"""Read and write ``LUNARTBL`` binary tables of float64 values.

A numeric table is a native binary archive (``.dat`` or ``.dat.gz``) with this layout, all integers
little-endian:

1. A 24-byte prefix (``struct`` format ``<8sIQI``): the magic bytes ``LUNARTBL``, the JSON header length in
   bytes (uint32), the row count (uint64) and the column count (uint32).
2. A UTF-8 JSON header with the keys ``artifactType`` (str), ``columns`` (list of column names) and
   ``metadata`` (mapping), written with sorted keys and compact separators.
3. The payload: ``rows * columns`` little-endian float64 values in row-major order.

Columns carry no units; their meaning and units are defined by the artifact type. Vectors and symmetric
matrices are stored as one-column tables of artifact type ``matrix`` whose metadata holds ``kind``
(``vector`` or ``lowerSymmetric``) and ``rows``; a lower-symmetric matrix stores its lower triangle row by
row, in ``numpy.tril_indices`` order.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .archive import atomic_binary_writer, open_binary_reader

_MAGIC = b"LUNARTBL"
_PREFIX = struct.Struct("<8sIQI")


def write_numeric_table(
    path: str | Path,
    artifact_type: str,
    columns: tuple[str, ...],
    values,
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write a ``LUNARTBL`` numeric table atomically.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.dat`` or ``.dat.gz``.
    artifact_type : str
        Artifact type stored in the JSON header.
    columns : tuple of str
        Unique, non-empty column names.
    values : array_like, shape (n, len(columns))
        Finite table values, converted to little-endian float64.
    metadata : dict or None, optional
        JSON-serializable metadata stored in the header; an empty mapping when None.

    Returns
    -------
    Path
        The written path with ``~`` expanded.

    Raises
    ------
    ValueError
        If ``values`` is not a finite 2-D array matching ``columns``, if the artifact type or a column name is
        empty, if column names repeat, or if the suffix is invalid.
    """
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
        stream.write(_PREFIX.pack(_MAGIC, len(header), len(data), data.shape[1]))
        stream.write(header)
        stream.write(np.ascontiguousarray(data).tobytes())
    return target


def read_numeric_table(path: str | Path, expected_type: str | None = None):
    """Read a ``LUNARTBL`` numeric table.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.dat`` or ``.dat.gz``.
    expected_type : str or None, optional
        Artifact type the header must declare; any type is accepted when None.

    Returns
    -------
    artifact_type : str
        Artifact type from the header.
    columns : tuple of str
        Column names.
    values : ndarray, shape (rows, len(columns))
        Table values as float64.
    metadata : dict
        Header metadata; empty when the header has none.

    Raises
    ------
    ValueError
        If the prefix or header is truncated, the magic bytes are wrong, the artifact type differs from
        ``expected_type``, the header is inconsistent with the prefix, or the payload size does not equal
        ``rows * columns * 8`` bytes.
    FileNotFoundError
        If the file does not exist.
    """
    with open_binary_reader(path) as stream:
        prefix = stream.read(_PREFIX.size)
        if len(prefix) != _PREFIX.size:
            raise ValueError("Truncated LunarOps numeric table header")
        magic, header_size, rows, column_count = _PREFIX.unpack(prefix)
        if magic != _MAGIC:
            raise ValueError("Invalid LunarOps numeric table")
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
    """Return the artifact type of a ``LUNARTBL`` numeric table without reading its payload.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.dat`` or ``.dat.gz``.

    Returns
    -------
    str
        Artifact type from the JSON header.

    Raises
    ------
    ValueError
        If the prefix is truncated or the magic bytes are wrong.
    TypeError
        If the header's ``artifactType`` is not a string.
    FileNotFoundError
        If the file does not exist.
    """
    with open_binary_reader(path) as stream:
        prefix = stream.read(_PREFIX.size)
        if len(prefix) != _PREFIX.size:
            raise ValueError("Truncated LunarOps numeric table header")
        magic, header_size, _, _ = _PREFIX.unpack(prefix)
        if magic != _MAGIC:
            raise ValueError("Invalid LunarOps numeric table")
        header = json.loads(stream.read(header_size))
    artifact_type = header.get("artifactType")
    if not isinstance(artifact_type, str):
        raise TypeError("Invalid LunarOps numeric table artifact type")
    return artifact_type


def write_matrix(path: str | Path, values, *, kind: str) -> Path:
    """Write a vector or symmetric matrix as a one-column ``matrix`` numeric table.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.dat`` or ``.dat.gz``.
    values : array_like, shape (n,) or (n, n)
        Finite values: a vector for ``kind="vector"``, a square matrix for ``kind="lowerSymmetric"``. Only the
        lower triangle of a matrix is stored; symmetry is not checked.
    kind : {"vector", "lowerSymmetric"}
        Storage kind, recorded in the table metadata with ``rows = n``.

    Returns
    -------
    Path
        The written path with ``~`` expanded.

    Raises
    ------
    ValueError
        If ``kind`` is unsupported or the values are not finite.
    """
    array = np.asarray(values, dtype="<f8")
    if kind == "vector":
        payload = array
    elif kind == "lowerSymmetric":
        payload = array[np.tril_indices(len(array))]
    else:
        raise ValueError(f"Unsupported matrix kind {kind!r}.")
    return write_numeric_table(path, "matrix", ("value",), payload[:, None],
                               metadata={"kind": kind, "rows": len(array)})


def read_matrix(path: str | Path, *, expected_kind: str | None = None) -> np.ndarray:
    """Read a vector or symmetric matrix from a ``matrix`` numeric table.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.dat`` or ``.dat.gz``.
    expected_kind : {"vector", "lowerSymmetric"} or None, optional
        Storage kind the table must declare; either kind is accepted when None.

    Returns
    -------
    ndarray, shape (rows,) or (rows, rows)
        The vector, or the full symmetric matrix rebuilt from its stored lower triangle.

    Raises
    ------
    ValueError
        If the table is not a one-column ``matrix`` table with a valid kind and non-negative integer ``rows``,
        if the kind differs from ``expected_kind``, or if the payload length does not match ``rows``.
    """
    _, columns, values, metadata = read_numeric_table(path, "matrix")
    kind, rows = metadata["kind"], metadata["rows"]
    if columns != ("value",) or kind not in {"vector", "lowerSymmetric"} or not isinstance(rows, int) or rows < 0:
        raise ValueError("Invalid matrix table schema.")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"Expected matrix kind {expected_kind!r}, found {kind!r}.")
    payload = values[:, 0]
    if payload.size != (rows if kind == "vector" else rows * (rows + 1) // 2):
        raise ValueError("Matrix payload length does not match header.")
    if kind == "vector":
        return payload
    matrix = np.zeros((rows, rows), dtype=payload.dtype)
    indices = np.tril_indices(rows)
    matrix[indices] = payload
    matrix[(indices[1], indices[0])] = payload
    return matrix


__all__ = ["read_matrix", "read_numeric_table", "read_numeric_table_type", "write_matrix", "write_numeric_table"]
