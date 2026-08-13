"""Typed LunarOps matrices with GROOPS-inspired ASCII and binary encodings."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .archive import (
    atomic_binary_writer,
    atomic_text_writer,
    data_lines,
    format_float,
    is_binary_path,
    open_binary_reader,
    open_text_reader,
    parse_float,
    parse_header,
)

_BINARY_MAGIC = b"LLRMTX01"
_DTYPE_TO_CODE = {np.dtype("<f8"): 1, np.dtype("<i8"): 2}
_CODE_TO_DTYPE = {1: np.dtype("<f8"), 2: np.dtype("<i8")}
_KIND_TO_CODE = {"dense": 1, "lowerSymmetric": 2, "vector": 3}
_CODE_TO_KIND = {value: key for key, value in _KIND_TO_CODE.items()}
_TEXT_FORMAT_VERSION = 1


def _normalize_kind(kind: str, array: np.ndarray) -> str:
    text = str(kind)
    if text not in _KIND_TO_CODE:
        raise ValueError(f"Unsupported matrix kind {kind!r}.")
    if text == "vector" and array.ndim != 1:
        raise ValueError("A vector matrix must be one-dimensional.")
    if text == "lowerSymmetric":
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("A lowerSymmetric matrix must be square.")
        if not np.allclose(array, array.T, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("A lowerSymmetric matrix must be symmetric.")
    if text == "dense" and array.ndim != 2:
        raise ValueError("A dense matrix must be two-dimensional.")
    return text


def _normalize_array(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind in {"i", "u"}:
        array = np.asarray(raw, dtype="<i8")
    else:
        array = np.asarray(raw, dtype="<f8")
        if not np.all(np.isfinite(array)):
            raise ValueError("Matrix values must be finite.")
    if array.ndim not in (1, 2):
        raise ValueError("Matrices must be one- or two-dimensional.")
    return array


def _flat_payload(array: np.ndarray, kind: str) -> np.ndarray:
    if kind == "lowerSymmetric":
        indices = np.tril_indices(array.shape[0])
        return np.asarray(array[indices], dtype=array.dtype)
    return array.reshape(-1)


def _restore_payload(
    payload: np.ndarray,
    *,
    rows: int,
    columns: int,
    kind: str,
    dtype: np.dtype,
) -> np.ndarray:
    if kind == "vector":
        if columns != 1 or payload.size != rows:
            raise ValueError("Vector payload length does not match header.")
        return payload.astype(dtype, copy=False)
    if kind == "dense":
        if payload.size != rows * columns:
            raise ValueError("Dense matrix payload length does not match header.")
        return payload.reshape(rows, columns).astype(dtype, copy=False)
    if kind == "lowerSymmetric":
        if rows != columns:
            raise ValueError("lowerSymmetric matrix header must be square.")
        expected = rows * (rows + 1) // 2
        if payload.size != expected:
            raise ValueError("Lower-symmetric payload length does not match header.")
        matrix = np.zeros((rows, rows), dtype=dtype)
        indices = np.tril_indices(rows)
        matrix[indices] = payload
        matrix[(indices[1], indices[0])] = payload
        return matrix
    raise ValueError(f"Unsupported matrix kind {kind!r}.")


def write_matrix(path: str | Path, values, *, kind: str = "dense") -> Path:
    """Write a matrix as ``.txt[.gz]`` or ``.dat[.gz]`` based on *path*."""
    target = Path(path).expanduser()
    array = _normalize_array(values)
    matrix_kind = _normalize_kind(kind, array)
    rows = int(array.shape[0])
    columns = 1 if matrix_kind == "vector" else int(array.shape[1])
    payload = _flat_payload(array, matrix_kind)

    if is_binary_path(target):
        code = _DTYPE_TO_CODE.get(array.dtype)
        if code is None:
            raise ValueError(f"Unsupported binary matrix dtype {array.dtype}.")
        with atomic_binary_writer(target) as stream:
            stream.write(_BINARY_MAGIC)
            stream.write(
                struct.pack(
                    "<IIIIQQQ",
                    1,
                    code,
                    _KIND_TO_CODE[matrix_kind],
                    0,
                    rows,
                    columns,
                    payload.size,
                )
            )
            stream.write(np.asarray(payload, dtype=array.dtype.newbyteorder("<")).tobytes(order="C"))
        return target

    with atomic_text_writer(target, "matrix", version=_TEXT_FORMAT_VERSION) as stream:
        dtype_name = "int64" if array.dtype.kind in {"i", "u"} else "float64"
        stream.write(f"dtype {dtype_name}\n")
        stream.write(f"matrixType {matrix_kind}\n")
        stream.write(f"rows {rows}\n")
        stream.write(f"columns {columns}\n")
        stream.write(f"valueCount {payload.size}\n")
        stream.write("data\n")
        if matrix_kind == "vector":
            for value in payload:
                stream.write((str(int(value)) if dtype_name == "int64" else format_float(value)) + "\n")
        elif matrix_kind == "lowerSymmetric":
            for row in range(rows):
                values_row = array[row, : row + 1]
                formatter = (lambda value: str(int(value))) if dtype_name == "int64" else format_float
                stream.write(" ".join(formatter(value) for value in values_row) + "\n")
        else:
            formatter = (lambda value: str(int(value))) if dtype_name == "int64" else format_float
            for row in array:
                stream.write(" ".join(formatter(value) for value in row) + "\n")
    return target


def _read_binary_matrix(path: Path) -> tuple[np.ndarray, str]:
    with open_binary_reader(path) as stream:
        magic = stream.read(len(_BINARY_MAGIC))
        if magic != _BINARY_MAGIC:
            raise ValueError(f"Invalid LunarOps binary matrix magic in {path}.")
        header = stream.read(struct.calcsize("<IIIIQQQ"))
        if len(header) != struct.calcsize("<IIIIQQQ"):
            raise ValueError(f"Truncated LunarOps binary matrix header in {path}.")
        version, dtype_code, kind_code, reserved, rows, columns, count = struct.unpack("<IIIIQQQ", header)
        if version != 1:
            raise ValueError(f"Unsupported LunarOps binary matrix version {version}.")
        dtype = _CODE_TO_DTYPE.get(dtype_code)
        kind = _CODE_TO_KIND.get(kind_code)
        if dtype is None or kind is None or reserved != 0:
            raise ValueError(f"Unsupported LunarOps binary matrix encoding in {path}.")
        raw = stream.read()
    expected_bytes = int(count) * dtype.itemsize
    if len(raw) != expected_bytes:
        raise ValueError(f"Binary matrix payload size mismatch in {path}.")
    payload = np.frombuffer(raw, dtype=dtype).copy()
    return _restore_payload(
        payload,
        rows=int(rows),
        columns=int(columns),
        kind=kind,
        dtype=dtype,
    ), kind


def _read_text_matrix(path: Path) -> tuple[np.ndarray, str]:
    with open_text_reader(path) as stream:
        parse_header(stream, "matrix", expected_version=_TEXT_FORMAT_VERSION)
        lines = iter(data_lines(stream))
        try:
            dtype_line = next(lines).split()
            kind_line = next(lines).split()
            rows_line = next(lines).split()
            columns_line = next(lines).split()
            count_line = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated text matrix header in {path}.") from exc
        if (
            len(dtype_line) != 2
            or dtype_line[0] != "dtype"
            or len(kind_line) != 2
            or kind_line[0] != "matrixType"
            or len(rows_line) != 2
            or rows_line[0] != "rows"
            or len(columns_line) != 2
            or columns_line[0] != "columns"
            or len(count_line) != 2
            or count_line[0] != "valueCount"
            or marker != "data"
        ):
            raise ValueError(f"Malformed text matrix header in {path}.")
        dtype = {"float64": np.dtype("<f8"), "int64": np.dtype("<i8")}.get(dtype_line[1])
        if dtype is None:
            raise ValueError(f"Unsupported text matrix dtype {dtype_line[1]!r}.")
        kind = kind_line[1]
        if kind not in _KIND_TO_CODE:
            raise ValueError(f"Unsupported text matrix kind {kind!r}.")
        try:
            rows, columns, count = (
                int(rows_line[1]),
                int(columns_line[1]),
                int(count_line[1]),
            )
        except ValueError as exc:
            raise ValueError(f"Invalid text matrix dimensions in {path}.") from exc
        if min(rows, columns, count) < 0:
            raise ValueError(f"Text matrix dimensions must be non-negative in {path}.")
        tokens = [token for line in lines for token in line.split()]
        if len(tokens) != count:
            raise ValueError(f"Text matrix declares {count} values but contains {len(tokens)} in {path}.")
        if dtype.kind == "f":
            payload = np.asarray([parse_float(token, field="matrix") for token in tokens], dtype=dtype)
        else:
            try:
                payload = np.asarray([int(token) for token in tokens], dtype=dtype)
            except ValueError as exc:
                raise ValueError(f"Invalid integer matrix value in {path}.") from exc
    return _restore_payload(payload, rows=rows, columns=columns, kind=kind, dtype=dtype), kind


def read_matrix(path: str | Path, *, expected_kind: str | None = None) -> np.ndarray:
    source = Path(path).expanduser()
    array, kind = _read_binary_matrix(source) if is_binary_path(source) else _read_text_matrix(source)
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"Expected matrix kind {expected_kind!r}, found {kind!r} in {source}.")
    return array


def matrix_kind(path: str | Path) -> str:
    source = Path(path).expanduser()
    _array, kind = _read_binary_matrix(source) if is_binary_path(source) else _read_text_matrix(source)
    return kind


__all__ = ["matrix_kind", "read_matrix", "write_matrix"]
