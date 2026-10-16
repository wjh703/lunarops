"""Typed LunarOps matrices in a GROOPS-inspired binary encoding."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .archive import atomic_binary_writer, open_binary_reader

_BINARY_MAGIC = b"LLRMTX01"
_HEADER = "<IIIIQQQ"
_FLOAT64_CODE = 1
_KIND_TO_CODE = {"lowerSymmetric": 2, "vector": 3}
_CODE_TO_KIND = {value: key for key, value in _KIND_TO_CODE.items()}


def _flat_payload(array: np.ndarray, kind: str) -> np.ndarray:
    if kind == "vector":
        if array.ndim != 1:
            raise ValueError("A vector matrix must be one-dimensional.")
        return array
    if kind == "lowerSymmetric":
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("A lowerSymmetric matrix must be square.")
        if not np.allclose(array, array.T, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("A lowerSymmetric matrix must be symmetric.")
        return array[np.tril_indices(array.shape[0])]
    raise ValueError(f"Unsupported matrix kind {kind!r}.")


def _restore_payload(payload: np.ndarray, *, rows: int, columns: int, kind: str) -> np.ndarray:
    if kind == "vector":
        if columns != 1 or payload.size != rows:
            raise ValueError("Vector payload length does not match header.")
        return payload.copy()
    if rows != columns:
        raise ValueError("lowerSymmetric matrix header must be square.")
    if payload.size != rows * (rows + 1) // 2:
        raise ValueError("Lower-symmetric payload length does not match header.")
    matrix = np.zeros((rows, rows), dtype=payload.dtype)
    indices = np.tril_indices(rows)
    matrix[indices] = payload
    matrix[(indices[1], indices[0])] = payload
    return matrix


def write_matrix(path: str | Path, values, *, kind: str) -> Path:
    """Write a float64 ``vector`` or ``lowerSymmetric`` matrix as ``.dat[.gz]``."""
    target = Path(path).expanduser()
    array = np.asarray(values, dtype="<f8")
    if not np.all(np.isfinite(array)):
        raise ValueError("Matrix values must be finite.")
    payload = _flat_payload(array, kind)
    rows = int(array.shape[0])
    columns = 1 if kind == "vector" else rows
    with atomic_binary_writer(target) as stream:
        stream.write(_BINARY_MAGIC)
        stream.write(struct.pack(_HEADER, 1, _FLOAT64_CODE, _KIND_TO_CODE[kind], 0, rows, columns, payload.size))
        stream.write(payload.tobytes(order="C"))
    return target


def read_matrix(path: str | Path, *, expected_kind: str | None = None) -> np.ndarray:
    source = Path(path).expanduser()
    with open_binary_reader(source) as stream:
        magic = stream.read(len(_BINARY_MAGIC))
        if magic != _BINARY_MAGIC:
            raise ValueError(f"Invalid LunarOps binary matrix magic in {source}.")
        header = stream.read(struct.calcsize(_HEADER))
        if len(header) != struct.calcsize(_HEADER):
            raise ValueError(f"Truncated LunarOps binary matrix header in {source}.")
        version, dtype_code, kind_code, reserved, rows, columns, count = struct.unpack(_HEADER, header)
        if version != 1:
            raise ValueError(f"Unsupported LunarOps binary matrix version {version}.")
        kind = _CODE_TO_KIND.get(kind_code)
        if dtype_code != _FLOAT64_CODE or kind is None or reserved != 0:
            raise ValueError(f"Unsupported LunarOps binary matrix encoding in {source}.")
        raw = stream.read()
    if len(raw) != int(count) * 8:
        raise ValueError(f"Binary matrix payload size mismatch in {source}.")
    array = _restore_payload(np.frombuffer(raw, dtype="<f8"), rows=int(rows), columns=int(columns), kind=kind)
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"Expected matrix kind {expected_kind!r}, found {kind!r} in {source}.")
    return array


__all__ = ["read_matrix", "write_matrix"]
