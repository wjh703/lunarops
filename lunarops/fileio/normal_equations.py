"""Normal equations as typed GROOPS-style file groups.

The native artifact is a directory containing ``info.txt``, a symmetric normal
matrix, a right-hand-side vector, and parameter names.  JSON sidecars and NPZ
payloads are not supported.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote, unquote

import numpy as np
import yaml

import lunarops._normal_equations_core as _normal_equations_core
from lunarops.base.parameter_name import ParameterName

from .archive import (
    atomic_text_writer,
    data_lines,
    decode_token,
    encode_token,
    format_float,
    open_text_reader,
    parse_float,
    parse_header,
    require_file_group_path,
    sha256_file,
)
from .matrix import read_matrix, write_matrix
from .parameters import parameter_unit, read_parameter_names, write_parameter_names
from .structured_text import plain_data


SparseNormalRow = tuple[Iterable[tuple[int, float]], float, float]


def _encode_metadata(value: object) -> str:
    text = yaml.safe_dump(
        plain_data(value),
        default_flow_style=True,
        sort_keys=True,
        allow_unicode=True,
    ).strip()
    return quote(text, safe="")


def _decode_metadata(value: str):
    return yaml.safe_load(unquote(value))


def _replace_directory(target: Path, temporary: Path) -> None:
    backup: Path | None = None
    try:
        if target.exists():
            if not target.is_dir():
                raise FileExistsError(f"Normal-equation target exists and is not a directory: {target}")
            backup = target.parent / f".{target.name}.old.{os.getpid()}"
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(target, backup)
        os.replace(temporary, target)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise


class NormalEquations:
    """Mutable normal-equation accumulator and persistence object."""

    __slots__ = (
        "parameter_names",
        "parameter_units",
        "N",
        "W",
        "lPl",
        "obs_count",
        "meta",
    )

    def __init__(
        self,
        parameter_names: List[ParameterName],
        N: np.ndarray,
        W: np.ndarray,
        lPl: float = 0.0,
        obs_count: int = 0,
        meta: Optional[Dict[str, object]] = None,
        parameter_units: Optional[Sequence[str]] = None,
    ) -> None:
        self.parameter_names = parameter_names
        self.parameter_units = (
            [parameter_unit(name) for name in parameter_names] if parameter_units is None else list(parameter_units)
        )
        self.N = N
        self.W = W
        self.lPl = lPl
        self.obs_count = obs_count
        self.meta = {} if meta is None else meta
        self.__post_init__()

    def __repr__(self) -> str:
        return (
            "NormalEquations("
            f"parameter_count={len(self.parameter_names)}, "
            f"obs_count={self.obs_count}, lPl={self.lPl!r}, "
            f"meta_keys={tuple(self.meta)!r})"
        )

    def __post_init__(self) -> None:
        names = list(self.parameter_names)
        if not all(isinstance(name, ParameterName) for name in names):
            raise TypeError("Normal-equation parameter names must be ParameterName objects.")
        if len(set(names)) != len(names):
            raise ValueError("Normal-equation parameter names must be unique.")
        units = [str(unit).strip() for unit in self.parameter_units]
        if len(units) != len(names) or any(not unit for unit in units):
            raise ValueError("Normal-equation parameter units must align and be non-empty.")

        normal_matrix = np.asarray(self.N, dtype=float)
        right_hand_side = np.asarray(self.W, dtype=float).reshape(-1)
        parameter_count = len(names)
        if normal_matrix.shape != (parameter_count, parameter_count):
            raise ValueError(
                f"Normal matrix has shape {normal_matrix.shape}, expected {(parameter_count, parameter_count)}."
            )
        if right_hand_side.shape != (parameter_count,):
            raise ValueError(
                f"Normal right-hand side has shape {right_hand_side.shape}, expected {(parameter_count,)}."
            )
        if not np.all(np.isfinite(normal_matrix)) or not np.all(np.isfinite(right_hand_side)):
            raise ValueError("Normal equations contain non-finite matrix values.")
        if not np.allclose(normal_matrix, normal_matrix.T, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("Normal matrix must be symmetric.")
        if not np.isfinite(self.lPl) or float(self.lPl) < 0.0:
            raise ValueError("Normal-equation lPl must be finite and non-negative.")
        if isinstance(self.obs_count, bool) or int(self.obs_count) != self.obs_count or int(self.obs_count) < 0:
            raise ValueError("Normal-equation observation count must be a non-negative integer.")
        self.parameter_names = names
        self.parameter_units = units
        self.N = normal_matrix
        self.W = right_hand_side
        self.lPl = float(self.lPl)
        self.obs_count = int(self.obs_count)
        metadata: dict[str, object] = {}
        for key, value in dict(self.meta).items():
            if not isinstance(key, str) or not key:
                raise ValueError("Normal-equation metadata keys must be non-empty strings.")
            metadata[key] = value
        self.meta = metadata

    @classmethod
    def zeros(
        cls,
        parameter_names: Sequence[ParameterName],
        *,
        parameter_units: Optional[Sequence[str]] = None,
        **meta,
    ) -> "NormalEquations":
        names = list(parameter_names)
        count = len(names)
        return cls(
            parameter_names=names,
            parameter_units=parameter_units,
            N=np.zeros((count, count), dtype=float),
            W=np.zeros(count, dtype=float),
            lPl=0.0,
            obs_count=0,
            meta=dict(meta),
        )

    def _normalize_sparse_row(
        self,
        entries: Iterable[tuple[int, float]],
        observation: float,
        weight: float,
    ) -> tuple[list[int], list[float], float, float]:
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Observation weight must be finite and non-negative, got {weight!r}.")
        observation = float(observation)
        if not np.isfinite(observation):
            raise ValueError("Reduced observation must be finite.")

        coalesced: dict[int, float] = {}
        parameter_count = len(self.parameter_names)
        for raw_index, raw_value in entries:
            index = int(raw_index)
            if index < 0 or index >= parameter_count:
                raise ValueError(f"Sparse design column {index} is outside [0, {parameter_count}).")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError("Sparse design values must be finite.")
            if value:
                coalesced[index] = coalesced.get(index, 0.0) + value
        indices = list(coalesced)
        values = [coalesced[index] for index in indices]
        return indices, values, observation, weight

    def accumulate_sparse_rows(self, rows: Iterable[SparseNormalRow]) -> None:
        """Validate and accumulate a bounded batch of sparse observation rows."""
        offsets = [0]
        indices: list[int] = []
        values: list[float] = []
        observations: list[float] = []
        weights: list[float] = []
        for entries, observation, weight in rows:
            row_indices, row_values, value, weight_value = self._normalize_sparse_row(
                entries,
                observation,
                weight,
            )
            indices.extend(row_indices)
            values.extend(row_values)
            offsets.append(len(indices))
            observations.append(value)
            weights.append(weight_value)
        if not observations:
            return
        self.lPl += float(
            _normal_equations_core.accumulate_sparse_batch(
                self.N,
                self.W,
                np.asarray(offsets, dtype=np.intp),
                np.asarray(indices, dtype=np.intp),
                np.asarray(values, dtype=np.float64),
                np.asarray(observations, dtype=np.float64),
                np.asarray(weights, dtype=np.float64),
            )
        )
        self.obs_count += len(observations)

    def accumulate(self, A: np.ndarray, l: np.ndarray, sigma: np.ndarray) -> None:
        design = np.asarray(A, dtype=float)
        observations = np.asarray(l, dtype=float).reshape(-1)
        sigmas = np.asarray(sigma, dtype=float).reshape(-1)
        if design.ndim != 2:
            raise ValueError("Design matrix A must be two-dimensional.")
        if design.shape[1] != len(self.parameter_names):
            raise ValueError(f"Design matrix has {design.shape[1]} columns, expected {len(self.parameter_names)}.")
        if design.shape[0] != observations.size or observations.size != sigmas.size:
            raise ValueError("A, l and sigma dimensions are inconsistent.")
        if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            raise ValueError("Observation sigmas must be positive and finite.")
        weights = 1.0 / sigmas**2
        self.N += design.T @ (weights[:, None] * design)
        self.W += design.T @ (weights * observations)
        self.lPl += float(np.dot(weights, observations**2))
        self.obs_count += observations.size

    def _assert_compatible(self, other: "NormalEquations") -> None:
        left = self.meta.get("compatibility")
        right = other.meta.get("compatibility")
        if left != right:
            raise ValueError("Normal equations have incompatible scientific conventions.")
        left_units = dict(zip(self.parameter_names, self.parameter_units))
        for name, unit in zip(other.parameter_names, other.parameter_units):
            if name in left_units and left_units[name] != unit:
                raise ValueError(f"Parameter {name} has incompatible units {left_units[name]!r} and {unit!r}.")

    def add(self, other: "NormalEquations") -> "NormalEquations":
        if not isinstance(other, NormalEquations):
            raise TypeError("Can only add another NormalEquations object.")
        self._assert_compatible(other)
        union: List[ParameterName] = list(self.parameter_names)
        unit_by_name = dict(zip(self.parameter_names, self.parameter_units))
        index = {name: position for position, name in enumerate(union)}
        for name, unit in zip(other.parameter_names, other.parameter_units):
            if name not in index:
                index[name] = len(union)
                union.append(name)
                unit_by_name[name] = unit
        count = len(union)
        matrix = np.zeros((count, count), dtype=float)
        rhs = np.zeros(count, dtype=float)

        def scatter(source: "NormalEquations") -> None:
            indices = np.array([index[name] for name in source.parameter_names], dtype=int)
            matrix[np.ix_(indices, indices)] += source.N
            rhs[indices] += source.W

        scatter(self)
        scatter(other)
        return NormalEquations(
            parameter_names=union,
            parameter_units=[unit_by_name[name] for name in union],
            N=matrix,
            W=rhs,
            lPl=self.lPl + other.lPl,
            obs_count=self.obs_count + other.obs_count,
            meta={**other.meta, **self.meta},
        )

    def solve(self) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
        matrix = np.asarray(self.N, dtype=float)
        rhs = np.asarray(self.W, dtype=float)
        if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-14):
            raise np.linalg.LinAlgError("Normal matrix is not symmetric.")
        symmetric = 0.5 * (matrix + matrix.T)
        lower = np.linalg.cholesky(symmetric)
        solution = np.linalg.solve(lower.T, np.linalg.solve(lower, rhs))
        identity = np.eye(matrix.shape[0])
        covariance = np.linalg.solve(lower.T, np.linalg.solve(lower, identity))
        residual_quadratic = self.lPl - float(rhs @ solution)
        tolerance = 1.0e-10 * max(1.0, self.lPl, abs(float(rhs @ solution)))
        if residual_quadratic < -tolerance:
            raise np.linalg.LinAlgError(
                f"Normal-equation negative residual quadratic is beyond roundoff: lPl-W.T@x={residual_quadratic:.6e}."
            )
        residual_quadratic = max(residual_quadratic, 0.0)
        degrees_of_freedom = self.obs_count - len(self.parameter_names)
        sigma0 = None if degrees_of_freedom <= 0 else float(np.sqrt(residual_quadratic / degrees_of_freedom))
        return solution, covariance, sigma0

    def save(self, path: str | Path) -> Path:
        target = require_file_group_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            matrix_path = temporary / "normalMatrix.dat.gz"
            rhs_path = temporary / "rightHandSide.dat.gz"
            names_path = temporary / "parameterNames.txt"
            write_matrix(matrix_path, self.N, kind="lowerSymmetric")
            write_matrix(rhs_path, self.W, kind="vector")
            write_parameter_names(names_path, self.parameter_names, self.parameter_units)
            with atomic_text_writer(temporary / "info.txt", "normalEquationInfo") as stream:
                stream.write(f"observationCount {self.obs_count}\n")
                stream.write(f"lPl {format_float(self.lPl)}\n")
                stream.write(f"parameterCount {len(self.parameter_names)}\n")
                stream.write("normalMatrixFile normalMatrix.dat.gz\n")
                stream.write("rightHandSideFile rightHandSide.dat.gz\n")
                stream.write("parameterNamesFile parameterNames.txt\n")
                stream.write(f"normalMatrixSha256 {sha256_file(matrix_path)}\n")
                stream.write(f"rightHandSideSha256 {sha256_file(rhs_path)}\n")
                stream.write(f"parameterNamesSha256 {sha256_file(names_path)}\n")
                items = sorted(self.meta.items())
                stream.write(f"metadataCount {len(items)}\n")
                for key, value in items:
                    stream.write(f"metadata {encode_token(key)} {_encode_metadata(value)}\n")
            _replace_directory(target, temporary)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "NormalEquations":
        source = require_file_group_path(path)
        if not source.is_dir():
            raise FileNotFoundError(f"Normal-equation file group not found: {source}")
        with open_text_reader(source / "info.txt") as stream:
            parse_header(stream, "normalEquationInfo")
            lines = iter(data_lines(stream))

            def pair(expected: str) -> str:
                try:
                    line = next(lines)
                except StopIteration as exc:
                    raise ValueError(f"Truncated normal-equation info in {source}.") from exc
                parts = line.split(maxsplit=1)
                if len(parts) != 2 or parts[0] != expected:
                    raise ValueError(f"Expected {expected!r} in normal-equation info, found {line!r}.")
                return parts[1]

            observation_count = int(pair("observationCount"))
            lpl = parse_float(pair("lPl"), field="lPl")
            parameter_count = int(pair("parameterCount"))
            matrix_name = pair("normalMatrixFile")
            rhs_name = pair("rightHandSideFile")
            names_name = pair("parameterNamesFile")
            if (
                matrix_name != "normalMatrix.dat.gz"
                or rhs_name != "rightHandSide.dat.gz"
                or names_name != "parameterNames.txt"
            ):
                raise ValueError(f"Normal-equation group uses unexpected payload names in {source}.")
            matrix_hash = pair("normalMatrixSha256")
            rhs_hash = pair("rightHandSideSha256")
            names_hash = pair("parameterNamesSha256")
            metadata_count = int(pair("metadataCount"))
            metadata: dict[str, object] = {}
            for _ in range(metadata_count):
                try:
                    metadata_line = next(lines)
                except StopIteration as exc:
                    raise ValueError(f"Truncated normal-equation metadata in {source}.") from exc
                fields = metadata_line.split()
                if len(fields) != 3 or fields[0] != "metadata":
                    raise ValueError(f"Malformed normal-equation metadata row {metadata_line!r}.")
                key = decode_token(fields[1])
                if key in metadata:
                    raise ValueError(f"Duplicate normal-equation metadata key {key!r}.")
                metadata[key] = _decode_metadata(fields[2])
            try:
                extra = next(lines)
            except StopIteration:
                extra = None
            if extra is not None:
                raise ValueError(f"Unexpected normal-equation info row {extra!r}.")

        matrix_path = source / matrix_name
        rhs_path = source / rhs_name
        names_path = source / names_name
        for payload, expected in (
            (matrix_path, matrix_hash),
            (rhs_path, rhs_hash),
            (names_path, names_hash),
        ):
            if sha256_file(payload) != expected:
                raise ValueError(f"Normal-equation payload checksum mismatch: {payload}")
        names, units = read_parameter_names(names_path)
        if len(names) != parameter_count:
            raise ValueError("Normal-equation parameter count does not match parameter names.")
        matrix = read_matrix(matrix_path, expected_kind="lowerSymmetric")
        rhs = read_matrix(rhs_path, expected_kind="vector")
        return cls(
            parameter_names=names,
            parameter_units=units,
            N=matrix,
            W=rhs,
            lPl=lpl,
            obs_count=observation_count,
            meta=metadata,
        )


__all__ = ["NormalEquations"]
