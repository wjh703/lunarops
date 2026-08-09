"""Read and write native GROOPS-style normal-equation file groups."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote, unquote

import yaml

from lunarops.base.serialization import plain_data as _plain_data
from lunarops.estimation.normal_equations import NormalEquations as _NormalEquations

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
from .parameters import read_parameter_names, write_parameter_names


def _encode_metadata(value: object) -> str:
    text = yaml.safe_dump(
        _plain_data(value),
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


def write_normal_equations(normals: _NormalEquations, path: str | Path) -> Path:
    """Persist one validated normal-equation system as a file group."""
    if not isinstance(normals, _NormalEquations):
        raise TypeError("normals must be a NormalEquations object.")
    target = require_file_group_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        matrix_path = temporary / "normalMatrix.dat.gz"
        rhs_path = temporary / "rightHandSide.dat.gz"
        names_path = temporary / "parameterNames.txt"
        write_matrix(matrix_path, normals.N, kind="lowerSymmetric")
        write_matrix(rhs_path, normals.W, kind="vector")
        write_parameter_names(names_path, normals.parameter_names, normals.parameter_units)
        with atomic_text_writer(temporary / "info.txt", "normalEquationInfo") as stream:
            stream.write(f"observationCount {normals.obs_count}\n")
            stream.write(f"lPl {format_float(normals.lPl)}\n")
            stream.write(f"parameterCount {len(normals.parameter_names)}\n")
            stream.write("normalMatrixFile normalMatrix.dat.gz\n")
            stream.write("rightHandSideFile rightHandSide.dat.gz\n")
            stream.write("parameterNamesFile parameterNames.txt\n")
            stream.write(f"normalMatrixSha256 {sha256_file(matrix_path)}\n")
            stream.write(f"rightHandSideSha256 {sha256_file(rhs_path)}\n")
            stream.write(f"parameterNamesSha256 {sha256_file(names_path)}\n")
            items = sorted(normals.meta.items())
            stream.write(f"metadataCount {len(items)}\n")
            for key, value in items:
                stream.write(f"metadata {encode_token(key)} {_encode_metadata(value)}\n")
        _replace_directory(target, temporary)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def read_normal_equations(path: str | Path) -> _NormalEquations:
    """Read and validate one native normal-equation file group."""
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
    return _NormalEquations(
        parameter_names=names,
        parameter_units=units,
        N=matrix,
        W=rhs,
        lPl=lpl,
        obs_count=observation_count,
        meta=metadata,
    )


__all__ = ["read_normal_equations", "write_normal_equations"]
