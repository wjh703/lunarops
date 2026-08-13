"""Read and write LunarOps normal-equation groups informed by GROOPS."""

from __future__ import annotations

from pathlib import Path
from lunarops.estimation.normal_equations import NormalEquations as _NormalEquations

from .archive import (
    atomic_directory_writer,
    atomic_text_writer,
    data_lines,
    format_float,
    open_text_reader,
    parse_float,
    parse_header,
    require_file_group_path,
    sha256_file,
)
from .matrix import read_matrix, write_matrix
from .parameter_names import read_parameter_names, write_parameter_names
from .yaml_artifact import read_structured_text, write_structured_text

FORMAT_VERSION = 2
_PAYLOAD_NAMES = (
    "normalMatrix.dat.gz",
    "rightHandSide.dat.gz",
    "x0.dat.gz",
    "parameterNames.txt",
    "metadata.txt",
)


def write_normal_equations(normals: _NormalEquations, path: str | Path) -> Path:
    """Persist one absolute system ``N x = W`` with its linearization ``x0``."""
    if not isinstance(normals, _NormalEquations):
        raise TypeError("normals must be a NormalEquations object.")
    target = require_file_group_path(path)
    with atomic_directory_writer(target) as directory:
        matrix_path = directory / "normalMatrix.dat.gz"
        rhs_path = directory / "rightHandSide.dat.gz"
        x0_path = directory / "x0.dat.gz"
        names_path = directory / "parameterNames.txt"
        write_matrix(matrix_path, normals.N, kind="lowerSymmetric")
        write_matrix(rhs_path, normals.W, kind="vector")
        write_matrix(x0_path, normals.x0, kind="vector")
        write_parameter_names(names_path, normals.parameter_names, normals.parameter_units)
        write_structured_text(directory / "metadata.txt", "normalEquationMetadata", normals.meta)
        with atomic_text_writer(
            directory / "info.txt", "normalEquationInfo", version=FORMAT_VERSION
        ) as stream:
            stream.write(f"observationCount {normals.obs_count}\n")
            stream.write(f"lPlAtX0 {format_float(normals.lPl)}\n")
            stream.write(f"parameterCount {len(normals.parameter_names)}\n")
            stream.write(f"payloadCount {len(_PAYLOAD_NAMES)}\n")
            for name in _PAYLOAD_NAMES:
                stream.write(f"payload {name} {sha256_file(directory / name)}\n")
    return target


def read_normal_equations(path: str | Path) -> _NormalEquations:
    """Read and validate one absolute-convention normal-equation group."""
    source = require_file_group_path(path)
    if not source.is_dir():
        raise FileNotFoundError(f"Normal-equation file group not found: {source}")
    with open_text_reader(source / "info.txt") as stream:
        parse_header(stream, "normalEquationInfo", expected_version=FORMAT_VERSION)
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
        lpl = parse_float(pair("lPlAtX0"), field="lPl at x0")
        parameter_count = int(pair("parameterCount"))
        payload_count = int(pair("payloadCount"))
        if min(observation_count, parameter_count, payload_count) < 0:
            raise ValueError("Normal-equation counts must be non-negative.")
        payloads: dict[str, str] = {}
        for _ in range(payload_count):
            try:
                payload_line = next(lines)
            except StopIteration as exc:
                raise ValueError(f"Truncated normal-equation payload list in {source}.") from exc
            fields = payload_line.split()
            if len(fields) != 3 or fields[0] != "payload" or fields[1] in payloads:
                raise ValueError(f"Malformed normal-equation payload row {payload_line!r}.")
            payloads[fields[1]] = fields[2]
        try:
            extra = next(lines)
        except StopIteration:
            extra = None
        if extra is not None:
            raise ValueError(f"Unexpected normal-equation info row {extra!r}.")
    if set(payloads) != set(_PAYLOAD_NAMES):
        raise ValueError(f"Normal-equation group has unexpected payloads: {sorted(payloads)!r}.")
    for name, expected in payloads.items():
        if sha256_file(source / name) != expected:
            raise ValueError(f"Normal-equation payload checksum mismatch: {name}")
    matrix_path = source / "normalMatrix.dat.gz"
    rhs_path = source / "rightHandSide.dat.gz"
    x0_path = source / "x0.dat.gz"
    names_path = source / "parameterNames.txt"
    metadata = read_structured_text(source / "metadata.txt", "normalEquationMetadata")
    names, units = read_parameter_names(names_path)
    if len(names) != parameter_count:
        raise ValueError("Normal-equation parameter count does not match parameter names.")
    matrix = read_matrix(matrix_path, expected_kind="lowerSymmetric")
    rhs = read_matrix(rhs_path, expected_kind="vector")
    x0 = read_matrix(x0_path, expected_kind="vector")
    if x0.size != parameter_count:
        raise ValueError("Normal-equation x0 does not match parameter count.")
    return _NormalEquations(
        parameter_names=names,
        parameter_units=units,
        N=matrix,
        W=rhs,
        lPl=lpl,
        obs_count=observation_count,
        meta=metadata,
        x0=x0,
    )


__all__ = ["read_normal_equations", "write_normal_equations"]
