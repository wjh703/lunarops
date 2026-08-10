"""Parameter-name, solution-vector, and covariance artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from lunarops.base.parameter_name import ParameterName, parameter_unit as _parameter_unit
from lunarops.estimation.parameter_products import CovarianceMatrix as _CovarianceMatrix
from lunarops.estimation.parameter_products import ParameterVector as _ParameterVector

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


def write_parameter_names(
    path: str | Path,
    names: Sequence[ParameterName],
    units: Sequence[str] | None = None,
) -> Path:
    values = list(names)
    if len(set(values)) != len(values):
        raise ValueError("Parameter names must be unique.")
    unit_values = list(units) if units is not None else [_parameter_unit(name) for name in values]
    if len(unit_values) != len(values):
        raise ValueError("Parameter units must match parameter names.")
    if any(not str(unit).strip() for unit in unit_values):
        raise ValueError("Parameter units must not be empty.")
    target = Path(path).expanduser()
    with atomic_text_writer(target, "parameterName") as stream:
        stream.write(f"parameterCount {len(values)}\n")
        stream.write("# object:type:temporal:interval unit\n")
        stream.write("data\n")
        for name, unit in zip(values, unit_values):
            stream.write(f"{encode_token(name)} {encode_token(unit)}\n")
    return target


def read_parameter_names(path: str | Path) -> tuple[list[ParameterName], list[str]]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, "parameterName")
        lines = iter(data_lines(stream))
        try:
            count_parts = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated parameter-name file {source}.") from exc
        if len(count_parts) != 2 or count_parts[0] != "parameterCount" or marker != "data":
            raise ValueError(f"Malformed parameter-name header in {source}.")
        count = int(count_parts[1])
        if count < 0:
            raise ValueError("Parameter count must be non-negative.")
        names: list[ParameterName] = []
        units: list[str] = []
        for line in lines:
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Malformed parameter-name row in {source}: {line!r}")
            names.append(ParameterName.parse(decode_token(fields[0])))
            units.append(decode_token(fields[1]))
    if len(names) != count:
        raise ValueError(f"Parameter-name file declares {count}, found {len(names)}.")
    if len(set(names)) != len(names):
        raise ValueError("Parameter-name file contains duplicates.")
    if any(not unit for unit in units):
        raise ValueError("Parameter-name file contains an empty unit.")
    return names, units


def write_parameter_vector(vector: _ParameterVector, path: str | Path) -> Path:
    target = Path(path).expanduser()
    with atomic_text_writer(target, "parameterVector") as stream:
        stream.write(f"vectorKind {encode_token(vector.kind)}\n")
        stream.write(f"parameterCount {len(vector.parameter_names)}\n")
        stream.write(f"hasUncertainty {1 if vector.uncertainties is not None else 0}\n")
        multiplier = (
            "~" if vector.uncertainty_sigma_multiplier is None else format_float(vector.uncertainty_sigma_multiplier)
        )
        stream.write(f"uncertaintySigmaMultiplier {multiplier}\n")
        stream.write("# parameter_name unit value uncertainty\n")
        stream.write("data\n")
        for index, (name, unit, value) in enumerate(zip(vector.parameter_names, vector.units, vector.values)):
            uncertainty = "~" if vector.uncertainties is None else format_float(vector.uncertainties[index])
            stream.write(f"{encode_token(name)} {encode_token(unit)} {format_float(value)} {uncertainty}\n")
    return target


def read_parameter_vector(path: str | Path) -> _ParameterVector:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, "parameterVector")
        lines = iter(data_lines(stream))
        try:
            kind_line = next(lines).split()
            count_line = next(lines).split()
            uncertainty_line = next(lines).split()
            multiplier_line = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated parameter vector {source}.") from exc
        if (
            len(kind_line) != 2
            or kind_line[0] != "vectorKind"
            or len(count_line) != 2
            or count_line[0] != "parameterCount"
            or len(uncertainty_line) != 2
            or uncertainty_line[0] != "hasUncertainty"
            or len(multiplier_line) != 2
            or multiplier_line[0] != "uncertaintySigmaMultiplier"
            or marker != "data"
        ):
            raise ValueError(f"Malformed parameter-vector header in {source}.")
        count = int(count_line[1])
        if count < 0 or uncertainty_line[1] not in {"0", "1"}:
            raise ValueError(f"Invalid parameter-vector count or hasUncertainty flag in {source}.")
        has_uncertainty = uncertainty_line[1] == "1"
        if has_uncertainty:
            if multiplier_line[1] == "~":
                raise ValueError("Parameter-vector uncertainty multiplier is missing despite hasUncertainty=1.")
            uncertainty_sigma_multiplier = parse_float(
                multiplier_line[1], field="parameter uncertainty sigma multiplier"
            )
        else:
            if multiplier_line[1] != "~":
                raise ValueError("Parameter-vector uncertainty multiplier is present despite hasUncertainty=0.")
            uncertainty_sigma_multiplier = None
        names: list[ParameterName] = []
        units: list[str] = []
        values: list[float] = []
        uncertainties: list[float] = []
        for line in lines:
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"Malformed parameter-vector row in {source}: {line!r}")
            names.append(ParameterName.parse(decode_token(fields[0])))
            units.append(decode_token(fields[1]))
            values.append(parse_float(fields[2], field="parameter value"))
            if has_uncertainty:
                if fields[3] == "~":
                    raise ValueError("Parameter-vector uncertainty is missing despite hasUncertainty=1.")
                uncertainties.append(parse_float(fields[3], field="parameter uncertainty"))
            elif fields[3] != "~":
                raise ValueError("Parameter-vector uncertainty is present despite hasUncertainty=0.")
    if len(names) != count:
        raise ValueError(f"Parameter vector declares {count}, found {len(names)}.")
    return _ParameterVector(
        parameter_names=tuple(names),
        values=np.asarray(values),
        units=tuple(units),
        uncertainties=(None if not has_uncertainty else np.asarray(uncertainties)),
        kind=decode_token(kind_line[1]),
        uncertainty_sigma_multiplier=uncertainty_sigma_multiplier,
    )


def _atomic_directory(target: Path, writer) -> Path:
    target = target.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    backup: Path | None = None
    try:
        writer(temporary)
        if target.exists():
            if not target.is_dir():
                raise FileExistsError(f"Artifact target exists and is not a directory: {target}")
            backup = target.parent / f".{target.name}.old.{os.getpid()}"
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(target, backup)
        os.replace(temporary, target)
        if backup is not None:
            shutil.rmtree(backup)
        return target
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_covariance(covariance: _CovarianceMatrix, path: str | Path) -> Path:
    target = require_file_group_path(path)

    def writer(directory: Path) -> None:
        write_parameter_names(
            directory / "parameterNames.txt",
            covariance.parameter_names,
            covariance.units,
        )
        write_matrix(directory / "covariance.dat.gz", covariance.matrix, kind="lowerSymmetric")
        with atomic_text_writer(directory / "info.txt", "covarianceInfo") as stream:
            stream.write(f"covarianceKind {encode_token(covariance.kind)}\n")
            stream.write(f"parameterCount {len(covariance.parameter_names)}\n")
            stream.write("matrixFile covariance.dat.gz\n")
            stream.write("parameterNamesFile parameterNames.txt\n")
            stream.write(f"matrixSha256 {sha256_file(directory / 'covariance.dat.gz')}\n")
            stream.write(f"parameterNamesSha256 {sha256_file(directory / 'parameterNames.txt')}\n")

    return _atomic_directory(target, writer)


def read_covariance(path: str | Path) -> _CovarianceMatrix:
    source = require_file_group_path(path)
    with open_text_reader(source / "info.txt") as stream:
        parse_header(stream, "covarianceInfo")
        lines = list(data_lines(stream))
    expected = [
        "covarianceKind",
        "parameterCount",
        "matrixFile",
        "parameterNamesFile",
        "matrixSha256",
        "parameterNamesSha256",
    ]
    if len(lines) != len(expected):
        raise ValueError(f"Malformed covariance info in {source}.")
    parsed = {}
    for line, key in zip(lines, expected):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[0] != key:
            raise ValueError(f"Malformed covariance info in {source}.")
        parsed[key] = parts[1]
    if parsed["matrixFile"] != "covariance.dat.gz" or parsed["parameterNamesFile"] != "parameterNames.txt":
        raise ValueError(f"Covariance group uses unexpected payload names in {source}.")
    matrix_path = source / parsed["matrixFile"]
    names_path = source / parsed["parameterNamesFile"]
    if sha256_file(matrix_path) != parsed["matrixSha256"]:
        raise ValueError("Covariance matrix checksum mismatch.")
    if sha256_file(names_path) != parsed["parameterNamesSha256"]:
        raise ValueError("Covariance parameter-name checksum mismatch.")
    names, units = read_parameter_names(names_path)
    if int(parsed["parameterCount"]) != len(names):
        raise ValueError("Covariance parameter count mismatch.")
    matrix = read_matrix(matrix_path, expected_kind="lowerSymmetric")
    return _CovarianceMatrix(tuple(names), matrix, tuple(units), decode_token(parsed["covarianceKind"]))


__all__ = [
    "read_covariance",
    "read_parameter_names",
    "read_parameter_vector",
    "write_covariance",
    "write_parameter_names",
    "write_parameter_vector",
]
