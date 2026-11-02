"""Read and write covariance matrix directory artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from lunarops.estimation.parameter_products import CovarianceMatrix as _CovarianceMatrix

from .archive import atomic_directory_writer, require_file_group_path
from .numeric_table import read_matrix, write_matrix
from .parameter_names import read_parameter_names, write_parameter_names
from .yaml_artifact import read_structured_text, write_structured_text


def write_covariance(covariance: _CovarianceMatrix, path: str | Path) -> Path:
    target = require_file_group_path(path)
    with atomic_directory_writer(target) as directory:
        write_parameter_names(directory / "parameterNames.txt", covariance.parameter_names, covariance.units)
        write_matrix(directory / "covariance.dat.gz", covariance.matrix, kind="lowerSymmetric")
        write_structured_text(directory / "info.txt", "covarianceInfo", {
            "covarianceKind": covariance.kind, "parameterCount": len(covariance.parameter_names),
        })
    return target


def read_covariance(path: str | Path) -> _CovarianceMatrix:
    source = require_file_group_path(path)
    info = read_structured_text(source / "info.txt", "covarianceInfo")
    names, units = read_parameter_names(source / "parameterNames.txt")
    if cast(int, info["parameterCount"]) != len(names):
        raise ValueError("Covariance parameter count mismatch.")
    matrix = read_matrix(source / "covariance.dat.gz", expected_kind="lowerSymmetric")
    return _CovarianceMatrix(tuple(names), matrix, tuple(units), cast(str, info["covarianceKind"]))


__all__ = ["read_covariance", "write_covariance"]
