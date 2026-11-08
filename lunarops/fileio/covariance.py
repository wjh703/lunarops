"""Read and write covariance matrix directory artifacts.

A covariance artifact is an extensionless file-group directory containing:

- ``info.txt``: structured text artifact ``covarianceInfo`` with ``covarianceKind`` (for example
  ``cofactor``) and ``parameterCount``.
- ``parameterNames.txt``: ``parameterName`` records table listing each parameter's name and unit.
- ``covariance.dat.gz``: ``LUNARTBL`` ``matrix`` table of kind ``lowerSymmetric`` holding the symmetric matrix,
  ordered like ``parameterNames.txt``. Element ``(i, j)`` is in the product of the units of parameters ``i``
  and ``j``.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from lunarops.estimation.lsq.products import CovarianceMatrix as _CovarianceMatrix

from .archive import atomic_directory_writer, require_file_group_path
from .numeric_table import read_matrix, write_matrix
from .parameter_names import read_parameter_names, write_parameter_names
from .yaml_artifact import read_structured_text, write_structured_text


def write_covariance(covariance: _CovarianceMatrix, path: str | Path) -> Path:
    """Write a covariance directory artifact atomically.

    Parameters
    ----------
    covariance : CovarianceMatrix
        Parameter names, units, symmetric matrix of shape ``(n, n)`` and covariance kind.
    path : str or Path
        Extensionless target directory.

    Returns
    -------
    Path
        The written directory.

    Raises
    ------
    ValueError
        If the path has a suffix.
    """
    target = require_file_group_path(path)
    with atomic_directory_writer(target) as directory:
        write_parameter_names(directory / "parameterNames.txt", covariance.parameter_names, covariance.units)
        write_matrix(directory / "covariance.dat.gz", covariance.matrix, kind="lowerSymmetric")
        write_structured_text(
            directory / "info.txt",
            "covarianceInfo",
            {
                "covarianceKind": covariance.kind,
                "parameterCount": len(covariance.parameter_names),
            },
        )
    return target


def read_covariance(path: str | Path) -> _CovarianceMatrix:
    """Read a covariance directory artifact.

    Parameters
    ----------
    path : str or Path
        Extensionless artifact directory.

    Returns
    -------
    CovarianceMatrix
        Parameter names, units, symmetric matrix as ``ndarray, shape (n, n)`` and covariance kind.

    Raises
    ------
    ValueError
        If the path has a suffix, ``parameterCount`` differs from the number of parameter names, the matrix
        is not stored as ``lowerSymmetric``, or a member file is invalid.
    """
    source = require_file_group_path(path)
    info = read_structured_text(source / "info.txt", "covarianceInfo")
    names, units = read_parameter_names(source / "parameterNames.txt")
    if cast(int, info["parameterCount"]) != len(names):
        raise ValueError("Covariance parameter count mismatch.")
    matrix = read_matrix(source / "covariance.dat.gz", expected_kind="lowerSymmetric")
    return _CovarianceMatrix(tuple(names), matrix, tuple(units), cast(str, info["covarianceKind"]))


__all__ = ["read_covariance", "write_covariance"]
