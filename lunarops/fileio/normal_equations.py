"""Read and write normal-equation directory artifacts.

A normal-equation artifact stores one absolute system ``N x = W`` with its linearization point ``x0`` in an
extensionless file-group directory containing:

- ``info.txt``: structured text artifact ``normalEquationInfo`` with ``observationCount``, ``lPlAtX0`` (the
  weighted reduced-observation quadratic form at ``x0``) and ``parameterCount``.
- ``metadata.txt``: structured text artifact ``normalEquationMetadata`` with the system's metadata mapping.
- ``parameterNames.txt``: ``parameterName`` records table listing each parameter's name and unit.
- ``normalMatrix.dat.gz``: ``LUNARTBL`` ``matrix`` table of kind ``lowerSymmetric`` holding ``N``.
- ``rightHandSide.dat.gz``: ``LUNARTBL`` ``matrix`` table of kind ``vector`` holding ``W``.
- ``x0.dat.gz``: ``LUNARTBL`` ``matrix`` table of kind ``vector`` holding ``x0``, in the units of
  ``parameterNames.txt``.

All arrays are ordered like ``parameterNames.txt``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from lunarops.estimation.normal_equations import NormalEquations as _NormalEquations

from .archive import atomic_directory_writer, require_file_group_path
from .numeric_table import read_matrix, write_matrix
from .parameter_names import read_parameter_names, write_parameter_names
from .yaml_artifact import read_structured_text, write_structured_text


def write_normal_equations(normals: _NormalEquations, path: str | Path) -> Path:
    """Write one absolute system ``N x = W`` with its linearization point ``x0`` atomically.

    Parameters
    ----------
    normals : NormalEquations
        Normal matrix ``N``, shape ``(n, n)``; right-hand side ``W``, shape ``(n,)``; linearization point
        ``x0``, shape ``(n,)``; parameter names and units; ``lPl``; observation count; and metadata.
    path : str or Path
        Extensionless target directory.

    Returns
    -------
    Path
        The written directory.

    Raises
    ------
    ValueError
        If the path has a suffix or an array is not finite.
    """
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
        write_structured_text(
            directory / "info.txt",
            "normalEquationInfo",
            {
                "observationCount": normals.obs_count,
                "lPlAtX0": normals.lPl,
                "parameterCount": len(normals.parameter_names),
            },
        )
    return target


def read_normal_equations(path: str | Path) -> _NormalEquations:
    """Read a normal-equation directory artifact.

    Parameters
    ----------
    path : str or Path
        Extensionless artifact directory.

    Returns
    -------
    NormalEquations
        System with ``N`` as ``ndarray, shape (n, n)``, ``W`` and ``x0`` as ``ndarray, shape (n,)``, parameter
        names and units, ``lPl``, observation count and metadata.

    Raises
    ------
    ValueError
        If the path has a suffix, ``observationCount`` or ``parameterCount`` is negative, the parameter-name
        count or an array shape differs from ``parameterCount``, ``lPlAtX0`` is not finite and non-negative,
        or a member file is invalid.
    """
    source = require_file_group_path(path)
    info = read_structured_text(source / "info.txt", "normalEquationInfo")
    observation_count = cast(int, info["observationCount"])
    lpl = cast(float, info["lPlAtX0"])
    parameter_count = cast(int, info["parameterCount"])
    if min(observation_count, parameter_count) < 0:
        raise ValueError("Normal-equation counts must be non-negative.")
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
    if matrix.shape != (parameter_count, parameter_count) or rhs.size != parameter_count:
        raise ValueError("Normal-equation arrays do not match parameter count.")
    if not math.isfinite(lpl) or lpl < 0:
        raise ValueError("Normal-equation lPl must be finite and non-negative.")
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
