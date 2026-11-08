"""Read and write ``parameterVector`` files of absolute parameter estimates.

The file is a records table (see :mod:`lunarops.fileio.result_table`) with artifact type ``parameterVector``.
Its header holds ``hasUncertainty`` (bool) and ``uncertaintySigmaMultiplier`` (the number of standard
deviations the stored uncertainties represent, or null when there are none). Fields, all with unit token
``1``:

- ``name`` (text): structured parameter name.
- ``unit`` (text): physical unit of ``value`` and ``uncertainty``.
- ``value`` (float): absolute parameter value.
- ``uncertainty`` (float): parameter uncertainty, ``~`` when the vector has none.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.estimation.lsq.products import ParameterVector as _ParameterVector

from .result_table import read_records, write_records

FIELDS = (("name", "text"), ("unit", "text"), ("value", "float"), ("uncertainty", "float"))


def write_parameter_vector(vector: _ParameterVector, path: str | Path) -> Path:
    """Write a ``parameterVector`` file.

    Parameters
    ----------
    vector : ParameterVector
        Parameter names, units, absolute values and optional uncertainties.
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    Path
        The written path.
    """
    rows = [
        {
            "name": str(name),
            "unit": unit,
            "value": value,
            "uncertainty": None if vector.uncertainties is None else vector.uncertainties[index],
        }
        for index, (name, unit, value) in enumerate(zip(vector.parameter_names, vector.units, vector.values))
    ]
    return write_records(
        path,
        "parameterVector",
        {
            "hasUncertainty": vector.uncertainties is not None,
            "uncertaintySigmaMultiplier": vector.uncertainty_sigma_multiplier,
        },
        FIELDS,
        rows,
    )


def read_parameter_vector(path: str | Path) -> _ParameterVector:
    """Read a ``parameterVector`` file.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    ParameterVector
        Parameter names, units, values as ``ndarray, shape (n,)``, and uncertainties as
        ``ndarray, shape (n,)`` or None.

    Raises
    ------
    ValueError
        If ``hasUncertainty`` is not a boolean, disagrees with the presence of ``uncertaintySigmaMultiplier``
        or with any row's ``uncertainty``, or if the records table is invalid.
    """
    header, rows = read_records(path, "parameterVector", expected_fields=FIELDS)
    has_uncertainty = header["hasUncertainty"]
    multiplier = header["uncertaintySigmaMultiplier"]
    if not isinstance(has_uncertainty, bool) or has_uncertainty != (multiplier is not None):
        raise ValueError("Invalid parameter-vector uncertainty flag or multiplier.")
    if any(has_uncertainty != (row["uncertainty"] is not None) for row in rows):
        raise ValueError("Parameter-vector uncertainty disagrees with its header.")
    return _ParameterVector(
        parameter_names=tuple(ParameterName.parse(row["name"]) for row in rows),
        values=np.asarray([row["value"] for row in rows]),
        units=tuple(row["unit"] for row in rows),
        uncertainties=np.asarray([row["uncertainty"] for row in rows]) if has_uncertainty else None,
        uncertainty_sigma_multiplier=multiplier,
    )
