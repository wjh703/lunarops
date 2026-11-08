"""Read and write ``parameterName`` files: structured parameter names with explicit units.

The file is a records table (see :mod:`lunarops.fileio.result_table`) with artifact type ``parameterName``,
an empty header and two text fields, ``name`` (the structured parameter name) and ``unit`` (that parameter's
physical unit). Both field unit tokens are ``1``; the physical unit is carried in the ``unit`` column.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lunarops.base.parameter_name import ParameterName
from lunarops.base.parameter_name import parameter_unit as _parameter_unit

from .result_table import read_records, write_records

FIELDS = (("name", "text"), ("unit", "text"))


def write_parameter_names(path: str | Path, names: Sequence[ParameterName], units: Sequence[str] | None = None) -> Path:
    """Write a ``parameterName`` file.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.
    names : Sequence of ParameterName
        Parameter names, one row each.
    units : Sequence of str or None, optional
        Unit of each parameter, paired with ``names`` by position. When None, each unit is the default unit
        of its parameter name.

    Returns
    -------
    Path
        The written path.
    """
    unit_values = units if units is not None else [_parameter_unit(name) for name in names]
    return write_records(
        path, "parameterName", {}, FIELDS, [{"name": str(name), "unit": unit} for name, unit in zip(names, unit_values)]
    )


def read_parameter_names(path: str | Path) -> tuple[list[ParameterName], list[str]]:
    """Read a ``parameterName`` file.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    names : list of ParameterName
        Parsed parameter names in file order.
    units : list of str
        Unit of each parameter.

    Raises
    ------
    ValueError
        If a name has no parameter type, names repeat, a unit is empty, or the records table is invalid.
    """
    _, rows = read_records(path, "parameterName", expected_fields=FIELDS)
    names = [ParameterName.parse(row["name"]) for row in rows]
    units = [row["unit"] for row in rows]
    if any(not name.parameter_type for name in names) or len(set(names)) != len(names):
        raise ValueError("Parameter names must have a type and be unique.")
    if any(not unit for unit in units):
        raise ValueError("Parameter-name file contains an empty unit.")
    return names, units
