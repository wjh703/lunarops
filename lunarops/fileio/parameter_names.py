"""Read and write structured parameter names with explicit units."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lunarops.base.parameter_name import ParameterName
from lunarops.base.parameter_name import parameter_unit as _parameter_unit

from .result_table import read_records, write_records

FIELDS = (("name", "text"), ("unit", "text"))


def write_parameter_names(path: str | Path, names: Sequence[ParameterName], units: Sequence[str] | None = None) -> Path:
    unit_values = units if units is not None else [_parameter_unit(name) for name in names]
    return write_records(path, "parameterName", {}, FIELDS,
                         [{"name": str(name), "unit": unit} for name, unit in zip(names, unit_values)])


def read_parameter_names(path: str | Path) -> tuple[list[ParameterName], list[str]]:
    _, rows = read_records(path, "parameterName", expected_fields=FIELDS)
    names = [ParameterName.parse(row["name"]) for row in rows]
    units = [row["unit"] for row in rows]
    if any(not name.parameter_type for name in names) or len(set(names)) != len(names):
        raise ValueError("Parameter names must have a type and be unique.")
    if any(not unit for unit in units):
        raise ValueError("Parameter-name file contains an empty unit.")
    return names, units
