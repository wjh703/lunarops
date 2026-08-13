"""Read and write structured parameter names with explicit units."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from lunarops.base.parameter_name import ParameterName, parameter_unit as _parameter_unit

from .archive import atomic_text_writer, data_lines, decode_token, encode_token, open_text_reader, parse_header

FORMAT_VERSION = 1


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
    with atomic_text_writer(target, "parameterName", version=FORMAT_VERSION) as stream:
        stream.write(f"parameterCount {len(values)}\n")
        stream.write("# object:type:temporal:interval unit\n")
        stream.write("data\n")
        for name, unit in zip(values, unit_values):
            stream.write(f"{encode_token(name)} {encode_token(unit)}\n")
    return target


def read_parameter_names(path: str | Path) -> tuple[list[ParameterName], list[str]]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, "parameterName", expected_version=FORMAT_VERSION)
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


__all__ = ["read_parameter_names", "write_parameter_names"]
