"""Read and write absolute parameter estimates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.estimation.parameter_products import ParameterVector as _ParameterVector

from .archive import atomic_text_writer, data_lines, decode_token, encode_token, format_float, open_text_reader, parse_float, parse_header

FORMAT_VERSION = 1


def write_parameter_vector(vector: _ParameterVector, path: str | Path) -> Path:
    target = Path(path).expanduser()
    with atomic_text_writer(target, "parameterVector", version=FORMAT_VERSION) as stream:
        stream.write(f"parameterCount {len(vector.parameter_names)}\n")
        stream.write(f"hasUncertainty {1 if vector.uncertainties is not None else 0}\n")
        multiplier = (
            "~" if vector.uncertainty_sigma_multiplier is None else format_float(vector.uncertainty_sigma_multiplier)
        )
        stream.write(f"uncertaintySigmaMultiplier {multiplier}\n")
        stream.write("# parameter_name unit absolute_value uncertainty\n")
        stream.write("data\n")
        for index, (name, unit, value) in enumerate(zip(vector.parameter_names, vector.units, vector.values)):
            uncertainty = "~" if vector.uncertainties is None else format_float(vector.uncertainties[index])
            stream.write(f"{encode_token(name)} {encode_token(unit)} {format_float(value)} {uncertainty}\n")
    return target


def read_parameter_vector(path: str | Path) -> _ParameterVector:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, "parameterVector", expected_version=FORMAT_VERSION)
        lines = iter(data_lines(stream))
        try:
            count_line = next(lines).split()
            uncertainty_line = next(lines).split()
            multiplier_line = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated parameter vector {source}.") from exc
        if (
            len(count_line) != 2
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
            raise ValueError(f"Invalid parameter-vector count or uncertainty flag in {source}.")
        has_uncertainty = uncertainty_line[1] == "1"
        if has_uncertainty:
            if multiplier_line[1] == "~":
                raise ValueError("Parameter-vector uncertainty multiplier is missing.")
            uncertainty_sigma_multiplier = parse_float(
                multiplier_line[1], field="parameter uncertainty sigma multiplier"
            )
        else:
            if multiplier_line[1] != "~":
                raise ValueError("Parameter-vector uncertainty multiplier is present without uncertainties.")
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
            values.append(parse_float(fields[2], field="absolute parameter value"))
            if has_uncertainty:
                if fields[3] == "~":
                    raise ValueError("Parameter-vector uncertainty is missing.")
                uncertainties.append(parse_float(fields[3], field="parameter uncertainty"))
            elif fields[3] != "~":
                raise ValueError("Parameter-vector uncertainty is present without uncertainties.")
    if len(names) != count:
        raise ValueError(f"Parameter vector declares {count}, found {len(names)}.")
    return _ParameterVector(
        parameter_names=tuple(names),
        values=np.asarray(values),
        units=tuple(units),
        uncertainties=(None if not has_uncertainty else np.asarray(uncertainties)),
        uncertainty_sigma_multiplier=uncertainty_sigma_multiplier,
    )


__all__ = ["read_parameter_vector", "write_parameter_vector"]
