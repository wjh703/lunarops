"""Native text artifacts for time-grid LLR predictions and visibility windows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .archive import (
    atomic_text_writer,
    data_lines,
    decode_token,
    encode_token,
    open_text_reader,
    parse_float,
    parse_header,
)

PREDICTION_ARTIFACT_TYPE = "observationPrediction"
WINDOW_ARTIFACT_TYPE = "predictionWindow"
PREDICTION_FORMAT_VERSION = 4
WINDOW_FORMAT_VERSION = 3

_PREDICTION_FIELDS = (
    ("utc_t1", "text"),
    ("local_t1", "text"),
    ("station", "text"),
    ("reflector", "text"),
    ("station_itrf_x_m", "float"),
    ("station_itrf_y_m", "float"),
    ("station_itrf_z_m", "float"),
    ("reflector_itrf_x_m", "float"),
    ("reflector_itrf_y_m", "float"),
    ("reflector_itrf_z_m", "float"),
    ("range_up_geometric_m", "float"),
    ("azimuth_deg", "float"),
    ("elevation_deg", "float"),
    ("observable", "bool"),
)

_WINDOW_FIELDS = (
    ("station", "text"),
    ("reflector", "text"),
    ("start_utc", "text"),
    ("end_utc", "text"),
    ("start_local", "text"),
    ("end_local", "text"),
    ("sample_count", "int"),
    ("duration_s", "float"),
)


def _format_float(value: object, field: str) -> str:
    number = float(cast(Any, value))
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"Prediction results reject non-finite float {value!r}.")
    if field.endswith("_m"):
        precision = ".3f" if field == "range_up_geometric_m" else ".6f"
    else:
        precision = ".6f"
    return format(number, precision)


def _format(value: object, kind: str, field: str) -> str:
    if kind == "text":
        return encode_token(value)
    if kind == "float":
        return _format_float(value, field)
    if kind == "int":
        return str(int(cast(Any, value)))
    if kind == "bool":
        return "1" if bool(value) else "0"
    raise AssertionError(kind)


def _parse(value: str, kind: str, field: str):
    if kind == "text":
        return decode_token(value)
    if kind == "float":
        return parse_float(value, field=field)
    if kind == "int":
        return int(value)
    if kind == "bool":
        if value not in {"0", "1"}:
            raise ValueError(f"Invalid boolean {value!r} for {field}.")
        return value == "1"
    raise AssertionError(kind)


def _write_rows(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
    *,
    artifact_type: str,
    fields: Sequence[tuple[str, str]],
) -> Path:
    target = Path(path).expanduser()
    version = PREDICTION_FORMAT_VERSION if artifact_type == PREDICTION_ARTIFACT_TYPE else WINDOW_FORMAT_VERSION
    with atomic_text_writer(target, artifact_type, version=version) as stream:
        stream.write(f"recordCount {len(rows)}\n")
        stream.write("fields " + " ".join(name for name, _ in fields) + "\n")
        stream.write("data\n")
        for row in rows:
            missing = [name for name, _ in fields if name not in row]
            if missing:
                raise ValueError(f"Prediction row is missing field(s): {missing}")
            stream.write(" ".join(_format(row[name], kind, name) for name, kind in fields) + "\n")
    return target


def _read_rows(
    path: str | Path,
    *,
    artifact_type: str,
    fields: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        version = PREDICTION_FORMAT_VERSION if artifact_type == PREDICTION_ARTIFACT_TYPE else WINDOW_FORMAT_VERSION
        parse_header(stream, artifact_type, expected_version=version)
        lines = iter(data_lines(stream))
        try:
            count_parts = next(lines).split()
            field_line = next(lines)
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated {artifact_type} file {source}.") from exc
        expected_field_line = "fields " + " ".join(name for name, _ in fields)
        if (
            len(count_parts) != 2
            or count_parts[0] != "recordCount"
            or field_line != expected_field_line
            or marker != "data"
        ):
            raise ValueError(f"Malformed {artifact_type} header in {source}.")
        count = int(count_parts[1])
        rows: list[dict[str, object]] = []
        for line_number, line in enumerate(lines, start=1):
            values = line.split()
            if len(values) != len(fields):
                raise ValueError(
                    f"{artifact_type} row {line_number} has {len(values)} fields; expected {len(fields)}."
                )
            rows.append({name: _parse(value, kind, name) for (name, kind), value in zip(fields, values)})
    if len(rows) != count:
        raise ValueError(f"{artifact_type} declares {count} rows, found {len(rows)}.")
    return rows


def write_prediction_results(rows: Sequence[Mapping[str, object]], path: str | Path) -> Path:
    return _write_rows(rows, path, artifact_type=PREDICTION_ARTIFACT_TYPE, fields=_PREDICTION_FIELDS)


def read_prediction_results(path: str | Path) -> list[dict[str, object]]:
    return _read_rows(path, artifact_type=PREDICTION_ARTIFACT_TYPE, fields=_PREDICTION_FIELDS)


def write_prediction_windows(rows: Sequence[Mapping[str, object]], path: str | Path) -> Path:
    return _write_rows(rows, path, artifact_type=WINDOW_ARTIFACT_TYPE, fields=_WINDOW_FIELDS)


def read_prediction_windows(path: str | Path) -> list[dict[str, object]]:
    return _read_rows(path, artifact_type=WINDOW_ARTIFACT_TYPE, fields=_WINDOW_FIELDS)


__all__ = [
    "PREDICTION_ARTIFACT_TYPE",
    "WINDOW_ARTIFACT_TYPE",
    "read_prediction_results",
    "read_prediction_windows",
    "write_prediction_results",
    "write_prediction_windows",
]
