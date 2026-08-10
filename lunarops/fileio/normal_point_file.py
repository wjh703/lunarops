"""Native GROOPS-style ASCII files for canonical LLR normal points."""

from __future__ import annotations

from pathlib import Path

from lunarops.classes.time import Epoch, TimeScale

from .archive import (
    atomic_text_writer,
    data_lines,
    decode_token,
    encode_token,
    format_float,
    open_text_reader,
    parse_float,
    parse_header,
    read_artifact_type,
)
from lunarops.classes.observation.normal_points import NptDataset as _NptDataset
from lunarops.classes.observation.normal_points import NptRecord as _NptRecord

ARTIFACT_TYPE = "normalPoint"


def is_normal_point_file(path: str | Path) -> bool:
    source = Path(path)
    if not source.is_file() or not (source.name.lower().endswith(".txt") or source.name.lower().endswith(".txt.gz")):
        return False
    try:
        return read_artifact_type(source) == ARTIFACT_TYPE
    except OSError, ValueError:
        return False


def _optional_token(value: object | None) -> str:
    return "~" if value is None or str(value) == "" else encode_token(value)


def _required_pair(line: str, key: str) -> str:
    parts = line.split(maxsplit=1)
    if len(parts) != 2 or parts[0] != key:
        raise ValueError(f"Expected {key!r} record, found {line!r}.")
    return parts[1]


def write_normal_point_file(dataset: _NptDataset, path: str | Path) -> Path:
    if not isinstance(dataset, _NptDataset):
        raise TypeError("dataset must be an NptDataset.")
    indices = [record.index for record in dataset.records]
    if len(set(indices)) != len(indices):
        raise ValueError("Normal-point record indices must be unique.")
    target = Path(path).expanduser()
    with atomic_text_writer(target, ARTIFACT_TYPE) as stream:
        stream.write(f"datasetName {encode_token(dataset.name or target.stem)}\n")
        stream.write("timeScale UTC\n")
        stream.write(f"recordCount {len(dataset.records)}\n")
        stream.write(f"inputRecordCount {int(dataset.n_input_records)}\n")
        stream.write(f"invalidRecordCount {int(dataset.n_invalid_records)}\n")
        stream.write(
            "# jd1_utc jd2_utc station reflector rtt_s uncertainty_two_way_s "
            "pressure_hPa temperature_K humidity_percent wavelength_nm index "
            "station_code reflector_code\n"
        )
        stream.write("data\n")
        for record in dataset.records:
            epoch = record.transmit_epoch.require_scale(TimeScale.UTC, name="transmit_epoch")
            fields = (
                format_float(epoch.jd1),
                format_float(epoch.jd2),
                encode_token(record.station_name),
                encode_token(record.reflector_name),
                format_float(record.round_trip_time_s),
                format_float(record.uncertainty_two_way_s),
                format_float(record.pressure_hpa),
                format_float(record.temperature_k),
                format_float(record.humidity_percent),
                format_float(record.wavelength_nm),
                str(int(record.index)),
                _optional_token(record.station_code),
                _optional_token(record.reflector_code),
            )
            stream.write(" ".join(fields) + "\n")
    return target


def read_normal_point_file(path: str | Path) -> _NptDataset:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, ARTIFACT_TYPE)
        lines = iter(data_lines(stream))
        try:
            dataset_name = decode_token(_required_pair(next(lines), "datasetName"))
            time_scale = _required_pair(next(lines), "timeScale")
            record_count = int(_required_pair(next(lines), "recordCount"))
            input_count = int(_required_pair(next(lines), "inputRecordCount"))
            invalid_count = int(_required_pair(next(lines), "invalidRecordCount"))
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated normal-point header in {source}.") from exc
        except ValueError as exc:
            raise ValueError(f"Invalid normal-point header in {source}: {exc}") from exc
        if time_scale != "UTC":
            raise ValueError(f"Normal-point timeScale must be UTC, found {time_scale!r}.")
        if min(record_count, input_count, invalid_count) < 0:
            raise ValueError("Normal-point counts must be non-negative.")
        if not dataset_name:
            raise ValueError("Normal-point datasetName must not be empty.")
        if input_count < record_count + invalid_count:
            raise ValueError("Normal-point input count must cover valid and invalid records.")
        if marker != "data":
            raise ValueError(f"Expected normal-point data marker, found {marker!r}.")

        records: list[_NptRecord] = []
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 13:
                raise ValueError(f"Normal-point data row {line_number} has {len(fields)} fields; expected 13.")
            try:
                station_code = None if fields[11] == "~" else decode_token(fields[11])
                reflector_code = None if fields[12] == "~" else decode_token(fields[12])
                records.append(
                    _NptRecord(
                        station_name=decode_token(fields[2]),
                        reflector_name=decode_token(fields[3]),
                        transmit_epoch=Epoch(
                            parse_float(fields[0], field="jd1_utc"),
                            parse_float(fields[1], field="jd2_utc"),
                            TimeScale.UTC,
                        ),
                        round_trip_time_s=parse_float(fields[4], field="rtt_s"),
                        uncertainty_two_way_s=parse_float(fields[5], field="uncertainty_two_way_s"),
                        pressure_hpa=parse_float(fields[6], field="pressure_hPa"),
                        temperature_k=parse_float(fields[7], field="temperature_K"),
                        humidity_percent=parse_float(fields[8], field="humidity_percent"),
                        wavelength_nm=parse_float(fields[9], field="wavelength_nm"),
                        index=int(fields[10]),
                        station_code=station_code,
                        reflector_code=reflector_code,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid normal-point data row {line_number} in {source}: {exc}") from exc
    if len(records) != record_count:
        raise ValueError(f"Normal-point header declares {record_count} records, found {len(records)}.")
    if len({record.index for record in records}) != len(records):
        raise ValueError("Normal-point file contains duplicate record indices.")
    return _NptDataset(
        records=records,
        name=dataset_name,
        n_input_records=input_count,
        n_invalid_records=invalid_count,
    )


__all__ = [
    "ARTIFACT_TYPE",
    "is_normal_point_file",
    "read_normal_point_file",
    "write_normal_point_file",
]
