"""Native LunarOps ASCII files for canonical LLR normal points.

The typed layout is informed by GROOPS file design, but is not a GROOPS file
format and is intentionally specific to LLR normal points.
"""

from __future__ import annotations

from pathlib import Path

from lunarops.classes.observation.normal_points import NptDataset as _NptDataset
from lunarops.classes.observation.normal_points import NptRecord as _NptRecord
from lunarops.classes.time import Epoch, TimeScale
from lunarops.fileio.formats.normal_point_record import read_normal_point_record

from .archive import read_artifact_type
from .result_table import read_records, write_records

ARTIFACT_TYPE = "normalPoint"


def is_normal_point_artifact(path: str | Path) -> bool:
    source = Path(path)
    if not source.is_file() or not (source.name.lower().endswith(".txt") or source.name.lower().endswith(".txt.gz")):
        return False
    try:
        return read_artifact_type(source) == ARTIFACT_TYPE
    except (OSError, ValueError):
        return False


FIELDS = (
    ("jd1_utc", "float"), ("jd2_utc", "float"), ("station_name", "text"), ("reflector_name", "text"),
    ("round_trip_time_s", "float"), ("uncertainty_two_way_s", "float"), ("pressure_hpa", "float"),
    ("temperature_k", "float"), ("humidity_percent", "float"), ("wavelength_nm", "float"),
    ("index", "int"), ("station_code", "text"), ("reflector_code", "text"),
)


def write_normal_points(dataset: _NptDataset, path: str | Path) -> Path:
    rows = []
    for record in dataset.records:
        epoch = record.transmit_epoch.require_scale(TimeScale.UTC, name="transmit_epoch")
        rows.append({"jd1_utc": epoch.jd1, "jd2_utc": epoch.jd2,
                     **{name: getattr(record, name) for name, _ in FIELDS[2:]}})
    return write_records(path, ARTIFACT_TYPE, {
        "datasetName": dataset.name or Path(path).stem, "timeScale": "UTC",
        "inputRecordCount": int(dataset.n_input_records), "invalidRecordCount": int(dataset.n_invalid_records),
        "importMetadata": dataset.import_metadata,
    }, FIELDS, rows)


def read_normal_points(path: str | Path) -> _NptDataset:
    header, rows = read_records(path, ARTIFACT_TYPE, expected_fields=FIELDS)
    if header["timeScale"] != "UTC":
        raise ValueError("Normal-point timeScale must be UTC.")
    input_count, invalid_count = header["inputRecordCount"], header["invalidRecordCount"]
    if min(input_count, invalid_count) < 0 or input_count < len(rows) + invalid_count:
        raise ValueError("Normal-point input count must cover valid and invalid records.")
    if not header["datasetName"] or not isinstance(header["importMetadata"], dict):
        raise ValueError("Invalid normal-point dataset name or import metadata.")
    records: list[_NptRecord] = []
    for row in rows:
        epoch = Epoch(row.pop("jd1_utc"), row.pop("jd2_utc"), TimeScale.UTC)
        records.append(read_normal_point_record(transmit_epoch=epoch, **row))
    if len({record.index for record in records}) != len(records):
        raise ValueError("Normal-point file contains duplicate record indices.")
    return _NptDataset(records=records, name=header["datasetName"], n_input_records=input_count,
                       n_invalid_records=invalid_count, import_metadata=header["importMetadata"])
