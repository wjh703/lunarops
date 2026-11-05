"""Read and write native ``normalPoint`` files of canonical LLR normal points.

The file is a records table (see :mod:`lunarops.fileio.result_table`) with artifact type ``normalPoint``,
a layout specific to LLR normal points. Header keys: ``datasetName`` (non-empty str), ``timeScale``
(always ``UTC``), ``inputRecordCount`` and ``invalidRecordCount`` (source records read and rejected at
import) and ``importMetadata`` (mapping describing the import). Fields, one row per normal point, with
their kind and unit token:

- ``jd1_utc``, ``jd2_utc`` (float, 1): two-part Julian date of the ground transmit epoch, UTC.
- ``station_name``, ``reflector_name`` (text, 1): station and reflector identifiers.
- ``round_trip_time_s`` (float, s): observed two-way light time.
- ``uncertainty_two_way_s`` (float, s): two-way light-time uncertainty.
- ``pressure_hpa`` (float, hPa): surface pressure at the station.
- ``temperature_k`` (float, K): surface temperature at the station.
- ``humidity_percent`` (float, %): relative humidity at the station.
- ``wavelength_nm`` (float, nm): laser wavelength.
- ``index`` (int, 1): record index, unique within the file.
- ``station_code``, ``reflector_code`` (text, 1): source station and reflector codes, ``~`` when absent.
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
    """Return whether a path is a native ``normalPoint`` file.

    Parameters
    ----------
    path : str or Path
        Candidate path.

    Returns
    -------
    bool
        True if the path is an existing ``.txt`` or ``.txt.gz`` file whose header declares ``normalPoint``;
        False otherwise, including when the file cannot be read or its header is invalid.
    """
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
    """Write a native ``normalPoint`` file.

    Parameters
    ----------
    dataset : NptDataset
        Normal points with UTC transmit epochs, dataset name, input and invalid record counts and import
        metadata. An empty dataset name is replaced by the stem of ``path``.
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    Path
        The written path.

    Raises
    ------
    ValueError
        If a transmit epoch is not in UTC or a value cannot be written.
    """
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
    """Read and validate a native ``normalPoint`` file.

    Besides the records schema and units, the reader requires ``timeScale`` to be ``UTC``, both record
    counts to be non-negative with ``inputRecordCount`` at least the number of rows plus
    ``invalidRecordCount``, a non-empty ``datasetName``, a mapping ``importMetadata`` and unique record
    indices. Every row is validated by ``read_normal_point_record``.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    NptDataset
        Normal points with UTC transmit epochs, in file order.

    Raises
    ------
    ValueError
        If any of these checks fails or the records table is invalid.
    """
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
