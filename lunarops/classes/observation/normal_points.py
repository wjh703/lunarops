"""Canonical source-independent LLR normal-point records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

from lunarops.base.constants import C
from lunarops.classes.time import Epoch, TimeScale


@dataclass
class NptRecord:
    station_name: str
    reflector_name: str
    transmit_epoch: Epoch
    round_trip_time_s: float
    uncertainty_two_way_s: float
    pressure_hpa: float
    temperature_k: float
    humidity_percent: float
    wavelength_nm: float
    index: int = 0
    station_code: Optional[str] = None
    reflector_code: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.transmit_epoch, Epoch):
            raise TypeError("transmit_epoch must be an Epoch.")
        self.transmit_epoch.require_scale(TimeScale.UTC, name="transmit_epoch")
        self.station_name = _compact_identity(self.station_name)
        self.reflector_name = _compact_identity(self.reflector_name)
        if not self.station_name or not self.reflector_name:
            raise ValueError("station_name and reflector_name must not be empty.")
        positive_fields = (
            "round_trip_time_s",
            "uncertainty_two_way_s",
            "pressure_hpa",
            "temperature_k",
            "wavelength_nm",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
            setattr(self, name, value)
        humidity = float(self.humidity_percent)
        if not math.isfinite(humidity) or not 0.0 <= humidity <= 100.0:
            raise ValueError("humidity_percent must be finite and in [0, 100].")
        self.humidity_percent = humidity
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("Normal-point index must be a non-negative integer.")
        self.index = int(self.index)
        self.station_code = _optional_text(self.station_code)
        self.reflector_code = _optional_text(self.reflector_code)

    @property
    def observed_round_trip_time_s(self) -> float:
        return float(self.round_trip_time_s)

    @property
    def range_uncertainty_one_way_m(self) -> float:
        return 0.5 * C * float(self.uncertainty_two_way_s)

    @property
    def temperature_c(self) -> float:
        return float(self.temperature_k) - 273.15

    @property
    def wavelength_um(self) -> float:
        return float(self.wavelength_nm) / 1000.0


class NptDataset:
    __slots__ = (
        "records",
        "name",
        "n_input_records",
        "n_invalid_records",
        "import_issues",
    )

    def __init__(
        self,
        records: List[NptRecord],
        name: Optional[str] = None,
        n_input_records: int = 0,
        n_invalid_records: int = 0,
        import_issues: Optional[List[dict[str, object]]] = None,
    ) -> None:
        if not isinstance(records, list) or not all(isinstance(record, NptRecord) for record in records):
            raise TypeError("NptDataset.records must be a list of NptRecord objects.")
        input_count = int(n_input_records)
        invalid_count = int(n_invalid_records)
        if input_count != n_input_records or invalid_count != n_invalid_records:
            raise TypeError("Normal-point record counts must be integers.")
        if input_count == 0 and records and invalid_count == 0:
            input_count = len(records)
        if invalid_count < 0 or input_count < len(records) + invalid_count:
            raise ValueError("Normal-point input count must cover valid and invalid records.")
        self.records = records
        self.name = name
        self.n_input_records = input_count
        self.n_invalid_records = invalid_count
        issues = list(import_issues or [])
        if not all(isinstance(issue, dict) for issue in issues):
            raise TypeError("Normal-point import issues must be mappings.")
        if len(issues) > invalid_count:
            raise ValueError("Normal-point import issues cannot exceed the invalid-record count.")
        self.import_issues = issues

    def __repr__(self) -> str:
        return (
            "NptDataset("
            f"record_count={len(self.records)}, name={self.name!r}, "
            f"n_input_records={self.n_input_records}, "
            f"n_invalid_records={self.n_invalid_records})"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[NptRecord]:
        return iter(self.records)

    def assign_indices(self, *, start: int = 0) -> "NptDataset":
        for offset, rec in enumerate(self.records):
            rec.index = int(start) + offset
        return self

    def filter_time(self, start_time_utc=None, end_time_utc=None) -> "NptDataset":
        start = parse_time_filter(start_time_utc)
        end = parse_time_filter(end_time_utc)
        if start is None and end is None:
            return self

        kept: List[NptRecord] = []
        for rec in self.records:
            epoch = rec.transmit_epoch
            if start is not None and epoch < start:
                continue
            if end is not None and epoch >= end:
                continue
            kept.append(rec)
        self.records = kept
        self.assign_indices(start=0)
        return self


def _optional_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _compact_identity(value: object) -> str:
    return "".join(str(value).split())


def parse_time_filter(value):
    """Parse an optional lower/upper UTC filter into an Epoch."""
    if value is None:
        return None
    if isinstance(value, Epoch):
        return value.require_scale(TimeScale.UTC, name="time filter")
    text = str(value).strip()
    if not text:
        return None
    return Epoch.from_isot(text, scale=TimeScale.UTC)


def combine_npt_datasets(
    datasets: Sequence[NptDataset],
    *,
    name: Optional[str] = None,
) -> NptDataset:
    merged: List[NptRecord] = []
    n_input_records = 0
    n_invalid_records = 0
    import_issues: List[dict[str, object]] = []
    for dataset in datasets:
        n_input_records += int(dataset.n_input_records)
        n_invalid_records += int(dataset.n_invalid_records)
        import_issues.extend(dataset.import_issues)
        for record in dataset.records:
            record.index = len(merged)
            merged.append(record)
    return NptDataset(
        records=merged,
        name=name,
        n_input_records=n_input_records,
        n_invalid_records=n_invalid_records,
        import_issues=import_issues,
    )


__all__ = [
    "NptDataset",
    "NptRecord",
    "combine_npt_datasets",
    "parse_time_filter",
]
