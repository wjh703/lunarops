"""Canonical source-independent LLR normal-point records."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

from lunarops.base.constants import C
from lunarops.classes.time import Epoch, TimeScale, parse_time_with_utc_offset


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
    station_code: str | None = None
    reflector_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transmit_epoch, Epoch):
            raise TypeError("transmit_epoch must be an Epoch.")
        self.transmit_epoch.require_scale(TimeScale.UTC, name="transmit_epoch")
        self.station_name = _exact_identity(self.station_name)
        self.reflector_name = _exact_identity(self.reflector_name)
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
        if self.station_code is not None:
            from lunarops.base.station_identity import station_id_from_ilrs_code
            station = station_id_from_ilrs_code(self.station_code)
            if self.station_name != station:
                raise ValueError("station_name contradicts station_code.")
        if self.reflector_code in {"0", "1", "2", "3", "4"}:
            reflector = ("APOLLO11", "LUNOKHOD1", "APOLLO14", "APOLLO15", "LUNOKHOD2")[int(self.reflector_code)]
            if self.reflector_name != reflector:
                raise ValueError("reflector_name contradicts reflector_code.")

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
        "import_issues",
        "import_metadata",
        "n_input_records",
        "n_invalid_records",
        "name",
        "records",
    )

    def __init__(
        self,
        records: list[NptRecord],
        name: str | None = None,
        n_input_records: int = 0,
        n_invalid_records: int = 0,
        import_issues: list[dict[str, object]] | None = None,
        import_metadata: dict[str, object] | None = None,
    ) -> None:
        input_count = int(n_input_records)
        invalid_count = int(n_invalid_records)
        if input_count == 0 and records and invalid_count == 0:
            input_count = len(records)
        if invalid_count < 0 or input_count < len(records) + invalid_count:
            raise ValueError("Normal-point input count must cover valid and invalid records.")
        self.records = records
        self.name = name
        self.n_input_records = input_count
        self.n_invalid_records = invalid_count
        issues = list(import_issues or [])
        if len(issues) > invalid_count:
            raise ValueError("Normal-point import issues cannot exceed the invalid-record count.")
        self.import_issues = issues
        from copy import deepcopy

        self.import_metadata = deepcopy(import_metadata or {})

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

    def assign_indices(self, *, start: int = 0) -> NptDataset:
        for offset, rec in enumerate(self.records):
            rec.index = int(start) + offset
        return self

    def filter_time(
        self,
        start_time_utc=None,
        end_time_utc=None,
        *,
        utc_offset_hours: object = 0.0,
    ) -> NptDataset:
        start = parse_time_filter(start_time_utc, utc_offset_hours=utc_offset_hours)
        end = parse_time_filter(end_time_utc, utc_offset_hours=utc_offset_hours)
        if start is None and end is None:
            return NptDataset(
                [replace(rec) for rec in self.records],
                self.name,
                self.n_input_records,
                self.n_invalid_records,
                [dict(issue) for issue in self.import_issues],
                self.import_metadata,
            )

        kept: list[NptRecord] = []
        for rec in self.records:
            epoch = rec.transmit_epoch
            if start is not None and epoch < start:
                continue
            if end is not None and epoch >= end:
                continue
            kept.append(rec)
        return NptDataset(
            [replace(rec, index=index) for index, rec in enumerate(kept)],
            self.name,
            self.n_input_records,
            self.n_invalid_records,
            [dict(issue) for issue in self.import_issues],
            self.import_metadata,
        )


def _optional_text(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Source codes must be non-empty exact strings or None.")
    return value


def _exact_identity(value: object) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError("Normal-point identifiers must be exact strings.")
    return value


def parse_time_filter(value, *, utc_offset_hours: object = 0.0):
    """Parse an optional civil-time filter into an internal UTC epoch."""
    if value is None:
        return None
    if isinstance(value, Epoch):
        return value.require_scale(TimeScale.UTC, name="time filter")
    return parse_time_with_utc_offset(value, utc_offset_hours=utc_offset_hours, name="time filter")


def combine_npt_datasets(
    datasets: Sequence[NptDataset],
    *,
    name: str | None = None,
) -> NptDataset:
    merged: list[NptRecord] = []
    n_input_records = 0
    n_invalid_records = 0
    import_issues: list[dict[str, object]] = []
    for dataset in datasets:
        n_input_records += int(dataset.n_input_records)
        n_invalid_records += int(dataset.n_invalid_records)
        import_issues.extend(dict(issue) for issue in dataset.import_issues)
        for record in dataset.records:
            merged.append(replace(record, index=len(merged)))
    return NptDataset(
        records=merged,
        name=name,
        n_input_records=n_input_records,
        n_invalid_records=n_invalid_records,
        import_issues=import_issues,
        import_metadata={"sources": [dataset.import_metadata for dataset in datasets]},
    )


__all__ = [
    "NptDataset",
    "NptRecord",
    "combine_npt_datasets",
    "parse_time_filter",
]
