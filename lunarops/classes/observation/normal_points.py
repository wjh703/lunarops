"""Canonical source-independent LLR normal-point records."""

from __future__ import annotations

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
        self.import_metadata = import_metadata if import_metadata is not None else {}

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
            return self

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


def parse_time_filter(value, *, utc_offset_hours: object = 0.0):
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
