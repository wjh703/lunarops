"""LLR normal-point records and datasets, independent of the input file format."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

from lunarops.base.constants import C
from lunarops.classes.time import Epoch, TimeScale, parse_time_with_utc_offset


@dataclass
class NptRecord:
    """One LLR normal point.

    Attributes
    ----------
    station_name : str
        Station name, used as the station catalog key unless overridden.
    reflector_name : str
        Reflector name, used as the reflector catalog key unless overridden.
    transmit_epoch : Epoch
        Laser transmit epoch, UTC.
    round_trip_time_s : float
        Observed two-way round-trip time, in seconds.
    uncertainty_two_way_s : float
        Reported uncertainty of the round-trip time, in seconds.
    pressure_hpa : float
        Surface pressure at the station, in hPa.
    temperature_k : float
        Surface temperature at the station, in kelvin.
    humidity_percent : float
        Relative humidity at the station, in percent.
    wavelength_nm : float
        Laser wavelength, in nanometres.
    index : int
        Position of the record in its dataset; it serves as the observation identity.
    station_code : str or None
        Station code from the source file.
    reflector_code : str or None
        Reflector code from the source file.
    """

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
        """A-priori one-way range sigma ``0.5 * c * uncertainty_two_way_s``, in metres."""
        return 0.5 * C * float(self.uncertainty_two_way_s)

    @property
    def temperature_c(self) -> float:
        """Surface temperature, in degrees Celsius."""
        return float(self.temperature_k) - 273.15

    @property
    def wavelength_um(self) -> float:
        """Laser wavelength, in micrometres."""
        return float(self.wavelength_nm) / 1000.0


class NptDataset:
    """Ordered collection of normal points with import bookkeeping.

    Iterating the dataset yields its records.

    Parameters
    ----------
    records : list[NptRecord]
        Valid records.
    name : str, optional
        Dataset label.
    n_input_records : int, default 0
        Number of records read from the source. When 0, no invalid records are given and ``records`` is not
        empty, it is set to ``len(records)``.
    n_invalid_records : int, default 0
        Number of source records rejected during import.
    import_issues : list[dict[str, object]], optional
        Descriptions of rejected records; at most ``n_invalid_records`` entries.
    import_metadata : dict[str, object], optional
        Source metadata.

    Raises
    ------
    ValueError
        If the counts are inconsistent or there are more issues than invalid records.
    """

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
        """Return a summary with the record counts."""
        return (
            "NptDataset("
            f"record_count={len(self.records)}, name={self.name!r}, "
            f"n_input_records={self.n_input_records}, "
            f"n_invalid_records={self.n_invalid_records})"
        )

    def __len__(self) -> int:
        """Return the number of valid records."""
        return len(self.records)

    def __iter__(self) -> Iterator[NptRecord]:
        """Iterate over the valid records."""
        return iter(self.records)

    def assign_indices(self, *, start: int = 0) -> NptDataset:
        """Assign consecutive indices from ``start`` to the records in place and return this dataset."""
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
        """Return the records with transmit epochs in ``[start_time_utc, end_time_utc)``.

        The kept records are copies renumbered from 0; import counts, issues and metadata are carried over.

        Parameters
        ----------
        start_time_utc : Epoch or str, optional
            Inclusive start. A UTC ``Epoch``, or a timestamp parsed with ``utc_offset_hours``.
        end_time_utc : Epoch or str, optional
            Exclusive end, in the same forms as ``start_time_utc``.
        utc_offset_hours : object, default 0.0
            Offset of local time from UTC, in hours, for timestamps given as text.

        Returns
        -------
        NptDataset
            Filtered dataset, or this dataset when both bounds are ``None``.
        """
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
    """Parse a time-filter bound to a UTC epoch.

    Parameters
    ----------
    value : Epoch, str or None
        A UTC ``Epoch`` (other scales are rejected), a timestamp, or ``None``.
    utc_offset_hours : object, default 0.0
        Offset of local time from UTC, in hours, applied to timestamps given as text.

    Returns
    -------
    Epoch or None
        Bound as a UTC epoch, or ``None`` when ``value`` is ``None``.
    """
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
    """Concatenate datasets into one, renumbering the records consecutively from 0.

    Parameters
    ----------
    datasets : Sequence[NptDataset]
        Datasets in output order.
    name : str, optional
        Label of the combined dataset.

    Returns
    -------
    NptDataset
        Combined dataset with summed import counts, concatenated issues and the source metadata listed under
        ``sources``.
    """
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
