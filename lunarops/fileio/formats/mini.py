"""
I/O for POLAC / OCA "MINI" fixed-width LLR normal-point files.

The MINI format is a one-line-per-normal-point fixed-width ASCII format.
Each line carries *exactly* the following fields (1-based columns):

    cols  width  field                          unit / encoding
    ----- ------ ------------------------------ -------------------------------
    1     1      format code                    1 = MINI                [required]
    2     1      laser color code               1 = green, 2 = infrared (optional)
    3-10  8      launch date (UTC)              YYYYMMDD                [required]
    11-23 13     launch time of day (UTC)       HHMMSSsssssss (100 ns)  [required]
    24-37 14     observed two-way light time    integer, 0.1 ps         [required]
    38    1      reflector id                   0=A11 1=L1 2=A14 3=A15 4=L2 [required]
    39-43 5      station id                     ILRS-style 5-char code  [required]
    44-46 3      number of returns              integer (optional)
    47-52 6      uncertainty (two-way)          integer, 0.1 ps         [required; used for weights]
    53-55 3      signal-to-noise ratio          integer, S/N * 10 (optional)
    56    1      quality code                   single char (optional)
    57-62 6      surface pressure               integer, hPa * 100, > 0 [required]
    63-66 4      surface temperature            integer, 0.1 deg        [required]
    67-68 2      relative humidity              integer percent, 0-100  [required]
    69-73 5      laser wavelength               integer, 0.1 nm, > 0    [required]
    74    1      version code                   single char (optional)
    75-78 4      session duration               integer seconds (optional)
    79-80 2      (unused / blank)
    81-89 9      source format tag              free text, e.g. original format

References:
    https://polac.obspm.fr/llrdatae.html
    http://www.geoazur.fr/astrogeo/observations/donnees/lune/mini-format.html

Everything the parser produces is either (a) one of the raw MINI fields above,
or (b) a derived quantity computed from them (SI scaling, the unified UTC
Epoch, station / reflector identifiers).  No CRD-style flags
(troposphere applied, center-of-mass applied, ...) exist in this module:
MINI data never carries such corrections, so they are never represented.

VALIDATION CONTRACT: every quantity carried by the MINI file and needed by
the downstream O-C / fit computation - launch epoch, light time, reflector,
station, MINI uncertainty, pressure, temperature, humidity, and laser wavelength
- is *guaranteed present and physically plausible* after parsing. Invalid
records never enter the returned NPT records; their line number, validation
reason, and original content are retained as structured import issues for the
converter report. Downstream code therefore never needs None checks, meteo
lookups, uncertainty fallbacks, or default wavelengths for MINI-owned fields.

NOTE on the temperature unit: this module follows the convention used by the
existing processing chain, ``temperature_c = raw / 10`` (i.e. 0.1 deg C).
Some MINI documentation describes the field as 0.1 K instead.  The conversion
is isolated in MINI-to-NPT conversion so that it
can be flipped in a single place if your data files use the 0.1 K convention.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from lunarops.classes.time import Epoch, TimeScale
from lunarops.base.station_identity import canonical_station_id
from lunarops.classes.observation.normal_points import NptDataset as _NptDataset
from lunarops.classes.observation.normal_points import NptRecord as _NptRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_UNIT_S = 1.0e-13  # light-time / uncertainty unit: 0.1 ps
TIME_100NS_S = 1.0e-7  # launch-time fractional unit: 100 ns
C_LIGHT_M_PER_S = 299_792_458.0

MINI_LINE_MIN_LENGTH = 78  # duration field ends at col 78
MINI_LINE_FULL_LENGTH = 89  # source-format field ends at col 89

REFLECTOR_NAMES = {
    0: "Apollo 11",
    1: "Lunokhod 1",
    2: "Apollo 14",
    3: "Apollo 15",
    4: "Lunokhod 2",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _open_text(path, *, encoding: str = "ascii", errors: str = "strict"):
    """Open a plain-text or .gz MINI file with universal newlines."""
    path = Path(path)
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding=encoding, errors=errors, newline=None)
    return path.open("r", encoding=encoding, errors=errors, newline=None)


def _blank_to_none(text: str) -> Optional[str]:
    value = str(text).strip()
    return value if value else None


def _parse_int(text: str, *, field: str, line_no: int, required: bool = True) -> Optional[int]:
    value = _blank_to_none(text)
    if value is None:
        if required:
            raise ValueError(f"line {line_no}: required MINI field {field!r} is blank")
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"line {line_no}: invalid MINI integer field {field!r}: {text!r}") from exc


def looks_like_mini_line(raw_line: str) -> bool:
    """Cheap structural test: does this line have the MINI fixed-width layout?"""
    raw = raw_line.rstrip("\r\n")
    if len(raw) < MINI_LINE_MIN_LENGTH or not raw.strip():
        return False
    padded = raw.ljust(MINI_LINE_FULL_LENGTH)
    return (
        padded[0:1].isdigit()
        and (padded[1:2].strip() == "" or padded[1:2].isdigit())
        and padded[2:10].isdigit()
        and padded[10:23].isdigit()
        and padded[23:37].strip().isdigit()
        and padded[37:38].strip().isdigit()
        and padded[38:43].strip().isdigit()
    )


def looks_like_mini_file(path) -> bool:
    """True if the first non-blank line of *path* looks like a MINI record."""
    with _open_text(path, encoding="ascii", errors="ignore") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            return looks_like_mini_line(raw)
    return False


# ---------------------------------------------------------------------------
# Record / dataset model
# ---------------------------------------------------------------------------
@dataclass
class MiniRecord:
    """One MINI normal point: the raw file fields plus derived conveniences.

    The stored attributes correspond 1:1 to the MINI columns; derived
    quantities (SI scalings, display names) are read-only properties so they
    cannot drift out of sync with the raw values.
    """

    # --- raw MINI fields, in column order -------------------------------
    format_code: int
    laser_color_code: Optional[int]
    launch_date: str  # YYYYMMDD as written in the file
    launch_time: str  # HHMMSSsssssss as written in the file
    light_time_raw: int  # two-way light time, 0.1 ps
    reflector_id: int
    station_id: str  # normalized 5-char ILRS code
    number_of_returns: Optional[int]
    uncertainty_raw: int  # original MINI two-way sigma, 0.1 ps
    signal_noise_ratio_raw: Optional[int]  # S/N * 10
    quality_code: Optional[str]
    pressure_raw: int  # hPa * 100 (> 0)
    temperature_raw: int  # 0.1 deg (see module docstring)
    humidity_percent: int  # %, 0..100
    wavelength_raw: int  # 0.1 nm (> 0)
    version_code: Optional[str]
    duration_s: Optional[int]
    source_format: Optional[str]

    # --- derived, computed once at parse time ---------------------------
    launch_epoch: Epoch = field(repr=False)
    seconds_of_day: float = 0.0
    index: int = 0  # 0-based record index within the file
    source_line_no: int = 0  # 1-based line number in the source MINI file
    source_line: str = ""  # original source line without trailing newline

    def __post_init__(self) -> None:
        if not isinstance(self.launch_epoch, Epoch):
            raise TypeError("launch_epoch must be an Epoch.")
        self.launch_epoch.require_scale(TimeScale.UTC, name="launch_epoch")

    # ------------------------------------------------------------------
    # Derived scalar conveniences (computed from the raw fields)
    # ------------------------------------------------------------------
    @property
    def observed_round_trip_time_s(self) -> float:
        """Observed two-way light time in seconds."""
        return float(self.light_time_raw) * TIME_UNIT_S

    @property
    def uncertainty_two_way_s(self) -> float:
        """Original MINI uncertainty as two-way round-trip light-time sigma [s]."""
        return float(self.uncertainty_raw) * TIME_UNIT_S

    @property
    def range_uncertainty_one_way_m(self) -> float:
        """Original MINI one-way range sigma [m], retained for diagnostics."""
        return 0.5 * C_LIGHT_M_PER_S * self.uncertainty_two_way_s

    @property
    def pressure_hpa(self) -> float:
        return float(self.pressure_raw) / 100.0

    @property
    def temperature_c(self) -> float:
        # See the module docstring for the 0.1 degC vs 0.1 K caveat.
        return float(self.temperature_raw) / 10.0

    @property
    def temperature_k(self) -> float:
        return self.temperature_c + 273.15

    @property
    def wavelength_nm(self) -> float:
        return float(self.wavelength_raw) / 10.0

    @property
    def wavelength_um(self) -> float:
        return self.wavelength_nm / 1000.0

    @property
    def reflector_name(self) -> str:
        return REFLECTOR_NAMES.get(self.reflector_id, str(self.reflector_id))

    @property
    def station_name(self) -> str:
        """Catalog token used to resolve the station in the builtin catalog."""
        return canonical_station_id(self.station_id)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_launch_epoch(
    date_text: str,
    time_text: str,
    *,
    line_no: int,
) -> tuple[Epoch, float]:
    """Return (UTC Epoch, seconds-of-day) for the MINI launch fields."""
    date_value = date_text.strip()
    time_value = time_text.strip()

    if len(date_value) != 8 or not date_value.isdigit():
        raise ValueError(f"line {line_no}: invalid MINI launch date {date_text!r}; expected YYYYMMDD")
    if len(time_value) != 13 or not time_value.isdigit():
        raise ValueError(f"line {line_no}: invalid MINI launch time {time_text!r}; expected HHMMSSsssssss")

    hour = int(time_value[0:2])
    minute = int(time_value[2:4])
    second = int(time_value[4:6])
    frac_100ns = int(time_value[6:13])

    seconds_of_day = hour * 3600.0 + minute * 60.0 + second + frac_100ns * TIME_100NS_S
    return (
        Epoch.from_date_seconds(date_value, seconds_of_day, scale=TimeScale.UTC),
        seconds_of_day,
    )


def parse_mini_line(raw_line: str, *, line_no: int = 0, index: int = 0) -> MiniRecord:
    """Parse one MINI fixed-width line into a :class:`MiniRecord`."""
    raw = raw_line.rstrip("\r\n")
    if len(raw) < MINI_LINE_MIN_LENGTH:
        raise ValueError(
            f"line {line_no}: MINI line is too short ({len(raw)} chars; expected at least {MINI_LINE_MIN_LENGTH})"
        )
    padded = raw.ljust(MINI_LINE_FULL_LENGTH)

    launch_date = padded[2:10].strip()
    launch_time = padded[10:23].strip()
    launch_epoch, seconds_of_day = _parse_launch_epoch(launch_date, launch_time, line_no=line_no)

    station_id_raw = padded[38:43].strip()
    if not station_id_raw:
        raise ValueError(f"line {line_no}: required MINI field 'station_id' is blank")
    # Some MINI files encode leading-zero station IDs with a blank first
    # column, e.g. " 1910" for Grasse.  Normalize numeric IDs to the
    # canonical 5-character code (01910 / 07941 / 08834 / ...).
    station_id = station_id_raw.zfill(5) if station_id_raw.isdigit() else station_id_raw

    # Fields the downstream computation depends on are *required*: a blank
    # value raises here, with the offending line number, so that no later
    # stage ever needs to handle missing MINI-owned data.
    format_code = _parse_int(padded[0:1], field="format_code", line_no=line_no)
    light_time_raw = _parse_int(padded[23:37], field="light_time", line_no=line_no)
    reflector_id = _parse_int(padded[37:38], field="reflector_id", line_no=line_no)
    uncertainty_raw = _parse_int(padded[46:52], field="uncertainty", line_no=line_no)
    pressure_raw = _parse_int(padded[56:62], field="pressure", line_no=line_no)
    temperature_raw = _parse_int(padded[62:66], field="temperature", line_no=line_no)
    humidity_percent = _parse_int(padded[66:68], field="humidity", line_no=line_no)
    wavelength_raw = _parse_int(padded[68:73], field="wavelength", line_no=line_no)
    required_values = (
        format_code,
        light_time_raw,
        reflector_id,
        uncertainty_raw,
        pressure_raw,
        temperature_raw,
        humidity_percent,
        wavelength_raw,
    )
    if any(value is None for value in required_values):
        raise AssertionError("Required MINI fields must not be None after parsing.")
    assert format_code is not None
    assert light_time_raw is not None
    assert reflector_id is not None
    assert uncertainty_raw is not None
    assert pressure_raw is not None
    assert temperature_raw is not None
    assert humidity_percent is not None
    assert wavelength_raw is not None

    # Physical sanity: zero / negative values in these fields are placeholders
    # for missing data and must be rejected just like blanks.
    if uncertainty_raw <= 0:
        raise ValueError(f"line {line_no}: MINI uncertainty must be > 0 (0.1 ps), got {uncertainty_raw}")
    if pressure_raw <= 0:
        raise ValueError(f"line {line_no}: MINI pressure must be > 0 (hPa*100), got {pressure_raw}")
    if not (0 <= humidity_percent <= 100):
        raise ValueError(f"line {line_no}: MINI humidity must be within 0..100 %, got {humidity_percent}")
    if wavelength_raw <= 0:
        raise ValueError(f"line {line_no}: MINI wavelength must be > 0 (0.1 nm), got {wavelength_raw}")

    return MiniRecord(
        format_code=format_code,
        laser_color_code=_parse_int(padded[1:2], field="laser_color_code", line_no=line_no, required=False),
        launch_date=launch_date,
        launch_time=launch_time,
        light_time_raw=light_time_raw,
        reflector_id=reflector_id,
        station_id=station_id,
        number_of_returns=_parse_int(padded[43:46], field="number_of_returns", line_no=line_no, required=False),
        uncertainty_raw=uncertainty_raw,
        signal_noise_ratio_raw=_parse_int(padded[52:55], field="signal_noise_ratio", line_no=line_no, required=False),
        quality_code=_blank_to_none(padded[55:56]),
        pressure_raw=pressure_raw,
        temperature_raw=temperature_raw,
        humidity_percent=humidity_percent,
        wavelength_raw=wavelength_raw,
        version_code=_blank_to_none(padded[73:74]),
        duration_s=_parse_int(padded[74:78], field="duration", line_no=line_no, required=False),
        source_format=_blank_to_none(padded[80:89]),
        launch_epoch=launch_epoch,
        seconds_of_day=seconds_of_day,
        index=index,
    )


def _npt_records_from_mini(records: Sequence[MiniRecord]) -> list[_NptRecord]:
    output = [
        _NptRecord(
            station_name=record.station_name,
            reflector_name=record.reflector_name,
            transmit_epoch=record.launch_epoch,
            round_trip_time_s=record.observed_round_trip_time_s,
            uncertainty_two_way_s=record.uncertainty_two_way_s,
            pressure_hpa=record.pressure_hpa,
            temperature_k=record.temperature_k,
            humidity_percent=float(record.humidity_percent),
            wavelength_nm=record.wavelength_nm,
            index=int(record.index),
            station_code=record.station_id,
            reflector_code=str(record.reflector_id),
        )
        for record in records
    ]
    for index, record in enumerate(output):
        record.index = index
    return output


def parse_mini_file(path):
    """Parse a MINI normal-point file (.dat / .mini, optionally gzipped).

    Invalid nonblank records are skipped and returned as structured import
    issues on the resulting dataset. ``NormalPointsConvert`` publishes those
    issues in its typed import report; the parser never creates an implicit log
    file.

    After this function returns, every record in the dataset is guaranteed to
    carry complete MINI-owned data: launch epoch, observed two-way light time,
    reflector id, station id, uncertainty, pressure, temperature, humidity,
    and wavelength. Catalog identities are resolved at the
    :meth:`LlrObservationProcessor.process` boundary.
    """
    path = Path(path)
    if not looks_like_mini_file(path):
        raise ValueError(f"Input does not look like a MINI fixed-width normal-point file: {path}")

    records: List[MiniRecord] = []
    n_input_records = 0
    n_invalid_records = 0
    import_issues: list[dict[str, object]] = []

    with _open_text(path, encoding="ascii", errors="strict") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            n_input_records += 1
            try:
                record = parse_mini_line(raw, line_no=line_no, index=len(records))
                record.source_line_no = line_no
                record.source_line = raw.rstrip("\r\n")
                records.append(record)
            except ValueError as exc:
                n_invalid_records += 1
                original = raw.rstrip("\r\n")
                import_issues.append(
                    {
                        "line": line_no,
                        "reason": str(exc),
                        "content": original,
                    }
                )

    if not records:
        raise ValueError(
            f"No valid MINI normal-point records found in {path}. "
            f"Data lines read={n_input_records}, invalid records skipped={n_invalid_records}."
        )

    return _NptDataset(
        records=_npt_records_from_mini(records),
        name=path.stem,
        n_input_records=n_input_records,
        n_invalid_records=n_invalid_records,
        import_issues=import_issues,
    )


__all__ = [
    "MiniRecord",
    "looks_like_mini_file",
    "looks_like_mini_line",
    "parse_mini_file",
    "parse_mini_line",
]
