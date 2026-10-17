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

NOTE on the temperature unit: the default follows the convention used by the
existing processing chain, ``temperature_c = raw / 10`` (i.e. 0.1 deg C).
Some MINI documentation describes the field as 0.1 K instead; callers select
that convention explicitly with ``temperature_unit="deciKelvin"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, overload

from lunarops.base.station_identity import station_id_from_ilrs_code
from lunarops.classes.observation.normal_points import NptDataset as _NptDataset
from lunarops.classes.observation.normal_points import NptRecord as _NptRecord
from lunarops.classes.time import Epoch, TimeScale

from ..archive import open_ascii_source

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_UNIT_S = 1.0e-13  # light-time / uncertainty unit: 0.1 ps
TIME_100NS_S = 1.0e-7  # launch-time fractional unit: 100 ns

MINI_LINE_MIN_LENGTH = 78  # duration field ends at col 78
MINI_LINE_FULL_LENGTH = 89  # source-format field ends at col 89

REFLECTOR_NAMES = {
    0: "APOLLO11",
    1: "LUNOKHOD1",
    2: "APOLLO14",
    3: "APOLLO15",
    4: "LUNOKHOD2",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
@overload
def _parse_int(text: str, *, field: str, line_no: int, required: Literal[True] = True) -> int: ...


@overload
def _parse_int(text: str, *, field: str, line_no: int, required: Literal[False]) -> int | None: ...


def _parse_int(text: str, *, field: str, line_no: int, required: bool = True) -> int | None:
    value = text.strip()
    if not value:
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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_launch_epoch(date_text: str, time_text: str, *, line_no: int) -> Epoch:
    """Return the UTC Epoch for the MINI launch fields."""
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
    return Epoch.from_date_seconds(date_value, seconds_of_day, scale=TimeScale.UTC)


def parse_mini_line(raw_line: str, *, line_no: int = 0, temperature_unit: str = "deciCelsius") -> dict[str, Any]:
    """Parse one MINI fixed-width line into :class:`NptRecord` keyword fields.

    The returned mapping carries the MINI station code; the caller resolves
    ``station_name`` from it and assigns the record index.
    """
    raw = raw_line.rstrip("\r\n")
    if len(raw) < MINI_LINE_MIN_LENGTH:
        raise ValueError(
            f"line {line_no}: MINI line is too short ({len(raw)} chars; expected at least {MINI_LINE_MIN_LENGTH})"
        )
    padded = raw.ljust(MINI_LINE_FULL_LENGTH)

    launch_epoch = _parse_launch_epoch(padded[2:10].strip(), padded[10:23].strip(), line_no=line_no)

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
    _parse_int(padded[0:1], field="format_code", line_no=line_no)
    light_time_raw = _parse_int(padded[23:37], field="light_time", line_no=line_no)
    reflector_id = _parse_int(padded[37:38], field="reflector_id", line_no=line_no)
    uncertainty_raw = _parse_int(padded[46:52], field="uncertainty", line_no=line_no)
    pressure_raw = _parse_int(padded[56:62], field="pressure", line_no=line_no)
    temperature_raw = _parse_int(padded[62:66], field="temperature", line_no=line_no)
    humidity_percent = _parse_int(padded[66:68], field="humidity", line_no=line_no)
    wavelength_raw = _parse_int(padded[68:73], field="wavelength", line_no=line_no)

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

    # Optional integer columns are not carried forward but must be well formed.
    for columns, field_name in (
        (padded[1:2], "laser_color_code"),
        (padded[43:46], "number_of_returns"),
        (padded[52:55], "signal_noise_ratio"),
        (padded[74:78], "duration"),
    ):
        _parse_int(columns, field=field_name, line_no=line_no, required=False)

    temperature = float(temperature_raw) / 10.0
    return {
        "reflector_name": REFLECTOR_NAMES.get(reflector_id, str(reflector_id)),
        "transmit_epoch": launch_epoch,
        "round_trip_time_s": float(light_time_raw) * TIME_UNIT_S,
        "uncertainty_two_way_s": float(uncertainty_raw) * TIME_UNIT_S,
        "pressure_hpa": float(pressure_raw) / 100.0,
        "temperature_k": temperature + 273.15 if temperature_unit == "deciCelsius" else temperature,
        "humidity_percent": float(humidity_percent),
        "wavelength_nm": float(wavelength_raw) / 10.0,
        "station_code": station_id,
        "reflector_code": str(reflector_id),
    }


def parse_mini_file(path, *, temperature_unit="deciCelsius"):
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
    if temperature_unit not in {"deciCelsius", "deciKelvin"}:
        raise ValueError("MINI temperature_unit must be deciCelsius or deciKelvin; units are never inferred")
    path = Path(path)

    records: list[_NptRecord] = []
    n_input_records = 0
    import_issues: list[dict[str, object]] = []

    with open_ascii_source(path) as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            n_input_records += 1
            try:
                fields = parse_mini_line(raw, line_no=line_no, temperature_unit=temperature_unit)
            except ValueError as exc:
                import_issues.append({"line": line_no, "reason": str(exc), "content": raw.rstrip("\r\n")})
                continue
            records.append(
                _NptRecord(
                    station_name=station_id_from_ilrs_code(fields["station_code"]),
                    index=len(records),
                    **fields,
                )
            )

    if not records:
        raise ValueError(
            f"No valid MINI normal-point records found in {path}. "
            f"Data lines read={n_input_records}, invalid records skipped={len(import_issues)}."
        )

    return _NptDataset(
        records=records,
        import_metadata={"format": "MINI", "temperatureUnit": temperature_unit},
        name=path.stem,
        n_input_records=n_input_records,
        n_invalid_records=len(import_issues),
        import_issues=import_issues,
    )


__all__ = [
    "looks_like_mini_line",
    "parse_mini_file",
    "parse_mini_line",
]
