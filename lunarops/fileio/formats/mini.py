"""Import POLAC / OCA MINI fixed-width LLR normal-point files.

A MINI file holds one normal point per line in fixed-width ASCII columns (1-based, inclusive)::

    cols   width  field                         unit / encoding                        read as
    1      1      format code                   integer                                required
    2      1      laser colour code             1 = green, 2 = infrared                optional
    3-10   8      launch date, UTC              YYYYMMDD                               required
    11-23  13     launch time of day, UTC       HHMMSS + 7 digits of 100 ns            required
    24-37  14     observed two-way light time   integer, 0.1 ps                        required
    38     1      reflector id                  0=A11 1=L1 2=A14 3=A15 4=L2            required
    39-43  5      station id                    ILRS station code, 5 characters        required
    44-46  3      number of returns             integer                                optional
    47-52  6      two-way uncertainty           integer, 0.1 ps, > 0                   required
    53-55  3      signal-to-noise ratio         integer, S/N * 10                      optional
    56     1      quality code                  single character                       not read
    57-62  6      surface pressure              integer, 0.01 hPa, > 0                 required
    63-66  4      surface temperature           integer, 0.1 degC or 0.1 K             required
    67-68  2      relative humidity             integer percent, 0-100                 required
    69-73  5      laser wavelength              integer, 0.1 nm, > 0                   required
    74     1      version code                  single character                       not read
    75-78  4      session duration              integer seconds                        optional
    79-80  2      unused
    81-89  9      source format tag             free text                              not read

Required fields must be present and valid; optional integer fields may be blank but must be integers when
present, and are not carried into the record. A line must be at least 78 characters long. Numeric station
ids written with a blank leading column are zero-padded to five digits.

The parser converts the fields to SI-based canonical values: the UTC launch (ground transmit) epoch, the
two-way light time and uncertainty in seconds, pressure in hPa, temperature in K, humidity in percent and
wavelength in nm. MINI records carry no correction flags (such as troposphere or centre-of-mass applied), so
none are represented.

By default the temperature field is read as 0.1 degC (``temperature_unit="deciCelsius"``). Some MINI
documentation describes the field as 0.1 K; ``temperature_unit="deciKelvin"`` selects that reading. The unit
is never inferred.

References
----------
- POLAC LLR data: https://polac.obspm.fr/llrdatae.html
- MINI format description: http://www.geoazur.fr/astrogeo/observations/donnees/lune/mini-format.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, overload

from lunarops.base.station_identity import station_id_from_ilrs_code
from lunarops.classes.observation.normal_points import NptDataset as _NptDataset
from lunarops.classes.observation.normal_points import NptRecord as _NptRecord
from lunarops.classes.time import Epoch, TimeScale
from lunarops.fileio.formats.normal_point_record import read_normal_point_record

from ..archive import open_ascii_source

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
    """Return whether a line has the MINI fixed-width layout.

    This is a structural test only: after removing the line ending the line must be nonblank and at least 78
    characters long, column 1 must be a digit, column 2 blank or a digit, columns 3-10 and 11-23 digits, and
    columns 24-37, 38 and 39-43 digits after stripping blanks. Field values are not validated.

    Parameters
    ----------
    raw_line : str
        Candidate line.

    Returns
    -------
    bool
        True if the layout matches.
    """
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
    """Parse one MINI fixed-width line into ``NptRecord`` keyword fields.

    The returned mapping carries the MINI station code; the caller resolves ``station_name`` from it and
    assigns the record index.

    Parameters
    ----------
    raw_line : str
        One MINI line, with or without its line ending.
    line_no : int, optional
        Source line number used in error messages.
    temperature_unit : str, optional
        ``"deciCelsius"`` reads the temperature field as 0.1 degC and converts it to kelvin; any other value
        reads it as 0.1 K. ``parse_mini_file`` accepts only ``"deciCelsius"`` and ``"deciKelvin"``.

    Returns
    -------
    dict
        Keys ``reflector_name`` (APOLLO11, LUNOKHOD1, APOLLO14, APOLLO15 or LUNOKHOD2 for ids 0-4),
        ``transmit_epoch`` (``Epoch``, UTC), ``round_trip_time_s`` and
        ``uncertainty_two_way_s`` (s), ``pressure_hpa`` (hPa), ``temperature_k`` (K), ``humidity_percent``
        (%), ``wavelength_nm`` (nm), ``station_code`` (5-character code) and ``reflector_code`` (id as text).

    Raises
    ------
    ValueError
        If the line is shorter than 78 characters, a required field is blank or not an integer, an optional
        integer field is malformed, the launch date or time is malformed, the reflector id is not 0-4, the
        light time, uncertainty, pressure or wavelength is not positive, or the humidity is outside 0-100 %.
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
    if reflector_id not in REFLECTOR_NAMES:
        raise ValueError(f"line {line_no}: MINI reflector id must be 0..{max(REFLECTOR_NAMES)}, got {reflector_id}")
    if light_time_raw <= 0:
        raise ValueError(f"line {line_no}: MINI light time must be > 0 (0.1 ps), got {light_time_raw}")
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
        "reflector_name": REFLECTOR_NAMES[reflector_id],
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
    """Parse a MINI normal-point file (``.dat`` or ``.mini``, optionally gzipped).

    Every nonblank line counts as an input record. Lines rejected by ``parse_mini_line`` or by
    ``read_normal_point_record`` are skipped and kept as import issues (line number, reason and original
    content) on the dataset; no log file is written. An unknown ILRS station code is an identity error and
    raises ``ValueError`` for the whole file. Every returned record carries a UTC launch epoch, two-way light
    time, reflector, station, uncertainty, pressure, temperature, humidity and wavelength.

    Parameters
    ----------
    path : str or Path
        MINI file, read as ASCII.
    temperature_unit : {"deciCelsius", "deciKelvin"}, optional
        Unit of the temperature field: 0.1 degC or 0.1 K.

    Returns
    -------
    NptDataset
        Records indexed from 0 in file order, named after the file stem, with input and invalid record
        counts, import issues and ``import_metadata`` ``{"format": "MINI", "temperatureUnit": ...}``.

    Raises
    ------
    ValueError
        If ``temperature_unit`` is not supported, no valid record is found, or a station code is unknown.
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
            # An unknown station code is an identity error and stops the import.
            station_name = station_id_from_ilrs_code(fields["station_code"])
            try:
                record = read_normal_point_record(station_name=station_name, index=len(records), **fields)
            except ValueError as exc:
                import_issues.append({"line": line_no, "reason": f"line {line_no}: {exc}", "content": raw.rstrip("\r\n")})
                continue
            records.append(record)

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
