"""
CRD normal-point reader.

Reads ILRS Consolidated Range Data (CRD, v1/v2) normal-point files directly
into canonical :class:`lunarops.classes.observation.normal_points.NptRecord` objects.  The
following CRD records are interpreted:

    H1  format header           (CRD version)
    H2  station header          (station name, pad / system / occupancy ids)
    H3  target header           (reflector name -> MINI reflector id)
    H4  session header          (session start date, for seconds-of-day anchor)
    C0  system configuration    (wavelength)
    11  normal point            (epoch, time of flight, window, returns, RMS)
    20  meteorological record   (pressure, temperature, humidity)

Caveats:
  * LLR record 11 epochs are interpreted as ground transmit times. The generic
    CRD epoch-event field is intentionally ignored at this import boundary.
  * Known station names and aliases are resolved by the central station
    identity registry. When no match is found, the CRD pad id is zero-padded
    to 5 characters.
  * CRD record 11 ``bin_rms`` is retained directly as the canonical two-way
    uncertainty. The normal-point window and number of returns are ignored and
    are not part of the canonical LLR artifact.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.classes.time import Epoch, TimeScale
from lunarops.base.station_identity import canonical_station_id, station_ilrs_code

# CRD target token -> (canonical catalog name, MINI interchange id).
CRD_REFLECTOR_IDENTITY_BY_NAME = {
    "APOLLO11": ("Apollo 11", 0),
    "A11": ("Apollo 11", 0),
    "AP11": ("Apollo 11", 0),
    "LUNOKHOD1": ("Lunokhod 1", 1),
    "LUNA17": ("Lunokhod 1", 1),
    "L1": ("Lunokhod 1", 1),
    "APOLLO14": ("Apollo 14", 2),
    "A14": ("Apollo 14", 2),
    "AP14": ("Apollo 14", 2),
    "APOLLO15": ("Apollo 15", 3),
    "A15": ("Apollo 15", 3),
    "AP15": ("Apollo 15", 3),
    "LUNOKHOD2": ("Lunokhod 2", 4),
    "LUNA21": ("Lunokhod 2", 4),
    "L2": ("Lunokhod 2", 4),
}


def _canonical(token: str) -> str:
    return "".join(ch for ch in str(token or "").upper() if ch.isalnum())


def _open_text(path):
    path = Path(path)
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="ascii", errors="replace", newline=None)
    return path.open("r", encoding="ascii", errors="replace", newline=None)


CRD_SUFFIXES = (".npt", ".crd", ".frd", ".npt.gz", ".crd.gz", ".frd.gz")


def looks_like_crd_file(path) -> bool:
    """Cheap CRD detection: known suffix, or an 'H1 CRD' / 'h1 crd' first line."""
    path = Path(path)
    name = path.name.lower()
    if any(name.endswith(suffix) for suffix in CRD_SUFFIXES):
        return True
    try:
        with _open_text(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                return line.upper().startswith("H1") and "CRD" in line.upper()
    except OSError:
        return False
    return False


def _to_float(text: str) -> Optional[float]:
    try:
        value = float(text)
    except TypeError, ValueError:
        return None
    return value


def _to_int(text: str) -> Optional[int]:
    try:
        return int(float(text))
    except TypeError, ValueError:
        return None


@dataclass
class _CrdMeteo:
    seconds_of_day: float
    pressure_hpa: Optional[float]
    temperature_k: Optional[float]
    humidity_percent: Optional[float]


@dataclass
class _CrdNormalPoint:
    seconds_of_day: float
    time_of_flight_s: float
    bin_rms_ps: Optional[float]
    snr: Optional[float]
    source_line_no: int
    source_line: str


@dataclass
class _CrdSession:
    crd_version: int = 1
    station_name: str = ""
    station_pad_id: str = ""
    target_name: str = ""
    start_epoch: Optional[Epoch] = None
    wavelength_nm: Optional[float] = None
    normal_points: List[_CrdNormalPoint] = field(default_factory=list)
    meteo: List[_CrdMeteo] = field(default_factory=list)
    input_record_count: int = 0
    import_issues: list[dict[str, object]] = field(default_factory=list)


def _circular_distance(a: float, b: float) -> float:
    d = abs(a - b) % SECONDS_PER_DAY
    return min(d, SECONDS_PER_DAY - d)


def parse_crd_sessions(path) -> List[_CrdSession]:
    """Parse the CRD records relevant to MINI conversion, session by session."""
    sessions: List[_CrdSession] = []
    current: Optional[_CrdSession] = None
    crd_version = 1

    with _open_text(path) as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            tag = fields[0].upper()

            if tag == "H1":
                # H1 CRD <version> <year> <month> <day> <hour>
                crd_version = _to_int(fields[2]) or 1

            elif tag == "H2":
                current = _CrdSession(crd_version=crd_version)
                sessions.append(current)
                current.station_name = fields[1]
                current.station_pad_id = fields[2] if len(fields) > 2 else ""

            elif tag == "H3" and current is not None:
                current.target_name = fields[1]

            elif tag == "H4" and current is not None:
                # H4 <data type> Y M D h m s  Y M D h m s ...
                try:
                    year, month, day = int(fields[2]), int(fields[3]), int(fields[4])
                    hour, minute, second = (
                        int(fields[5]),
                        int(fields[6]),
                        int(fields[7]),
                    )
                    current.start_epoch = Epoch.from_calendar(
                        year, month, day, hour, minute, second, scale=TimeScale.UTC
                    )
                except IndexError, ValueError:
                    current.start_epoch = None

            elif tag == "C0" and current is not None:
                # C0 <detail type> <wavelength nm> <component ids...>
                wavelength = _to_float(fields[2]) if len(fields) > 2 else None
                if wavelength is not None:
                    current.wavelength_nm = wavelength

            elif tag == "11" and current is not None:
                current.input_record_count += 1
                # v1: 11 sod tof sysconfig epoch_event np_window n_ranges
                #     bin_rms skew kurtosis peak-mean return_rate ch
                # v2 adds snr at the end.
                seconds_of_day = _to_float(fields[1]) if len(fields) > 1 else None
                tof = _to_float(fields[2]) if len(fields) > 2 else None
                bin_rms = _to_float(fields[7]) if len(fields) > 7 else None
                if seconds_of_day is None or tof is None or bin_rms is None or bin_rms <= 0.0:
                    current.import_issues.append(
                        {
                            "line": line_no,
                            "reason": "CRD record 11 requires valid seconds-of-day, time-of-flight, and positive bin RMS.",
                            "content": line,
                        }
                    )
                    continue
                snr = _to_float(fields[13]) if (crd_version >= 2 and len(fields) > 13) else None
                current.normal_points.append(
                    _CrdNormalPoint(
                        seconds_of_day=seconds_of_day,
                        time_of_flight_s=tof,
                        bin_rms_ps=bin_rms,
                        snr=snr,
                        source_line_no=line_no,
                        source_line=line,
                    )
                )

            elif tag == "20" and current is not None:
                # 20 sod pressure(hPa) temperature(K) humidity(%) origin
                current.meteo.append(
                    _CrdMeteo(
                        seconds_of_day=_to_float(fields[1]) or 0.0,
                        pressure_hpa=_to_float(fields[2]) if len(fields) > 2 else None,
                        temperature_k=_to_float(fields[3]) if len(fields) > 3 else None,
                        humidity_percent=_to_float(fields[4]) if len(fields) > 4 else None,
                    )
                )

    return [s for s in sessions if s.input_record_count]


def _station_identity(session: _CrdSession) -> tuple[str, str]:
    pad = str(session.station_pad_id or "").strip()
    for candidate in (session.station_name, pad):
        try:
            station = canonical_station_id(candidate)
            return station, station_ilrs_code(station)
        except ValueError:
            pass
    if pad.isdigit():
        return canonical_station_id(session.station_name), pad.zfill(5)[:5]
    raise ValueError(
        f"Cannot map CRD station {session.station_name!r} (pad {session.station_pad_id!r}) to a canonical identity."
    )


def _reflector_identity(session: _CrdSession) -> tuple[str, int]:
    token = _canonical(session.target_name)
    if token in CRD_REFLECTOR_IDENTITY_BY_NAME:
        return CRD_REFLECTOR_IDENTITY_BY_NAME[token]
    raise ValueError(
        f"Cannot map CRD target {session.target_name!r} to a canonical identity; extend CRD_REFLECTOR_IDENTITY_BY_NAME."
    )


def _nearest_meteo(meteo: Sequence[_CrdMeteo], seconds_of_day: float) -> Optional[_CrdMeteo]:
    if not meteo:
        return None
    return min(meteo, key=lambda rec: _circular_distance(rec.seconds_of_day, seconds_of_day))


@dataclass
class _CrdObservation:
    station_name: str
    station_code: str
    reflector_name: str
    reflector_id: int
    transmit_epoch: Epoch
    time_of_flight_s: float
    uncertainty_two_way_s: float
    pressure_hpa: float
    temperature_k: float
    humidity_percent: float
    wavelength_nm: float
    signal_noise_ratio: Optional[float]
    source_format: str
    source_record: str


def _crd_observations(
    sessions: Sequence[_CrdSession],
    import_issues: list[dict[str, object]] | None = None,
) -> List[_CrdObservation]:
    observations: List[_CrdObservation] = []
    for session_index, session in enumerate(sessions, start=1):
        issues = session.import_issues if import_issues is None else import_issues
        try:
            if session.start_epoch is None:
                raise ValueError("CRD session is missing the H4 start epoch; cannot anchor seconds-of-day.")
            station_name, station_code = _station_identity(session)
            reflector_name, reflector_id = _reflector_identity(session)
        except ValueError as exc:
            for record in session.normal_points:
                issues.append({"line": record.source_line_no, "reason": str(exc), "content": record.source_line})
            continue
        assert session.start_epoch is not None
        day_anchor = Epoch.from_date_seconds(
            session.start_epoch.date_iso(),
            0.0,
            scale=TimeScale.UTC,
        )
        session_start_sod = day_anchor.seconds_until(session.start_epoch)

        for record_index, np_rec in enumerate(session.normal_points, start=1):
            seconds = float(np_rec.seconds_of_day)
            day_offset = 1 if seconds + 1.0 < float(session_start_sod) else 0
            epoch = day_anchor.shifted(day_offset * SECONDS_PER_DAY + seconds)

            label = f"CRD NP at {epoch.isot(scale=TimeScale.UTC)} (station {station_code}, reflector {reflector_id})"
            try:
                meteo = _nearest_meteo(session.meteo, seconds)
                if meteo is None:
                    raise ValueError(f"{label}: the CRD session has no '20' meteorological record.")
                if meteo.pressure_hpa is None or meteo.temperature_k is None or meteo.humidity_percent is None:
                    raise ValueError(f"{label}: the nearest CRD '20' record is incomplete.")
                if session.wavelength_nm is None or session.wavelength_nm <= 0.0:
                    raise ValueError(f"{label}: the CRD session 'C0' record carries no usable laser wavelength.")
            except ValueError as exc:
                issues.append(
                    {"line": np_rec.source_line_no, "reason": str(exc), "content": np_rec.source_line}
                )
                continue
            assert meteo is not None
            assert np_rec.bin_rms_ps is not None

            observations.append(
                _CrdObservation(
                    station_name=station_name,
                    station_code=station_code,
                    reflector_name=reflector_name,
                    reflector_id=reflector_id,
                    transmit_epoch=epoch,
                    time_of_flight_s=float(np_rec.time_of_flight_s),
                    uncertainty_two_way_s=float(np_rec.bin_rms_ps) * 1.0e-12,
                    pressure_hpa=float(meteo.pressure_hpa),
                    temperature_k=float(meteo.temperature_k),
                    humidity_percent=float(meteo.humidity_percent),
                    wavelength_nm=float(session.wavelength_nm),
                    signal_noise_ratio=np_rec.snr,
                    source_format=f"crd-v{session.crd_version}",
                    source_record=f"session:{session_index}/normal-point:{record_index}",
                )
            )

    observations.sort(
        key=lambda observation: (
            observation.transmit_epoch.jd1,
            observation.transmit_epoch.jd2,
        )
    )
    return observations


def crd_sessions_to_npt_records(
    sessions: Sequence[_CrdSession],
):
    """Convert parsed CRD sessions directly to canonical NptRecord objects."""
    from lunarops.classes.observation.normal_points import NptRecord

    return [
        NptRecord(
            station_name=observation.station_name,
            reflector_name=observation.reflector_name,
            transmit_epoch=observation.transmit_epoch,
            round_trip_time_s=observation.time_of_flight_s,
            uncertainty_two_way_s=observation.uncertainty_two_way_s,
            pressure_hpa=observation.pressure_hpa,
            temperature_k=observation.temperature_k,
            humidity_percent=observation.humidity_percent,
            wavelength_nm=observation.wavelength_nm,
            index=index,
            station_code=observation.station_code,
            reflector_code=str(observation.reflector_id),
        )
        for index, observation in enumerate(_crd_observations(sessions))
    ]


def parse_crd_file(path):
    """Parse a CRD v1/v2 file directly into a canonical NptDataset."""
    from lunarops.classes.observation.normal_points import NptDataset

    source = Path(path)
    sessions = parse_crd_sessions(source)
    if not sessions:
        raise ValueError(f"No CRD normal-point sessions found in {source}")
    import_issues = [issue for session in sessions for issue in session.import_issues]
    observations = _crd_observations(sessions, import_issues)
    from lunarops.classes.observation.normal_points import NptRecord

    records = [
        NptRecord(
            station_name=item.station_name,
            reflector_name=item.reflector_name,
            transmit_epoch=item.transmit_epoch,
            round_trip_time_s=item.time_of_flight_s,
            uncertainty_two_way_s=item.uncertainty_two_way_s,
            pressure_hpa=item.pressure_hpa,
            temperature_k=item.temperature_k,
            humidity_percent=item.humidity_percent,
            wavelength_nm=item.wavelength_nm,
            index=index,
            station_code=item.station_code,
            reflector_code=str(item.reflector_id),
        )
        for index, item in enumerate(observations)
    ]
    if not records:
        raise ValueError(f"No valid CRD normal points found in {source}.")
    input_count = sum(session.input_record_count for session in sessions)
    return NptDataset(
        records=records,
        name=source.stem,
        n_input_records=input_count,
        n_invalid_records=len(import_issues),
        import_issues=import_issues,
    )


__all__ = [
    "CRD_REFLECTOR_IDENTITY_BY_NAME",
    "CRD_SUFFIXES",
    "crd_sessions_to_npt_records",
    "looks_like_crd_file",
    "parse_crd_file",
    "parse_crd_sessions",
]
