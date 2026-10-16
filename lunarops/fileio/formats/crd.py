"""
CRD normal-point reader.

Reads ILRS Consolidated Range Data (CRD, v2) normal-point files directly
into canonical :class:`lunarops.classes.observation.normal_points.NptRecord` objects.  The
following CRD records are interpreted:

    H1  format header           (CRD version)
    H2  station header          (station name, pad / system / occupancy ids)
    H3  target header           (reflector name -> MINI reflector id)
    H4  session header          (session start date, for seconds-of-day anchor)
    C0  system configuration    (wavelength)
    11  normal point            (epoch, time of flight, window, returns, RMS)
    20  meteorological record   (pressure, temperature, humidity)

Input contract: only epoch-event 2 (ground transmit) is supported. Other
 events are rejected with their source line. Meteorology uses absolute session
 UTC epochs and a bounded nearest neighbour. Bin RMS is a source quality
 statistic: callers must explicitly select binRmsAsSigma or fixed uncertainty.
"""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.base.station_identity import station_ilrs_code
from lunarops.classes.time import Epoch, TimeScale

# Exact CRD H2 names and their declared pad IDs. Five-digit station codes
# identify the corresponding MINI station/system; they are checked explicitly.
_CRD_STATIONS = {
    "APOL": ("APOLLO", ("7045", "70610")),
    "GRSM": ("GRASSE", ("7845", "1910", "01910")),
    "HALE": ("HALEAKALA", ("5661", "56610")),
    "MATM": ("MATERA", ("7941", "07941")),
    "MDOL": ("MCDONALD", ("7111", "71110")),
    "MLRS1": ("MLRS1", ("7111", "71111")),
    "MLRS2": ("MLRS2", ("7112", "71112")),
    "WETL": ("WETTZELL", ("8834", "08834")),
}
CRD_REFLECTOR_IDENTITY_BY_NAME = {
    "APOLLO11": ("APOLLO11", 0),
    "LUNOKHOD1": ("LUNOKHOD1", 1),
    "APOLLO14": ("APOLLO14", 2),
    "APOLLO15": ("APOLLO15", 3),
    "LUNOKHOD2": ("LUNOKHOD2", 4),
}


def _open_text(path):
    path = Path(path)
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="ascii", errors="strict", newline=None)
    return path.open("r", encoding="ascii", errors="strict", newline=None)


CRD_SUFFIXES = (".npt", ".crd", ".frd", ".npt.gz", ".crd.gz", ".frd.gz")


def looks_like_crd_file(path) -> bool:
    """Cheap CRD detection: known suffix, or an 'H1 CRD' / 'h1 crd' first line."""
    path = Path(path)
    try:
        with _open_text(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                return line.upper().split()[:2] == ["H1", "CRD"]
    except OSError:
        return False
    return False


def _to_float(text: str) -> Optional[float]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _to_int(text: str) -> Optional[int]:
    try:
        return int(text)
    except (TypeError, ValueError):
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
    configuration_id: str
    epoch_event: int = 2


@dataclass
class _CrdSession:
    station_name: str = ""
    station_pad_id: str = ""
    target_name: str = ""
    start_epoch: Optional[Epoch] = None
    wavelength_nm: Optional[float] = None
    end_epoch: Optional[Epoch] = None
    configurations: dict[str, float] = field(default_factory=dict)
    meteorology_issues: list[dict[str, object]] = field(default_factory=list)
    normal_points: List[_CrdNormalPoint] = field(default_factory=list)
    meteo: List[_CrdMeteo] = field(default_factory=list)
    input_record_count: int = 0
    import_issues: list[dict[str, object]] = field(default_factory=list)


def _session_epoch(session: _CrdSession, seconds: float) -> Epoch:
    if session.start_epoch is None or session.end_epoch is None:
        raise ValueError("CRD requires a complete H4 start/end interval.")
    if not 0.0 <= seconds < SECONDS_PER_DAY:
        raise ValueError("CRD seconds-of-day must be in [0, 86400). Leap-second records are unsupported.")
    anchor = Epoch.from_date_seconds(session.start_epoch.date_iso(), seconds, scale=TimeScale.UTC)
    if anchor < session.start_epoch:
        anchor = Epoch.from_date_seconds(session.end_epoch.date_iso(), seconds, scale=TimeScale.UTC)
    if anchor < session.start_epoch or anchor > session.end_epoch:
        raise ValueError("CRD epoch falls outside H4 session bounds.")
    return anchor


def _header_epoch(fields, offset):
    values = [int(value) for value in fields[offset : offset + 5]]
    return Epoch.from_calendar(*values, float(fields[offset + 5]), scale=TimeScale.UTC)


def parse_crd_sessions(path) -> List[_CrdSession]:
    """Read v2 sessions; structural failures abort and bad NP rows are counted."""
    sessions: List[_CrdSession] = []
    current = None
    station = None
    target = None
    header_seen = False
    file_ended = False
    with _open_text(path) as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            tag = fields[0].upper()
            try:
                if tag == "00":
                    continue
                if file_ended:
                    raise ValueError("record encountered after H9")
                if tag == "H1":
                    if header_seen:
                        raise ValueError("CRD permits exactly one H1 header")
                    header_seen = True
                    if current is not None:
                        raise ValueError("H1 encountered before session H8")
                    if len(fields) < 7 or fields[1].upper() != "CRD" or fields[2] != "2":
                        raise ValueError("H1 must declare CRD version 2")
                    _header_epoch(fields, 3) if len(fields) >= 9 else Epoch.from_calendar(
                        int(fields[3]), int(fields[4]), int(fields[5]), int(fields[6]), scale=TimeScale.UTC
                    )
                    station = target = None
                elif tag == "H2":
                    if not header_seen or len(fields) < 6 or current is not None:
                        raise ValueError("H2 requires H1 and a complete station header outside a session")
                    # Supported UTC realizations are declared by CRD H2 codes 3/4/7.
                    if fields[5] not in {"3", "4", "7"}:
                        raise ValueError("H2 requires supported UTC time scale 3 (USNO), 4 (GPS), or 7 (BIPM)")
                    _crd_station_identity(fields[1], fields[2])
                    station = (fields[1], fields[2])
                elif tag == "H3":
                    if station is None or len(fields) < 6 or current is not None:
                        raise ValueError("H3 requires H2 and a complete target header outside a session")
                    target = fields[1]
                    if target not in CRD_REFLECTOR_IDENTITY_BY_NAME:
                        raise ValueError(f"Unknown CRD H3 target name {target!r}.")
                elif tag == "H4":
                    if current is not None or station is None or target is None:
                        raise ValueError("H4 requires H1/H2/H3 and a closed preceding session")
                    if len(fields) < 14 or fields[1] != "1":
                        raise ValueError("H4 must declare normal-point data type 1 and start/end epochs")
                    start, end = _header_epoch(fields, 2), _header_epoch(fields, 8)
                    duration = start.seconds_until(end)
                    if duration < 0 or duration >= SECONDS_PER_DAY:
                        raise ValueError(
                            "H4 interval must be nonnegative and shorter than one day; multi-day sessions are unsupported"
                        )
                    current = _CrdSession(
                        station_name=station[0],
                        station_pad_id=station[1],
                        target_name=target,
                        start_epoch=start,
                        end_epoch=end,
                    )
                    sessions.append(current)
                elif tag == "C0":
                    if current is None or len(fields) < 4:
                        raise ValueError("C0 requires H4 and a wavelength/configuration id")
                    wavelength = _to_float(fields[2])
                    if wavelength is None or wavelength <= 0 or fields[3] in current.configurations:
                        raise ValueError("C0 requires positive finite wavelength and unique configuration id")
                    current.configurations[fields[3]] = wavelength
                elif tag == "11":
                    if current is None:
                        raise ValueError("record 11 requires an active H4 session")
                    current.input_record_count += 1
                    try:
                        if len(fields) != 14:
                            raise ValueError("CRD v2 record 11 requires exactly 14 fields including SNR")
                        event = _to_int(fields[4])
                        if event != 2:
                            raise ValueError(
                                f"unsupported epoch-event {fields[4]}; only ground transmit event 2 is supported"
                            )
                        if fields[3] not in current.configurations:
                            raise ValueError(f"unknown C0 configuration id {fields[3]!r}")
                        sod, tof, rms = (_to_float(fields[i]) for i in (1, 2, 7))
                        if sod is None or tof is None or tof <= 0 or rms is None or rms <= 0:
                            raise ValueError(
                                "record 11 requires finite seconds-of-day, positive time-of-flight and bin RMS"
                            )
                        _session_epoch(current, sod)
                        current.normal_points.append(
                            _CrdNormalPoint(
                                sod,
                                tof,
                                rms,
                                _to_float(fields[13]),
                                line_no,
                                line,
                                fields[3],
                                event,
                            )
                        )
                    except ValueError as exc:
                        current.import_issues.append(
                            {
                                "line": line_no,
                                "reason": str(exc),
                                "content": line,
                                "epochEvent": fields[4] if len(fields) > 4 else None,
                            }
                        )
                elif tag == "20":
                    if current is None:
                        raise ValueError("record 20 requires H4")
                    try:
                        if len(fields) < 6:
                            raise ValueError("incomplete meteorological record")
                        sod, pressure, temperature, humidity = (_to_float(fields[i]) for i in (1, 2, 3, 4))
                        if sod is None or pressure is None or temperature is None or humidity is None:
                            raise ValueError("meteorological fields must be finite numbers")
                        if pressure <= 0 or temperature <= 0 or not 0 <= humidity <= 100:
                            raise ValueError("meteorological fields are outside physical bounds")
                        _session_epoch(current, sod)
                        current.meteo.append(_CrdMeteo(sod, pressure, temperature, humidity))
                    except ValueError as exc:
                        current.meteorology_issues.append({"line": line_no, "reason": str(exc), "content": line})
                elif tag == "H8":
                    if current is None:
                        raise ValueError("H8 requires an active session")
                    current = None
                elif tag == "H9":
                    if current is not None:
                        raise ValueError("H9 requires H8")
                    file_ended = True
                    station = target = None
                elif tag in {"10", "12", "21", "30", "40", "41", "42", "50", "60"} or tag in {
                    f"C{i}" for i in range(1, 8)
                }:
                    if current is None:
                        raise ValueError(f"{tag} requires H4")
                else:
                    raise ValueError(f"unsupported CRD record {tag!r}")
            except (ValueError, IndexError, TypeError) as exc:
                raise ValueError(f"{path}: line {line_no}: {exc}") from exc
    if current is not None:
        raise ValueError(f"{path}: missing H8 session terminator")
    return [session for session in sessions if session.input_record_count]


def _crd_station_identity(name: str, pad_id: str) -> tuple[str, str]:
    try:
        station, pad_ids = _CRD_STATIONS[name]
    except KeyError:
        raise ValueError(f"Unknown CRD H2 station name {name!r}.") from None
    if pad_id not in pad_ids:
        raise ValueError(f"CRD H2 station name {name!r} contradicts pad ID {pad_id!r}.")
    return station, station_ilrs_code(station)


def _station_identity(session: _CrdSession) -> tuple[str, str]:
    return _crd_station_identity(session.station_name, session.station_pad_id)


def _reflector_identity(session: _CrdSession) -> tuple[str, int]:
    token = session.target_name
    if token in CRD_REFLECTOR_IDENTITY_BY_NAME:
        return CRD_REFLECTOR_IDENTITY_BY_NAME[token]
    raise ValueError(
        f"Cannot map CRD target {session.target_name!r} to a canonical identity; extend CRD_REFLECTOR_IDENTITY_BY_NAME."
    )


def _nearest_meteo(session: _CrdSession, epoch: Epoch, max_age_s: float) -> Optional[_CrdMeteo]:
    if not session.meteo:
        return None
    ages = [(abs(_session_epoch(session, rec.seconds_of_day).seconds_until(epoch)), rec) for rec in session.meteo]
    age, record = min(ages, key=lambda item: item[0])
    return record if age <= max_age_s else None


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
    *,
    uncertainty_policy: str | None = None,
    uncertainty_two_way_s: float | None = None,
    meteorology_max_age_s: float = 1800.0,
) -> List[_CrdObservation]:
    if uncertainty_policy not in {"binRmsAsSigma", "fixed"}:
        raise ValueError(
            "CRD requires explicit uncertainty_policy: binRmsAsSigma or fixed; bin RMS is not inherently NP sigma."
        )
    if uncertainty_policy == "fixed" and (
        uncertainty_two_way_s is None or not math.isfinite(uncertainty_two_way_s) or uncertainty_two_way_s <= 0
    ):
        raise ValueError("fixed CRD uncertainty requires positive finite uncertainty_two_way_s")
    if not math.isfinite(meteorology_max_age_s) or meteorology_max_age_s < 0:
        raise ValueError("meteorology_max_age_s must be nonnegative and finite")
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
        for record_index, np_rec in enumerate(session.normal_points, start=1):
            epoch = _session_epoch(session, np_rec.seconds_of_day)
            label = f"CRD NP at {epoch.isot(scale=TimeScale.UTC)} (station {station_code}, reflector {reflector_id})"
            try:
                meteo = _nearest_meteo(session, epoch, meteorology_max_age_s)
                if meteo is None:
                    raise ValueError(f"{label}: no valid CRD '20' record within the meteorology age limit.")
                if meteo.pressure_hpa is None or meteo.temperature_k is None or meteo.humidity_percent is None:
                    raise ValueError(f"{label}: the nearest CRD '20' record is incomplete.")
                if np_rec.configuration_id not in session.configurations:
                    raise ValueError(f"{label}: the CRD session 'C0' record carries no usable laser wavelength.")
            except ValueError as exc:
                issues.append({"line": np_rec.source_line_no, "reason": str(exc), "content": np_rec.source_line})
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
                    uncertainty_two_way_s=(
                        float(np_rec.bin_rms_ps) * 1.0e-12
                        if uncertainty_policy == "binRmsAsSigma"
                        else float(uncertainty_two_way_s)
                    ),
                    pressure_hpa=float(meteo.pressure_hpa),
                    temperature_k=float(meteo.temperature_k),
                    humidity_percent=float(meteo.humidity_percent),
                    wavelength_nm=session.configurations[np_rec.configuration_id],
                    signal_noise_ratio=np_rec.snr,
                    source_format="crd-v2",
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
    **options,
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
        for index, observation in enumerate(_crd_observations(sessions, **options))
    ]


def parse_crd_file(path, **options):
    """Parse a CRD v2 file directly into a canonical NptDataset."""
    from lunarops.classes.observation.normal_points import NptDataset

    source = Path(path)
    sessions = parse_crd_sessions(source)
    if not sessions:
        raise ValueError(f"No CRD normal-point sessions found in {source}")
    import_issues = [issue for session in sessions for issue in session.import_issues]
    observations = _crd_observations(sessions, import_issues, **options)
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
        raise ValueError(f"No valid CRD normal points found in {source}: {import_issues}")
    input_count = sum(session.input_record_count for session in sessions)
    return NptDataset(
        records=records,
        name=source.stem,
        n_input_records=input_count,
        n_invalid_records=len(import_issues),
        import_issues=import_issues,
        import_metadata={
            "format": "CRD",
            "epochEvent": 2,
            "uncertaintyPolicy": options.get("uncertainty_policy"),
            "uncertaintyTwoWayS": options.get("uncertainty_two_way_s"),
            "meteorologyPolicy": "absolute-UTC nearest neighbour",
            "meteorologyMaxAgeS": options.get("meteorology_max_age_s", 1800.0),
            "meteorologyIssues": [issue for session in sessions for issue in session.meteorology_issues],
            "sourceQuality": [
                {
                    "line": rec.source_line_no,
                    "binRmsPs": rec.bin_rms_ps,
                    "epochEvent": rec.epoch_event,
                    "configurationId": rec.configuration_id,
                }
                for session in sessions
                for rec in session.normal_points
            ],
        },
    )


__all__ = [
    "CRD_REFLECTOR_IDENTITY_BY_NAME",
    "CRD_SUFFIXES",
    "crd_sessions_to_npt_records",
    "looks_like_crd_file",
    "parse_crd_file",
    "parse_crd_sessions",
]
