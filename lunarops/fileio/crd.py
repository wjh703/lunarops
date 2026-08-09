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
  * The canonical launch epoch is the ground *transmit* time.  CRD record 11 field
    "epoch event" tells which event the timestamp refers to; event 2
    (transmit) converts directly, event 1 (bounce) is shifted by half the
    time of flight (an approximation good to ~ (tau_up - tau_down)/2; refine
    with a light-time solution if you need it exact).
  * Known station names and aliases are resolved by the central station
    identity registry. When no match is found, the CRD pad id is zero-padded
    to 5 characters.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.base.epoch import Epoch, TimeScale
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
    epoch_event: int
    np_window_s: Optional[float]
    number_of_returns: Optional[int]
    bin_rms_ps: Optional[float]
    snr: Optional[float]


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


def _circular_distance(a: float, b: float) -> float:
    d = abs(a - b) % SECONDS_PER_DAY
    return min(d, SECONDS_PER_DAY - d)


def parse_crd_sessions(path) -> List[_CrdSession]:
    """Parse the CRD records relevant to MINI conversion, session by session."""
    sessions: List[_CrdSession] = []
    current: Optional[_CrdSession] = None
    crd_version = 1

    with _open_text(path) as fh:
        for raw in fh:
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
                # v1: 11 sod tof sysconfig epoch_event np_window n_ranges
                #     bin_rms skew kurtosis peak-mean return_rate ch
                # v2 adds snr at the end.
                seconds_of_day = _to_float(fields[1])
                tof = _to_float(fields[2])
                if seconds_of_day is None or tof is None:
                    continue
                epoch_event = _to_int(fields[4]) if len(fields) > 4 else 2
                np_window = _to_float(fields[5]) if len(fields) > 5 else None
                n_ranges = _to_int(fields[6]) if len(fields) > 6 else None
                bin_rms = _to_float(fields[7]) if len(fields) > 7 else None
                snr = _to_float(fields[13]) if (crd_version >= 2 and len(fields) > 13) else None
                current.normal_points.append(
                    _CrdNormalPoint(
                        seconds_of_day=seconds_of_day,
                        time_of_flight_s=tof,
                        epoch_event=epoch_event if epoch_event is not None else 2,
                        np_window_s=np_window,
                        number_of_returns=n_ranges,
                        bin_rms_ps=bin_rms,
                        snr=snr,
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

    return [s for s in sessions if s.normal_points]


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
    number_of_returns: Optional[int]
    signal_noise_ratio: Optional[float]
    duration_s: Optional[float]
    source_format: str
    source_record: str


def _crd_observations(sessions: Sequence[_CrdSession]) -> List[_CrdObservation]:
    observations: List[_CrdObservation] = []
    for session_index, session in enumerate(sessions, start=1):
        if session.start_epoch is None:
            raise ValueError("CRD session is missing the H4 start epoch; cannot anchor seconds-of-day.")

        station_name, station_code = _station_identity(session)
        reflector_name, reflector_id = _reflector_identity(session)
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

            if np_rec.epoch_event == 1:
                epoch = epoch.shifted(-0.5 * np_rec.time_of_flight_s)
            elif np_rec.epoch_event not in (1, 2):
                raise ValueError(
                    f"Unsupported CRD epoch event {np_rec.epoch_event}; only 1 (bounce) and 2 (transmit) are handled."
                )

            label = f"CRD NP at {epoch.isot(scale=TimeScale.UTC)} (station {station_code}, reflector {reflector_id})"
            meteo = _nearest_meteo(session.meteo, seconds)
            if meteo is None:
                raise ValueError(f"{label}: the CRD session has no '20' meteorological record.")
            if meteo.pressure_hpa is None or meteo.temperature_k is None or meteo.humidity_percent is None:
                raise ValueError(
                    f"{label}: the nearest CRD '20' record is incomplete "
                    f"(pressure={meteo.pressure_hpa!r}, "
                    f"temperature={meteo.temperature_k!r}, "
                    f"humidity={meteo.humidity_percent!r})."
                )
            if session.wavelength_nm is None or session.wavelength_nm <= 0.0:
                raise ValueError(f"{label}: the CRD session 'C0' record carries no usable laser wavelength.")
            if np_rec.bin_rms_ps is None or np_rec.bin_rms_ps <= 0.0:
                raise ValueError(f"{label}: the CRD '11' record carries no usable bin RMS (uncertainty).")

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
                    number_of_returns=np_rec.number_of_returns,
                    signal_noise_ratio=np_rec.snr,
                    duration_s=np_rec.np_window_s,
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
    records = crd_sessions_to_npt_records(sessions)
    return NptDataset(
        records=records,
        name=source.stem,
        n_input_records=sum(len(session.normal_points) for session in sessions),
        n_invalid_records=0,
    )


__all__ = [
    "CRD_REFLECTOR_IDENTITY_BY_NAME",
    "CRD_SUFFIXES",
    "crd_sessions_to_npt_records",
    "looks_like_crd_file",
    "parse_crd_file",
    "parse_crd_sessions",
]
