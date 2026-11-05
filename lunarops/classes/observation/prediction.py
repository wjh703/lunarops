"""Time-grid LLR pointing and visibility prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import erfa
import numpy as np

from lunarops.classes.frames import ReferenceFrameSystem
from lunarops.classes.time import (
    Epoch,
    TimeScale,
    format_time_with_utc_offset,
    parse_time_with_utc_offset,
)

from .catalogs import ReflectorRecord, StationRecord
from .light_time import LightTimeRequest, LightTimeSolver

_J2000_JD = 2_451_545.0
_FULL_CIRCLE_DEG = 360.0


@dataclass(frozen=True, slots=True)
class PredictionCriteria:
    """Thresholds for geometric visibility of a reflector from a station.

    Attributes
    ----------
    minimum_elevation_deg : float
        Minimum vacuum elevation of the Moon reflector at the station, in degrees, in [0, 90].
    minimum_reflector_elevation_deg : float
        Minimum elevation of the station above the reflector's local horizon, in degrees, in [-90, 90].
    maximum_sun_elevation_deg : float
        Maximum Sun elevation at the station, in degrees, in [-90, 90].
    allowed_elongation_ranges_deg : tuple[tuple[float, float], ...]
        Allowed ranges ``(start, end)`` of the mean elongation D of the Moon from the Sun, in degrees, each
        endpoint in [0, 360]. A range with ``start > end`` wraps through 0.

    Raises
    ------
    ValueError
        If a threshold is not finite or out of range, or no elongation range is given.
    """

    minimum_elevation_deg: float = 20.0
    minimum_reflector_elevation_deg: float = 0.0
    maximum_sun_elevation_deg: float = -6.0
    allowed_elongation_ranges_deg: tuple[tuple[float, float], ...] = ((0.0, 360.0),)

    def __post_init__(self) -> None:
        """Validate the thresholds and convert them to floats."""
        minimum = float(self.minimum_elevation_deg)
        minimum_reflector = float(self.minimum_reflector_elevation_deg)
        sun_maximum = float(self.maximum_sun_elevation_deg)
        if not np.isfinite(minimum) or not 0.0 <= minimum <= 90.0:
            raise ValueError("minimum_elevation_deg must be finite and in [0, 90].")
        if not np.isfinite(minimum_reflector) or not -90.0 <= minimum_reflector <= 90.0:
            raise ValueError("minimum_reflector_elevation_deg must be finite and in [-90, 90].")
        if not np.isfinite(sun_maximum) or not -90.0 <= sun_maximum <= 90.0:
            raise ValueError("maximum_sun_elevation_deg must be finite and in [-90, 90].")
        ranges: list[tuple[float, float]] = []
        for item in self.allowed_elongation_ranges_deg:
            if len(item) != 2:
                raise ValueError("Each elongation range must contain [startDeg, endDeg].")
            start, end = float(item[0]), float(item[1])
            if not np.isfinite(start) or not np.isfinite(end) or not 0.0 <= start <= 360.0 or not 0.0 <= end <= 360.0:
                raise ValueError("Elongation range endpoints must be finite and in [0, 360].")
            ranges.append((start, end))
        if not ranges:
            raise ValueError("At least one elongation range is required.")
        object.__setattr__(self, "minimum_elevation_deg", minimum)
        object.__setattr__(self, "minimum_reflector_elevation_deg", minimum_reflector)
        object.__setattr__(self, "maximum_sun_elevation_deg", sun_maximum)
        object.__setattr__(self, "allowed_elongation_ranges_deg", tuple(ranges))

    def elongation_allowed(self, elongation_deg: float) -> bool:
        """Return whether an elongation in degrees (taken modulo 360) lies in an allowed range, endpoints included."""
        value = float(elongation_deg) % _FULL_CIRCLE_DEG
        for start, end in self.allowed_elongation_ranges_deg:
            if start == 0.0 and end == _FULL_CIRCLE_DEG:
                return True
            if start <= end:
                if start <= value <= end:
                    return True
            elif value >= start or value <= end:
                return True
        return False


@dataclass(frozen=True, slots=True)
class PredictionMeteorology:
    """Representative station weather and laser wavelength for the troposphere model in predictions.

    Attributes
    ----------
    pressure_hpa : float
        Surface pressure, in hPa; positive.
    temperature_k : float
        Surface temperature, in kelvin; positive.
    relative_humidity_percent : float
        Relative humidity, in percent, in [0, 100].
    wavelength_nm : float
        Laser wavelength, in nanometres; positive.

    Raises
    ------
    ValueError
        If a value is not finite or out of range.
    """

    pressure_hpa: float = 900.0
    temperature_k: float = 285.0
    relative_humidity_percent: float = 25.0
    wavelength_nm: float = 532.0

    def __post_init__(self) -> None:
        """Validate the values and convert them to floats."""
        for name in ("pressure_hpa", "temperature_k", "relative_humidity_percent", "wavelength_nm"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.pressure_hpa <= 0.0 or self.temperature_k <= 0.0 or self.wavelength_nm <= 0.0:
            raise ValueError("pressure_hpa, temperature_k, and wavelength_nm must be positive.")
        if not 0.0 <= self.relative_humidity_percent <= 100.0:
            raise ValueError("relative_humidity_percent must be in [0, 100].")


def _mean_elongation_deg(epoch_utc: Epoch, frames: ReferenceFrameSystem) -> float:
    """Mean elongation D of the Moon from the Sun (IERS Conventions 2003) at the TT epoch, in [0, 360) degrees."""
    tt = frames.time_scale_converter.utc2tt(epoch_utc)
    centuries = (tt.jd - _J2000_JD) / 36_525.0
    return float(np.rad2deg(erfa.fad03(float(centuries))) % _FULL_CIRCLE_DEG)


class LlrObservationPredictor:
    """Pointing and visibility predictor for one station and reflector at arbitrary UTC epochs.

    Parameters
    ----------
    frames : ReferenceFrameSystem
        Frame and time-scale transformations.
    light_time_solver : LightTimeSolver
        Light-time solver; only the uplink is solved.
    station : StationRecord
        Station record.
    reflector : ReflectorRecord
        Reflector record (Moon PA coordinates).
    station_key : str
        Station catalog key written to the rows.
    reflector_key : str
        Reflector catalog key written to the rows.
    criteria : PredictionCriteria
        Visibility thresholds.
    meteorology : PredictionMeteorology
        Station weather for the troposphere model.
    utc_offset_hours : float, default 0.0
        Offset of local time from UTC, in hours, for the ``local_t1`` column.
    """

    def __init__(
        self,
        frames: ReferenceFrameSystem,
        light_time_solver: LightTimeSolver,
        station: StationRecord,
        reflector: ReflectorRecord,
        *,
        station_key: str,
        reflector_key: str,
        criteria: PredictionCriteria,
        meteorology: PredictionMeteorology,
        utc_offset_hours: float = 0.0,
    ) -> None:
        self.frames = frames
        self.light_time_solver = light_time_solver
        self.station = station
        self.reflector = reflector
        self.station_key = str(station_key)
        self.reflector_key = str(reflector_key)
        self.criteria = criteria
        self.meteorology = meteorology
        self.utc_offset_hours = float(utc_offset_hours)

    def _request(self, epoch_utc: Epoch) -> LightTimeRequest:
        return LightTimeRequest.from_station(
            self.station,
            self.reflector.moon_fixed_xyz_m,
            epoch_utc,
            station_key=self.station_key,
            pressure_hpa=self.meteorology.pressure_hpa,
            temperature_k=self.meteorology.temperature_k,
            relative_humidity_percent=self.meteorology.relative_humidity_percent,
            wavelength_um=self.meteorology.wavelength_nm / 1000.0,
        )

    def evaluate(self, epoch_utc: Epoch) -> dict[str, object]:
        """Return the prediction row for one transmit epoch.

        The azimuth and elevation are those of the uplink BCRS vector (station at transmit to reflector at
        bounce) in the station's local frame, without refraction. The Sun elevation uses the BCRS Sun position at
        the transmit epoch (TDB). The row is ``observable`` when all four criteria hold.

        Parameters
        ----------
        epoch_utc : Epoch
            Transmit epoch ``t1``, UTC.

        Returns
        -------
        dict[str, object]
            Row with the UTC and local transmit times (millisecond precision), station and reflector keys, the
            station ITRF position at transmit and the reflector ITRF position at bounce (metres), the uplink
            geometric range (metres), azimuth and elevation (degrees), the reflector-horizon elevation of the
            station, the Sun elevation and the mean elongation (degrees), and the visibility flags.

        Raises
        ------
        ValueError
            If ``epoch_utc`` is not in UTC.
        RuntimeError
            If the light-time iteration does not converge.
        """
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        request = self._request(epoch_utc)
        solution = self.light_time_solver.solve(request, two_way=False)
        if not solution.light_time_converged:
            raise RuntimeError(f"Light-time iteration did not converge at {epoch_utc.isot()}.")

        station_itrf_m = solution.station_itrf_transmit_m
        azimuth_deg, elevation_deg = solution.topocentric_pointing(
            self.frames,
            epoch_utc,
            solution.uplink_vector_bcrs_m,
        )

        sun_bcrs_m = self.frames.relativistic_transform.body_state("SUN", solution.transmit_epoch_tdb).position_m
        sun_vector_bcrs_m = sun_bcrs_m - solution.station_bcrs_transmit_m
        _, sun_elevation_deg = solution.topocentric_pointing(self.frames, epoch_utc, sun_vector_bcrs_m)
        reflector_elevation_deg = solution.reflector_elevation_deg(self.frames, self.reflector.moon_fixed_xyz_m)
        elongation_deg = _mean_elongation_deg(epoch_utc, self.frames)
        elevation_ok = elevation_deg >= self.criteria.minimum_elevation_deg
        reflector_elevation_ok = reflector_elevation_deg >= self.criteria.minimum_reflector_elevation_deg
        sun_ok = sun_elevation_deg <= self.criteria.maximum_sun_elevation_deg
        elongation_ok = self.criteria.elongation_allowed(elongation_deg)
        bounce_utc = self.light_time_solver.event_epoch_utc(request, solution.bounce_epoch_tdb)
        reflector_itrf_m = solution.reflector_itrf_m(self.frames, bounce_utc)
        return {
            "utc_t1": format_time_with_utc_offset(
                epoch_utc,
                utc_offset_hours=0.0,
                precision=3,
            ),
            "local_t1": format_time_with_utc_offset(
                epoch_utc,
                utc_offset_hours=self.utc_offset_hours,
                precision=3,
            ),
            "station": self.station_key,
            "reflector": self.reflector_key,
            "station_itrf_x_m": float(station_itrf_m[0]),
            "station_itrf_y_m": float(station_itrf_m[1]),
            "station_itrf_z_m": float(station_itrf_m[2]),
            "reflector_itrf_x_m": float(reflector_itrf_m[0]),
            "reflector_itrf_y_m": float(reflector_itrf_m[1]),
            "reflector_itrf_z_m": float(reflector_itrf_m[2]),
            "range_up_geometric_m": float(solution.uplink.geometric_range_m),
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "observable": bool(elevation_ok and reflector_elevation_ok and sun_ok and elongation_ok),
            "station_elevation_ok": bool(elevation_ok),
            "reflector_elevation_ok": bool(reflector_elevation_ok),
            "sun_elevation_ok": bool(sun_ok),
            "elongation_ok": bool(elongation_ok),
            "reflector_elevation_deg": reflector_elevation_deg,
            "sun_elevation_deg": sun_elevation_deg,
            "elongation_deg": elongation_deg,
        }


def build_visibility_windows(
    rows: Sequence[Mapping[str, object]],
    *,
    step_seconds: float,
) -> list[dict[str, object]]:
    """Merge consecutive observable prediction rows into visibility windows.

    Rows are consecutive when they have the same station and reflector and their ``utc_t1`` epochs are one grid
    step apart. A window runs from its first to its last observable sample, so its duration is a multiple of
    the step and excludes the time up to the next unobservable sample.

    Parameters
    ----------
    rows : Sequence[Mapping[str, object]]
        Prediction rows in time order, as returned by ``LlrObservationPredictor.evaluate``.
    step_seconds : float
        Grid step, in seconds; finite and positive.

    Returns
    -------
    list[dict[str, object]]
        Windows with station, reflector, UTC and local start and end times, ``sample_count`` and
        ``duration_s`` (seconds, rounded to 1 ns).

    Raises
    ------
    ValueError
        If ``step_seconds`` is not finite and positive or a row has no ``utc_t1``.
    """
    step = float(step_seconds)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_seconds must be finite and positive.")
    windows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    previous_epoch: Epoch | None = None
    for row in rows:
        epoch = parse_time_with_utc_offset(row["utc_t1"], name="prediction utc_t1")
        if epoch is None:
            raise ValueError("Prediction rows must contain a non-empty utc_t1.")
        is_observable = bool(row["observable"])
        contiguous = (
            current is not None
            and previous_epoch is not None
            and current["station"] == row["station"]
            and current["reflector"] == row["reflector"]
            and abs(previous_epoch.seconds_until(epoch) - step) < max(1.0e-6, step * 1.0e-9)
        )
        if is_observable and contiguous:
            assert current is not None
            current["end_utc"] = str(row["utc_t1"])
            current["end_local"] = str(row["local_t1"])
            current["sample_count"] = int(cast(Any, current["sample_count"])) + 1
        elif is_observable:
            if current is not None:
                windows.append(current)
            current = {
                "station": row["station"],
                "reflector": row["reflector"],
                "start_utc": str(row["utc_t1"]),
                "end_utc": str(row["utc_t1"]),
                "start_local": str(row["local_t1"]),
                "end_local": str(row["local_t1"]),
                "sample_count": 1,
            }
        elif current is not None:
            windows.append(current)
            current = None
        previous_epoch = epoch
    if current is not None:
        windows.append(current)
    for window in windows:
        start = parse_time_with_utc_offset(window["start_utc"], name="prediction window start_utc")
        end = parse_time_with_utc_offset(window["end_utc"], name="prediction window end_utc")
        if start is None or end is None:
            raise ValueError("Prediction windows must contain non-empty UTC timestamps.")
        # UTC leap-second aware arithmetic can leave a few ulps of floating
        # noise for an otherwise integral grid interval.
        window["duration_s"] = float(round(start.seconds_until(end), 9))
    return windows


__all__ = [
    "LlrObservationPredictor",
    "PredictionCriteria",
    "PredictionMeteorology",
    "build_visibility_windows",
]
