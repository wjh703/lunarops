"""Time-grid LLR pointing and visibility prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import erfa
import numpy as np

from lunarops.classes.displacement.terrestrial_geometry import itrf2enu, itrf2geodetic
from lunarops.classes.frames import ReferenceFrameSystem
from lunarops.classes.time import (
    Epoch,
    TimeScale,
    format_time_with_utc_offset,
    parse_time_with_utc_offset,
    validate_utc_offset_hours,
)

from .catalogs import ReflectorRecord, StationRecord
from .light_time import LightTimeRequest, LightTimeSolver, TroposphereEnvironment

_J2000_JD = 2_451_545.0
_FULL_CIRCLE_DEG = 360.0


@dataclass(frozen=True, slots=True)
class PredictionCriteria:
    """Thresholds for the first-generation geometric visibility decision."""

    minimum_elevation_deg: float = 20.0
    minimum_reflector_elevation_deg: float = 0.0
    maximum_sun_elevation_deg: float = -6.0
    allowed_elongation_ranges_deg: tuple[tuple[float, float], ...] = ((0.0, 360.0),)

    def __post_init__(self) -> None:
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
    """Representative station weather used when an optical troposphere model is enabled."""

    pressure_hpa: float = 900.0
    temperature_k: float = 285.0
    relative_humidity_percent: float = 25.0
    wavelength_nm: float = 532.0

    def __post_init__(self) -> None:
        for name in ("pressure_hpa", "temperature_k", "relative_humidity_percent", "wavelength_nm"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.pressure_hpa <= 0.0 or self.temperature_k <= 0.0 or self.wavelength_nm <= 0.0:
            raise ValueError("pressure_hpa, temperature_k, and wavelength_nm must be positive.")
        if not 0.0 <= self.relative_humidity_percent <= 100.0:
            raise ValueError("relative_humidity_percent must be in [0, 100].")


def _enu_angles(enu_m: np.ndarray) -> tuple[float, float]:
    east, north, up = (float(value) for value in enu_m)
    if float(np.linalg.norm(enu_m)) <= 0.0:
        raise RuntimeError("Cannot compute pointing for a zero-length ENU vector.")
    azimuth_deg = float(np.rad2deg(np.arctan2(east, north)) % 360.0)
    elevation_deg = float(np.rad2deg(np.arctan2(up, np.hypot(east, north))))
    return azimuth_deg, elevation_deg


def _mean_elongation_deg(epoch_utc: Epoch, frames: ReferenceFrameSystem) -> float:
    tt = frames.time_scale_converter.utc2tt(epoch_utc)
    centuries = (tt.jd - _J2000_JD) / 36_525.0
    return float(np.rad2deg(erfa.fad03(float(centuries))) % _FULL_CIRCLE_DEG)


class LlrObservationPredictor:
    """Evaluate one reflector from a station over arbitrary UTC epochs."""

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
        if not isinstance(frames, ReferenceFrameSystem):
            raise TypeError("frames must be a ReferenceFrameSystem.")
        if not isinstance(light_time_solver, LightTimeSolver):
            raise TypeError("light_time_solver must be a LightTimeSolver.")
        if not isinstance(station, StationRecord) or not isinstance(reflector, ReflectorRecord):
            raise TypeError("station and reflector must be catalog records.")
        self.frames = frames
        self.light_time_solver = light_time_solver
        self.station = station
        self.reflector = reflector
        self.station_key = str(station_key)
        self.reflector_key = str(reflector_key)
        self.criteria = criteria
        self.meteorology = meteorology
        self.utc_offset_hours = validate_utc_offset_hours(utc_offset_hours)

    def _request(self, epoch_utc: Epoch) -> LightTimeRequest:
        station_position = self.station.itrf_xyz_at(epoch_utc)
        geodetic = itrf2geodetic(station_position)
        return LightTimeRequest(
            reflector_reference_pa_m=np.asarray(self.reflector.moon_fixed_xyz_m, dtype=float),
            transmit_epoch_utc=epoch_utc,
            troposphere_environment=TroposphereEnvironment(
                pressure_hpa=self.meteorology.pressure_hpa,
                temperature_k=self.meteorology.temperature_k,
                relative_humidity_percent=self.meteorology.relative_humidity_percent,
                latitude_rad=geodetic.latitude_rad,
                ellipsoidal_height_m=geodetic.ellipsoidal_height_m,
                wavelength_um=self.meteorology.wavelength_nm / 1000.0,
            ),
            station_reference_itrf_at_utc=self.station.itrf_xyz_at,
            station_key=self.station_key,
        )

    def _topocentric_pointing(
        self,
        vector_bcrs_m: np.ndarray,
        epoch_utc: Epoch,
        epoch_tdb: Epoch,
        station_itrf_m: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        vector_gcrs_m = self.frames.bcrs_vector2gcrs(vector_bcrs_m, epoch_tdb)
        vector_itrf_m = self.frames.gcrs2itrf(vector_gcrs_m, epoch_utc)
        geodetic = itrf2geodetic(station_itrf_m)
        enu_m = itrf2enu(
            vector_itrf_m,
            latitude_rad=geodetic.latitude_rad,
            longitude_rad=geodetic.longitude_rad,
        )
        azimuth_deg, elevation_deg = _enu_angles(enu_m)
        enu_unit = enu_m / np.linalg.norm(enu_m)
        return azimuth_deg, elevation_deg, enu_unit

    def evaluate(self, epoch_utc: Epoch) -> dict[str, object]:
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        request = self._request(epoch_utc)
        solution = self.light_time_solver.solve(request)
        if not solution.light_time_converged:
            raise RuntimeError(f"Light-time iteration did not converge at {epoch_utc.isot()}.")

        station_itrf_m = self.light_time_solver.station_position_itrf_m(request, epoch_utc)
        up_vector_bcrs_m = solution.reflector_bcrs_bounce_m - solution.station_bcrs_transmit_m
        azimuth_deg, elevation_deg, _ = self._topocentric_pointing(
            up_vector_bcrs_m,
            epoch_utc,
            solution.transmit_epoch_tdb,
            station_itrf_m,
        )

        sun_bcrs_m = self.frames.ephemeris.body_position_bcrs("SUN", solution.transmit_epoch_tdb)
        sun_vector_bcrs_m = sun_bcrs_m - solution.station_bcrs_transmit_m
        _, sun_elevation_deg, _ = self._topocentric_pointing(
            sun_vector_bcrs_m,
            epoch_utc,
            solution.transmit_epoch_tdb,
            station_itrf_m,
        )
        reflector_pa_m = (
            np.asarray(self.reflector.moon_fixed_xyz_m, dtype=float)
            + solution.reflector_displacement_bounce_pa_m
        )
        down_vector_bcrs_m = solution.station_bcrs_transmit_m - solution.reflector_bcrs_bounce_m
        down_vector_lcrs_m = self.frames.bcrs_vector2lcrs(
            down_vector_bcrs_m,
            solution.bounce_epoch_tdb,
        )
        down_vector_pa_m = self.frames.lcrs2pa(down_vector_lcrs_m, solution.bounce_epoch_tdb)
        reflector_normal_pa = reflector_pa_m / np.linalg.norm(reflector_pa_m)
        reflector_sine_elevation = float(
            np.dot(down_vector_pa_m / np.linalg.norm(down_vector_pa_m), reflector_normal_pa)
        )
        reflector_elevation_deg = float(
            np.rad2deg(np.arcsin(np.clip(reflector_sine_elevation, -1.0, 1.0)))
        )
        elongation_deg = _mean_elongation_deg(epoch_utc, self.frames)
        elevation_ok = elevation_deg >= self.criteria.minimum_elevation_deg
        reflector_elevation_ok = reflector_elevation_deg >= self.criteria.minimum_reflector_elevation_deg
        sun_ok = sun_elevation_deg <= self.criteria.maximum_sun_elevation_deg
        elongation_ok = self.criteria.elongation_allowed(elongation_deg)
        bounce_utc = self.light_time_solver.event_epoch_utc(request, solution.bounce_epoch_tdb)
        reflector_itrf_m = self.frames.gcrs2itrf(
            self.frames.bcrs2gcrs(solution.reflector_bcrs_bounce_m, solution.bounce_epoch_tdb),
            bounce_utc,
        )
        return {
            "utc_t1": format_time_with_utc_offset(
                epoch_utc,
                utc_offset_hours=0.0,
                precision=9,
            ),
            "local_t1": format_time_with_utc_offset(
                epoch_utc,
                utc_offset_hours=self.utc_offset_hours,
                precision=9,
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
        }


def build_visibility_windows(
    rows: Sequence[Mapping[str, object]],
    *,
    step_seconds: float,
) -> list[dict[str, object]]:
    """Merge consecutive true samples into coarse grid-defined windows."""
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
