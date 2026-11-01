"""Two-way lunar laser ranging light-time solution using unified epochs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, overload

import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.constants import C
from lunarops.classes.delays import (
    Iers2010MendesPavlisTroposphere,
    Iers2010ShapiroDelay,
    TroposphereInput,
)
from lunarops.classes.displacement import (
    LunarSolidTide,
    ReflectorDisplacementInput,
    StationDisplacement,
    StationDisplacementInput,
)
from lunarops.classes.displacement.terrestrial_geometry import itrf2enu, itrf2geodetic, local_up_unit_itrf
from lunarops.classes.frames import ReferenceFrameSystem
from lunarops.classes.time import Epoch, TdbTopocentricArguments, TimeScale

from .catalogs import StationRecord

_MAX_LIGHT_TIME_ITERATIONS = 12
_ROUND_TRIP_TIME_TOLERANCE_S = 1.0e-12
_INITIAL_UPLINK_LIGHT_TIME_S = 1.2
_PRE_1972_UTC_RATE_START = Epoch.from_isot("1968-02-01T00:00:00", scale=TimeScale.UTC)
_PRE_1972_UTC_RATE_END = Epoch.from_isot("1972-01-01T00:00:00", scale=TimeScale.UTC)


@dataclass(frozen=True, slots=True, eq=False)
class LightTimeRequest:
    """One station/reflector light-time problem.

    ``troposphere_input`` carries the station meteorology; each leg
    evaluates it with its own elevation.
    """

    reflector_reference_pa_m: np.ndarray
    transmit_epoch_utc: Epoch
    troposphere_input: TroposphereInput
    station: StationRecord
    station_key: str

    @classmethod
    def from_station(
        cls,
        station,
        reflector_reference_pa_m,
        epoch_utc,
        *,
        station_key,
        pressure_hpa,
        temperature_k,
        relative_humidity_percent,
        wavelength_um,
    ):
        geodetic = itrf2geodetic(station.itrf_xyz_at(epoch_utc))
        return cls(
            reflector_reference_pa_m=reflector_reference_pa_m,
            transmit_epoch_utc=epoch_utc,
            station=station,
            station_key=station_key,
            troposphere_input=TroposphereInput(
                pressure_hpa=pressure_hpa,
                temperature_k=temperature_k,
                relative_humidity_percent=relative_humidity_percent,
                latitude_rad=geodetic.latitude_rad,
                height_m=geodetic.ellipsoidal_height_m,
                wavelength_um=wavelength_um,
            ),
        )

class LightTimeLeg(NamedTuple):
    geometric_range_m: float
    gravitational_path_delay_m: float
    tropospheric_path_delay_m: float
    vacuum_elevation_rad: float
    troposphere_elevation_used_rad: float | None = None
    troposphere_elevation_clamped: bool = False

    @property
    def path_length_m(self) -> float:
        return self.geometric_range_m + self.gravitational_path_delay_m + self.tropospheric_path_delay_m

    @property
    def travel_time_s(self) -> float:
        return self.path_length_m / C


@dataclass(frozen=True, slots=True, eq=False)
class UplinkSolution:
    """Common physical evaluation of station, reflector, and uplink events."""

    transmit_epoch_tdb: Epoch
    bounce_epoch_tdb: Epoch
    uplink: LightTimeLeg
    station_itrf_transmit_m: np.ndarray
    station_displacement_transmit_itrf_m: np.ndarray
    reflector_displacement_bounce_pa_m: np.ndarray
    station_bcrs_transmit_m: np.ndarray
    reflector_bcrs_bounce_m: np.ndarray
    iteration_count: int
    light_time_converged: bool

    @property
    def uplink_vector_bcrs_m(self) -> np.ndarray:
        return self.reflector_bcrs_bounce_m - self.station_bcrs_transmit_m

    @property
    def elevation_up_deg(self) -> float:
        return float(np.rad2deg(self.uplink.vacuum_elevation_rad))

    def topocentric_pointing(self, frames, epoch_utc, vector_bcrs_m) -> tuple[float, float]:
        vector_gcrs_m = frames.relativistic_transform.bcrs_vector2gcrs(vector_bcrs_m, self.transmit_epoch_tdb)
        vector_itrf_m = frames.terrestrial_transform.gcrs2itrf(vector_gcrs_m, epoch_utc)
        geodetic = itrf2geodetic(self.station_itrf_transmit_m)
        enu_m = itrf2enu(
            vector_itrf_m,
            latitude_rad=geodetic.latitude_rad,
            longitude_rad=geodetic.longitude_rad,
        )
        east, north, up = (float(value) for value in enu_m)
        if float(np.linalg.norm(enu_m)) <= 0.0:
            raise RuntimeError("Cannot compute pointing for a zero-length ENU vector.")
        return (
            float(np.rad2deg(np.arctan2(east, north)) % 360.0),
            float(np.rad2deg(np.arctan2(up, np.hypot(east, north)))),
        )

    def reflector_elevation_deg(self, frames, reflector_reference_pa_m) -> float:
        reflector_pa_m = np.asarray(reflector_reference_pa_m, dtype=float) + self.reflector_displacement_bounce_pa_m
        incident_bcrs_m = self.station_bcrs_transmit_m - self.reflector_bcrs_bounce_m
        incident_lcrs_m = frames.relativistic_transform.bcrs_vector2lcrs(incident_bcrs_m, self.bounce_epoch_tdb)
        incident_pa_m = (frames.lunar_orientation.pa_to_lcrs_matrix(self.bounce_epoch_tdb).T @ np.asarray(incident_lcrs_m))
        normal_pa = reflector_pa_m / np.linalg.norm(reflector_pa_m)
        sine_elevation = float(np.dot(incident_pa_m / np.linalg.norm(incident_pa_m), normal_pa))
        return float(np.rad2deg(np.arcsin(np.clip(sine_elevation, -1.0, 1.0))))

    def reflector_itrf_m(self, frames, bounce_epoch_utc) -> np.ndarray:
        return frames.terrestrial_transform.gcrs2itrf(
            frames.relativistic_transform.bcrs2gcrs(self.reflector_bcrs_bounce_m, self.bounce_epoch_tdb),
            bounce_epoch_utc,
        )


@dataclass(frozen=True, slots=True, eq=False)
class LightTimeSolution(UplinkSolution):
    """Round-trip evaluation extends the uplink with reception and clock corrections."""

    receive_epoch_tdb: Epoch
    computed_observable_round_trip_time_s: float
    tdb_coordinate_round_trip_time_s: float
    tt_minus_tdb_interval_correction_s: float
    pre_1972_utc_rate_offset: float
    downlink: LightTimeLeg
    station_displacement_receive_itrf_m: np.ndarray
    station_bcrs_receive_m: np.ndarray

    @property
    def event_closure_residuals_s(self) -> tuple[float, float, float]:
        up = self.transmit_epoch_tdb.seconds_until(self.bounce_epoch_tdb)
        down = self.bounce_epoch_tdb.seconds_until(self.receive_epoch_tdb)
        total = self.transmit_epoch_tdb.seconds_until(self.receive_epoch_tdb)
        return (
            up - self.uplink.travel_time_s,
            down - self.downlink.travel_time_s,
            total - self.uplink.travel_time_s - self.downlink.travel_time_s,
        )


class _StationEventState(NamedTuple):
    epoch_utc: Epoch
    reference_itrf_m: np.ndarray
    displacement_itrf_m: np.ndarray
    position_itrf_m: np.ndarray
    position_gcrs_m: np.ndarray


class LightTimeSolver:
    def __init__(
        self,
        frame_system: ReferenceFrameSystem,
        *,
        gravitational_delay_models: tuple[Iers2010ShapiroDelay, ...],
        troposphere_delay_models: tuple[Iers2010MendesPavlisTroposphere, ...],
        station_displacement_model: StationDisplacement,
        reflector_displacement_models: tuple[LunarSolidTide, ...],
    ) -> None:
        self._station_request: LightTimeRequest | None = None
        self._station_cache: dict[tuple[float, float], _StationEventState] = {}
        self.frame_system = frame_system
        self.time_scale_converter = frame_system.time_scale_converter
        self.gravitational_delay_models = gravitational_delay_models
        self.troposphere_delay_models = troposphere_delay_models
        self.station_displacement_model = station_displacement_model
        self.reflector_displacement_models = reflector_displacement_models

    def _reflector_state_lcrs_m(
        self,
        reflector_reference_pa_m: ArrayLike,
        epoch_tdb: Epoch,
    ) -> tuple[np.ndarray, np.ndarray]:
        reflector_lcrs = (self.frame_system.lunar_orientation.pa_to_lcrs_matrix(epoch_tdb) @ np.asarray(reflector_reference_pa_m))
        displacement_lcrs = np.zeros(3, dtype=float)
        data = ReflectorDisplacementInput(reference_position_lcrs_m=reflector_lcrs, epoch_tdb=epoch_tdb)
        for model in self.reflector_displacement_models:
            displacement_lcrs += model.displacement_lcrs_m(data)
        return reflector_lcrs + displacement_lcrs, displacement_lcrs

    def _station_state_at_utc(
        self,
        request: LightTimeRequest,
        epoch_utc: Epoch,
    ) -> _StationEventState:
        if self._station_request is not request:
            self._station_cache.clear()
            self._station_request = request
        key = (epoch_utc.jd1, epoch_utc.jd2)
        if key in self._station_cache:
            return self._station_cache[key]
        reference = request.station.itrf_xyz_at(epoch_utc)
        displacement = self.station_displacement_model.displacement_itrf_m(
            StationDisplacementInput(
                reference_position_itrf_m=reference,
                epoch_utc=epoch_utc,
                station_id=request.station_key,
            )
        )
        position = reference + displacement
        gcrs = self.frame_system.terrestrial_transform.itrf2gcrs(position, epoch_utc)
        state = _StationEventState(
            epoch_utc=epoch_utc,
            reference_itrf_m=reference,
            displacement_itrf_m=displacement,
            position_itrf_m=position,
            position_gcrs_m=gcrs,
        )
        self._station_cache[key] = state
        return state

    def _topocentric_observer(
        self,
        request: LightTimeRequest,
    ) -> Callable[[Epoch], TdbTopocentricArguments]:
        """Return the UTC-dependent station context needed by ERFA ``dtdb``."""

        def arguments_at_utc(epoch_utc: Epoch) -> TdbTopocentricArguments:
            # Reference coordinates avoid computing tide/loading displacements
            # for dtdb alone; time rounding can change the resulting residuals.
            return self.frame_system.terrestrial_transform.tdb_topocentric_arguments(
                request.station.itrf_xyz_at(epoch_utc),
                epoch_utc,
            )

        return arguments_at_utc

    def event_epoch_utc(self, request: LightTimeRequest, epoch_tdb: Epoch) -> Epoch:
        """Convert a solved TDB event epoch to UTC with the request's station context."""
        return self.time_scale_converter.convert(
            epoch_tdb,
            TimeScale.UTC,
            topocentric_observer=self._topocentric_observer(request),
        )

    def _station_state_from_tdb(
        self,
        request: LightTimeRequest,
        epoch_tdb: Epoch,
    ) -> _StationEventState:
        """Resolve the station UTC epoch including the topocentric TDB-TT term."""
        return self._station_state_at_utc(request, self.event_epoch_utc(request, epoch_tdb))

    def _vacuum_elevation_rad(
        self,
        station_itrf_m: np.ndarray,
        line_of_sight_bcrs_m: np.ndarray,
        station_epoch_utc: Epoch,
        station_epoch_tdb: Epoch,
    ) -> float:
        """Vacuum elevation from a light path's two BCRS endpoint events."""
        line_of_sight_gcrs = self.frame_system.relativistic_transform.bcrs_vector2gcrs(
            line_of_sight_bcrs_m,
            station_epoch_tdb,
        )
        los_itrf = self.frame_system.terrestrial_transform.gcrs2itrf(line_of_sight_gcrs, station_epoch_utc)
        distance = float(np.linalg.norm(los_itrf))
        if distance <= 0.0:
            raise RuntimeError("Cannot compute elevation for a zero-length topocentric vector.")
        up = local_up_unit_itrf(station_itrf_m)
        sine_elevation = float(np.dot(los_itrf / distance, up))
        return float(np.arcsin(np.clip(sine_elevation, -1.0, 1.0)))

    def _troposphere_evaluation_elevation(self, elevation_rad: float) -> tuple[float, bool]:
        for model in self.troposphere_delay_models:
            floor_rad = model.elevation_floor_rad
            if float(elevation_rad) < floor_rad:
                return floor_rad, True
        return float(elevation_rad), False

    @staticmethod
    def _pre_1972_utc_rate_offset(epoch_utc: Epoch) -> float:
        return 3.0e-8 if _PRE_1972_UTC_RATE_START <= epoch_utc < _PRE_1972_UTC_RATE_END else 0.0

    def _transmit_state(
        self,
        request: LightTimeRequest,
    ) -> tuple[Epoch, _StationEventState, Epoch, np.ndarray]:
        transmit_utc = request.transmit_epoch_utc
        transmit_station = self._station_state_at_utc(request, transmit_utc)
        transmit_tdb = self.time_scale_converter.convert(
            transmit_utc,
            TimeScale.TDB,
            topocentric_observer=self._topocentric_observer(request),
        )
        station_bcrs_transmit = self.frame_system.relativistic_transform.gcrs2bcrs(
            transmit_station.position_gcrs_m,
            transmit_tdb,
        )
        return transmit_utc, transmit_station, transmit_tdb, station_bcrs_transmit

    def _leg(self, request, station, station_tdb, station_bcrs, reflector_bcrs, bounce_tdb, *, uplink):
        start, end = (station_bcrs, reflector_bcrs) if uplink else (reflector_bcrs, station_bcrs)
        geometric_range_m = float(np.linalg.norm(end - start))
        gravitational_delay_m = float(sum(model.path_delay_m(start, end, bounce_tdb) for model in self.gravitational_delay_models))
        vacuum_elevation_rad = self._vacuum_elevation_rad(
            station.position_itrf_m, reflector_bcrs - station_bcrs, station.epoch_utc, station_tdb,
        )
        troposphere_elevation_rad, troposphere_clamped = self._troposphere_evaluation_elevation(vacuum_elevation_rad)
        tropospheric_delay_m = float(sum(model.slant_delay_m(
            request.troposphere_input, elevation_rad=troposphere_elevation_rad
        ) for model in self.troposphere_delay_models))
        return LightTimeLeg(
            geometric_range_m, gravitational_delay_m, tropospheric_delay_m, vacuum_elevation_rad,
            troposphere_elevation_rad, troposphere_clamped,
        )

    @overload
    def solve(self, request: LightTimeRequest, *, two_way: Literal[True] = True) -> LightTimeSolution: ...

    @overload
    def solve(self, request: LightTimeRequest, *, two_way: Literal[False]) -> UplinkSolution: ...

    def solve(self, request: LightTimeRequest, *, two_way: bool = True) -> UplinkSolution:
        transmit_utc, transmit_station, transmit_tdb, station_bcrs_transmit = self._transmit_state(request)
        initial_rtt_s = 2.0 * _INITIAL_UPLINK_LIGHT_TIME_S
        bounce_tdb = transmit_tdb.shifted(_INITIAL_UPLINK_LIGHT_TIME_S)
        receive_tdb = transmit_tdb.shifted(initial_rtt_s)
        previous_rtt_s = initial_rtt_s
        converged = False
        for iteration_count in range(1, _MAX_LIGHT_TIME_ITERATIONS + 1):
            evaluated_bounce_tdb = bounce_tdb
            if two_way:
                receive_station = self._station_state_from_tdb(request, receive_tdb)
            reflector_lcrs, reflector_displacement_lcrs = self._reflector_state_lcrs_m(
                request.reflector_reference_pa_m, bounce_tdb,
            )
            reflector_bcrs = self.frame_system.relativistic_transform.lcrs2bcrs(reflector_lcrs, bounce_tdb)
            uplink = self._leg(
                request, transmit_station, transmit_tdb, station_bcrs_transmit, reflector_bcrs, bounce_tdb,
                uplink=True,
            )
            if two_way:
                station_bcrs_receive = self.frame_system.relativistic_transform.gcrs2bcrs(
                    receive_station.position_gcrs_m, receive_tdb,
                )
                downlink = self._leg(
                    request, receive_station, receive_tdb, station_bcrs_receive, reflector_bcrs, bounce_tdb,
                    uplink=False,
                )
            new_bounce_tdb = transmit_tdb.shifted(uplink.travel_time_s)
            if two_way:
                receive_tdb = new_bounce_tdb.shifted(downlink.travel_time_s)
                new_rtt_s = transmit_tdb.seconds_until(receive_tdb)
                change_s = abs(new_rtt_s - previous_rtt_s)
                previous_rtt_s = new_rtt_s
                bounce_tdb = new_bounce_tdb
            else:
                change_s = abs(bounce_tdb.seconds_until(new_bounce_tdb))
            if change_s < _ROUND_TRIP_TIME_TOLERANCE_S:
                converged = True
                break
            if not two_way:
                bounce_tdb = new_bounce_tdb

        if not two_way:
            bounce_tdb = evaluated_bounce_tdb
        reflector_displacement_pa = (self.frame_system.lunar_orientation.pa_to_lcrs_matrix(bounce_tdb).T @ np.asarray(reflector_displacement_lcrs))
        common = {
            "transmit_epoch_tdb": transmit_tdb,
            "bounce_epoch_tdb": bounce_tdb,
            "uplink": uplink,
            "station_itrf_transmit_m": transmit_station.position_itrf_m,
            "station_displacement_transmit_itrf_m": transmit_station.displacement_itrf_m,
            "reflector_displacement_bounce_pa_m": reflector_displacement_pa,
            "station_bcrs_transmit_m": station_bcrs_transmit,
            "reflector_bcrs_bounce_m": reflector_bcrs,
            "iteration_count": iteration_count,
            "light_time_converged": converged,
        }
        if not two_way:
            return UplinkSolution(**common)
        transmit_tt = self.time_scale_converter.utc2tt(transmit_utc)
        receive_tt = self.time_scale_converter.utc2tt(receive_station.epoch_utc)
        coordinate_rtt_s = transmit_tdb.seconds_until(receive_tdb)
        tt_rtt_s = transmit_tt.seconds_until(receive_tt)
        tt_minus_tdb_s = tt_rtt_s - coordinate_rtt_s
        zeta = self._pre_1972_utc_rate_offset(transmit_utc)
        observable_rtt_s = tt_rtt_s / (1.0 + zeta)
        return LightTimeSolution(
            **common,
            receive_epoch_tdb=receive_tdb,
            computed_observable_round_trip_time_s=float(observable_rtt_s),
            tdb_coordinate_round_trip_time_s=float(coordinate_rtt_s),
            tt_minus_tdb_interval_correction_s=float(tt_minus_tdb_s),
            pre_1972_utc_rate_offset=float(zeta),
            downlink=downlink,
            station_displacement_receive_itrf_m=receive_station.displacement_itrf_m,
            station_bcrs_receive_m=station_bcrs_receive,
        )


__all__ = [
    "LightTimeLeg",
    "LightTimeRequest",
    "LightTimeSolution",
    "LightTimeSolver",
    "UplinkSolution",
]
