"""Two-way lunar laser ranging light-time solution using unified epochs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.array_validation import finite_array
from lunarops.base.constants import C
from lunarops.classes.delays import (
    GravitationalDelay,
    TroposphereDelay,
    TroposphereInput,
)
from lunarops.classes.displacement import (
    ReflectorDisplacement,
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

    ``troposphere_input`` carries the station meteorology at zenith; each leg
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
                elevation_rad=0.5 * np.pi,
                pressure_hpa=pressure_hpa,
                temperature_k=temperature_k,
                relative_humidity_percent=relative_humidity_percent,
                latitude_rad=geodetic.latitude_rad,
                height_m=geodetic.ellipsoidal_height_m,
                wavelength_um=wavelength_um,
            ),
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reflector_reference_pa_m",
            finite_array(self.reflector_reference_pa_m, name="reflector_reference_pa_m", size=3, copy=True, readonly=True),
        )
        self.transmit_epoch_utc.require_scale(TimeScale.UTC, name="transmit_epoch_utc")


@dataclass(frozen=True, slots=True)
class LightTimeLeg:
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
class UplinkLightTimeSolution:
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
class LightTimeSolution(UplinkLightTimeSolution):
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


@dataclass(frozen=True, slots=True)
class _IterationState:
    bounce_epoch_tdb: Epoch
    receive_epoch_tdb: Epoch
    uplink: LightTimeLeg
    downlink: LightTimeLeg
    receive_station: _StationEventState
    station_bcrs_receive_m: np.ndarray
    reflector_bcrs_bounce_m: np.ndarray
    reflector_displacement_lcrs_bounce_m: np.ndarray


@dataclass(frozen=True, slots=True)
class _UplinkIterationState:
    bounce_epoch_tdb: Epoch
    uplink: LightTimeLeg
    reflector_bcrs_bounce_m: np.ndarray
    reflector_displacement_lcrs_bounce_m: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class _StationEventState:
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
        gravitational_delay_model: GravitationalDelay,
        troposphere_delay_model: TroposphereDelay,
        station_displacement_model: StationDisplacement,
        reflector_displacement_model: ReflectorDisplacement,
    ) -> None:
        self._station_request: LightTimeRequest | None = None
        self._station_cache: dict[tuple[float, float], _StationEventState] = {}
        self.frame_system = frame_system
        self.time_scale_converter = frame_system.time_scale_converter
        self.gravitational_delay_model = gravitational_delay_model
        self.troposphere_delay_model = troposphere_delay_model
        self.station_displacement_model = station_displacement_model
        self.reflector_displacement_model = reflector_displacement_model

    def _reflector_state_lcrs_m(
        self,
        reflector_reference_pa_m: ArrayLike,
        epoch_tdb: Epoch,
    ) -> tuple[np.ndarray, np.ndarray]:
        reflector_lcrs = (self.frame_system.lunar_orientation.pa_to_lcrs_matrix(epoch_tdb) @ np.asarray(reflector_reference_pa_m))
        displacement_lcrs = self.reflector_displacement_model.displacement_lcrs_m(
            ReflectorDisplacementInput(
                reference_position_lcrs_m=reflector_lcrs,
                epoch_tdb=epoch_tdb,
            )
        )
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
        floor_rad = self.troposphere_delay_model.elevation_floor_rad
        if floor_rad is None:
            return float(elevation_rad), False
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

    def _uplink_state(
        self,
        request: LightTimeRequest,
        *,
        transmit_utc: Epoch,
        transmit_station: _StationEventState,
        transmit_tdb: Epoch,
        station_bcrs_transmit_m: np.ndarray,
        bounce_tdb: Epoch,
    ) -> _UplinkIterationState:
        reflector_lcrs_bounce, reflector_displacement_lcrs_bounce = self._reflector_state_lcrs_m(
            request.reflector_reference_pa_m,
            bounce_tdb,
        )
        reflector_bcrs_bounce = self.frame_system.relativistic_transform.lcrs2bcrs(
            reflector_lcrs_bounce,
            bounce_tdb,
        )
        geometric_range_m = float(np.linalg.norm(reflector_bcrs_bounce - station_bcrs_transmit_m))
        gravitational_delay_m = float(
            self.gravitational_delay_model.path_delay_m(
                station_bcrs_transmit_m,
                reflector_bcrs_bounce,
                bounce_tdb,
            )
        )
        vacuum_elevation_rad = self._vacuum_elevation_rad(
            transmit_station.position_itrf_m,
            reflector_bcrs_bounce - station_bcrs_transmit_m,
            transmit_utc,
            transmit_tdb,
        )
        troposphere_elevation_rad, troposphere_clamped = self._troposphere_evaluation_elevation(vacuum_elevation_rad)
        tropospheric_delay_m = float(
            self.troposphere_delay_model.slant_delay_m(
                replace(request.troposphere_input, elevation_rad=troposphere_elevation_rad)
            )
        )
        return _UplinkIterationState(
            bounce_epoch_tdb=bounce_tdb,
            uplink=LightTimeLeg(
                geometric_range_m=geometric_range_m,
                gravitational_path_delay_m=gravitational_delay_m,
                tropospheric_path_delay_m=tropospheric_delay_m,
                vacuum_elevation_rad=vacuum_elevation_rad,
                troposphere_elevation_used_rad=troposphere_elevation_rad,
                troposphere_elevation_clamped=troposphere_clamped,
            ),
            reflector_bcrs_bounce_m=reflector_bcrs_bounce,
            reflector_displacement_lcrs_bounce_m=reflector_displacement_lcrs_bounce,
        )

    @staticmethod
    def _fixed_point(value, update):
        for iteration in range(1, _MAX_LIGHT_TIME_ITERATIONS + 1):
            value, state, change_s = update(value, iteration)
            if change_s < _ROUND_TRIP_TIME_TOLERANCE_S:
                return state, iteration, True
        return state, iteration, False

    def solve_uplink(self, request: LightTimeRequest) -> UplinkLightTimeSolution:
        """Solve only the outgoing station-to-reflector light path.

        Pointing prediction needs the transmit event and the reflector bounce
        event, but not the later station receive event. The fixed-point solve
        is therefore independent of the downlink geometry.
        """
        transmit_utc, transmit_station, transmit_tdb, station_bcrs_transmit = self._transmit_state(request)

        def update(bounce_tdb, iteration):
            state = self._uplink_state(
                request,
                transmit_utc=transmit_utc,
                transmit_station=transmit_station,
                transmit_tdb=transmit_tdb,
                station_bcrs_transmit_m=station_bcrs_transmit,
                bounce_tdb=bounce_tdb,
            )
            corrected = transmit_tdb.shifted(state.uplink.travel_time_s)
            return corrected, state, abs(bounce_tdb.seconds_until(corrected))

        final_state, iteration_count, converged = self._fixed_point(
            transmit_tdb.shifted(_INITIAL_UPLINK_LIGHT_TIME_S),
            update,
        )

        reflector_displacement_pa_bounce = (self.frame_system.lunar_orientation.pa_to_lcrs_matrix(final_state.bounce_epoch_tdb).T @ np.asarray(final_state.reflector_displacement_lcrs_bounce_m))
        return UplinkLightTimeSolution(
            transmit_epoch_tdb=transmit_tdb,
            bounce_epoch_tdb=final_state.bounce_epoch_tdb,
            uplink=final_state.uplink,
            station_itrf_transmit_m=transmit_station.position_itrf_m,
            station_displacement_transmit_itrf_m=transmit_station.displacement_itrf_m,
            reflector_displacement_bounce_pa_m=reflector_displacement_pa_bounce,
            station_bcrs_transmit_m=station_bcrs_transmit,
            reflector_bcrs_bounce_m=final_state.reflector_bcrs_bounce_m,
            iteration_count=iteration_count,
            light_time_converged=converged,
        )

    def solve(self, request: LightTimeRequest) -> LightTimeSolution:
        transmit_utc, transmit_station, transmit_tdb, station_bcrs_transmit = self._transmit_state(request)
        initial_rtt_s = 2.0 * _INITIAL_UPLINK_LIGHT_TIME_S
        bounce_tdb = transmit_tdb.shifted(_INITIAL_UPLINK_LIGHT_TIME_S)
        receive_tdb = transmit_tdb.shifted(initial_rtt_s)
        previous_rtt_s = initial_rtt_s

        def update(events, iteration):
            bounce_tdb, receive_tdb, previous_rtt_s = events
            receive_station = self._station_state_from_tdb(request, receive_tdb)
            receive_utc = receive_station.epoch_utc
            uplink_state = self._uplink_state(
                request,
                transmit_utc=transmit_utc,
                transmit_station=transmit_station,
                transmit_tdb=transmit_tdb,
                station_bcrs_transmit_m=station_bcrs_transmit,
                bounce_tdb=bounce_tdb,
            )
            station_bcrs_receive = self.frame_system.relativistic_transform.gcrs2bcrs(
                receive_station.position_gcrs_m,
                receive_tdb,
            )
            reflector_bcrs_bounce = uplink_state.reflector_bcrs_bounce_m
            geometric_down_m = float(np.linalg.norm(station_bcrs_receive - reflector_bcrs_bounce))
            gravitational_down_m = float(
                self.gravitational_delay_model.path_delay_m(
                    reflector_bcrs_bounce,
                    station_bcrs_receive,
                    bounce_tdb,
                )
            )
            elevation_down_rad = self._vacuum_elevation_rad(
                receive_station.position_itrf_m,
                reflector_bcrs_bounce - station_bcrs_receive,
                receive_utc,
                receive_tdb,
            )
            tropo_elevation_down_rad, tropo_down_clamped = self._troposphere_evaluation_elevation(elevation_down_rad)
            troposphere_down_m = float(
                self.troposphere_delay_model.slant_delay_m(
                    replace(request.troposphere_input, elevation_rad=tropo_elevation_down_rad)
                )
            )

            uplink = uplink_state.uplink
            downlink = LightTimeLeg(
                geometric_range_m=geometric_down_m,
                gravitational_path_delay_m=gravitational_down_m,
                tropospheric_path_delay_m=troposphere_down_m,
                vacuum_elevation_rad=elevation_down_rad,
                troposphere_elevation_used_rad=tropo_elevation_down_rad,
                troposphere_elevation_clamped=tropo_down_clamped,
            )
            new_bounce_tdb = transmit_tdb.shifted(uplink.travel_time_s)
            new_receive_tdb = new_bounce_tdb.shifted(downlink.travel_time_s)
            new_rtt_s = transmit_tdb.seconds_until(new_receive_tdb)

            final_state = _IterationState(
                bounce_epoch_tdb=new_bounce_tdb,
                receive_epoch_tdb=new_receive_tdb,
                uplink=uplink,
                downlink=downlink,
                receive_station=receive_station,
                station_bcrs_receive_m=station_bcrs_receive,
                reflector_bcrs_bounce_m=reflector_bcrs_bounce,
                reflector_displacement_lcrs_bounce_m=uplink_state.reflector_displacement_lcrs_bounce_m,
            )
            return (new_bounce_tdb, new_receive_tdb, new_rtt_s), final_state, abs(new_rtt_s - previous_rtt_s)

        final_state, iteration_count, converged = self._fixed_point(
            (bounce_tdb, receive_tdb, previous_rtt_s),
            update,
        )

        # Reuse the last evaluated events and their UTC context. Their epochs
        # precede the final fixed-point update; avoiding reevaluation and TDB-TT
        # round trips intentionally changes the observable's rounding path.
        receive_station = final_state.receive_station
        reflector_displacement_pa_bounce = (self.frame_system.lunar_orientation.pa_to_lcrs_matrix(final_state.bounce_epoch_tdb).T @ np.asarray(final_state.reflector_displacement_lcrs_bounce_m))
        transmit_tt = self.time_scale_converter.utc2tt(transmit_utc)
        receive_tt = self.time_scale_converter.utc2tt(receive_station.epoch_utc)

        coordinate_rtt_s = transmit_tdb.seconds_until(final_state.receive_epoch_tdb)
        tt_rtt_s = transmit_tt.seconds_until(receive_tt)
        tt_minus_tdb_s = tt_rtt_s - coordinate_rtt_s
        zeta = self._pre_1972_utc_rate_offset(transmit_utc)
        observable_rtt_s = tt_rtt_s / (1.0 + zeta)

        return LightTimeSolution(
            transmit_epoch_tdb=transmit_tdb,
            bounce_epoch_tdb=final_state.bounce_epoch_tdb,
            receive_epoch_tdb=final_state.receive_epoch_tdb,
            computed_observable_round_trip_time_s=float(observable_rtt_s),
            tdb_coordinate_round_trip_time_s=float(coordinate_rtt_s),
            tt_minus_tdb_interval_correction_s=float(tt_minus_tdb_s),
            pre_1972_utc_rate_offset=float(zeta),
            uplink=final_state.uplink,
            downlink=final_state.downlink,
            station_itrf_transmit_m=transmit_station.position_itrf_m,
            station_displacement_transmit_itrf_m=transmit_station.displacement_itrf_m,
            station_displacement_receive_itrf_m=receive_station.displacement_itrf_m,
            reflector_displacement_bounce_pa_m=reflector_displacement_pa_bounce,
            station_bcrs_transmit_m=station_bcrs_transmit,
            station_bcrs_receive_m=final_state.station_bcrs_receive_m,
            reflector_bcrs_bounce_m=final_state.reflector_bcrs_bounce_m,
            iteration_count=iteration_count,
            light_time_converged=converged,
        )


__all__ = [
    "LightTimeLeg",
    "LightTimeRequest",
    "LightTimeSolution",
    "LightTimeSolver",
    "UplinkLightTimeSolution",
]
