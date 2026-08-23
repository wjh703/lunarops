"""Two-way lunar laser ranging light-time solution using unified epochs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.array_validation import readonly_vector3, vector3
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
from lunarops.classes.displacement.terrestrial_geometry import local_up_unit_itrf
from lunarops.classes.frames import ReferenceFrameSystem
from lunarops.classes.time import Epoch, TdbTopocentricArguments, TimeScale

_MAX_LIGHT_TIME_ITERATIONS = 12
_ROUND_TRIP_TIME_TOLERANCE_S = 1.0e-12


@dataclass(frozen=True, slots=True)
class TroposphereEnvironment:
    pressure_hpa: float
    temperature_k: float
    relative_humidity_percent: float
    latitude_rad: float
    ellipsoidal_height_m: float
    wavelength_um: float

    def __post_init__(self) -> None:
        for name in (
            "pressure_hpa",
            "temperature_k",
            "relative_humidity_percent",
            "latitude_rad",
            "ellipsoidal_height_m",
            "wavelength_um",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.pressure_hpa <= 0.0:
            raise ValueError("pressure_hpa must be positive.")
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive.")
        if not 0.0 <= self.relative_humidity_percent <= 100.0:
            raise ValueError("relative_humidity_percent must be in [0, 100].")
        if not -0.5 * np.pi <= self.latitude_rad <= 0.5 * np.pi:
            raise ValueError("latitude_rad must be in [-pi/2, pi/2].")
        if self.wavelength_um <= 0.0:
            raise ValueError("wavelength_um must be positive.")

    def troposphere_input(self, elevation_rad: float) -> TroposphereInput:
        return TroposphereInput(
            elevation_rad=float(elevation_rad),
            pressure_hpa=self.pressure_hpa,
            temperature_k=self.temperature_k,
            relative_humidity_percent=self.relative_humidity_percent,
            latitude_rad=self.latitude_rad,
            height_m=self.ellipsoidal_height_m,
            wavelength_um=self.wavelength_um,
        )


@dataclass(frozen=True, slots=True, eq=False)
class LightTimeRequest:
    reflector_reference_pa_m: np.ndarray
    transmit_epoch_utc: Epoch
    troposphere_environment: TroposphereEnvironment
    station_reference_itrf_at_utc: Callable[[Epoch], ArrayLike]
    station_key: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reflector_reference_pa_m",
            readonly_vector3(self.reflector_reference_pa_m, name="reflector_reference_pa_m"),
        )
        if not isinstance(self.transmit_epoch_utc, Epoch):
            raise TypeError("transmit_epoch_utc must be an Epoch.")
        self.transmit_epoch_utc.require_scale(TimeScale.UTC, name="transmit_epoch_utc")
        if not isinstance(self.troposphere_environment, TroposphereEnvironment):
            raise TypeError("troposphere_environment must be a TroposphereEnvironment.")
        if not callable(self.station_reference_itrf_at_utc):
            raise TypeError("station_reference_itrf_at_utc must be callable.")
        if not isinstance(self.station_key, str) or not self.station_key.strip():
            raise TypeError("station_key must be a non-empty string.")

    def station_reference_itrf_at(self, epoch_utc: Epoch) -> np.ndarray:
        if not isinstance(epoch_utc, Epoch):
            raise TypeError("epoch_utc must be an Epoch.")
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        return readonly_vector3(
            self.station_reference_itrf_at_utc(epoch_utc),
            name="station_reference_itrf_at_utc result",
        )


@dataclass(frozen=True, slots=True)
class LightTimeLeg:
    geometric_range_m: float
    gravitational_path_delay_m: float
    tropospheric_path_delay_m: float
    vacuum_elevation_rad: float
    troposphere_elevation_used_rad: float | None = None
    troposphere_elevation_clamped: bool = False

    def __post_init__(self) -> None:
        for name in (
            "geometric_range_m",
            "gravitational_path_delay_m",
            "tropospheric_path_delay_m",
            "vacuum_elevation_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.geometric_range_m < 0.0:
            raise ValueError("geometric_range_m must be non-negative.")
        if not -0.5 * np.pi <= self.vacuum_elevation_rad <= 0.5 * np.pi:
            raise ValueError("vacuum_elevation_rad must be in [-pi/2, pi/2].")
        if self.troposphere_elevation_used_rad is not None:
            used = float(self.troposphere_elevation_used_rad)
            if not np.isfinite(used) or not -0.5 * np.pi <= used <= 0.5 * np.pi:
                raise ValueError("troposphere_elevation_used_rad must be in [-pi/2, pi/2].")
            object.__setattr__(self, "troposphere_elevation_used_rad", used)
        if not isinstance(self.troposphere_elevation_clamped, bool):
            raise TypeError("troposphere_elevation_clamped must be a bool.")

    @property
    def path_length_m(self) -> float:
        return self.geometric_range_m + self.gravitational_path_delay_m + self.tropospheric_path_delay_m

    @property
    def travel_time_s(self) -> float:
        return self.path_length_m / C


@dataclass(frozen=True, slots=True, eq=False)
class LightTimeSolution:
    """TDB event epochs and light-path diagnostics.

    UTC copies are deliberately not stored.  Callers convert an event through
    the shared ``TimeScaleConverter`` only where UTC is actually required.
    """

    transmit_epoch_tdb: Epoch
    bounce_epoch_tdb: Epoch
    receive_epoch_tdb: Epoch
    computed_observable_round_trip_time_s: float
    tdb_coordinate_round_trip_time_s: float
    tt_minus_tdb_interval_correction_s: float
    pre_1972_utc_rate_offset: float
    uplink: LightTimeLeg
    downlink: LightTimeLeg
    station_displacement_transmit_itrf_m: np.ndarray
    station_displacement_receive_itrf_m: np.ndarray
    reflector_displacement_bounce_pa_m: np.ndarray
    station_bcrs_transmit_m: np.ndarray
    station_bcrs_receive_m: np.ndarray
    reflector_bcrs_bounce_m: np.ndarray
    iteration_count: int
    light_time_converged: bool

    def __post_init__(self) -> None:
        for name in ("transmit_epoch_tdb", "bounce_epoch_tdb", "receive_epoch_tdb"):
            epoch = getattr(self, name)
            if not isinstance(epoch, Epoch):
                raise TypeError(f"{name} must be an Epoch.")
            epoch.require_scale(TimeScale.TDB, name=name)
        for name in (
            "station_displacement_transmit_itrf_m",
            "station_displacement_receive_itrf_m",
            "reflector_displacement_bounce_pa_m",
            "station_bcrs_transmit_m",
            "station_bcrs_receive_m",
            "reflector_bcrs_bounce_m",
        ):
            object.__setattr__(
                self,
                name,
                readonly_vector3(getattr(self, name), name=name),
            )
        for name in (
            "computed_observable_round_trip_time_s",
            "tdb_coordinate_round_trip_time_s",
            "tt_minus_tdb_interval_correction_s",
            "pre_1972_utc_rate_offset",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if isinstance(self.iteration_count, bool) or not isinstance(self.iteration_count, int):
            raise TypeError("iteration_count must be an integer.")
        if self.iteration_count <= 0:
            raise ValueError("iteration_count must be positive.")
        if not isinstance(self.light_time_converged, bool):
            raise TypeError("light_time_converged must be a bool.")
        for name in ("uplink", "downlink"):
            if not isinstance(getattr(self, name), LightTimeLeg):
                raise TypeError(f"{name} must be a LightTimeLeg.")


@dataclass(frozen=True, slots=True)
class _IterationState:
    transmit_epoch_tdb: Epoch
    bounce_epoch_tdb: Epoch
    receive_epoch_tdb: Epoch
    uplink: LightTimeLeg
    downlink: LightTimeLeg
    iteration_count: int


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
        if not isinstance(frame_system, ReferenceFrameSystem):
            raise TypeError("frame_system must be a ReferenceFrameSystem.")
        if not isinstance(gravitational_delay_model, GravitationalDelay):
            raise TypeError("gravitational_delay_model must be a GravitationalDelay.")
        if not isinstance(troposphere_delay_model, TroposphereDelay):
            raise TypeError("troposphere_delay_model must be a TroposphereDelay.")
        if not isinstance(station_displacement_model, StationDisplacement):
            raise TypeError("station_displacement_model must be a StationDisplacement.")
        if not isinstance(reflector_displacement_model, ReflectorDisplacement):
            raise TypeError("reflector_displacement_model must be a ReflectorDisplacement.")
        self.frame_system = frame_system
        self.time_scale_converter = frame_system.time_scale_converter
        self.gravitational_delay_model = gravitational_delay_model
        self.troposphere_delay_model = troposphere_delay_model
        self.station_displacement_model = station_displacement_model
        self.reflector_displacement_model = reflector_displacement_model

    def _station_displacement_itrf_m(
        self,
        station_reference_itrf_m: ArrayLike,
        epoch_utc: Epoch,
        station_key: str,
    ) -> np.ndarray:
        reference = vector3(station_reference_itrf_m, name="station_reference_itrf_m")
        return np.asarray(
            self.station_displacement_model.displacement_itrf_m(
                StationDisplacementInput(
                    reference_position_itrf_m=reference,
                    epoch_utc=epoch_utc,
                    station_id=station_key,
                )
            ),
            dtype=np.float64,
        ).reshape(3)

    def _reflector_state_lcrs_m(
        self,
        reflector_reference_pa_m: ArrayLike,
        epoch_tdb: Epoch,
    ) -> tuple[np.ndarray, np.ndarray]:
        reflector_lcrs = self.frame_system.pa2lcrs(reflector_reference_pa_m, epoch_tdb)
        displacement_lcrs = np.asarray(
            self.reflector_displacement_model.displacement_lcrs_m(
                ReflectorDisplacementInput(
                    reference_position_lcrs_m=reflector_lcrs,
                    epoch_tdb=epoch_tdb,
                )
            ),
            dtype=float,
        ).reshape(3)
        return reflector_lcrs + displacement_lcrs, displacement_lcrs

    def _station_state_at_utc(
        self,
        request: LightTimeRequest,
        epoch_utc: Epoch,
    ) -> _StationEventState:
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        reference = request.station_reference_itrf_at(epoch_utc)
        displacement = self._station_displacement_itrf_m(
            reference,
            epoch_utc,
            request.station_key,
        )
        position = reference + displacement
        gcrs = self.frame_system.itrf2gcrs(position, epoch_utc)
        return _StationEventState(
            epoch_utc=epoch_utc,
            reference_itrf_m=reference,
            displacement_itrf_m=displacement,
            position_itrf_m=position,
            position_gcrs_m=gcrs,
        )

    def _topocentric_observer(
        self,
        request: LightTimeRequest,
    ) -> Callable[[Epoch], TdbTopocentricArguments]:
        """Return the UTC-dependent station context needed by ERFA ``dtdb``."""

        def arguments_at_utc(epoch_utc: Epoch) -> TdbTopocentricArguments:
            station = self._station_state_at_utc(request, epoch_utc)
            return self.frame_system.terrestrial_transform.tdb_topocentric_arguments(
                station.position_itrf_m,
                epoch_utc,
            )

        return arguments_at_utc

    def _station_state_from_tdb(
        self,
        request: LightTimeRequest,
        epoch_tdb: Epoch,
    ) -> _StationEventState:
        """Resolve the station UTC epoch including the topocentric TDB-TT term."""
        epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
        epoch_utc = self.time_scale_converter.convert(
            epoch_tdb,
            TimeScale.UTC,
            topocentric_observer=self._topocentric_observer(request),
        )
        return self._station_state_at_utc(request, epoch_utc)

    def event_epoch_utc(self, request: LightTimeRequest, epoch_tdb: Epoch) -> Epoch:
        """Convert a solved TDB event epoch to UTC with the request's station context."""
        if not isinstance(request, LightTimeRequest):
            raise TypeError("request must be a LightTimeRequest.")
        epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
        return self.time_scale_converter.convert(
            epoch_tdb,
            TimeScale.UTC,
            topocentric_observer=self._topocentric_observer(request),
        )

    def station_position_itrf_m(self, request: LightTimeRequest, epoch_utc: Epoch) -> np.ndarray:
        """Return the reference plus modeled displacement of a station event."""
        if not isinstance(request, LightTimeRequest):
            raise TypeError("request must be a LightTimeRequest.")
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        return np.array(self._station_state_at_utc(request, epoch_utc).position_itrf_m, copy=True)

    def _vacuum_elevation_rad(
        self,
        station_itrf_m: ArrayLike,
        line_of_sight_bcrs_m: ArrayLike,
        station_epoch_utc: Epoch,
        station_epoch_tdb: Epoch,
    ) -> float:
        """Vacuum elevation from a light path's two BCRS endpoint events."""
        station_itrf = vector3(station_itrf_m, name="station_itrf_m")
        line_of_sight_bcrs = vector3(line_of_sight_bcrs_m, name="line_of_sight_bcrs_m")
        line_of_sight_gcrs = self.frame_system.bcrs_vector2gcrs(
            line_of_sight_bcrs,
            station_epoch_tdb,
        )
        los_itrf = self.frame_system.gcrs2itrf(line_of_sight_gcrs, station_epoch_utc)
        distance = float(np.linalg.norm(los_itrf))
        if distance <= 0.0:
            raise RuntimeError("Cannot compute elevation for a zero-length topocentric vector.")
        up = local_up_unit_itrf(station_itrf)
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
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        start = Epoch.from_isot("1968-02-01T00:00:00", scale=TimeScale.UTC)
        end = Epoch.from_isot("1972-01-01T00:00:00", scale=TimeScale.UTC)
        return 3.0e-8 if start <= epoch_utc < end else 0.0

    def solve(self, request: LightTimeRequest) -> LightTimeSolution:
        if not isinstance(request, LightTimeRequest):
            raise TypeError("request must be a LightTimeRequest.")

        transmit_utc = request.transmit_epoch_utc
        transmit_station = self._station_state_at_utc(request, transmit_utc)
        transmit_tdb = self.time_scale_converter.convert(
            transmit_utc,
            TimeScale.TDB,
            topocentric_observer=self._topocentric_observer(request),
        )

        station_bcrs_transmit = self.frame_system.gcrs2bcrs(
            transmit_station.position_gcrs_m,
            transmit_tdb,
        )

        initial_rtt_s = 2.4
        bounce_tdb = transmit_tdb.shifted(0.5 * initial_rtt_s)
        receive_tdb = transmit_tdb.shifted(initial_rtt_s)
        previous_rtt_s = float(initial_rtt_s)
        final_state: _IterationState | None = None
        converged = False

        for iteration in range(1, _MAX_LIGHT_TIME_ITERATIONS + 1):
            receive_station = self._station_state_from_tdb(request, receive_tdb)
            receive_utc = receive_station.epoch_utc
            reflector_lcrs_bounce, _ = self._reflector_state_lcrs_m(
                request.reflector_reference_pa_m,
                bounce_tdb,
            )

            station_bcrs_receive = self.frame_system.gcrs2bcrs(
                receive_station.position_gcrs_m,
                receive_tdb,
            )
            reflector_bcrs_bounce = self.frame_system.lcrs2bcrs(
                reflector_lcrs_bounce,
                bounce_tdb,
            )

            geometric_up_m = float(np.linalg.norm(reflector_bcrs_bounce - station_bcrs_transmit))
            geometric_down_m = float(np.linalg.norm(station_bcrs_receive - reflector_bcrs_bounce))
            gravitational_up_m = float(
                self.gravitational_delay_model.path_delay_m(
                    station_bcrs_transmit,
                    reflector_bcrs_bounce,
                    bounce_tdb,
                )
            )
            gravitational_down_m = float(
                self.gravitational_delay_model.path_delay_m(
                    reflector_bcrs_bounce,
                    station_bcrs_receive,
                    bounce_tdb,
                )
            )
            elevation_up_rad = self._vacuum_elevation_rad(
                transmit_station.position_itrf_m,
                reflector_bcrs_bounce - station_bcrs_transmit,
                transmit_utc,
                transmit_tdb,
            )
            elevation_down_rad = self._vacuum_elevation_rad(
                receive_station.position_itrf_m,
                reflector_bcrs_bounce - station_bcrs_receive,
                receive_utc,
                receive_tdb,
            )
            tropo_elevation_up_rad, tropo_up_clamped = self._troposphere_evaluation_elevation(elevation_up_rad)
            tropo_elevation_down_rad, tropo_down_clamped = self._troposphere_evaluation_elevation(elevation_down_rad)
            troposphere_up_m = float(
                self.troposphere_delay_model.slant_delay_m(
                    request.troposphere_environment.troposphere_input(tropo_elevation_up_rad)
                )
            )
            troposphere_down_m = float(
                self.troposphere_delay_model.slant_delay_m(
                    request.troposphere_environment.troposphere_input(tropo_elevation_down_rad)
                )
            )

            uplink = LightTimeLeg(
                geometric_range_m=geometric_up_m,
                gravitational_path_delay_m=gravitational_up_m,
                tropospheric_path_delay_m=troposphere_up_m,
                vacuum_elevation_rad=elevation_up_rad,
                troposphere_elevation_used_rad=tropo_elevation_up_rad,
                troposphere_elevation_clamped=tropo_up_clamped,
            )
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
                transmit_epoch_tdb=transmit_tdb,
                bounce_epoch_tdb=new_bounce_tdb,
                receive_epoch_tdb=new_receive_tdb,
                uplink=uplink,
                downlink=downlink,
                iteration_count=iteration,
            )
            bounce_tdb = new_bounce_tdb
            receive_tdb = new_receive_tdb
            if abs(new_rtt_s - previous_rtt_s) < _ROUND_TRIP_TIME_TOLERANCE_S:
                converged = True
                break
            previous_rtt_s = new_rtt_s

        if final_state is None:
            raise RuntimeError("Light-time solver failed before the first iteration.")

        receive_station = self._station_state_from_tdb(request, final_state.receive_epoch_tdb)
        reflector_lcrs_bounce, reflector_displacement_lcrs_bounce = self._reflector_state_lcrs_m(
            request.reflector_reference_pa_m,
            final_state.bounce_epoch_tdb,
        )
        reflector_displacement_pa_bounce = self.frame_system.lcrs2pa(
            reflector_displacement_lcrs_bounce,
            final_state.bounce_epoch_tdb,
        )
        station_bcrs_transmit_final = station_bcrs_transmit
        station_bcrs_receive_final = self.frame_system.gcrs2bcrs(
            receive_station.position_gcrs_m,
            final_state.receive_epoch_tdb,
        )
        reflector_bcrs_bounce_final = self.frame_system.lcrs2bcrs(
            reflector_lcrs_bounce,
            final_state.bounce_epoch_tdb,
        )

        transmit_tt = self.time_scale_converter.tdb2tt(
            final_state.transmit_epoch_tdb,
            topocentric_observer=self._topocentric_observer(request),
        )
        receive_tt = self.time_scale_converter.tdb2tt(
            final_state.receive_epoch_tdb,
            topocentric_observer=self._topocentric_observer(request),
        )

        coordinate_rtt_s = final_state.transmit_epoch_tdb.seconds_until(final_state.receive_epoch_tdb)
        tt_rtt_s = transmit_tt.seconds_until(receive_tt)
        tt_minus_tdb_s = tt_rtt_s - coordinate_rtt_s
        zeta = self._pre_1972_utc_rate_offset(transmit_utc)
        observable_rtt_s = tt_rtt_s / (1.0 + zeta)

        return LightTimeSolution(
            transmit_epoch_tdb=final_state.transmit_epoch_tdb,
            bounce_epoch_tdb=final_state.bounce_epoch_tdb,
            receive_epoch_tdb=final_state.receive_epoch_tdb,
            computed_observable_round_trip_time_s=float(observable_rtt_s),
            tdb_coordinate_round_trip_time_s=float(coordinate_rtt_s),
            tt_minus_tdb_interval_correction_s=float(tt_minus_tdb_s),
            pre_1972_utc_rate_offset=float(zeta),
            uplink=final_state.uplink,
            downlink=final_state.downlink,
            station_displacement_transmit_itrf_m=transmit_station.displacement_itrf_m,
            station_displacement_receive_itrf_m=receive_station.displacement_itrf_m,
            reflector_displacement_bounce_pa_m=reflector_displacement_pa_bounce,
            station_bcrs_transmit_m=station_bcrs_transmit_final,
            station_bcrs_receive_m=station_bcrs_receive_final,
            reflector_bcrs_bounce_m=reflector_bcrs_bounce_final,
            iteration_count=final_state.iteration_count,
            light_time_converged=converged,
        )


__all__ = [
    "LightTimeLeg",
    "LightTimeRequest",
    "LightTimeSolution",
    "LightTimeSolver",
    "TroposphereEnvironment",
]
