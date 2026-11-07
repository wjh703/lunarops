"""Two-way lunar laser ranging light-time solution.

The station is propagated in ITRF and transformed to GCRS and BCRS; the reflector is rotated from the Moon PA
frame to LCRS and transformed to BCRS. Starting from the transmit epoch (UTC), the bounce and receive epochs
(TDB) are iterated until the round-trip coordinate time (uplink-only solutions: the bounce epoch) changes by less
than 1 ps, for at most 12 iterations. Each leg's travel time is its BCRS geometric range
plus the gravitational and tropospheric path delays, divided by ``c``. The computed observable is the
round-trip time on the station clock (TT interval, with the 1968-1972 UTC rate offset applied).
"""

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

    ``troposphere_input`` carries the station meteorology; each leg evaluates it with its own elevation.

    Attributes
    ----------
    reflector_reference_pa_m : ndarray, shape (3,)
        Reference reflector position in the Moon PA frame, in metres.
    transmit_epoch_utc : Epoch
        Laser transmit epoch, UTC.
    troposphere_input : TroposphereInput
        Station meteorology, geodetic latitude, ellipsoidal height and laser wavelength.
    station : StationRecord
        Station reference coordinates and velocity.
    station_key : str
        Station catalog key, passed to station-displacement models.
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
        """Build a request for a station record and per-observation meteorology.

        The troposphere latitude and height are the WGS84 geodetic coordinates of the station reference position
        propagated to ``epoch_utc``.

        Parameters
        ----------
        station : StationRecord
            Station record.
        reflector_reference_pa_m : array_like, shape (3,)
            Reference reflector position in the Moon PA frame, in metres.
        epoch_utc : Epoch
            Transmit epoch, UTC.
        station_key : str
            Station catalog key.
        pressure_hpa : float
            Surface pressure, in hPa.
        temperature_k : float
            Surface temperature, in kelvin.
        relative_humidity_percent : float
            Relative humidity, in percent.
        wavelength_um : float
            Laser wavelength, in micrometres.

        Returns
        -------
        LightTimeRequest
            The request.
        """
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
    """Path-length terms of one light-time leg (uplink or downlink).

    Attributes
    ----------
    geometric_range_m : float
        BCRS distance between the station and reflector events of the leg, in metres.
    gravitational_path_delay_m : float
        Sum of the gravitational (Shapiro) path delays, in metres.
    tropospheric_path_delay_m : float
        Sum of the tropospheric slant delays, in metres.
    vacuum_elevation_rad : float
        Elevation of the reflector above the station's local horizon (ITRF ellipsoidal up direction), without
        refraction, in radians.
    troposphere_elevation_used_rad : float or None
        Elevation at which the troposphere models were evaluated, in radians.
    troposphere_elevation_clamped : bool
        Whether the vacuum elevation was below a troposphere model's elevation floor and was replaced by it.
    """

    geometric_range_m: float
    gravitational_path_delay_m: float
    tropospheric_path_delay_m: float
    vacuum_elevation_rad: float
    troposphere_elevation_used_rad: float | None = None
    troposphere_elevation_clamped: bool = False

    @property
    def path_length_m(self) -> float:
        """Geometric range plus gravitational and tropospheric delays, in metres."""
        return self.geometric_range_m + self.gravitational_path_delay_m + self.tropospheric_path_delay_m

    @property
    def travel_time_s(self) -> float:
        """Path length divided by ``c``, in seconds of TDB coordinate time."""
        return self.path_length_m / C


@dataclass(frozen=True, slots=True, eq=False)
class UplinkSolution:
    """Solved transmit and bounce events and the uplink leg.

    Attributes
    ----------
    transmit_epoch_tdb : Epoch
        Transmit epoch, TDB.
    bounce_epoch_tdb : Epoch
        Reflection epoch at the reflector, TDB.
    uplink : LightTimeLeg
        Station-to-reflector leg.
    station_itrf_transmit_m : ndarray, shape (3,)
        Displaced station ITRF position at transmit, in metres.
    station_displacement_transmit_itrf_m : ndarray, shape (3,)
        Sum of the station displacements at transmit, ITRF, in metres.
    reflector_displacement_bounce_pa_m : ndarray, shape (3,)
        Sum of the reflector displacements at bounce, Moon PA frame, in metres.
    station_bcrs_transmit_m : ndarray, shape (3,)
        Station BCRS position at transmit, in metres.
    reflector_bcrs_bounce_m : ndarray, shape (3,)
        Displaced reflector BCRS position at bounce, in metres.
    iteration_count : int
        Number of light-time iterations performed.
    light_time_converged : bool
        Whether the iteration met its 1 ps tolerance.
    """

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
        """BCRS vector from the station at transmit to the reflector at bounce, in metres, shape ``(3,)``."""
        return self.reflector_bcrs_bounce_m - self.station_bcrs_transmit_m

    @property
    def elevation_up_deg(self) -> float:
        """Vacuum elevation of the uplink at the station, in degrees."""
        return float(np.rad2deg(self.uplink.vacuum_elevation_rad))

    def topocentric_pointing(self, frames, epoch_utc, vector_bcrs_m) -> tuple[float, float]:
        """Return the topocentric azimuth and elevation of a BCRS direction seen from the station at transmit.

        The vector is transformed BCRS to GCRS at the transmit epoch (TDB), GCRS to ITRF at ``epoch_utc``, and
        into the local east-north-up frame of the station's transmit position.

        Parameters
        ----------
        frames : ReferenceFrameSystem
            Frame transformations.
        epoch_utc : Epoch
            Station epoch, UTC, used for the GCRS-to-ITRF rotation.
        vector_bcrs_m : ndarray, shape (3,)
            BCRS direction vector, in metres.

        Returns
        -------
        tuple[float, float]
            Azimuth in degrees, from north towards east in [0, 360), and elevation in degrees, without refraction.

        Raises
        ------
        RuntimeError
            If the vector has zero length.
        """
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
        """Return the elevation of the station above the reflector's local horizon at bounce.

        The horizon is normal to the displaced reflector PA position vector (spherical Moon). The incident
        direction is the BCRS vector from the reflector at bounce to the station at transmit, transformed to
        LCRS and rotated into the Moon PA frame.

        Parameters
        ----------
        frames : ReferenceFrameSystem
            Frame transformations.
        reflector_reference_pa_m : array_like, shape (3,)
            Reference reflector position in the Moon PA frame, in metres.

        Returns
        -------
        float
            Elevation in degrees, negative when the station is below the reflector horizon.
        """
        reflector_pa_m = np.asarray(reflector_reference_pa_m, dtype=float) + self.reflector_displacement_bounce_pa_m
        incident_bcrs_m = self.station_bcrs_transmit_m - self.reflector_bcrs_bounce_m
        incident_lcrs_m = frames.relativistic_transform.bcrs_vector2lcrs(incident_bcrs_m, self.bounce_epoch_tdb)
        incident_pa_m = frames.lunar_orientation.pa_to_lcrs_matrix(self.bounce_epoch_tdb).T @ np.asarray(
            incident_lcrs_m
        )
        normal_pa = reflector_pa_m / np.linalg.norm(reflector_pa_m)
        sine_elevation = float(np.dot(incident_pa_m / np.linalg.norm(incident_pa_m), normal_pa))
        return float(np.rad2deg(np.arcsin(np.clip(sine_elevation, -1.0, 1.0))))

    def reflector_itrf_m(self, frames, bounce_epoch_utc) -> np.ndarray:
        """Return the reflector position at bounce in ITRF, in metres, shape ``(3,)``.

        The BCRS position is transformed to GCRS at the bounce epoch (TDB) and rotated to ITRF at
        ``bounce_epoch_utc``.
        """
        return frames.terrestrial_transform.gcrs2itrf(
            frames.relativistic_transform.bcrs2gcrs(self.reflector_bcrs_bounce_m, self.bounce_epoch_tdb),
            bounce_epoch_utc,
        )


@dataclass(frozen=True, slots=True, eq=False)
class LightTimeSolution(UplinkSolution):
    """Solved two-way light-time problem: uplink events plus the receive event and the computed observable.

    Attributes
    ----------
    receive_epoch_tdb : Epoch
        Receive epoch, TDB.
    receive_epoch_utc : Epoch
        Receive epoch, UTC.
    computed_observable_round_trip_time_s : float
        Computed round-trip time on the station clock, in seconds: the TT interval from transmit to receive
        divided by ``1 + pre_1972_utc_rate_offset``. The range-bias correction is not included.
    tdb_coordinate_round_trip_time_s : float
        TDB coordinate-time interval from transmit to receive, in seconds.
    tt_minus_tdb_interval_correction_s : float
        TT round-trip interval minus the TDB round-trip interval, in seconds.
    pre_1972_utc_rate_offset : float
        Fractional UTC rate offset, 3e-8 for transmit epochs from 1968-02-01 to 1972-01-01 UTC, otherwise 0.
    downlink : LightTimeLeg
        Reflector-to-station leg.
    station_displacement_receive_itrf_m : ndarray, shape (3,)
        Sum of the station displacements at receive, ITRF, in metres.
    station_bcrs_receive_m : ndarray, shape (3,)
        Station BCRS position at receive, in metres.
    """

    receive_epoch_tdb: Epoch
    receive_epoch_utc: Epoch
    computed_observable_round_trip_time_s: float
    tdb_coordinate_round_trip_time_s: float
    tt_minus_tdb_interval_correction_s: float
    pre_1972_utc_rate_offset: float
    downlink: LightTimeLeg
    station_displacement_receive_itrf_m: np.ndarray
    station_bcrs_receive_m: np.ndarray

    @property
    def event_closure_residuals_s(self) -> tuple[float, float, float]:
        """Event-interval minus leg travel-time differences (uplink, downlink, round trip), in seconds of TDB."""
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
    """Iterative two-way light-time solver for LLR.

    The solver caches station states per request and transmit/receive UTC epoch, so a request object should
    not be mutated between calls.

    Parameters
    ----------
    frame_system : ReferenceFrameSystem
        Frame and time-scale transformations, ephemeris and lunar orientation.
    gravitational_delay_models : tuple[Iers2010ShapiroDelay, ...]
        Gravitational path-delay models, summed per leg; each is evaluated at the bounce epoch (TDB).
    troposphere_delay_models : tuple[Iers2010MendesPavlisTroposphere, ...]
        Tropospheric slant-delay models, summed per leg at the leg's vacuum elevation (clamped to each model's
        elevation floor).
    station_displacement_model : StationDisplacement
        ITRF station displacement added to the propagated reference position.
    reflector_displacement_models : tuple[LunarSolidTide, ...]
        LCRS reflector displacement models, summed at the bounce epoch.
    """

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
        reflector_lcrs = self.frame_system.lunar_orientation.pa_to_lcrs_matrix(epoch_tdb) @ np.asarray(
            reflector_reference_pa_m
        )
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
        state = self._displaced_station_state(epoch_utc, reference, displacement)
        self._station_cache[key] = state
        return state

    def _displaced_station_state(self, epoch_utc, reference, displacement) -> _StationEventState:
        position = reference + displacement
        gcrs = self.frame_system.terrestrial_transform.itrf2gcrs(position, epoch_utc)
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
            # Reference coordinates avoid computing tide/loading displacements
            # for dtdb alone; time rounding can change the resulting residuals.
            return self.frame_system.terrestrial_transform.tdb_topocentric_arguments(
                request.station.itrf_xyz_at(epoch_utc),
                epoch_utc,
            )

        return arguments_at_utc

    def event_epoch_utc(self, request: LightTimeRequest, epoch_tdb: Epoch) -> Epoch:
        """Convert a TDB event epoch to UTC with the topocentric TDB-TT term of the request's station.

        Parameters
        ----------
        request : LightTimeRequest
            Request whose station reference position defines the topocentric term.
        epoch_tdb : Epoch
            Event epoch, TDB.

        Returns
        -------
        Epoch
            Event epoch, UTC.
        """
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
        gravitational_delay_m = float(
            sum(model.path_delay_m(start, end, bounce_tdb) for model in self.gravitational_delay_models)
        )
        vacuum_elevation_rad = self._vacuum_elevation_rad(
            station.position_itrf_m,
            reflector_bcrs - station_bcrs,
            station.epoch_utc,
            station_tdb,
        )
        troposphere_elevation_rad, troposphere_clamped = self._troposphere_evaluation_elevation(vacuum_elevation_rad)
        tropospheric_delay_m = float(
            sum(
                model.slant_delay_m(request.troposphere_input, elevation_rad=troposphere_elevation_rad)
                for model in self.troposphere_delay_models
            )
        )
        return LightTimeLeg(
            geometric_range_m,
            gravitational_delay_m,
            tropospheric_delay_m,
            vacuum_elevation_rad,
            troposphere_elevation_rad,
            troposphere_clamped,
        )

    @overload
    def solve(self, request: LightTimeRequest, *, two_way: Literal[True] = True) -> LightTimeSolution: ...

    @overload
    def solve(self, request: LightTimeRequest, *, two_way: Literal[False]) -> UplinkSolution: ...

    def solve(self, request: LightTimeRequest, *, two_way: bool = True) -> UplinkSolution:
        """Solve the light-time problem of one request.

        From the third iteration on, the receive-event station displacement is held at its value from the
        second iteration while the station reference position follows the receive epoch.

        Parameters
        ----------
        request : LightTimeRequest
            Station, reflector, transmit epoch (UTC) and meteorology.
        two_way : bool, default True
            Solve the full round trip; ``False`` solves the uplink only.

        Returns
        -------
        LightTimeSolution or UplinkSolution
            ``LightTimeSolution`` with the computed round-trip observable when ``two_way`` is true, otherwise
            ``UplinkSolution``. ``light_time_converged`` is false when the iteration limit was reached.
        """
        self.frame_system.relativistic_transform.begin_observation()
        transmit_utc, transmit_station, transmit_tdb, station_bcrs_transmit = self._transmit_state(request)
        initial_rtt_s = 2.0 * _INITIAL_UPLINK_LIGHT_TIME_S
        bounce_tdb = transmit_tdb.shifted(_INITIAL_UPLINK_LIGHT_TIME_S)
        receive_tdb = transmit_tdb.shifted(initial_rtt_s)
        previous_rtt_s = initial_rtt_s
        converged = False
        for iteration_count in range(1, _MAX_LIGHT_TIME_ITERATIONS + 1):
            evaluated_bounce_tdb = bounce_tdb
            if two_way:
                if iteration_count <= 2:
                    receive_station = self._station_state_from_tdb(request, receive_tdb)
                else:
                    receive_utc = self.event_epoch_utc(request, receive_tdb)
                    receive_station = self._displaced_station_state(
                        receive_utc,
                        request.station.itrf_xyz_at(receive_utc),
                        receive_station.displacement_itrf_m,
                    )
            reflector_lcrs, reflector_displacement_lcrs = self._reflector_state_lcrs_m(
                request.reflector_reference_pa_m,
                bounce_tdb,
            )
            reflector_bcrs = self.frame_system.relativistic_transform.lcrs2bcrs(reflector_lcrs, bounce_tdb)
            uplink = self._leg(
                request,
                transmit_station,
                transmit_tdb,
                station_bcrs_transmit,
                reflector_bcrs,
                bounce_tdb,
                uplink=True,
            )
            if two_way:
                station_bcrs_receive = self.frame_system.relativistic_transform.gcrs2bcrs(
                    receive_station.position_gcrs_m,
                    receive_tdb,
                )
                downlink = self._leg(
                    request,
                    receive_station,
                    receive_tdb,
                    station_bcrs_receive,
                    reflector_bcrs,
                    bounce_tdb,
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
        reflector_displacement_pa = self.frame_system.lunar_orientation.pa_to_lcrs_matrix(bounce_tdb).T @ np.asarray(
            reflector_displacement_lcrs
        )
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
            receive_epoch_utc=receive_station.epoch_utc,
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
