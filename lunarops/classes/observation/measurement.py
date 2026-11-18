"""Evaluation of one LLR normal point into a linearized equation and an optional result row.

The observable is the two-way round-trip time. The computed round-trip time is the light-time solution minus
the range-bias correction (centimetres of two-way light distance, converted with ``c``). The residual is the
one-way O-C ``0.5 * c * (observed - computed)`` in metres and the a-priori sigma is
``0.5 * c * uncertainty_two_way_s`` in metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

import erfa
import numpy as np

from lunarops.base.constants import C2, J2000_JD, C
from lunarops.classes.displacement.base import ReflectorDisplacementInput
from lunarops.classes.ephemerides.lunar_orientation import euler_313_matrix
from lunarops.classes.frames import ReferenceFrameSystem
from lunarops.classes.frames.coordinate_scaling import MOON_EXTERNAL_POTENTIAL_BODIES
from lunarops.classes.range_bias.table import RangeBiasTable

from .equations import (
    REFLECTOR_DESIGN_OUTPUT_FIELDS,
    EquationBatch,
    ObservationOptions,
    ObservationResultDetail,
    diagnostic_values,
)
from .light_time import LightTimeRequest, LightTimeSolution, LightTimeSolver
from .normal_points import NptRecord
from .resolver import ResolvedObservation

_J2000_OBLIQUITY_RAD = float(erfa.obl06(J2000_JD, 0.0))
_MEAN_LUNAR_INCLINATION_RAD = float(np.deg2rad(1.543))
_ECLIPTIC_DIFFERENCE_STEP_RAD = 1.0e-6
_MAS_PER_DEG = 3_600_000.0
_PICOSECONDS_PER_SECOND = 1.0e12


def _ecliptic_libration_partials(pa_to_lcrs: np.ndarray, orientation_partial: np.ndarray) -> np.ndarray:
    """d(range)/d(rho, I*sigma, tau) per radian for physical-libration corrections.

    The physical librations are referred to the J2000 mean ecliptic (frame bias
    neglected): ecliptic->PA = Rz(psi_e) Rx(theta_e) Rz(phi_e) with
    theta_e = I + rho, phi_e = node + sigma and psi_e compensated by -sigma so
    that tau is unchanged.  ``I*sigma`` uses the mean inclination I = 1.543 deg.
    Each perturbation is converted to a small PA-frame rotation vector w and
    projected with ``orientation_partial`` (= r_pa x d(range)/d(r_pa)).
    """
    icrf_to_pa = np.asarray(pa_to_lcrs, dtype=float).T
    ecliptic_to_icrf = erfa.rx(_J2000_OBLIQUITY_RAD, np.eye(3)).T
    matrix = icrf_to_pa @ ecliptic_to_icrf
    theta = float(np.arccos(np.clip(matrix[2, 2], -1.0, 1.0)))
    psi = float(np.arctan2(matrix[0, 2], matrix[1, 2]))
    phi = float(np.arctan2(matrix[2, 0], -matrix[2, 1]))
    h = _ECLIPTIC_DIFFERENCE_STEP_RAD
    steps = ((0.0, h, 0.0), (h, 0.0, -h), (0.0, 0.0, h))  # rho, sigma, tau
    partials = np.empty(3, dtype=float)
    for index, (d_phi, d_theta, d_psi) in enumerate(steps):
        plus = euler_313_matrix(phi + d_phi, theta + d_theta, psi + d_psi)
        minus = euler_313_matrix(phi - d_phi, theta - d_theta, psi - d_psi)
        rotation = (plus - minus) @ matrix.T / (2.0 * h)
        w = np.array([rotation[1, 2], rotation[2, 0], rotation[0, 1]])
        partials[index] = float(w @ orientation_partial)
    partials[1] /= _MEAN_LUNAR_INCLINATION_RAD
    return partials


@dataclass(frozen=True, slots=True)
class LlrObservationEvaluation:
    """Result of evaluating one normal point.

    Attributes
    ----------
    equation : EquationBatch
        One-row linearized equation.
    result_row : dict[str, object] or None
        Result-table row; ``None`` when no result detail was requested.
    below_elevation_limit : bool
        Whether the uplink or downlink vacuum elevation is below ``min_elevation_deg`` or, with the
        ``rejectClamped`` policy, a troposphere elevation was clamped.
    """

    equation: EquationBatch
    result_row: dict[str, object] | None = None
    below_elevation_limit: bool = False


class LlrObservationModel:
    """LLR observation model for one normal point at a time.

    Parameters
    ----------
    frame_system : ReferenceFrameSystem
        Frame and time-scale transformations, ephemeris and lunar orientation.
    light_time_solver : LightTimeSolver
        Two-way light-time solver.
    range_bias_model : RangeBiasTable
        Station range-bias corrections, looked up by station key and transmit UTC date, in centimetres of
        two-way light distance.
    """

    def __init__(
        self,
        frame_system: ReferenceFrameSystem,
        light_time_solver: LightTimeSolver,
        range_bias_model: RangeBiasTable,
    ) -> None:
        self.frame_system = frame_system
        self.light_time_solver = light_time_solver
        self.range_bias_model = range_bias_model

    def _reflector_position_partial_pa(
        self,
        solution: LightTimeSolution,
        uplink_unit: np.ndarray,
        downlink_unit: np.ndarray,
    ) -> np.ndarray:
        """Return d(one-way range)/d(reflector Moon PA coordinates), shape (3,).

        The mean of the BCRS uplink and downlink unit vectors is projected through the Jacobian of the BCRS
        reflector position with respect to its PA coordinates: the PA-to-LCRS rotation scaled by the LCRS-to-BCRS
        coordinate scale, plus the first-order velocity term of the Moon's BCRS motion.
        """
        unit_sum = uplink_unit + downlink_unit

        pa2lcrs = self.frame_system.lunar_orientation.pa_to_lcrs_matrix(solution.bounce_epoch_tdb)
        moon_velocity = self.frame_system.relativistic_transform.body_state(
            "MOON", solution.bounce_epoch_tdb
        ).velocity_mps
        external_potential = self.frame_system.relativistic_transform.external_gravitational_potential_m2_s2(
            "MOON",
            solution.bounce_epoch_tdb,
            MOON_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 - self.frame_system.lunar_scale.l_b_minus_l_l - external_potential / C2
        jacobian = scale * pa2lcrs - 0.5 * np.outer(moon_velocity, moon_velocity @ pa2lcrs) / C2
        return np.asarray(0.5 * unit_sum @ jacobian, dtype=float).reshape(3)

    def _lunar_h2_partial(self, solution: LightTimeSolution, reflector_pa_m, reflector_gradient_pa) -> float:
        """Return d(one-way range)/d(h2) in metres; zero when no lunar solid tide is modelled.

        The tide displacement is linear in h2, so the partial is the PA-coordinate gradient of the range
        projected on the displacement of unit h2 at the bounce epoch.
        """
        models = [
            model
            for model in self.light_time_solver.reflector_displacement_models
            if hasattr(model, "displacement_per_h2_lcrs_m")
        ]
        if not models:
            return 0.0
        pa2lcrs = self.frame_system.lunar_orientation.pa_to_lcrs_matrix(solution.bounce_epoch_tdb)
        data = ReflectorDisplacementInput(
            reference_position_lcrs_m=pa2lcrs @ np.asarray(reflector_pa_m, dtype=float),
            epoch_tdb=solution.bounce_epoch_tdb,
        )
        per_h2_lcrs = sum((model.displacement_per_h2_lcrs_m(data) for model in models), start=np.zeros(3))
        return float(reflector_gradient_pa @ (pa2lcrs.T @ per_h2_lcrs))

    def _station_position_partial_itrf(
        self,
        solution: LightTimeSolution,
        transmit_epoch_utc,
        uplink_unit: np.ndarray,
        downlink_unit: np.ndarray,
    ) -> np.ndarray:
        """d(one-way range)/d(station ITRF reference position).

        The uplink depends on the transmit-event station and the downlink on
        the receive-event station; both BCRS line-of-sight unit vectors are
        rotated into ITRF with the GCRS-to-ITRF matrix at their own event
        epochs.  The common BCRS/GCRS scale (~1e-8) is neglected.
        """
        receive_epoch_utc = solution.receive_epoch_utc
        terrestrial = self.frame_system.terrestrial_transform
        transmit_matrix = terrestrial.gcrs2itrf_matrix(transmit_epoch_utc)
        receive_matrix = terrestrial.gcrs2itrf_matrix(receive_epoch_utc)
        return np.asarray(
            -0.5 * (transmit_matrix @ uplink_unit + receive_matrix @ downlink_unit),
            dtype=float,
        ).reshape(3)

    def _partials(
        self,
        resolved_observation: ResolvedObservation,
        solution: LightTimeSolution,
        options: ObservationOptions,
    ) -> _Partials:
        """Return the partial-derivative blocks requested by ``options``; the others are empty."""
        partials = _Partials()
        if not options.includes_any_partials:
            return partials
        uplink_unit, downlink_unit = _line_of_sight_units(solution)
        if options.include_reflector_position_partials:
            partials = partials._replace(
                reflector_position_pa=self._reflector_position_partial_pa(solution, uplink_unit, downlink_unit)
            )
        if options.include_lunar_orientation_partials or options.include_lunar_tide_partials:
            reflector_gradient_pa = (
                partials.reflector_position_pa
                if options.include_reflector_position_partials
                else self._reflector_position_partial_pa(solution, uplink_unit, downlink_unit)
            )
        if options.include_lunar_orientation_partials:
            # A small PA-frame rotation eps about axis a moves the reflector by
            # eps * (a x r_pa), so d(range)/d(eps) = a . (r_pa x g) with g the
            # PA-coordinate partial.  Components are x, y (latitude) and z (psi).
            reflector_pa_m = np.asarray(resolved_observation.reflector.moon_fixed_xyz_m, dtype=float) + np.asarray(
                solution.reflector_displacement_bounce_pa_m, dtype=float
            )
            orientation_partial = np.cross(reflector_pa_m, reflector_gradient_pa)
            partials = partials._replace(
                lunar_orientation_pa=orientation_partial,
                lunar_ecliptic_libration=_ecliptic_libration_partials(
                    self.frame_system.lunar_orientation.pa_to_lcrs_matrix(solution.bounce_epoch_tdb),
                    orientation_partial,
                ),
            )
        if options.include_moon_position_partials:
            # With the Earth-Moon barycentre fixed, a change dr of the Moon-minus-Earth vector moves the
            # reflector by GM_E / (GM_E + GM_M) dr and the stations by -GM_M / (GM_E + GM_M) dr, so the
            # reflector moves by dr relative to the station of either leg. Both are BCRS displacements; the
            # LCRS coordinate scale of the reflector's selenocentric position does not apply.
            partials = partials._replace(moon_position_bcrs=0.5 * (uplink_unit + downlink_unit))
        if options.include_lunar_tide_partials:
            h2_partial = self._lunar_h2_partial(
                solution, resolved_observation.reflector.moon_fixed_xyz_m, reflector_gradient_pa
            )
            partials = partials._replace(lunar_h2=np.asarray([h2_partial]))
        if options.include_station_position_partials:
            partials = partials._replace(
                station_position_itrf=self._station_position_partial_itrf(
                    solution,
                    resolved_observation.transmit_epoch_utc,
                    uplink_unit,
                    downlink_unit,
                )
            )
        return partials

    def evaluate(
        self,
        resolved_observation: ResolvedObservation,
        options: ObservationOptions,
        *,
        result_detail: ObservationResultDetail | None = None,
    ) -> LlrObservationEvaluation:
        """Evaluate one resolved normal point.

        Parameters
        ----------
        resolved_observation : ResolvedObservation
            Normal point with its station and reflector records.
        options : ObservationOptions
            Elevation limits and the partial-derivative blocks to compute.
        result_detail : ObservationResultDetail, optional
            Detail level of the result row; ``None`` builds no row.

        Returns
        -------
        LlrObservationEvaluation
            One-row equation (one-way O-C in metres, a-priori one-way sigma in metres and the requested
            partials of the one-way range), the optional result row and the elevation flag.
        """
        record = resolved_observation.normal_point
        solution = self.light_time_solver.solve(
            LightTimeRequest.from_station(
                resolved_observation.station,
                resolved_observation.reflector.moon_fixed_xyz_m,
                resolved_observation.transmit_epoch_utc,
                station_key=resolved_observation.station_key,
                pressure_hpa=record.pressure_hpa,
                temperature_k=record.temperature_k,
                relative_humidity_percent=float(record.humidity_percent),
                wavelength_um=record.wavelength_um,
            )
        )

        bias_cm, bias_components = self.range_bias_model.lookup(
            resolved_observation.station_key,
            date.fromisoformat(resolved_observation.transmit_epoch_utc.date_iso()),
        )
        computed_before_range_bias_s = float(solution.computed_observable_round_trip_time_s)
        computed_s = computed_before_range_bias_s - (0.01 * bias_cm) / C
        oc_m = 0.5 * C * (record.round_trip_time_s - computed_s)
        partials = self._partials(resolved_observation, solution, options)
        clamped = solution.uplink.troposphere_elevation_clamped or solution.downlink.troposphere_elevation_clamped
        equation = EquationBatch.single(
            oc=oc_m,
            sigma=record.range_uncertainty_one_way_m,
            observation_id=int(record.index),
            station=resolved_observation.station_key,
            reflector=resolved_observation.reflector_key,
            epoch=resolved_observation.transmit_epoch_utc,
            receive_epoch=solution.receive_epoch_utc,
            converged=solution.light_time_converged,
            wavelength_nm=record.wavelength_nm,
            clamped=clamped,
            reflector_position_pa=() if partials.reflector_position_pa is None else partials.reflector_position_pa,
            station_position_itrf=partials.station_position_itrf,
            lunar_orientation_pa=partials.lunar_orientation_pa,
            lunar_ecliptic_libration=partials.lunar_ecliptic_libration,
            moon_position_bcrs=partials.moon_position_bcrs,
            lunar_h2=partials.lunar_h2,
            bounce_epoch_tdb=solution.bounce_epoch_tdb,
        )
        below_elevation_limit = (
            solution.elevation_up_deg < options.min_elevation_deg
            or _elevation_down_deg(solution) < options.min_elevation_deg
            or (options.low_elevation_policy == "rejectClamped" and clamped)
        )
        if result_detail is None:
            return LlrObservationEvaluation(equation, below_elevation_limit=below_elevation_limit)
        row = self._result_row(
            resolved_observation,
            solution,
            equation,
            partials.reflector_position_pa,
            computed_s=computed_s,
            oc_m=oc_m,
            below_elevation_limit=below_elevation_limit,
        )
        if result_detail is ObservationResultDetail.FULL:
            diagnostics = _DiagnosticContext(
                frames=self.frame_system,
                resolved_observation=resolved_observation,
                solution=solution,
                station_reference_itrf_m=resolved_observation.station.itrf_xyz_at(
                    resolved_observation.transmit_epoch_utc
                ),
                range_bias_diagnostics=self.range_bias_model.diagnostics(
                    resolved_observation.station_key, bias_cm, bias_components
                ),
                computed_before_range_bias_s=computed_before_range_bias_s,
                oc_one_way_m=oc_m,
                sigma_one_way_m=equation.sigma[0].item(),
                below_elevation_limit=below_elevation_limit,
                libration_rad=float(
                    self.frame_system.lunar_orientation.longitude_libration_correction_rad(solution.bounce_epoch_tdb)
                ),
            )
            row.update(diagnostic_values(FULL_DIAGNOSTIC_FIELDS, diagnostics))
        return LlrObservationEvaluation(equation, row, below_elevation_limit)

    @staticmethod
    def _result_row(
        resolved_observation: ResolvedObservation,
        solution: LightTimeSolution,
        equation: EquationBatch,
        reflector_position_partial_pa: np.ndarray | None,
        *,
        computed_s: float,
        oc_m: float,
        below_elevation_limit: bool,
    ) -> dict[str, object]:
        """Return the standard result row, with the reflector partials when they were computed."""
        record = resolved_observation.normal_point
        if below_elevation_limit:
            status = "below_elevation_limit"
        elif not solution.light_time_converged:
            status = "light_time_not_converged"
        else:
            status = "ok"
        row: dict[str, object] = {
            "normal_point_index": equation.observation_id[0],
            "station_id": record.station_code or record.station_name,
            "station_name": resolved_observation.station.name,
            "reflector_id": record.reflector_code,
            "reflector_name": resolved_observation.reflector.name,
            "observed_rtt_s": record.round_trip_time_s,
            "computed_rtt_s": computed_s,
            "oc_one_way_m": oc_m,
            "observation_sigma_one_way_m": equation.sigma[0].item(),
            "elevation_up_deg": solution.elevation_up_deg,
            "light_time_converged": solution.light_time_converged,
            "status": status,
        }
        if reflector_position_partial_pa is not None:
            row.update(
                {
                    name: float(value)
                    for name, value in zip(REFLECTOR_DESIGN_OUTPUT_FIELDS, reflector_position_partial_pa)
                }
            )
        return row


__all__ = ["LlrObservationEvaluation", "LlrObservationModel"]


def _line_of_sight_units(solution: LightTimeSolution) -> tuple[np.ndarray, np.ndarray]:
    """Return the BCRS unit vectors of the uplink and downlink lines of sight, both from station to reflector."""
    uplink_vector = solution.uplink_vector_bcrs_m
    downlink_vector = solution.reflector_bcrs_bounce_m - solution.station_bcrs_receive_m
    return (
        uplink_vector / float(np.linalg.norm(uplink_vector)),
        downlink_vector / float(np.linalg.norm(downlink_vector)),
    )


def _elevation_down_deg(solution: LightTimeSolution) -> float:
    """Vacuum elevation of the downlink at the station, in degrees."""
    return float(np.rad2deg(solution.downlink.vacuum_elevation_rad))


_NOT_REQUESTED = np.empty(0)


class _Partials(NamedTuple):
    """Partial-derivative blocks of one normal point; an empty array (``None`` for the reflector) when not requested."""

    reflector_position_pa: np.ndarray | None = None
    station_position_itrf: np.ndarray = _NOT_REQUESTED
    lunar_orientation_pa: np.ndarray = _NOT_REQUESTED
    lunar_ecliptic_libration: np.ndarray = _NOT_REQUESTED
    moon_position_bcrs: np.ndarray = _NOT_REQUESTED
    lunar_h2: np.ndarray = _NOT_REQUESTED


class _DiagnosticContext(NamedTuple):
    """Values the ``FULL_DIAGNOSTIC_FIELDS`` getters read; the remaining fields derive from ``solution``."""

    frames: ReferenceFrameSystem
    resolved_observation: ResolvedObservation
    solution: LightTimeSolution
    station_reference_itrf_m: np.ndarray
    range_bias_diagnostics: dict[str, object]
    computed_before_range_bias_s: float
    oc_one_way_m: float
    sigma_one_way_m: float
    below_elevation_limit: bool
    libration_rad: float

    @property
    def record(self) -> NptRecord:
        return self.resolved_observation.normal_point

    @property
    def coordinate_rtt_s(self) -> float:
        return float(self.solution.tdb_coordinate_round_trip_time_s)

    @property
    def tt_minus_tdb_s(self) -> float:
        return float(self.solution.tt_minus_tdb_interval_correction_s)

    @property
    def utc_rate_correction_s(self) -> float:
        return self.computed_before_range_bias_s - (self.coordinate_rtt_s + self.tt_minus_tdb_s)

    @property
    def observed_minus_computed_before_range_bias_rtt_s(self) -> float:
        return self.record.round_trip_time_s - self.computed_before_range_bias_s


def _degrees_or_none(angle_rad: float | None) -> float | None:
    return None if angle_rad is None else float(np.rad2deg(angle_rad))


FULL_DIAGNOSTIC_FIELDS = (
    ("sigma_definition", "1", lambda d: "0.5 * c * uncertainty_two_way_s"),
    ("pressure_hpa", "hPa", lambda d: d.record.pressure_hpa),
    ("temperature_c", "degC", lambda d: d.record.temperature_c),
    ("humidity_percent", "%", lambda d: d.record.humidity_percent),
    ("wavelength_nm", "nm", lambda d: d.record.wavelength_nm),
    ("computed_rtt_before_range_bias_s", "s", lambda d: d.computed_before_range_bias_s),
    ("coordinate_rtt_tdb_s", "s", lambda d: d.coordinate_rtt_s),
    ("uplink_event_closure_s", "s", lambda d: d.solution.event_closure_residuals_s[0]),
    ("downlink_event_closure_s", "s", lambda d: d.solution.event_closure_residuals_s[1]),
    ("round_trip_event_closure_s", "s", lambda d: d.solution.event_closure_residuals_s[2]),
    ("range_bias_model_label", "1", lambda d: d.range_bias_diagnostics["range_bias_model_label"]),
    ("range_bias_lookup_status", "1", lambda d: d.range_bias_diagnostics["range_bias_lookup_status"]),
    ("range_bias_station_id", "1", lambda d: d.range_bias_diagnostics["range_bias_station_id"]),
    ("range_bias_active_component_count", "1", lambda d: d.range_bias_diagnostics["range_bias_active_component_count"]),
    ("range_bias_sources", "1", lambda d: d.range_bias_diagnostics["range_bias_sources"]),
    ("range_bias_correction_two_way_cm", "cm", lambda d: d.range_bias_diagnostics["range_bias_correction_two_way_cm"]),
    ("range_bias_correction_two_way_m", "m", lambda d: d.range_bias_diagnostics["range_bias_correction_two_way_m"]),
    (
        "range_bias_correction_round_trip_time_s",
        "s",
        lambda d: d.range_bias_diagnostics["range_bias_correction_round_trip_time_s"],
    ),
    ("range_bias_correction_one_way_m", "m", lambda d: d.range_bias_diagnostics["range_bias_correction_one_way_m"]),
    ("tt_minus_tdb_correction_s", "s", lambda d: d.tt_minus_tdb_s),
    ("tt_minus_tdb_correction_one_way_m", "m", lambda d: 0.5 * C * d.tt_minus_tdb_s),
    ("pre_1972_utc_rate_offset", "1", lambda d: d.solution.pre_1972_utc_rate_offset),
    ("utc_rate_correction_s", "s", lambda d: d.utc_rate_correction_s),
    ("utc_rate_correction_one_way_m", "m", lambda d: 0.5 * C * d.utc_rate_correction_s),
    ("longitude_libration_correction_type", "1", lambda d: d.frames.lunar_orientation.correction.value),
    ("longitude_libration_correction_mas", "1", lambda d: float(np.rad2deg(d.libration_rad) * _MAS_PER_DEG)),
    ("longitude_libration_correction_rad", "rad", lambda d: d.libration_rad),
    ("lunar_relativistic_scale_convention", "1", lambda d: d.frames.lunar_scale.convention.value),
    ("l_b_minus_l_l", "1", lambda d: d.frames.lunar_scale.l_b_minus_l_l),
    ("transmit_jd1", "1", lambda d: d.solution.transmit_epoch_tdb.jd1),
    ("transmit_jd2", "1", lambda d: d.solution.transmit_epoch_tdb.jd2),
    ("transmit_scale", "1", lambda d: d.solution.transmit_epoch_tdb.scale.value),
    ("bounce_jd1", "1", lambda d: d.solution.bounce_epoch_tdb.jd1),
    ("bounce_jd2", "1", lambda d: d.solution.bounce_epoch_tdb.jd2),
    ("bounce_scale", "1", lambda d: d.solution.bounce_epoch_tdb.scale.value),
    ("receive_jd1", "1", lambda d: d.solution.receive_epoch_tdb.jd1),
    ("receive_jd2", "1", lambda d: d.solution.receive_epoch_tdb.jd2),
    ("receive_scale", "1", lambda d: d.solution.receive_epoch_tdb.scale.value),
    ("oc_rtt_s", "s", lambda d: 2.0 * d.oc_one_way_m / C),
    (
        "observed_minus_computed_before_range_bias_rtt_s",
        "s",
        lambda d: d.observed_minus_computed_before_range_bias_rtt_s,
    ),
    (
        "observed_minus_computed_before_range_bias_one_way_m",
        "m",
        lambda d: 0.5 * C * d.observed_minus_computed_before_range_bias_rtt_s,
    ),
    ("range_uncertainty_one_way_m", "m", lambda d: d.sigma_one_way_m),
    ("uncertainty_two_way_s", "s", lambda d: 2.0 * d.sigma_one_way_m / C),
    (
        "uncertainty_two_way_ps",
        "1",
        lambda d: 2.0 * _PICOSECONDS_PER_SECOND * d.sigma_one_way_m / C,
    ),
    ("geometric_range_up_m", "m", lambda d: d.solution.uplink.geometric_range_m),
    ("geometric_range_down_m", "m", lambda d: d.solution.downlink.geometric_range_m),
    ("gravitational_path_delay_up_m", "m", lambda d: d.solution.uplink.gravitational_path_delay_m),
    ("gravitational_path_delay_down_m", "m", lambda d: d.solution.downlink.gravitational_path_delay_m),
    ("tropospheric_path_delay_up_m", "m", lambda d: d.solution.uplink.tropospheric_path_delay_m),
    ("tropospheric_path_delay_down_m", "m", lambda d: d.solution.downlink.tropospheric_path_delay_m),
    (
        "tropo_elevation_up_used_deg",
        "deg",
        lambda d: _degrees_or_none(d.solution.uplink.troposphere_elevation_used_rad),
    ),
    (
        "tropo_elevation_down_used_deg",
        "deg",
        lambda d: _degrees_or_none(d.solution.downlink.troposphere_elevation_used_rad),
    ),
    ("tropo_up_clamped", "1", lambda d: d.solution.uplink.troposphere_elevation_clamped),
    ("tropo_down_clamped", "1", lambda d: d.solution.downlink.troposphere_elevation_clamped),
    (
        "tropo_clamped",
        "1",
        lambda d: bool(
            d.solution.uplink.troposphere_elevation_clamped or d.solution.downlink.troposphere_elevation_clamped
        ),
    ),
    ("elevation_down_deg", "deg", lambda d: _elevation_down_deg(d.solution)),
    (
        "station_displacement_transmit_itrf_x_m",
        "m",
        lambda d: float(d.solution.station_displacement_transmit_itrf_m[0]),
    ),
    (
        "station_displacement_transmit_itrf_y_m",
        "m",
        lambda d: float(d.solution.station_displacement_transmit_itrf_m[1]),
    ),
    (
        "station_displacement_transmit_itrf_z_m",
        "m",
        lambda d: float(d.solution.station_displacement_transmit_itrf_m[2]),
    ),
    ("station_displacement_receive_itrf_x_m", "m", lambda d: float(d.solution.station_displacement_receive_itrf_m[0])),
    ("station_displacement_receive_itrf_y_m", "m", lambda d: float(d.solution.station_displacement_receive_itrf_m[1])),
    ("station_displacement_receive_itrf_z_m", "m", lambda d: float(d.solution.station_displacement_receive_itrf_m[2])),
    ("reflector_displacement_bounce_pa_x_m", "m", lambda d: float(d.solution.reflector_displacement_bounce_pa_m[0])),
    ("reflector_displacement_bounce_pa_y_m", "m", lambda d: float(d.solution.reflector_displacement_bounce_pa_m[1])),
    ("reflector_displacement_bounce_pa_z_m", "m", lambda d: float(d.solution.reflector_displacement_bounce_pa_m[2])),
    ("iteration_count", "1", lambda d: d.solution.iteration_count),
    ("station_reference_itrf_x_m", "m", lambda d: float(d.station_reference_itrf_m[0])),
    ("station_reference_itrf_y_m", "m", lambda d: float(d.station_reference_itrf_m[1])),
    ("station_reference_itrf_z_m", "m", lambda d: float(d.station_reference_itrf_m[2])),
    ("valid_geometry", "1", lambda d: not d.below_elevation_limit),
    ("below_elevation_limit", "1", lambda d: d.below_elevation_limit),
    ("station_catalog_key", "1", lambda d: d.resolved_observation.station_key),
    ("reflector_catalog_key", "1", lambda d: d.resolved_observation.reflector_key),
)

FULL_DIAGNOSTIC_UNITS = {name: unit for name, unit, _ in FULL_DIAGNOSTIC_FIELDS}
