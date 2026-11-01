"""Single LLR measurement evaluation from resolved input to equation or row."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import NamedTuple, cast

import erfa
import numpy as np

from lunarops.base.constants import C2, C
from lunarops.classes.ephemerides import Ephemeris
from lunarops.classes.ephemerides.lunar_orientation import LunarOrientationProvider
from lunarops.classes.frames import ReferenceFrameSystem
from lunarops.classes.frames.coordinate_scaling import MOON_EXTERNAL_POTENTIAL_BODIES
from lunarops.classes.range_bias.table import RangeBiasTable

from .equations import (
    REFLECTOR_DESIGN_OUTPUT_FIELDS,
    ObservationEquation,
    ObservationOptions,
    ObservationResultDetail,
)
from .light_time import LightTimeRequest, LightTimeSolution, LightTimeSolver
from .normal_points import NptRecord
from .resolver import ResolvedObservation

_J2000_OBLIQUITY_RAD = float(erfa.obl06(2_451_545.0, 0.0))
_MEAN_LUNAR_INCLINATION_RAD = float(np.deg2rad(1.543))
_ECLIPTIC_DIFFERENCE_STEP_RAD = 1.0e-6


def _ecliptic_to_pa_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    return erfa.rz(psi, np.eye(3)) @ erfa.rx(theta, np.eye(3)) @ erfa.rz(phi, np.eye(3))


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
        plus = _ecliptic_to_pa_matrix(phi + d_phi, theta + d_theta, psi + d_psi)
        minus = _ecliptic_to_pa_matrix(phi - d_phi, theta - d_theta, psi - d_psi)
        rotation = (plus - minus) @ matrix.T / (2.0 * h)
        w = np.array([rotation[1, 2], rotation[2, 0], rotation[0, 1]])
        partials[index] = float(w @ orientation_partial)
    partials[1] /= _MEAN_LUNAR_INCLINATION_RAD
    return partials


@dataclass(frozen=True, slots=True)
class LlrObservationEvaluation:
    equation: ObservationEquation
    result_row: dict[str, object] | None = None
    below_elevation_limit: bool = False


class LlrObservationModel:
    def __init__(
        self,
        frame_system: ReferenceFrameSystem,
        light_time_solver: LightTimeSolver,
        range_bias_model: RangeBiasTable,
    ) -> None:
        self.frame_system = frame_system
        self.light_time_solver = light_time_solver
        self.range_bias_model = range_bias_model

    @property
    def ephemeris(self) -> Ephemeris:
        return self.frame_system.ephemeris

    def _reflector_position_partial_pa(
        self,
        solution: LightTimeSolution,
        uplink_unit: np.ndarray,
        downlink_unit: np.ndarray,
    ) -> np.ndarray:
        unit_sum = uplink_unit + downlink_unit

        pa2lcrs = self.frame_system.lunar_orientation.pa_to_lcrs_matrix(solution.bounce_epoch_tdb)
        moon_velocity = self.ephemeris.body_state_bcrs("MOON", solution.bounce_epoch_tdb).velocity_mps
        external_potential = self.frame_system.relativistic_transform.external_gravitational_potential_m2_s2(
            "MOON",
            solution.bounce_epoch_tdb,
            MOON_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 - self.frame_system.lunar_scale.l_b_minus_l_l - external_potential / C2
        jacobian = scale * pa2lcrs - 0.5 * np.outer(moon_velocity, moon_velocity @ pa2lcrs) / C2
        return np.asarray(0.5 * unit_sum @ jacobian, dtype=float).reshape(3)

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
        receive_epoch_utc = transmit_epoch_utc.shifted(solution.tdb_coordinate_round_trip_time_s)
        terrestrial = self.frame_system.terrestrial_transform
        transmit_matrix = terrestrial.gcrs2itrf_matrix(transmit_epoch_utc)
        receive_matrix = terrestrial.gcrs2itrf_matrix(receive_epoch_utc)
        return np.asarray(
            -0.5 * (transmit_matrix @ uplink_unit + receive_matrix @ downlink_unit),
            dtype=float,
        ).reshape(3)

    def evaluate(
        self,
        resolved_observation: ResolvedObservation,
        options: ObservationOptions,
        *,
        result_detail: ObservationResultDetail | None = None,
    ) -> LlrObservationEvaluation:
        record = resolved_observation.normal_point
        station = resolved_observation.station
        solution = self.light_time_solver.solve(
            LightTimeRequest.from_station(
                station,
                resolved_observation.reflector.moon_fixed_xyz_m,
                resolved_observation.transmit_epoch_utc,
                station_key=resolved_observation.station_key,
                pressure_hpa=record.pressure_hpa,
                temperature_k=record.temperature_k,
                relative_humidity_percent=float(record.humidity_percent),
                wavelength_um=record.wavelength_um,
            )
        )

        elevation_up_deg = solution.elevation_up_deg
        elevation_down_deg = float(np.rad2deg(solution.downlink.vacuum_elevation_rad))
        bias_cm, bias_components = self.range_bias_model.lookup(
            resolved_observation.station_key, date.fromisoformat(resolved_observation.transmit_epoch_utc.date_iso()),
        )
        computed_before_range_bias_s = float(solution.computed_observable_round_trip_time_s)
        computed_s = computed_before_range_bias_s - (0.01 * bias_cm) / C
        observed_minus_computed_before_range_bias_rtt_s = (
            record.round_trip_time_s - computed_before_range_bias_s
        )
        oc_s = record.round_trip_time_s - computed_s
        observed_minus_computed_before_range_bias_one_way_m = 0.5 * C * observed_minus_computed_before_range_bias_rtt_s
        oc_m = 0.5 * C * oc_s
        if (options.include_reflector_position_partials or options.include_station_position_partials
                or options.include_lunar_orientation_partials):
            uplink_vector = solution.uplink_vector_bcrs_m
            downlink_vector = solution.reflector_bcrs_bounce_m - solution.station_bcrs_receive_m
            uplink_unit = uplink_vector / float(np.linalg.norm(uplink_vector))
            downlink_unit = downlink_vector / float(np.linalg.norm(downlink_vector))
        reflector_position_partial_pa = (
            self._reflector_position_partial_pa(solution, uplink_unit, downlink_unit) if options.include_reflector_position_partials else None
        )
        design_partials = (
            {} if reflector_position_partial_pa is None else {"reflector_position_pa": reflector_position_partial_pa}
        )
        if options.include_lunar_orientation_partials:
            # A small PA-frame rotation eps about axis a moves the reflector by
            # eps * (a x r_pa), so d(range)/d(eps) = a . (r_pa x g) with g the
            # PA-coordinate partial.  Components are x, y (latitude) and z (psi).
            reflector_gradient_pa = (
                reflector_position_partial_pa
                if reflector_position_partial_pa is not None
                else self._reflector_position_partial_pa(solution, uplink_unit, downlink_unit)
            )
            reflector_pa_m = (
                np.asarray(resolved_observation.reflector.moon_fixed_xyz_m, dtype=float)
                + np.asarray(solution.reflector_displacement_bounce_pa_m, dtype=float)
            )
            orientation_partial = np.cross(reflector_pa_m, reflector_gradient_pa)
            design_partials["lunar_orientation_pa"] = orientation_partial
            design_partials["lunar_ecliptic_libration"] = _ecliptic_libration_partials(
                self.frame_system.lunar_orientation.pa_to_lcrs_matrix(solution.bounce_epoch_tdb),
                orientation_partial,
            )
        if options.include_station_position_partials:
            design_partials["station_position_itrf"] = self._station_position_partial_itrf(
                solution,
                resolved_observation.transmit_epoch_utc,
                uplink_unit, downlink_unit,
            )
        clamped = solution.uplink.troposphere_elevation_clamped or solution.downlink.troposphere_elevation_clamped
        equation = ObservationEquation(
            observed_minus_computed_one_way_m=oc_m,
            sigma_one_way_m=record.range_uncertainty_one_way_m,
            design_partials=design_partials,
            observation_id=int(record.index),
            station_key=resolved_observation.station_key,
            reflector_key=resolved_observation.reflector_key,
            transmit_epoch_utc=resolved_observation.transmit_epoch_utc,
            light_time_converged=solution.light_time_converged,
            wavelength_nm=record.wavelength_nm,
            troposphere_clamped=clamped,
        )
        below_elevation_limit = (
            elevation_up_deg < options.min_elevation_deg
            or elevation_down_deg < options.min_elevation_deg
            or (options.low_elevation_policy == "rejectClamped" and clamped)
        )
        if result_detail is None:
            return LlrObservationEvaluation(equation, below_elevation_limit=below_elevation_limit)

        level = result_detail
        status = (
            "below_elevation_limit"
            if below_elevation_limit
            else "light_time_not_converged"
            if not solution.light_time_converged
            else "ok"
        )
        row: dict[str, object] = {
            "normal_point_index": equation.observation_id,
            "station_id": record.station_code or record.station_name,
            "station_name": station.name,
            "reflector_id": record.reflector_code,
            "reflector_name": resolved_observation.reflector.name,
            "observed_rtt_s": record.round_trip_time_s,
            "computed_rtt_s": computed_s,
            "oc_one_way_m": oc_m,
            "observation_sigma_one_way_m": equation.sigma_one_way_m,
            "elevation_up_deg": elevation_up_deg,
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
        if level is ObservationResultDetail.FULL:
            self._add_full_diagnostics(
                row,
                resolved_observation,
                solution,
                station.itrf_xyz_at(resolved_observation.transmit_epoch_utc),
                self.range_bias_model.diagnostics(resolved_observation.station_key, bias_cm, bias_components),
                computed_before_range_bias_s,
                observed_minus_computed_before_range_bias_rtt_s,
                observed_minus_computed_before_range_bias_one_way_m,
                elevation_down_deg,
                below_elevation_limit,
            )
        return LlrObservationEvaluation(equation, row, below_elevation_limit)

    def _add_full_diagnostics(
        self,
        row: dict[str, object],
        resolved_observation: ResolvedObservation,
        solution: LightTimeSolution,
        station_itrf_m: np.ndarray,
        range_bias_diagnostics: dict[str, object],
        computed_before_range_bias_s: float,
        observed_minus_computed_before_range_bias_rtt_s: float,
        observed_minus_computed_before_range_bias_one_way_m: float,
        elevation_down_deg: float,
        below_elevation_limit: bool,
    ) -> None:
        record = resolved_observation.normal_point
        coordinate_rtt_s = float(solution.tdb_coordinate_round_trip_time_s)
        tt_minus_tdb_s = float(solution.tt_minus_tdb_interval_correction_s)
        utc_rate_correction_s = computed_before_range_bias_s - (coordinate_rtt_s + tt_minus_tdb_s)
        lunar_orientation = self.frame_system.lunar_orientation
        libration_rad = float(lunar_orientation.longitude_libration_correction_rad(solution.bounce_epoch_tdb))
        tropo_up_used = solution.uplink.troposphere_elevation_used_rad
        tropo_down_used = solution.downlink.troposphere_elevation_used_rad
        diagnostics = _DiagnosticContext(
            row, self.frame_system, record, resolved_observation,
            solution, station_itrf_m, range_bias_diagnostics,
            computed_before_range_bias_s, observed_minus_computed_before_range_bias_rtt_s, observed_minus_computed_before_range_bias_one_way_m,
            elevation_down_deg, below_elevation_limit, coordinate_rtt_s,
            tt_minus_tdb_s, utc_rate_correction_s, lunar_orientation,
            libration_rad, tropo_up_used, tropo_down_used,
        )
        row.update((name, getter(diagnostics)) for name, _, getter in FULL_DIAGNOSTIC_FIELDS)


__all__ = ["LlrObservationEvaluation", "LlrObservationModel"]


class _DiagnosticContext(NamedTuple):
    row: dict[str, object]
    frames: ReferenceFrameSystem
    record: NptRecord
    resolved_observation: ResolvedObservation
    solution: LightTimeSolution
    station_itrf_m: np.ndarray
    range_bias_diagnostics: dict[str, object]
    computed_before_range_bias_s: float
    observed_minus_computed_before_range_bias_rtt_s: float
    observed_minus_computed_before_range_bias_one_way_m: float
    elevation_down_deg: float
    below_elevation_limit: bool
    coordinate_rtt_s: float
    tt_minus_tdb_s: float
    utc_rate_correction_s: float
    lunar_orientation: LunarOrientationProvider
    libration_rad: float
    tropo_up_used: float | None
    tropo_down_used: float | None


FULL_DIAGNOSTIC_FIELDS = (
    ("sigma_definition", "1", lambda d: '0.5 * c * uncertainty_two_way_s'),
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
    ("range_bias_correction_round_trip_time_s", "s", lambda d: d.range_bias_diagnostics["range_bias_correction_round_trip_time_s"]),
    ("range_bias_correction_one_way_m", "m", lambda d: d.range_bias_diagnostics["range_bias_correction_one_way_m"]),
    ("tt_minus_tdb_correction_s", "s", lambda d: d.tt_minus_tdb_s),
    ("tt_minus_tdb_correction_one_way_m", "m", lambda d: 0.5 * C * d.tt_minus_tdb_s),
    ("pre_1972_utc_rate_offset", "1", lambda d: d.solution.pre_1972_utc_rate_offset),
    ("utc_rate_correction_s", "s", lambda d: d.utc_rate_correction_s),
    ("utc_rate_correction_one_way_m", "m", lambda d: 0.5 * C * d.utc_rate_correction_s),
    ("longitude_libration_correction_type", "1", lambda d: d.lunar_orientation.correction.value),
    ("longitude_libration_correction_mas", "1", lambda d: float(np.rad2deg(d.libration_rad) * 3600000.0)),
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
    ("oc_rtt_s", "s", lambda d: 2.0 * cast(float, d.row['oc_one_way_m']) / C),
    ("observed_minus_computed_before_range_bias_rtt_s", "s", lambda d: d.observed_minus_computed_before_range_bias_rtt_s),
    ("observed_minus_computed_before_range_bias_one_way_m", "m", lambda d: d.observed_minus_computed_before_range_bias_one_way_m),
    ("range_uncertainty_one_way_m", "m", lambda d: d.row['observation_sigma_one_way_m']),
    ("uncertainty_two_way_s", "s", lambda d: 2.0 * cast(float, d.row['observation_sigma_one_way_m']) / C),
    ("uncertainty_two_way_ps", "1", lambda d: 2000000000000.0 * cast(float, d.row['observation_sigma_one_way_m']) / C),
    ("geometric_range_up_m", "m", lambda d: d.solution.uplink.geometric_range_m),
    ("geometric_range_down_m", "m", lambda d: d.solution.downlink.geometric_range_m),
    ("gravitational_path_delay_up_m", "m", lambda d: d.solution.uplink.gravitational_path_delay_m),
    ("gravitational_path_delay_down_m", "m", lambda d: d.solution.downlink.gravitational_path_delay_m),
    ("tropospheric_path_delay_up_m", "m", lambda d: d.solution.uplink.tropospheric_path_delay_m),
    ("tropospheric_path_delay_down_m", "m", lambda d: d.solution.downlink.tropospheric_path_delay_m),
    ("tropo_elevation_up_used_deg", "deg", lambda d: None if d.tropo_up_used is None else float(np.rad2deg(d.tropo_up_used))),
    ("tropo_elevation_down_used_deg", "deg", lambda d: None if d.tropo_down_used is None else float(np.rad2deg(d.tropo_down_used))),
    ("tropo_up_clamped", "1", lambda d: d.solution.uplink.troposphere_elevation_clamped),
    ("tropo_down_clamped", "1", lambda d: d.solution.downlink.troposphere_elevation_clamped),
    ("tropo_clamped", "1", lambda d: bool(d.solution.uplink.troposphere_elevation_clamped or d.solution.downlink.troposphere_elevation_clamped)),
    ("elevation_down_deg", "deg", lambda d: d.elevation_down_deg),
    ("station_displacement_transmit_itrf_x_m", "m", lambda d: float(d.solution.station_displacement_transmit_itrf_m[0])),
    ("station_displacement_transmit_itrf_y_m", "m", lambda d: float(d.solution.station_displacement_transmit_itrf_m[1])),
    ("station_displacement_transmit_itrf_z_m", "m", lambda d: float(d.solution.station_displacement_transmit_itrf_m[2])),
    ("station_displacement_receive_itrf_x_m", "m", lambda d: float(d.solution.station_displacement_receive_itrf_m[0])),
    ("station_displacement_receive_itrf_y_m", "m", lambda d: float(d.solution.station_displacement_receive_itrf_m[1])),
    ("station_displacement_receive_itrf_z_m", "m", lambda d: float(d.solution.station_displacement_receive_itrf_m[2])),
    ("reflector_displacement_bounce_pa_x_m", "m", lambda d: float(d.solution.reflector_displacement_bounce_pa_m[0])),
    ("reflector_displacement_bounce_pa_y_m", "m", lambda d: float(d.solution.reflector_displacement_bounce_pa_m[1])),
    ("reflector_displacement_bounce_pa_z_m", "m", lambda d: float(d.solution.reflector_displacement_bounce_pa_m[2])),
    ("iteration_count", "1", lambda d: d.solution.iteration_count),
    ("station_reference_itrf_x_m", "m", lambda d: float(d.station_itrf_m[0])),
    ("station_reference_itrf_y_m", "m", lambda d: float(d.station_itrf_m[1])),
    ("station_reference_itrf_z_m", "m", lambda d: float(d.station_itrf_m[2])),
    ("valid_geometry", "1", lambda d: not d.below_elevation_limit),
    ("below_elevation_limit", "1", lambda d: d.below_elevation_limit),
    ("station_catalog_key", "1", lambda d: d.resolved_observation.station_key),
    ("reflector_catalog_key", "1", lambda d: d.resolved_observation.reflector_key),
)

FULL_DIAGNOSTIC_UNITS = {name: unit for name, unit, _ in FULL_DIAGNOSTIC_FIELDS}
