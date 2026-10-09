"""Single LLR measurement evaluation from resolved input to equation or row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from lunarops.base.constants import C2, C
from lunarops.classes.ephemerides import Ephemeris
from lunarops.classes.frames import ReferenceFrameSystem
from lunarops.classes.range_bias.models import RangeBiasCorrection, RangeBiasModel, RangeBiasRequest
from lunarops.classes.relativistic.constants import MOON_EXTERNAL_POTENTIAL_BODIES

from .equations import (
    REFLECTOR_DESIGN_OUTPUT_FIELDS,
    ObservationEquation,
    ObservationResultDetail,
)
from .light_time import LightTimeRequest, LightTimeSolution, LightTimeSolver
from .resolver import ResolvedObservation


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
        range_bias_model: RangeBiasModel,
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
    ) -> np.ndarray:
        uplink_vector = solution.uplink_vector_bcrs_m
        downlink_vector = solution.reflector_bcrs_bounce_m - solution.station_bcrs_receive_m
        uplink_range = max(float(np.linalg.norm(uplink_vector)), 1.0e-30)
        downlink_range = max(float(np.linalg.norm(downlink_vector)), 1.0e-30)
        unit_sum = uplink_vector / uplink_range + downlink_vector / downlink_range

        pa2lcrs = self.frame_system.lunar_orientation.pa_to_lcrs_matrix(solution.bounce_epoch_tdb)
        moon_velocity = self.ephemeris.body_state_bcrs("MOON", solution.bounce_epoch_tdb).velocity_mps
        external_potential = self.frame_system.external_gravitational_potential_m2_s2(
            "MOON",
            solution.bounce_epoch_tdb,
            MOON_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 - self.frame_system.lunar_scale.l_b_minus_l_l - external_potential / C2
        jacobian = scale * pa2lcrs - 0.5 * np.outer(moon_velocity, moon_velocity @ pa2lcrs) / C2
        return np.asarray(0.5 * unit_sum @ jacobian, dtype=float).reshape(3)

    def evaluate(
        self,
        resolved_observation: ResolvedObservation,
        *,
        min_elevation_deg: float,
        include_reflector_position_partials: bool = False,
        result_detail: ObservationResultDetail | None = None,
    ) -> LlrObservationEvaluation:
        min_elevation = float(min_elevation_deg)
        if not np.isfinite(min_elevation):
            raise ValueError("min_elevation_deg must be finite.")
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
        range_bias_request = RangeBiasRequest(
            station_identifiers=resolved_observation.station_identity_candidates,
            observation_epoch_utc=resolved_observation.transmit_epoch_utc,
        )
        range_bias_correction = self.range_bias_model.evaluate(range_bias_request)
        computed_before_range_bias_s = float(solution.computed_observable_round_trip_time_s)
        computed_s = range_bias_correction.apply_to_computed_round_trip_time_s(computed_before_range_bias_s)
        observed_minus_computed_before_range_bias_rtt_s = (
            float(record.observed_round_trip_time_s) - computed_before_range_bias_s
        )
        oc_s = float(record.observed_round_trip_time_s) - computed_s
        observed_minus_computed_before_range_bias_one_way_m = 0.5 * C * observed_minus_computed_before_range_bias_rtt_s
        oc_m = 0.5 * C * oc_s
        reflector_position_partial_pa = (
            self._reflector_position_partial_pa(solution) if include_reflector_position_partials else None
        )
        design_partials = (
            {} if reflector_position_partial_pa is None else {"reflector_position_pa": reflector_position_partial_pa}
        )
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
        )
        below_elevation_limit = elevation_up_deg < min_elevation or elevation_down_deg < min_elevation
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
            "obs_time_utc": equation.transmit_epoch_utc.isot(),
            "normal_point_index": equation.observation_id,
            "station_id": record.station_code or record.station_name,
            "station_name": station.name,
            "reflector_id": record.reflector_code,
            "reflector_name": resolved_observation.reflector.name,
            "observed_rtt_s": record.observed_round_trip_time_s,
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
                range_bias_correction,
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
        range_bias_correction: RangeBiasCorrection,
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
        row.update(
            {
                "sigma_definition": "0.5 * c * uncertainty_two_way_s",
                "pressure_hpa": record.pressure_hpa,
                "temperature_c": record.temperature_c,
                "humidity_percent": record.humidity_percent,
                "wavelength_nm": record.wavelength_nm,
                "computed_rtt_before_range_bias_s": computed_before_range_bias_s,
                "coordinate_rtt_tdb_s": coordinate_rtt_s,
                "range_bias_model_label": range_bias_correction.model_label,
                "range_bias_lookup_status": range_bias_correction.lookup.status.value,
                "range_bias_station_id": range_bias_correction.lookup.matched_station_id,
                "range_bias_active_component_count": len(range_bias_correction.lookup.active_components),
                "range_bias_sources": range_bias_correction.lookup.sources,
                "range_bias_correction_two_way_cm": range_bias_correction.correction_two_way_cm,
                "range_bias_correction_two_way_m": range_bias_correction.correction_two_way_m,
                "range_bias_correction_round_trip_time_s": range_bias_correction.correction_round_trip_time_s,
                "range_bias_correction_one_way_m": range_bias_correction.correction_one_way_m,
                "tt_minus_tdb_correction_s": tt_minus_tdb_s,
                "tt_minus_tdb_correction_one_way_m": 0.5 * C * tt_minus_tdb_s,
                "pre_1972_utc_rate_offset": solution.pre_1972_utc_rate_offset,
                "utc_rate_correction_s": utc_rate_correction_s,
                "utc_rate_correction_one_way_m": 0.5 * C * utc_rate_correction_s,
                "longitude_libration_correction_type": lunar_orientation.correction.value,
                "longitude_libration_correction_mas": float(np.rad2deg(libration_rad) * 3_600_000.0),
                "longitude_libration_correction_rad": libration_rad,
                "lunar_relativistic_scale_convention": self.frame_system.lunar_scale.convention.value,
                "l_b_minus_l_l": self.frame_system.lunar_scale.l_b_minus_l_l,
                "transmit_jd1": solution.transmit_epoch_tdb.jd1,
                "transmit_jd2": solution.transmit_epoch_tdb.jd2,
                "transmit_scale": solution.transmit_epoch_tdb.scale.value,
                "bounce_jd1": solution.bounce_epoch_tdb.jd1,
                "bounce_jd2": solution.bounce_epoch_tdb.jd2,
                "bounce_scale": solution.bounce_epoch_tdb.scale.value,
                "receive_jd1": solution.receive_epoch_tdb.jd1,
                "receive_jd2": solution.receive_epoch_tdb.jd2,
                "receive_scale": solution.receive_epoch_tdb.scale.value,
                "oc_rtt_s": 2.0 * cast(float, row["oc_one_way_m"]) / C,
                "observed_minus_computed_before_range_bias_rtt_s": observed_minus_computed_before_range_bias_rtt_s,
                "observed_minus_computed_before_range_bias_one_way_m": (
                    observed_minus_computed_before_range_bias_one_way_m
                ),
                "range_uncertainty_one_way_m": row["observation_sigma_one_way_m"],
                "uncertainty_two_way_s": 2.0 * cast(float, row["observation_sigma_one_way_m"]) / C,
                "uncertainty_two_way_ps": 2.0e12 * cast(float, row["observation_sigma_one_way_m"]) / C,
                "geometric_range_up_m": solution.uplink.geometric_range_m,
                "geometric_range_down_m": solution.downlink.geometric_range_m,
                "gravitational_path_delay_up_m": solution.uplink.gravitational_path_delay_m,
                "gravitational_path_delay_down_m": solution.downlink.gravitational_path_delay_m,
                "tropospheric_path_delay_up_m": solution.uplink.tropospheric_path_delay_m,
                "tropospheric_path_delay_down_m": solution.downlink.tropospheric_path_delay_m,
                "tropo_elevation_up_used_deg": None if tropo_up_used is None else float(np.rad2deg(tropo_up_used)),
                "tropo_elevation_down_used_deg": None
                if tropo_down_used is None
                else float(np.rad2deg(tropo_down_used)),
                "tropo_up_clamped": solution.uplink.troposphere_elevation_clamped,
                "tropo_down_clamped": solution.downlink.troposphere_elevation_clamped,
                "tropo_clamped": bool(
                    solution.uplink.troposphere_elevation_clamped or solution.downlink.troposphere_elevation_clamped
                ),
                "elevation_down_deg": elevation_down_deg,
                "station_displacement_transmit_itrf_x_m": float(solution.station_displacement_transmit_itrf_m[0]),
                "station_displacement_transmit_itrf_y_m": float(solution.station_displacement_transmit_itrf_m[1]),
                "station_displacement_transmit_itrf_z_m": float(solution.station_displacement_transmit_itrf_m[2]),
                "station_displacement_receive_itrf_x_m": float(solution.station_displacement_receive_itrf_m[0]),
                "station_displacement_receive_itrf_y_m": float(solution.station_displacement_receive_itrf_m[1]),
                "station_displacement_receive_itrf_z_m": float(solution.station_displacement_receive_itrf_m[2]),
                "reflector_displacement_bounce_pa_x_m": float(solution.reflector_displacement_bounce_pa_m[0]),
                "reflector_displacement_bounce_pa_y_m": float(solution.reflector_displacement_bounce_pa_m[1]),
                "reflector_displacement_bounce_pa_z_m": float(solution.reflector_displacement_bounce_pa_m[2]),
                "iteration_count": solution.iteration_count,
                "station_reference_itrf_x_m": float(station_itrf_m[0]),
                "station_reference_itrf_y_m": float(station_itrf_m[1]),
                "station_reference_itrf_z_m": float(station_itrf_m[2]),
                "valid_geometry": not below_elevation_limit,
                "below_elevation_limit": below_elevation_limit,
                "station_catalog_key": resolved_observation.station_key,
                "reflector_catalog_key": resolved_observation.reflector_key,
            }
        )


__all__ = ["LlrObservationEvaluation", "LlrObservationModel"]
