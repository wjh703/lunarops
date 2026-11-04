"""Two-way light time, its uplink-only form and the per-leg model contributions."""

from typing import Any, cast

import numpy as np
import pytest

from lunarops.base.constants import C
from lunarops.classes.delays import TroposphereInput
from lunarops.classes.delays.troposphere import Iers2010MendesPavlisTroposphere
from lunarops.classes.displacement import Iers2010SolidEarthTide
from lunarops.classes.displacement.terrestrial_geometry import itrf2geodetic
from lunarops.classes.observation import (
    LightTimeRequest,
    LlrObservationPredictor,
    NptDataset,
    ObservationOptions,
    ObservationResultDetail,
    PredictionCriteria,
    PredictionMeteorology,
)
from tests.support.observation import (
    TRANSMIT_UTC,
    npt_record,
    observation_processor,
    static_frames,
    station_facing_the_moon,
)
from tests.tolerances import DISPLACEMENT_M, LIGHT_TIME_S, Tolerance, assert_close

ALL_ELEVATIONS = ObservationOptions(min_elevation_deg=-90.0)


def _full_row(processor, options=ALL_ELEVATIONS) -> dict[str, Any]:
    observation = processor.resolver.resolve(npt_record())
    return cast(dict[str, Any], processor.observation_model.evaluate(
        observation, options=options, result_detail=ObservationResultDetail.FULL).result_row)


def _request(processor):
    observation = processor.resolver.resolve(npt_record())
    record = npt_record()
    return LightTimeRequest.from_station(
        observation.station, observation.reflector.moon_fixed_xyz_m, TRANSMIT_UTC, station_key="APOLLO",
        pressure_hpa=record.pressure_hpa, temperature_k=record.temperature_k,
        relative_humidity_percent=record.humidity_percent, wavelength_um=record.wavelength_nm / 1000.0)


def test_processor_builds_one_equation_and_a_matching_result_row():
    processor = observation_processor()
    equation = processor.equations(NptDataset([npt_record()]), options=ObservationOptions(
        min_elevation_deg=-90.0, include_reflector_position_partials=True))
    row = processor.rows(NptDataset([npt_record()]), options=ALL_ELEVATIONS)[0]

    assert (equation.observation_id[0], equation.station[0], equation.reflector[0]) == (4, "APOLLO", "APOLLO15")
    assert equation.converged[0]
    # One-way sigma of a two-way time uncertainty.
    assert equation.sigma[0] == pytest.approx(0.5 * C * 100.0e-12)
    assert equation.reflector_position_pa[0].shape == (3,)
    assert row["status"] == "ok"
    assert_close(row["oc_one_way_m"], equation.oc[0], DISPLACEMENT_M)


def test_uplink_solution_equals_the_uplink_of_the_two_way_solution():
    processor = observation_processor()
    solver = processor.observation_model.light_time_solver
    request = _request(processor)
    full, uplink = solver.solve(request), solver.solve(request, two_way=False)

    assert uplink.light_time_converged
    assert_close(uplink.bounce_epoch_tdb.seconds_until(full.bounce_epoch_tdb), 0.0, LIGHT_TIME_S)
    assert_close(uplink.uplink.geometric_range_m, full.uplink.geometric_range_m, Tolerance("RANGE_M", atol=1e-4))
    assert_close(uplink.station_bcrs_transmit_m, full.station_bcrs_transmit_m, Tolerance("STATION_M", atol=1e-8))
    assert_close(uplink.reflector_bcrs_bounce_m, full.reflector_bcrs_bounce_m, Tolerance("REFLECTOR_M", atol=1e-3))
    assert_close(full.uplink_vector_bcrs_m, full.reflector_bcrs_bounce_m - full.station_bcrs_transmit_m,
                 Tolerance("UPLINK_M", atol=1e-6))
    assert full.elevation_up_deg == pytest.approx(np.rad2deg(full.uplink.vacuum_elevation_rad))


def test_prediction_row_reports_the_uplink_geometry():
    processor = observation_processor()
    observation = processor.resolver.resolve(npt_record())
    predictor = LlrObservationPredictor(
        processor.observation_model.frame_system, processor.observation_model.light_time_solver,
        observation.station, observation.reflector, station_key="APOLLO", reflector_key="APOLLO15",
        criteria=PredictionCriteria(minimum_elevation_deg=0.0, minimum_reflector_elevation_deg=-90.0,
                                    maximum_sun_elevation_deg=90.0),
        meteorology=PredictionMeteorology(),
    )
    row = predictor.evaluate(TRANSMIT_UTC)
    uplink = processor.observation_model.light_time_solver.solve(_request(processor), two_way=False)

    assert (row["station"], row["reflector"], row["utc_t1"]) == ("APOLLO", "APOLLO15", "2020-01-01T00:00:00.000")
    assert_close(row["range_up_geometric_m"], uplink.uplink.geometric_range_m, Tolerance("RANGE_M", atol=1e-4))
    assert np.isfinite(row["azimuth_deg"]) and np.isfinite(row["elevation_deg"])


def test_solid_earth_tide_displaces_both_legs_and_shifts_oc_by_half_the_rtt_change():
    frames = static_frames()
    with_tide = _full_row(observation_processor(frames=frames, station_displacement=Iers2010SolidEarthTide(frames)))
    without = _full_row(observation_processor())

    for leg in ("transmit", "receive"):
        displacement = [with_tide[f"station_displacement_{leg}_itrf_{axis}_m"] for axis in "xyz"]
        assert np.linalg.norm(displacement) > 1e-6
    delta_rtt = with_tide["computed_rtt_before_range_bias_s"] - without["computed_rtt_before_range_bias_s"]
    delta_oc = (with_tide["observed_minus_computed_before_range_bias_one_way_m"]
                - without["observed_minus_computed_before_range_bias_one_way_m"])
    assert abs(delta_rtt) > 1e-15
    assert_close(delta_oc, -0.5 * C * delta_rtt, Tolerance("ONE_WAY_M", atol=1e-9))


def test_troposphere_uses_each_leg_elevation_and_adds_its_delays_to_the_rtt():
    frames = static_frames()
    station = station_facing_the_moon(frames, zenith_angle_deg=60.0)
    model = Iers2010MendesPavlisTroposphere()
    row = _full_row(observation_processor(frames=frames, troposphere=model, station_itrf_m=station))
    vacuum = _full_row(observation_processor(frames=frames, station_itrf_m=station))
    record = npt_record()
    site = itrf2geodetic(station)
    weather = TroposphereInput(pressure_hpa=record.pressure_hpa, temperature_k=record.temperature_k,
                               relative_humidity_percent=record.humidity_percent, latitude_rad=site.latitude_rad,
                               height_m=site.ellipsoidal_height_m, wavelength_um=record.wavelength_nm / 1000.0)

    assert row["light_time_converged"]
    assert 25.0 < row["tropo_elevation_up_used_deg"] < 35.0
    for leg in ("up", "down"):
        elevation = np.deg2rad(row[f"tropo_elevation_{leg}_used_deg"])
        assert_close(row[f"tropospheric_path_delay_{leg}_m"], model.slant_delay_m(weather, elevation_rad=elevation),
                     DISPLACEMENT_M, err_msg=leg)
    # Earth rotation during the 2.5 s flight changes the receive elevation, so the legs differ.
    assert abs(row["tropospheric_path_delay_up_m"] - row["tropospheric_path_delay_down_m"]) > 1e-6
    # The delays move the receive epoch by 20 ns; the geometry changes by much less than a picosecond.
    delta_rtt = row["computed_rtt_before_range_bias_s"] - vacuum["computed_rtt_before_range_bias_s"]
    expected = (row["tropospheric_path_delay_up_m"] + row["tropospheric_path_delay_down_m"]) / C
    assert_close(delta_rtt, expected, Tolerance("TROPOSPHERE_RTT_S", atol=1e-12, unit="s"))


def test_geometry_below_the_elevation_limit_is_flagged_and_dropped():
    processor = observation_processor()
    options = ObservationOptions(min_elevation_deg=91.0)
    observation = processor.resolver.resolve(npt_record())
    result = processor.observation_model.evaluate(observation, options=options,
                                                  result_detail=ObservationResultDetail.FULL)
    row = cast(dict[str, Any], result.result_row)

    assert result.below_elevation_limit and row["below_elevation_limit"]
    assert not row["valid_geometry"]
    assert row["status"] == "below_elevation_limit"
    assert row["computed_rtt_s"] == row["computed_rtt_before_range_bias_s"]
    assert (row["range_bias_model_label"], row["range_bias_lookup_status"]) == ("none", "explicit_zero")
    assert row["range_bias_correction_two_way_cm"] == 0.0
    assert (row["lunar_relativistic_scale_convention"], row["l_b_minus_l_l"]) == ("alreadyScaled", 0.0)
    assert len(processor.equations(NptDataset([npt_record()]), options=options)) == 0
    assert processor.rows(NptDataset([npt_record()]), options=options) == []


def test_resolver_reports_every_unresolved_record():
    processor = observation_processor()
    missing_station, missing_reflector = npt_record(1), npt_record(2)
    missing_station.station_name, missing_station.station_code = "UNKNOWN", None
    missing_reflector.reflector_name, missing_reflector.reflector_code = "UNKNOWN", None

    with pytest.raises(ValueError, match="2 record") as error:
        processor.resolver.resolve_all([missing_station, missing_reflector])
    assert "record_index=0" in str(error.value) and "record_index=1" in str(error.value)
