from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pytest

from lunarops.base.constants import C
from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.delays import ZeroGravitationalDelay, ZeroTroposphereDelay
from lunarops.classes.delays.troposphere import Iers2010MendesPavlisTroposphere
from lunarops.classes.displacement import (
    Iers2010SolidEarthTide,
    ZeroReflectorDisplacement,
    ZeroStationDisplacement,
)
from lunarops.classes.ephemerides import BodyState, Ephemeris
from lunarops.classes.frames import EarthOrientationProvider, PolarMotion, ReferenceFrameSystem
from lunarops.classes.observation import (
    LightTimeSolver,
    LlrObservationModel,
    LlrObservationProcessor,
    ObservationCatalogState,
    ObservationProcessingOptions,
    ObservationResolver,
    ObservationResultDetail,
    NptDataset,
    NptRecord,
    ReflectorRecord,
    StationRecord,
)
from lunarops.classes.parametrization.reflector_position import (
    ReflectorPositionParametrization,
)
from lunarops.classes.range_bias.models import ZeroRangeBiasModel
from lunarops.classes.observation_factory import ensure_registered
from lunarops.config.registry import validate_class_config


class _Ephemeris(Ephemeris):
    _POSITIONS: ClassVar[dict[str, np.ndarray]] = {
        "SSB": np.zeros(3),
        "EARTH": np.zeros(3),
        "MOON": np.array([384_400_000.0, 0.0, 0.0]),
        "SUN": np.array([149_597_870_700.0, 0.0, 0.0]),
        "MERCURY BARYCENTER": np.array([5.0e10, 2.0e10, 0.0]),
        "VENUS BARYCENTER": np.array([1.0e11, 3.0e10, 0.0]),
        "MARS BARYCENTER": np.array([2.0e11, 4.0e10, 0.0]),
        "JUPITER BARYCENTER": np.array([7.0e11, 5.0e10, 0.0]),
        "SATURN BARYCENTER": np.array([1.4e12, 6.0e10, 0.0]),
        "URANUS BARYCENTER": np.array([2.8e12, 7.0e10, 0.0]),
        "NEPTUNE BARYCENTER": np.array([4.5e12, 8.0e10, 0.0]),
    }

    @property
    def source_file_path(self) -> Path:
        return Path("test.eph")

    def body_state_bcrs(self, body_name: str, epoch_tdb: Epoch) -> BodyState:
        epoch_tdb.require_scale(TimeScale.TDB)
        return BodyState(self._POSITIONS[body_name.upper()], np.zeros(3))

    def pa2lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        epoch_tdb.require_scale(TimeScale.TDB)
        return np.eye(3)

    def geocentric_tdb_minus_tt_s(self, epoch_tdb: Epoch) -> float:
        epoch_tdb.require_scale(TimeScale.TDB)
        return 0.0


class _EarthOrientation(EarthOrientationProvider):
    @property
    def source_file_path(self) -> Path:
        return Path("test.eop")

    def polar_motion(self, epoch_utc: Epoch) -> PolarMotion:
        return PolarMotion(0.0, 0.0)

    def ut1_minus_utc_s(self, epoch_utc: Epoch) -> float:
        return 0.0


def _record(index: int = 4) -> NptRecord:
    return NptRecord(
        station_name="APOL",
        reflector_name="Apollo 15",
        transmit_epoch=Epoch.from_isot(
            "2020-01-01T00:00:00",
            scale=TimeScale.UTC,
        ),
        round_trip_time_s=2.55,
        uncertainty_two_way_s=100.0e-12,
        pressure_hpa=900.0,
        temperature_k=285.0,
        humidity_percent=25.0,
        wavelength_nm=532.0,
        index=index,
        station_code="70610",
        reflector_code="A15",
    )


def _pipeline(troposphere_delay=None, *, frames=None, station_displacement=None):
    if frames is None:
        frames = ReferenceFrameSystem(_Ephemeris(), _EarthOrientation())
    solver = LightTimeSolver(
        frames,
        gravitational_delay_model=ZeroGravitationalDelay(),
        troposphere_delay_model=(ZeroTroposphereDelay() if troposphere_delay is None else troposphere_delay),
        station_displacement_model=station_displacement or ZeroStationDisplacement(),
        reflector_displacement_model=ZeroReflectorDisplacement(),
    )
    state = ObservationCatalogState(
        {
            "APOLLO": StationRecord(
                name="Apache Point",
                aliases=("APOL", "70610"),
                itrf_xyz_m=(6_378_137.0, 0.0, 0.0),
            )
        },
        {
            "APOLLO15": ReflectorRecord(
                name="Apollo 15",
                aliases=("A15",),
                moon_fixed_xyz_m=(1_737_400.0, 0.0, 0.0),
            )
        },
    )
    resolver = ObservationResolver(state)
    return LlrObservationProcessor(
        resolver,
        LlrObservationModel(frames, solver, ZeroRangeBiasModel()),
    )


class _RecordingStationDisplacement:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = []

    def displacement_itrf_m(self, data):
        self.calls.append(data)
        return self.delegate.displacement_itrf_m(data)


def test_full_observation_pipeline_builds_one_equation():
    processor = _pipeline()
    equation = processor.equations(
        NptDataset([_record()]),
        options=ObservationProcessingOptions(
            min_elevation_deg=-90.0,
            include_reflector_position_partials=True,
        ),
    )[0]

    assert equation.observation_id == 4
    assert equation.station_key == "APOLLO"
    assert equation.reflector_key == "APOLLO15"
    assert equation.light_time_converged
    assert equation.sigma_one_way_m == pytest.approx(0.5 * C * 100.0e-12)
    assert equation.design_partials["reflector_position_pa"].shape == (3,)
    row = processor.rows(
        NptDataset([_record()]),
        options=ObservationProcessingOptions(min_elevation_deg=-90.0),
    )[0]
    assert row["status"] == "ok"
    assert row["oc_one_way_m"] == pytest.approx(equation.observed_minus_computed_one_way_m)


def test_light_time_starts_from_fixed_round_trip_time(monkeypatch):
    processor = _pipeline()
    observation = processor.resolver.resolve(_record())
    solver = processor.observation_model.light_time_solver
    calls = []
    original = LightTimeSolver._station_state_from_tdb

    def record_receive_epoch(self, request, epoch_tdb):
        calls.append((request, epoch_tdb))
        return original(self, request, epoch_tdb)

    monkeypatch.setattr(LightTimeSolver, "_station_state_from_tdb", record_receive_epoch)
    processor.observation_model.evaluate(observation, min_elevation_deg=-90.0)

    request, initial_receive_tdb = calls[0]
    transmit_station = solver._station_state_at_utc(request, request.transmit_epoch_utc)
    transmit_tdb = solver.time_scale_converter.convert(
        request.transmit_epoch_utc,
        TimeScale.TDB,
        station_gcrs_m=transmit_station.position_gcrs_m,
    )
    assert transmit_tdb.seconds_until(initial_receive_tdb) == pytest.approx(2.4, abs=1.0e-13)


def test_native_solid_earth_tide_enters_transmit_and_receive_light_time():
    frames = ReferenceFrameSystem(_Ephemeris(), _EarthOrientation())
    recorder = _RecordingStationDisplacement(Iers2010SolidEarthTide(frames))
    with_tide = _pipeline(frames=frames, station_displacement=recorder)
    observation = with_tide.resolver.resolve(_record())

    native = cast(
        dict[str, Any],
        with_tide.observation_model.evaluate(
            observation,
            min_elevation_deg=-90.0,
            result_detail=ObservationResultDetail.FULL,
        ).result_row,
    )
    zero = cast(
        dict[str, Any],
        _pipeline()
        .observation_model.evaluate(
            observation,
            min_elevation_deg=-90.0,
            result_detail=ObservationResultDetail.FULL,
        )
        .result_row,
    )

    assert native["light_time_converged"]
    assert len(recorder.calls) >= 3
    assert {call.station_id for call in recorder.calls} == {"APOLLO"}
    assert recorder.calls[0].epoch_utc == observation.transmit_epoch_utc
    assert any(call.epoch_utc != observation.transmit_epoch_utc for call in recorder.calls)
    assert (
        np.linalg.norm(
            [
                native["station_displacement_transmit_itrf_x_m"],
                native["station_displacement_transmit_itrf_y_m"],
                native["station_displacement_transmit_itrf_z_m"],
            ]
        )
        > 1.0e-6
    )
    assert native["computed_rtt_before_range_bias_s"] != pytest.approx(
        zero["computed_rtt_before_range_bias_s"],
        rel=0.0,
        abs=1.0e-15,
    )


def test_end_to_end_contribution_changes_rtt_and_oc_separately():
    frames = ReferenceFrameSystem(_Ephemeris(), _EarthOrientation())
    with_tide = _pipeline(
        frames=frames,
        station_displacement=Iers2010SolidEarthTide(frames),
    )
    without_tide = _pipeline()
    observation = with_tide.resolver.resolve(_record())

    tide = cast(
        dict[str, Any],
        with_tide.observation_model.evaluate(
            observation,
            min_elevation_deg=-90.0,
            result_detail=ObservationResultDetail.FULL,
        ).result_row,
    )
    zero = cast(
        dict[str, Any],
        without_tide.observation_model.evaluate(
            without_tide.resolver.resolve(_record()),
            min_elevation_deg=-90.0,
            result_detail=ObservationResultDetail.FULL,
        ).result_row,
    )

    delta_rtt_s = tide["computed_rtt_before_range_bias_s"] - zero["computed_rtt_before_range_bias_s"]
    delta_oc_m = (
        tide["observed_minus_computed_before_range_bias_one_way_m"]
        - zero["observed_minus_computed_before_range_bias_one_way_m"]
    )
    assert abs(delta_rtt_s) > 1.0e-15
    assert delta_oc_m == pytest.approx(-0.5 * C * delta_rtt_s, abs=1.0e-9)
    assert (
        np.linalg.norm(
            [
                tide["station_displacement_transmit_itrf_x_m"],
                tide["station_displacement_transmit_itrf_y_m"],
                tide["station_displacement_transmit_itrf_z_m"],
            ]
        )
        > 1.0e-6
    )
    assert (
        np.linalg.norm(
            [
                tide["station_displacement_receive_itrf_x_m"],
                tide["station_displacement_receive_itrf_y_m"],
                tide["station_displacement_receive_itrf_z_m"],
            ]
        )
        > 1.0e-6
    )


def test_cython_troposphere_contributes_to_both_light_time_legs(monkeypatch):
    transmit_epoch = _record().transmit_epoch

    def controlled_elevation(
        self,
        station_itrf_m,
        target_bcrs_m,
        station_epoch_utc,
        target_epoch_tdb,
    ):
        del self, station_itrf_m, target_bcrs_m, target_epoch_tdb
        return np.deg2rad(30.0 if station_epoch_utc == transmit_epoch else 45.0)

    monkeypatch.setattr(LightTimeSolver, "_vacuum_elevation_rad", controlled_elevation)
    model = Iers2010MendesPavlisTroposphere()
    processor = _pipeline(model)
    observation = processor.resolver.resolve(_record())
    with_troposphere = cast(
        dict[str, Any],
        processor.observation_model.evaluate(
            observation,
            min_elevation_deg=-90.0,
            result_detail=ObservationResultDetail.FULL,
        ).result_row,
    )
    without_troposphere = cast(
        dict[str, Any],
        _pipeline()
        .observation_model.evaluate(
            observation,
            min_elevation_deg=-90.0,
            result_detail=ObservationResultDetail.FULL,
        )
        .result_row,
    )

    assert with_troposphere["light_time_converged"]
    assert with_troposphere["tropospheric_path_delay_up_m"] > 0.0
    assert with_troposphere["tropospheric_path_delay_down_m"] > 0.0
    assert with_troposphere["tropo_elevation_up_used_deg"] == pytest.approx(30.0)
    assert with_troposphere["tropo_elevation_down_used_deg"] == pytest.approx(45.0)
    assert with_troposphere["tropospheric_path_delay_up_m"] != pytest.approx(
        with_troposphere["tropospheric_path_delay_down_m"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert (
        with_troposphere["computed_rtt_before_range_bias_s"] > without_troposphere["computed_rtt_before_range_bias_s"]
    )


def test_measurement_marks_geometry_below_requested_elevation():
    processor = _pipeline()
    observation = processor.resolver.resolve(_record())
    result = processor.observation_model.evaluate(
        observation,
        min_elevation_deg=91.0,
        result_detail=ObservationResultDetail.FULL,
    )
    row = cast(dict[str, Any], result.result_row)

    assert result.below_elevation_limit
    assert row["below_elevation_limit"]
    assert not row["valid_geometry"]
    assert row["status"] == "below_elevation_limit"
    assert row["computed_rtt_s"] == pytest.approx(row["computed_rtt_before_range_bias_s"])
    assert row["range_bias_model_label"] == "none"
    assert row["range_bias_lookup_status"] == "explicit_zero"
    assert row["range_bias_correction_two_way_cm"] == 0.0
    assert row["lunar_relativistic_scale_convention"] == "alreadyScaled"
    assert row["l_b_minus_l_l"] == 0.0

    options = ObservationProcessingOptions(min_elevation_deg=91.0)
    dataset = NptDataset([_record()])
    assert processor.equations(dataset, options=options) == []
    assert processor.rows(dataset, options=options) == []


def test_resolver_reports_all_unresolved_records():
    processor = _pipeline()
    missing_station = _record(1)
    missing_station.station_name = "UNKNOWN"
    missing_station.station_code = None
    missing_reflector = _record(2)
    missing_reflector.reflector_name = "UNKNOWN"
    missing_reflector.reflector_code = None

    with pytest.raises(ValueError, match="2 record") as error:
        processor.resolver.resolve_all([missing_station, missing_reflector])
    assert "record_index=0" in str(error.value)
    assert "record_index=1" in str(error.value)


def test_reflector_parametrization_updates_explicit_model_state():
    processor = _pipeline()
    equation = processor.equations(
        NptDataset([_record()]),
        options=ObservationProcessingOptions(
            min_elevation_deg=-90.0,
            include_reflector_position_partials=True,
        ),
    )[0]
    block = ReflectorPositionParametrization(reflectors=["APOLLO15"])
    original = processor.model_state.reflector_catalog["APOLLO15"]

    block.setup([equation], processor.model_state)
    assert block.reference_values() == pytest.approx([1_737_400.0, 0.0, 0.0])
    block.apply_update(np.array([1.0, 2.0, 3.0]))

    updated = processor.model_state.reflector_catalog["APOLLO15"]
    assert updated is not original
    assert updated.moon_fixed_xyz_m == pytest.approx([1_737_401.0, 2.0, 3.0])
    assert block.reference_values() == pytest.approx([1_737_401.0, 2.0, 3.0])
    assert processor.resolver.resolve(_record()).reflector is updated


@pytest.mark.parametrize(
    "reflectors, error",
    [
        ("APOLLO15", TypeError),
        (123, TypeError),
        ([], ValueError),
        (["APOLLO15", "APOLLO15"], ValueError),
    ],
)
def test_reflector_parametrization_validates_explicit_selection(reflectors, error):
    with pytest.raises(error):
        ReflectorPositionParametrization(reflectors=reflectors)


def test_reflector_parametrization_rejects_unknown_config_keys():
    ensure_registered()
    with pytest.raises(ValueError, match="unknown configuration key"):
        validate_class_config(
            "parametrization",
            {"type": "reflectorPosition", "reflector": ["APOLLO15"]},
        )


def test_reflector_parametrization_rejects_unobserved_catalog_key():
    processor = _pipeline()
    equation = processor.equations(
        NptDataset([_record()]),
        options=ObservationProcessingOptions(
            min_elevation_deg=-90.0,
            include_reflector_position_partials=True,
        ),
    )[0]
    processor.model_state.reflector_catalog["UNOBSERVED"] = ReflectorRecord(
        name="Unobserved",
        moon_fixed_xyz_m=[1.0, 2.0, 3.0],
    )
    block = ReflectorPositionParametrization(reflectors=["UNOBSERVED"])

    with pytest.raises(ValueError, match="have no observations"):
        block.setup([equation], processor.model_state)


def test_reflector_parametrization_missing_partial_message_names_current_option():
    processor = _pipeline()
    equation = processor.equations(
        NptDataset([_record()]),
        options=ObservationProcessingOptions(min_elevation_deg=-90.0),
    )[0]
    block = ReflectorPositionParametrization(reflectors=["APOLLO15"])
    block.setup([equation], processor.model_state)

    with pytest.raises(KeyError, match="include_reflector_position_partials=True"):
        block.design_columns(equation)
