from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pytest

from lunarops.classes.displacement import (
    CompositeStationDisplacement,
    Iers2010OceanPoleTide,
    Iers2010SolidEarthPoleTide,
    Iers2010SolidEarthTide,
    LunarSolidTide,
    OceanPoleTideGrid,
    OceanPoleTideResult,
    PolarWobble,
    PoleTideResult,
    ReflectorDisplacementInput,
    StationDisplacementInput,
    ZeroReflectorDisplacement,
    ZeroStationDisplacement,
    secular_pole_2018_arcsec,
)
from lunarops.classes.displacement.terrestrial_geometry import enu2itrf
from lunarops.classes.ephemerides import BodyState, Ephemeris
from lunarops.classes.frames import EarthOrientationProvider, PolarMotion, ReferenceFrameSystem
from lunarops.classes.observation_factory import _compose_station_displacements, ensure_registered
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.context import RunContext
from lunarops.config.registry import available, validate_global_class_configs


class _ConstantDisplacement:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, dtype=float)

    def displacement_itrf_m(self, data: StationDisplacementInput):
        assert isinstance(data, StationDisplacementInput)
        return self.xyz


def _station_input() -> StationDisplacementInput:
    return StationDisplacementInput(
        reference_position_itrf_m=np.asarray([6_378_137.0, 0.0, 0.0]),
        epoch_utc=Epoch.from_isot("2000-01-01T00:00:00", scale=TimeScale.UTC),
    )


def test_displacement_inputs_are_frozen_slotted_and_read_only():
    station = _station_input()
    reflector = ReflectorDisplacementInput(
        reference_position_lcrs_m=np.array([1_737_400.0, 0.0, 0.0]),
        epoch_tdb=Epoch(2451544.5, 0.0, TimeScale.TDB),
    )

    assert not hasattr(station, "__dict__")
    assert not hasattr(reflector, "__dict__")
    assert not station.reference_position_itrf_m.flags.writeable
    assert not reflector.reference_position_lcrs_m.flags.writeable
    with pytest.raises(FrozenInstanceError):
        cast(Any, station).epoch_utc = station.epoch_utc
    with pytest.raises(ValueError):
        station.reference_position_itrf_m[0] = 0.0
    with pytest.raises(ValueError):
        reflector.reference_position_lcrs_m[0] = 0.0


def test_zero_displacement_models_return_three_component_vectors():
    assert np.allclose(
        ZeroStationDisplacement().displacement_itrf_m(_station_input()),
        np.zeros(3),
    )
    reflector_data = ReflectorDisplacementInput(
        reference_position_lcrs_m=np.array([1.0, 0.0, 0.0]),
        epoch_tdb=Epoch(2451544.5, 0.0, TimeScale.TDB),
    )
    assert np.allclose(
        ZeroReflectorDisplacement().displacement_lcrs_m(reflector_data),
        np.zeros(3),
    )


def test_solid_earth_tide_uses_native_single_epoch_call():
    frames = ReferenceFrameSystem(
        ephemeris=_FakeEphemeris(),
        earth_orientation_provider=_FakeEarthOrientation(),
    )
    displacement = Iers2010SolidEarthTide(frames).displacement_itrf_m(_station_input())
    assert displacement.shape == (3,)
    assert np.all(np.isfinite(displacement))


def test_solid_earth_tide_rejects_exact_utc_leap_second():
    epoch = Epoch.from_isot("2016-12-31T23:59:60", scale=TimeScale.UTC)
    with pytest.raises(ValueError, match="exact UTC leap-second label"):
        Iers2010SolidEarthTide._utc_calendar(epoch)


def test_composite_station_displacement_sums_components():
    model = CompositeStationDisplacement(
        components=(
            _ConstantDisplacement([1.0, 2.0, 3.0]),
            ZeroStationDisplacement(),
            _ConstantDisplacement([-0.5, 0.5, 1.0]),
        )
    )
    assert np.allclose(
        model.displacement_itrf_m(_station_input()),
        [0.5, 2.5, 4.0],
    )


def test_composite_station_displacement_rejects_invalid_components():
    with pytest.raises(TypeError, match="cannot contain None"):
        CompositeStationDisplacement(components=cast(Any, (None,)))
    with pytest.raises(ValueError, match="at least one component"):
        CompositeStationDisplacement(components=())


def test_station_displacement_config_components_are_automatically_summed():
    components = {
        "solid": _ConstantDisplacement([1.0, 2.0, 3.0]),
        "pole": _ConstantDisplacement([-0.5, 0.5, 1.0]),
    }

    class _FactoryContext:
        def create_class(self, category, config, *, cache):
            assert category == "stationDisplacement"
            assert cache is True
            return components[config["type"]]

    model = _compose_station_displacements(
        _FactoryContext(),
        [{"type": "solid"}, {"type": "pole"}],
    )

    assert np.allclose(model.displacement_itrf_m(_station_input()), [0.5, 2.5, 4.0])


def test_station_displacement_global_is_a_nonempty_component_list():
    ensure_registered()
    resolved = validate_global_class_configs(
        {"stationDisplacement": ["none", {"type": "none"}]}
    )
    assert resolved["stationDisplacement"] == [
        {"type": "none"},
        {"type": "none"},
    ]
    assert "sum" not in available("stationDisplacement")

    with pytest.raises(TypeError, match="list of class configs"):
        validate_global_class_configs({"stationDisplacement": {"type": "none"}})
    with pytest.raises(ValueError, match="at least 1 item"):
        validate_global_class_configs({"stationDisplacement": []})

    context = RunContext()
    first = context.create_class("stationDisplacement", "none", cache=True)
    second = context.create_class("stationDisplacement", "none", cache=True)
    assert first is second


def test_calceph_factory_requires_explicit_lunar_scale_convention(tmp_path):
    ensure_registered()
    context = RunContext(working_dir=str(tmp_path))

    with pytest.raises(ValueError, match="lunarRelativisticScaleConvention"):
        context.create_class(
            "ephemerides",
            {"type": "calceph", "file": "renamed_kernel.bsp"},
            cache=False,
        )


def test_dependent_factories_require_an_assembled_observation_context():
    ensure_registered()
    with pytest.raises(RuntimeError, match="build_observation_processor"):
        RunContext().create_class("relativity", "iersShapiro", cache=False)


class _FakeEarthOrientation(EarthOrientationProvider):
    @property
    def source_file_path(self):
        return None

    def polar_motion(self, epoch_utc):
        return PolarMotion(0.1, 0.2)

    def ut1_minus_utc_s(self, epoch_utc):
        return 0.0


class _FakeEphemeris(Ephemeris):
    @property
    def source_file_path(self):
        from pathlib import Path

        return Path("fake.eph")

    def body_state_bcrs(self, body_name, epoch_tdb: Epoch):
        positions = {
            "MOON": np.zeros(3),
            "EARTH": np.array([384_400_000.0, 0.0, 0.0]),
            "SUN": np.array([149_597_870_700.0, 0.0, 0.0]),
            "MERCURY BARYCENTER": np.array([5.0e10, 1.0e10, 0.0]),
            "VENUS BARYCENTER": np.array([1.0e11, 2.0e10, 0.0]),
            "MARS BARYCENTER": np.array([2.0e11, 3.0e10, 0.0]),
            "JUPITER BARYCENTER": np.array([7.0e11, 4.0e10, 0.0]),
            "SATURN BARYCENTER": np.array([1.4e12, 5.0e10, 0.0]),
            "URANUS BARYCENTER": np.array([2.8e12, 6.0e10, 0.0]),
            "NEPTUNE BARYCENTER": np.array([4.5e12, 7.0e10, 0.0]),
        }
        position = positions[body_name]
        return BodyState(position, np.zeros(3))

    def pa2lcrs_matrix(self, epoch_tdb: Epoch):
        return np.eye(3)

def test_pole_tide_exposes_typed_evaluation_result():
    model = Iers2010SolidEarthPoleTide(earth_orientation_provider=_FakeEarthOrientation())
    result = model.evaluate(_station_input())
    assert result.displacement_itrf_m.shape == (3,)
    assert result.displacement_enu_m.shape == (3,)
    assert np.all(np.isfinite(result.displacement_itrf_m))
    assert np.allclose(
        model.displacement_itrf_m(_station_input()),
        result.displacement_itrf_m,
    )
    assert not result.displacement_itrf_m.flags.writeable
    assert not result.displacement_enu_m.flags.writeable


def test_pole_tide_result_validates_vectors_when_constructed_directly():
    wobble = PolarWobble(0.1, 0.2, 0.05, 0.3, 0.05, 0.1)
    result = PoleTideResult(
        displacement_itrf_m=np.array([1.0, 2.0, 3.0]),
        displacement_enu_m=np.array([4.0, 5.0, 6.0]),
        wobble=wobble,
        geocentric_latitude_rad=0.1,
        longitude_rad=0.2,
    )
    assert not result.displacement_itrf_m.flags.writeable
    with pytest.raises(ValueError, match="exactly 3"):
        PoleTideResult(np.array([1.0]), np.array([1.0, 2.0, 3.0]), wobble, 0.1, 0.2)


def test_pole_tide_matches_independent_reference_vector():
    # Spherical coordinates make the independent geocentric-latitude reference
    # exact while still exercising longitude wrapping and ENU-to-ITRF rotation.
    latitude_rad = np.deg2rad(30.0)
    longitude_rad = np.deg2rad(-75.0)
    station_itrf_m = 6_371_000.0 * np.array(
        [
            np.cos(latitude_rad) * np.cos(longitude_rad),
            np.cos(latitude_rad) * np.sin(longitude_rad),
            np.sin(latitude_rad),
        ]
    )
    data = StationDisplacementInput(
        reference_position_itrf_m=station_itrf_m,
        epoch_utc=Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC),
    )
    result = Iers2010SolidEarthPoleTide(_FakeEarthOrientation()).evaluate(data)

    assert result.geocentric_latitude_rad == pytest.approx(latitude_rad)
    assert result.longitude_rad == pytest.approx(longitude_rad)
    assert result.wobble.secular_x_arcsec == pytest.approx(0.088537704312115)
    assert result.wobble.secular_y_arcsec == pytest.approx(0.389695263518138)
    assert result.wobble.m1_arcsec == pytest.approx(0.011462295687885)
    assert result.wobble.m2_arcsec == pytest.approx(0.189695263518138)
    np.testing.assert_allclose(
        result.displacement_enu_m,
        [-0.000270758134790, 0.000811192021795, 0.005151761253627],
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.displacement_itrf_m,
        [0.000788227447308, -0.003987833981558, 0.003278393525035],
        rtol=0.0,
        atol=2.0e-15,
    )


def test_ocean_pole_tide_grid_and_model(tmp_path):
    coefficient_file = tmp_path / "ocean_pole_tide.txt"
    coefficient_file.write_text("0 -90 1 0 2 0 3 0\n180 -90 1 0 2 0 3 0\n0 90 1 0 2 0 3 0\n180 90 1 0 2 0 3 0\n")
    grid = OceanPoleTideGrid(coefficient_file)
    model = Iers2010OceanPoleTide(
        grid=grid,
        earth_orientation_provider=_FakeEarthOrientation(),
    )
    result = model.evaluate(_station_input())
    assert grid.info.latitude_nodes == 2
    assert grid.info.longitude_nodes == 2
    assert np.all(np.isfinite(result.displacement_itrf_m))
    direct_result = OceanPoleTideResult(
        displacement_itrf_m=np.array([1.0, 2.0, 3.0]),
        displacement_enu_m=np.array([4.0, 5.0, 6.0]),
        coefficients=result.coefficients,
        wobble=result.wobble,
    )
    assert not direct_result.displacement_itrf_m.flags.writeable
    assert not direct_result.displacement_enu_m.flags.writeable


@pytest.mark.parametrize(
    ("mjd", "m1_arcsec", "m2_arcsec", "expected_enu_m"),
    (
        (52640.0, -0.1449316, 0.1808667, (3.133391e-05, 4.801230e-05, 3.690801e-04)),
        (52913.0, 0.2014101, 0.06780107, (1.544616e-05, 7.515046e-05, 1.114796e-03)),
        (53370.0, 0.09200257, 0.1377043, (2.678392e-05, 8.177465e-05, 1.053583e-03)),
    ),
)
def test_ocean_pole_tide_matches_official_test_vectors(
    tmp_path,
    mjd,
    m1_arcsec,
    m2_arcsec,
    expected_enu_m,
):
    # Four cells are sufficient because the official test location is an exact
    # grid node; the negative longitude exercises the 0..360 wrapping path.
    coefficient_file = tmp_path / "opoleloadcoefcmcor.txt"
    rows = (
        "232.25 -43.75 0.216192 0.285652 0.012697 0.024930 0.000868 0.010389",
        "232.75 -43.75 0.216192 0.285652 0.012697 0.024930 0.000868 0.010389",
        "232.25 -43.25 0.216192 0.285652 0.012697 0.024930 0.000868 0.010389",
        "232.75 -43.25 0.216192 0.285652 0.012697 0.024930 0.000868 0.010389",
    )
    coefficient_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    grid = OceanPoleTideGrid(coefficient_file)

    class _OfficialTestEarthOrientation(_FakeEarthOrientation):
        def polar_motion(self, epoch_utc):
            secular_x, secular_y = secular_pole_2018_arcsec(epoch_utc)
            return PolarMotion(secular_x + m1_arcsec, secular_y - m2_arcsec)

    model = Iers2010OceanPoleTide(
        grid=grid,
        earth_orientation_provider=_OfficialTestEarthOrientation(),
    )
    latitude_rad = np.deg2rad(-43.75)
    longitude_rad = np.deg2rad(232.25)
    equatorial_radius_m = 6_378_137.0
    eccentricity_squared = 6.6943799901413165e-3
    prime_vertical_radius_m = equatorial_radius_m / np.sqrt(1.0 - eccentricity_squared * np.sin(latitude_rad) ** 2)
    station_itrf_m = np.array(
        [
            prime_vertical_radius_m * np.cos(latitude_rad) * np.cos(longitude_rad),
            prime_vertical_radius_m * np.cos(latitude_rad) * np.sin(longitude_rad),
            prime_vertical_radius_m * (1.0 - eccentricity_squared) * np.sin(latitude_rad),
        ]
    )
    result = model.evaluate(
        StationDisplacementInput(
            reference_position_itrf_m=station_itrf_m,
            epoch_utc=Epoch(2_400_000.5, mjd, TimeScale.UTC),
        )
    )

    np.testing.assert_allclose(result.displacement_enu_m, expected_enu_m, rtol=0.0, atol=3.0e-7)
    np.testing.assert_allclose(
        result.displacement_itrf_m,
        enu2itrf(
            expected_enu_m,
            latitude_rad=latitude_rad,
            longitude_rad=longitude_rad,
        ),
        rtol=0.0,
        atol=3.0e-7,
    )


def test_lunar_solid_tide_requires_no_runtime_backend_injection():
    model = LunarSolidTide(ephemeris=_FakeEphemeris())
    data = ReflectorDisplacementInput(
        reference_position_lcrs_m=np.array([1_737_400.0, 0.0, 0.0]),
        epoch_tdb=Epoch(2451544.5, 0.0, TimeScale.TDB),
    )
    displacement = model.displacement_lcrs_m(data)
    assert displacement.shape == (3,)
    assert np.all(np.isfinite(displacement))


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("h2", np.nan, "h2 must be finite"),
        ("l2", np.inf, "l2 must be finite"),
        ("moon_radius_m", 0.0, "moon_radius_m must be positive"),
    ),
)
def test_lunar_solid_tide_validates_scalar_parameters(name, value, message):
    with pytest.raises(ValueError, match=message):
        LunarSolidTide(ephemeris=_FakeEphemeris(), **{name: value})


@pytest.mark.parametrize(
    ("latitude_rad", "longitude_rad", "message"),
    (
        (np.nan, 0.0, "latitude_rad must be finite"),
        (0.0, np.inf, "longitude_rad must be finite"),
        (np.pi, 0.0, "latitude_rad must be in"),
    ),
)
def test_enu2itrf_validates_coordinates(latitude_rad, longitude_rad, message):
    with pytest.raises(ValueError, match=message):
        enu2itrf(
            [1.0, 2.0, 3.0],
            latitude_rad=latitude_rad,
            longitude_rad=longitude_rad,
        )
