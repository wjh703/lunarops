"""Station and reflector displacement models."""

import gzip

import numpy as np
import pytest

from lunarops import _iers2010
from lunarops.classes.displacement import (
    CompositeStationDisplacement,
    Iers2010OceanPoleTide,
    Iers2010SolidEarthPoleTide,
    Iers2010SolidEarthTide,
    LunarSolidTide,
    OceanPoleTideGrid,
    ReflectorDisplacementInput,
    StationDisplacementInput,
    secular_pole_2018_arcsec,
)
from lunarops.classes.displacement.constants import LUNAR_H2, MOON_REFERENCE_RADIUS_M
from lunarops.classes.displacement.terrestrial_geometry import enu2itrf, geodetic2itrf, itrf2enu, itrf2geodetic
from lunarops.classes.frames import EarthOrientationValues
from lunarops.classes.frames.coordinate_scaling import GM_EARTH, GM_MOON, GM_SUN
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.class_configs import available, validate_model_configs
from lunarops.config.context import RunContext
from lunarops.config.registry import resolve
from tests.support.observation import EARTH_MOON_M, STATIC_BODIES, ZeroEarthOrientation, static_frames
from tests.tolerances import DISPLACEMENT_M, REL, Tolerance, assert_close, assert_vector_close

EPOCH_UTC = Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC)


class _Pole(ZeroEarthOrientation):
    def __init__(self, xp_arcsec=0.1, yp_arcsec=0.2):
        self.values = EarthOrientationValues(xp_arcsec, yp_arcsec, 0.0)

    def interpolate(self, epoch_utc):
        return self.values


def _station(position=(6_378_137.0, 0.0, 0.0), epoch=EPOCH_UTC):
    return StationDisplacementInput(reference_position_itrf_m=np.asarray(position, dtype=float), epoch_utc=epoch)


def test_geodetic_and_local_frame_conversions_are_inverse():
    site = itrf2geodetic(
        geodetic2itrf(latitude_rad=np.deg2rad(40.0), longitude_rad=np.deg2rad(-105.0), ellipsoidal_height_m=1600.0)
    )
    assert site.latitude_deg == pytest.approx(40.0, abs=1e-10)
    assert site.longitude_deg == pytest.approx(-105.0, abs=1e-10)
    assert site.ellipsoidal_height_m == pytest.approx(1600.0, abs=1e-5)
    enu = np.array([12.5, -31.0, 4.25])
    assert_close(
        itrf2enu(enu2itrf(enu, latitude_rad=0.43, longitude_rad=-1.17), latitude_rad=0.43, longitude_rad=-1.17),
        enu,
        DISPLACEMENT_M,
    )


@pytest.mark.parametrize(
    ("latitude_rad", "longitude_rad", "message"),
    [
        (np.nan, 0.0, "latitude_rad must be finite"),
        (0.0, np.inf, "longitude_rad must be finite"),
        (np.pi, 0.0, "latitude_rad must be in"),
    ],
)
def test_enu2itrf_rejects_invalid_coordinates(latitude_rad, longitude_rad, message):
    with pytest.raises(ValueError, match=message):
        enu2itrf([1.0, 2.0, 3.0], latitude_rad=latitude_rad, longitude_rad=longitude_rad)


def test_composite_displacement_is_the_sum_of_its_components():
    class Constant:
        def __init__(self, xyz):
            self.xyz = np.asarray(xyz, dtype=float)

        def displacement_itrf_m(self, data):
            return self.xyz

    model = CompositeStationDisplacement((Constant([1.0, 2.0, 3.0]), Constant([-0.5, 0.5, 1.0])))
    assert_close(model.displacement_itrf_m(_station()), [0.5, 2.5, 4.0], REL)
    assert_close(CompositeStationDisplacement(()).displacement_itrf_m(_station()), np.zeros(3), REL)


def test_station_displacement_is_declared_as_a_non_empty_component_list():
    resolved = validate_model_configs({"stationDisplacement": [{"type": "none"}, {"type": "none"}]})

    assert resolved["stationDisplacement"] == [{"type": "none"}, {"type": "none"}]
    assert "sum" not in available("stationDisplacement")["stationDisplacement"]
    with pytest.raises(TypeError, match="must be a sequence"):
        validate_model_configs({"stationDisplacement": {"type": "none"}})
    with pytest.raises(ValueError, match="at least 1 item"):
        validate_model_configs({"stationDisplacement": []})
    assert RunContext().create_class("reflectorDisplacement", {"type": "none"}) is None
    with pytest.raises(ValueError, match="lunarRelativisticScaleConvention"):
        resolve("ephemerides", {"type": "chebyshev", "directory": "kernels"})


def test_solid_earth_tide_passes_itrf_sun_moon_and_utc_calendar_to_dehanttideinel():
    frames = static_frames()
    data = _station(epoch=Epoch.from_isot("2020-01-01T06:30:00", scale=TimeScale.UTC))
    tdb = frames.time_scale_converter.convert(data.epoch_utc, TimeScale.TDB)
    matrix = frames.terrestrial_transform.gcrs2itrf_matrix(data.epoch_utc)
    sun, moon = (
        matrix @ frames.relativistic_transform.bcrs2gcrs(np.asarray(STATIC_BODIES[name][0]), tdb)
        for name in ("SUN", "MOON")
    )

    expected = _iers2010.dehanttideinel(data.reference_position_itrf_m, 2020, 1, 1, 6.5, sun, moon)

    assert_close(Iers2010SolidEarthTide(frames).displacement_itrf_m(data), expected, DISPLACEMENT_M)


def test_solid_earth_tide_rejects_an_exact_utc_leap_second_label():
    with pytest.raises(ValueError, match="exact UTC leap-second label"):
        Iers2010SolidEarthTide(static_frames()).displacement_itrf_m(
            _station(epoch=Epoch.from_isot("2016-12-31T23:59:60", scale=TimeScale.UTC))
        )


def test_solid_earth_pole_tide_matches_an_independent_reference_vector():
    # A spherical site makes the geocentric-latitude reference exact.
    latitude, longitude = np.deg2rad(30.0), np.deg2rad(-75.0)
    position = 6_371_000.0 * np.array(
        [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)]
    )
    assert_close(
        Iers2010SolidEarthPoleTide(_Pole()).evaluate(_station(position)),
        [0.000788227447308, -0.003987833981558, 0.003278393525035],
        DISPLACEMENT_M,
    )


@pytest.mark.parametrize(
    ("mjd", "m1_arcsec", "m2_arcsec", "expected_enu_m"),
    [
        (52640.0, -0.1449316, 0.1808667, (3.133391e-05, 4.801230e-05, 3.690801e-04)),
        (52913.0, 0.2014101, 0.06780107, (1.544616e-05, 7.515046e-05, 1.114796e-03)),
        (53370.0, 0.09200257, 0.1377043, (2.678392e-05, 8.177465e-05, 1.053583e-03)),
    ],
)
def test_ocean_pole_tide_matches_the_official_test_vectors(tmp_path, mjd, m1_arcsec, m2_arcsec, expected_enu_m):
    # The official site is an exact grid node; its negative longitude exercises the 0..360 wrap.
    path = tmp_path / "opoleloadcoefcmcor.txt"
    path.write_text(
        "".join(
            f"{lon} {lat} 0.216192 0.285652 0.012697 0.024930 0.000868 0.010389\n"
            for lat in (-43.75, -43.25)
            for lon in (232.25, 232.75)
        ),
        encoding="utf-8",
    )

    class OfficialPole(ZeroEarthOrientation):
        def interpolate(self, epoch_utc):
            secular_x, secular_y = secular_pole_2018_arcsec(epoch_utc)
            return EarthOrientationValues(secular_x + m1_arcsec, secular_y - m2_arcsec, 0.0)

    latitude, longitude = np.deg2rad(-43.75), np.deg2rad(232.25)
    site = geodetic2itrf(latitude_rad=latitude, longitude_rad=longitude, ellipsoidal_height_m=0.0)
    model = Iers2010OceanPoleTide(grid=OceanPoleTideGrid(path), earth_orientation_provider=OfficialPole())

    result = model.evaluate(_station(site, Epoch(2_400_000.5, mjd, TimeScale.UTC)))

    # The test vectors are printed with seven significant digits.
    assert_close(
        result,
        enu2itrf(expected_enu_m, latitude_rad=latitude, longitude_rad=longitude),
        Tolerance("PRINTED_VECTOR_M", atol=3e-7, unit="m"),
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_ocean_pole_grid_reads_only_complete_numeric_rows_and_keeps_signed_zero(tmp_path, compressed):
    path = tmp_path / ("grid.txt.gz" if compressed else "grid.txt")
    text = (
        "unmarked published header with at least eight text fields\n"
        "# 0 -90 999 0 0 0 0 0\n"
        "0 -90 999 invalid 0 0 0 0\n"
        "0 -90 999\n"
        "180 90 4_0 -0.0 2 3 4 5 trailing text\n"
        "0 -90 1_0 -0.0 2 3 4 5\n"
        "180 -90 2e1 -0.0 2 3 4 5\n"
        "0 90 ٣0 -0.0 2 3 4 5\n"
    )
    if compressed:
        with gzip.open(path, "wt") as stream:
            stream.write(text)
    else:
        path.write_text(text)

    grid = OceanPoleTideGrid(path)

    # Values are parsed from text, so they must be exact, including the sign of zero.
    np.testing.assert_array_equal(grid.radial_grid.real, [[10, 20], [30, 40]])
    assert np.all(np.signbit(grid.radial_grid.imag))
    np.testing.assert_array_equal(grid.north_grid, np.full((2, 2), complex(2, 3)))
    np.testing.assert_array_equal(grid.east_grid, np.full((2, 2), complex(4, 5)))


def test_lunar_solid_tide_is_radial_with_earth_and_sun_on_the_reflector_axis():
    model = LunarSolidTide(transform=static_frames().relativistic_transform)
    data = ReflectorDisplacementInput(
        reference_position_lcrs_m=np.array([1_737_400.0, 0.0, 0.0]), epoch_tdb=Epoch(2451544.5, 0.0, TimeScale.TDB)
    )
    sun_distance = STATIC_BODIES["SUN"][0][0] - EARTH_MOON_M
    # Pavlov et al. (2016) Eq. (24) with cos(theta) = +-1: h2 R^4 / GM_M * sum GM_j / d_j^3 along the reflector.
    radial = LUNAR_H2 * MOON_REFERENCE_RADIUS_M**4 / GM_MOON * (GM_EARTH / EARTH_MOON_M**3 + GM_SUN / sun_distance**3)

    # LCRS distances differ from BCRS ones by about 1e-8.
    assert_vector_close(model.displacement_lcrs_m(data), [radial, 0.0, 0.0], Tolerance("LCRS_SCALE", rtol=1e-7))
