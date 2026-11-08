"""Earth-orientation tables, IERS high-frequency corrections and the GCRS/ITRS matrix."""

import pickle

import erfa
import numpy as np
import pytest

from lunarops.classes.frames import EarthOrientationSample, TabulatedEarthOrientation, TerrestrialFrameTransform
from lunarops.classes.frames.earth_orientation import read_iers_c04, read_iers_rapid
from lunarops.classes.frames.high_frequency_eop import (
    earth_rotation_libration_eop_correction,
    high_frequency_eop_correction,
    ocean_tide_eop_correction,
)
from lunarops.classes.time import Epoch, TimeScale, utc2tt
from tests.tolerances import ANGLE_RAD, EOP_UAS, TIME_S, Tolerance, assert_close

ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)
# The IERS libration examples print their results to these precisions.
PRINTED_LIBRATION_PM_UAS = Tolerance("PRINTED_LIBRATION_PM_UAS", atol=2e-5, unit="uas")
PRINTED_LIBRATION_UT_US = Tolerance("PRINTED_LIBRATION_UT_US", atol=2e-8, unit="us")
FINALS_ROW = (
    "73 1 2 41684.00 I  0.120733 0.009786  0.136966 0.015902  I 0.8084178 0.0002710  0.0000 0.1916  P    -0.766"
    "    0.199    -0.720    0.300   .143000   .137000   .8075000   -18.637    -3.667  \n"
)


def _erfa_gcrs2itrs(epoch, xp_arcsec, yp_arcsec, dut1_s, dx_arcsec=0.0, dy_arcsec=0.0):
    """IERS 2010 CIO-based chain written directly with ERFA."""
    tt = utc2tt(epoch)
    ut11, ut12 = erfa.utcut1(epoch.jd1, epoch.jd2, dut1_s)
    x, y, s = erfa.xys06a(tt.jd1, tt.jd2)
    rc2i = erfa.c2ixys(x + dx_arcsec * ARCSEC_TO_RAD, y + dy_arcsec * ARCSEC_TO_RAD, s)
    rpom = erfa.pom00(xp_arcsec * ARCSEC_TO_RAD, yp_arcsec * ARCSEC_TO_RAD, erfa.sp00(tt.jd1, tt.jd2))
    return np.asarray(erfa.c2tcio(rc2i, erfa.era00(ut11, ut12), rpom))


def test_ocean_tide_correction_reproduces_ortho_eop_example():
    correction = ocean_tide_eop_correction(Epoch(2_400_000.5, 47_100.0, TimeScale.UTC), background_ut1_minus_utc_s=0.0)
    actual = [
        correction.ocean_delta_xp_arcsec * 1e6,
        correction.ocean_delta_yp_arcsec * 1e6,
        correction.ocean_delta_ut1_s * 1e6,
    ]
    assert_close(actual, (-162.8386373279636530, 117.7907525842668974, -23.39092370609808214), EOP_UAS)


def test_libration_corrections_reproduce_iers_examples():
    polar_motion = earth_rotation_libration_eop_correction(Epoch(2_400_000.5, 54_335.0, TimeScale.TT))
    assert_close(
        [polar_motion.libration_delta_xp_arcsec * 1e6, polar_motion.libration_delta_yp_arcsec * 1e6],
        [24.83144238273364834, -14.09240692041837661],
        PRINTED_LIBRATION_PM_UAS,
    )
    for mjd, expected_us in ((44_239.1, 2.441143834386761746), (55_227.4, -2.655705844335680244)):
        correction = earth_rotation_libration_eop_correction(Epoch(2_400_000.5, mjd, TimeScale.TT))
        assert_close(correction.libration_delta_ut1_s * 1e6, expected_us, PRINTED_LIBRATION_UT_US)


def test_high_frequency_correction_is_the_sum_of_ocean_and_libration_terms():
    epoch = Epoch(2_400_000.5, 55_227.4, TimeScale.UTC)
    result = high_frequency_eop_correction(epoch, background_ut1_minus_utc_s=0.0)
    ocean = ocean_tide_eop_correction(epoch, background_ut1_minus_utc_s=0.0)
    libration = earth_rotation_libration_eop_correction(utc2tt(epoch))

    assert result.delta_xp_arcsec == ocean.delta_xp_arcsec + libration.delta_xp_arcsec
    assert result.delta_yp_arcsec == ocean.delta_yp_arcsec + libration.delta_yp_arcsec
    assert result.delta_ut1_s == ocean.delta_ut1_s + libration.delta_ut1_s
    assert result.libration_delta_lod_s_per_day == libration.libration_delta_lod_s_per_day
    assert result.libration_delta_lod_s_per_day == pytest.approx(27.0861697892672e-6)


def test_high_frequency_corrections_require_their_time_scales_and_background_dut1():
    with pytest.raises(ValueError, match="epoch_utc must use the UTC scale"):
        ocean_tide_eop_correction(Epoch(2_400_000.5, 47_100.0, TimeScale.TT), background_ut1_minus_utc_s=0.0)
    with pytest.raises(ValueError, match="requires a TT or TDB Epoch"):
        earth_rotation_libration_eop_correction(Epoch(2_400_000.5, 47_100.0, TimeScale.UTC))
    with pytest.raises(TypeError):
        ocean_tide_eop_correction(Epoch(2_400_000.5, 55_227.4, TimeScale.UTC))  # type: ignore[call-arg]


def test_tabulated_eop_interpolates_and_pickles():
    samples = (EarthOrientationSample(60000.0, 0.1, 0.2, 0.3), EarthOrientationSample(60001.0, 0.4, 0.5, 0.6))
    eop = TabulatedEarthOrientation(samples, source_file_path="eop.txt")
    restored = pickle.loads(pickle.dumps(eop))

    assert eop.mjd_utc_range == (60000.0, 60001.0)
    values = eop.interpolate(Epoch(2_400_000.5, 60000.5, TimeScale.UTC))
    assert (values.xp_arcsec, values.yp_arcsec) == pytest.approx((0.25, 0.35))
    assert restored.source_file_path == eop.source_file_path
    assert restored.samples == eop.samples


def test_ut1_interpolation_removes_the_leap_second_step():
    eop = TabulatedEarthOrientation(
        (
            EarthOrientationSample(57_753.0, 0.0, 0.0, -0.4),
            EarthOrientationSample(57_754.0, 0.0, 0.0, 0.6),
        )
    )
    midday = Epoch.from_calendar(2016, 12, 31, 12, scale=TimeScale.UTC)
    assert_close(eop.interpolate(midday).ut1_minus_utc_s, -0.4, TIME_S)


def test_duplicate_mjd_is_rejected():
    samples = (EarthOrientationSample(60000.0, 0.1, 0.2, 0.3), EarthOrientationSample(60000.0, 0.4, 0.5, 0.6))
    with pytest.raises(ValueError, match="duplicate MJD"):
        TabulatedEarthOrientation(samples)


def test_c04_reader_supports_both_layouts_with_pole_offsets_and_lod(tmp_path):
    path = tmp_path / "eopc04.txt"
    path.write_text(
        "# header\n"
        "2020 1 1 58849 0.076 0.282 -0.177 0.001 0.0002 -0.0003 0 0 0 0 0 0\n"
        "2020 1 2 0 58850 0.077 0.283 -0.178 0.0004 -0.0005 0 0 0.0012 0 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    first, second = read_iers_c04(path)

    assert (first.mjd_utc, first.xp_arcsec, first.yp_arcsec, first.ut1_minus_utc_s) == (58849.0, 0.076, 0.282, -0.177)
    assert (first.dx_arcsec, first.dy_arcsec) == (0.0002, -0.0003)
    assert (second.dx_arcsec, second.dy_arcsec, second.lod_s) == (0.0004, -0.0005, 0.0012)


def test_finals_reader_uses_bulletin_a_columns(tmp_path):
    path = tmp_path / "finals2000A.all"
    path.write_text(FINALS_ROW, encoding="ascii")
    (sample,) = read_iers_rapid(path)

    assert (sample.mjd_utc, sample.xp_arcsec, sample.yp_arcsec, sample.ut1_minus_utc_s) == (
        41684.0,
        0.120733,
        0.136966,
        0.8084178,
    )
    assert (sample.dx_arcsec, sample.dy_arcsec) == pytest.approx((-0.000766, -0.000720))


def test_c04_parse_errors_name_the_file_and_line(tmp_path):
    path = tmp_path / "eopc04.txt"
    path.write_text("2020 1 1 58849 0.076 0.282 -0.177 0.001 0.0002 -0.0003 0 0 0 0 0 0\n2020 1 2 58850 0.1\n")
    with pytest.raises(ValueError, match=r"eopc04\.txt: line 2: C04 requires"):
        read_iers_c04(path)


def test_gcrs_to_itrs_matrix_matches_the_erfa_cio_chain():
    epoch = Epoch.from_calendar(2020, 1, 1, 6, scale=TimeScale.UTC)
    eop = TabulatedEarthOrientation((EarthOrientationSample(epoch.mjd, 0.076, 0.282, -0.177, 0.0002, -0.0003),))
    hf = high_frequency_eop_correction(epoch, background_ut1_minus_utc_s=-0.177)

    actual = TerrestrialFrameTransform(eop).gcrs2itrf_matrix(epoch)

    expected = _erfa_gcrs2itrs(
        epoch, 0.076 + hf.delta_xp_arcsec, 0.282 + hf.delta_yp_arcsec, -0.177 + hf.delta_ut1_s, 0.0002, -0.0003
    )
    assert_close(actual, expected, ANGLE_RAD)
    assert_close(actual @ actual.T, np.eye(3), ANGLE_RAD)


def test_gcrs_itrs_vector_transforms_are_inverse_and_the_cached_matrix_is_read_only():
    epoch = Epoch.from_calendar(2020, 1, 1, 6, scale=TimeScale.UTC)
    transform = TerrestrialFrameTransform(
        TabulatedEarthOrientation((EarthOrientationSample(epoch.mjd, 0.076, 0.282, -0.177),))
    )
    gcrs = np.array([6_378_137.0, -1.2e6, 3.4e5])

    itrf = transform.gcrs2itrf(gcrs, epoch)

    assert_close(itrf, transform.gcrs2itrf_matrix(epoch) @ gcrs, Tolerance("ITRF_M", atol=1e-8, unit="m"))
    assert_close(transform.itrf2gcrs(itrf, epoch), gcrs, Tolerance("ROUND_TRIP_M", atol=1e-8, unit="m"))
    with pytest.raises(ValueError):
        transform.gcrs2itrf_matrix(epoch)[0, 0] = 0.0
