import erfa
import numpy as np
import pytest
from typing import Any, cast

from lunarops.classes.time import Epoch, TimeScale, utc2tt
from lunarops.classes.frames import EarthOrientationSample, TabulatedEarthOrientation
from lunarops.classes.frames.earth_orientation import read_iers_eop
from lunarops.classes.frames.high_frequency_eop import (
    HighFrequencyEopCorrection,
    earth_rotation_libration_eop_correction,
    high_frequency_eop_correction,
    ocean_tide_eop_correction,
)
from lunarops.classes.frames.terrestrial import TerrestrialFrameTransform


_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)


def test_ocean_tide_correction_matches_iers_ortho_eop_reference():
    correction = ocean_tide_eop_correction(
        Epoch(2_400_000.5, 47_100.0, TimeScale.UTC),
        background_ut1_minus_utc_s=0.0,
    )

    np.testing.assert_allclose(
        correction.ocean_delta_xp_arcsec * 1.0e6,
        -162.8386373279636530,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        correction.ocean_delta_yp_arcsec * 1.0e6,
        117.7907525842668974,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        correction.ocean_delta_ut1_s * 1.0e6,
        -23.39092370609808214,
        rtol=0.0,
        atol=2.0e-13,
    )


def test_libration_corrections_match_iers_reference_values():
    polar_motion = earth_rotation_libration_eop_correction(Epoch(2_400_000.5, 54_335.0, TimeScale.TT))
    np.testing.assert_allclose(
        [polar_motion.libration_delta_xp_arcsec * 1.0e6, polar_motion.libration_delta_yp_arcsec * 1.0e6],
        [24.83144238273364834, -14.09240692041837661],
        rtol=0.0,
        atol=2.0e-5,
    )

    np.testing.assert_allclose(
        earth_rotation_libration_eop_correction(Epoch(2_400_000.5, 44_239.1, TimeScale.TT)).libration_delta_ut1_s
        * 1.0e6,
        2.441143834386761746,
        rtol=0.0,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        earth_rotation_libration_eop_correction(Epoch(2_400_000.5, 55_227.4, TimeScale.TT)).libration_delta_ut1_s
        * 1.0e6,
        -2.655705844335680244,
        rtol=0.0,
        atol=2.0e-8,
    )


def test_high_frequency_result_retains_named_components_and_lod():
    epoch = Epoch(2_400_000.5, 55_227.4, TimeScale.UTC)
    result = high_frequency_eop_correction(
        epoch,
        background_ut1_minus_utc_s=0.0,
    )
    ocean = ocean_tide_eop_correction(epoch, background_ut1_minus_utc_s=0.0)
    libration = earth_rotation_libration_eop_correction(utc2tt(epoch))

    assert result.ocean_delta_xp_arcsec == ocean.ocean_delta_xp_arcsec
    assert result.libration_delta_yp_arcsec == libration.libration_delta_yp_arcsec
    assert result.libration_delta_lod_s_per_day == libration.libration_delta_lod_s_per_day
    assert result.delta_xp_arcsec == ocean.delta_xp_arcsec + libration.delta_xp_arcsec
    assert result.delta_yp_arcsec == ocean.delta_yp_arcsec + libration.delta_yp_arcsec
    assert result.delta_ut1_s == ocean.delta_ut1_s + libration.delta_ut1_s
    np.testing.assert_allclose(result.libration_delta_lod_s_per_day, 27.0861697892672e-6)


def test_high_frequency_requires_explicit_utc_epoch():
    with pytest.raises(TypeError, match="requires an Epoch"):
        cast(Any, ocean_tide_eop_correction)(47_100.0, background_ut1_minus_utc_s=0.0)
    with pytest.raises(ValueError, match="epoch_utc must use the UTC scale"):
        ocean_tide_eop_correction(
            Epoch(2_400_000.5, 47_100.0, TimeScale.TT),
            background_ut1_minus_utc_s=0.0,
        )
    with pytest.raises(ValueError, match="requires a TT or TDB Epoch"):
        earth_rotation_libration_eop_correction(Epoch(2_400_000.5, 47_100.0, TimeScale.UTC))


def test_c04_parser_retains_dx_dy_for_supported_layouts(tmp_path):
    path = tmp_path / "eopc04.txt"
    path.write_text(
        "2020 1 1 58849 0.076 0.282 -0.177 0.001 0.0002 -0.0003\n"
        "2020 1 2 0 58850 0.077 0.283 -0.178 0.0004 -0.0005 0.0\n",
        encoding="utf-8",
    )

    first, second = read_iers_eop(path)
    assert (first.dx_arcsec, first.dy_arcsec) == (0.0002, -0.0003)
    assert (second.dx_arcsec, second.dy_arcsec) == (0.0004, -0.0005)


def test_ut1_interpolation_removes_leap_second_discontinuity():
    eop = TabulatedEarthOrientation(
        (
            EarthOrientationSample(57_753.0, 0.0, 0.0, -0.4),
            EarthOrientationSample(57_754.0, 0.0, 0.0, 0.6),
        )
    )
    midday = Epoch.from_calendar(2016, 12, 31, 12, scale=TimeScale.UTC)

    np.testing.assert_allclose(eop.ut1_minus_utc_s(midday), -0.4, rtol=0.0, atol=2.0e-11)


def test_terrestrial_matrix_applies_celestial_pole_offsets(monkeypatch):
    import lunarops.classes.frames.terrestrial as terrestrial_module

    epoch = Epoch.from_calendar(2020, 1, 1, 6, scale=TimeScale.UTC)
    eop = TabulatedEarthOrientation(
        (
            EarthOrientationSample(
                epoch.mjd,
                0.076,
                0.282,
                -0.177,
                0.0002,
                -0.0003,
            ),
        )
    )
    monkeypatch.setattr(
        terrestrial_module,
        "high_frequency_eop_correction",
        lambda epoch, **kwargs: HighFrequencyEopCorrection(),
    )

    actual = TerrestrialFrameTransform(eop).gcrs2itrf_matrix(epoch)

    tt = utc2tt(epoch)
    ut11, ut12 = erfa.utcut1(epoch.jd1, epoch.jd2, -0.177)
    x, y, s = erfa.xys06a(tt.jd1, tt.jd2)
    rc2i = erfa.c2ixys(x + 0.0002 * _ARCSEC_TO_RAD, y - 0.0003 * _ARCSEC_TO_RAD, s)
    rpom = erfa.pom00(
        0.076 * _ARCSEC_TO_RAD,
        0.282 * _ARCSEC_TO_RAD,
        erfa.sp00(tt.jd1, tt.jd2),
    )
    expected = erfa.c2tcio(rc2i, erfa.era00(ut11, ut12), rpom)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(actual @ actual.T, np.eye(3), rtol=0.0, atol=2.0e-15)


def test_terrestrial_matrix_applies_native_high_frequency_eop():
    epoch = Epoch.from_calendar(2020, 1, 1, 6, scale=TimeScale.UTC)
    eop = TabulatedEarthOrientation((EarthOrientationSample(epoch.mjd, 0.076, 0.282, -0.177),))
    actual = TerrestrialFrameTransform(eop).gcrs2itrf_matrix(epoch)

    high_frequency = high_frequency_eop_correction(
        epoch,
        background_ut1_minus_utc_s=-0.177,
    )
    tt = utc2tt(epoch)
    dut1_s = -0.177 + high_frequency.delta_ut1_s
    ut11, ut12 = erfa.utcut1(epoch.jd1, epoch.jd2, dut1_s)
    x, y, s = erfa.xys06a(tt.jd1, tt.jd2)
    rc2i = erfa.c2ixys(x, y, s)
    rpom = erfa.pom00(
        (0.076 + high_frequency.delta_xp_arcsec) * _ARCSEC_TO_RAD,
        (0.282 + high_frequency.delta_yp_arcsec) * _ARCSEC_TO_RAD,
        erfa.sp00(tt.jd1, tt.jd2),
    )
    expected = np.asarray(erfa.c2tcio(rc2i, erfa.era00(ut11, ut12), rpom), dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)
