"""IERS Conventions v1.3.0 routines against the test cases printed in their source headers."""

import pytest

from lunarops import _iers2010
from tests.tolerances import DISPLACEMENT_M, EOP_UAS, REL, Tolerance, assert_close

# ERFA's IAU 2003 fundamental-argument constants differ from the printed IERS
# routine constants in their final digits.
ERFA_CONSTANTS_UAS = Tolerance("ERFA_CONSTANTS_UAS", atol=2e-9, unit="uas")
ERFA_CONSTANTS_RAD = Tolerance("ERFA_CONSTANTS_RAD", atol=2e-11, unit="rad")


def test_fcul_a_mapping_function():
    assert_close(_iers2010.fcul_a(30.67166667, 2075.0, 300.15, 15.0), 3.800243667312344087, REL)


def test_fculzd_hpa_zenith_delays():
    # The source header lists 2010.344 m, but its three outputs are reproduced with 2003.344 m.
    actual = _iers2010.fculzd_hpa(30.67166667, 2003.344, 798.4188, 14.322, 0.532)
    assert_close(actual, (1.935225924846803114, 1.932992176591644462, 0.002233748255158703871), DISPLACEMENT_M)


def test_ortho_eop_ocean_tide_terms():
    expected = (-162.8386373279636530, 117.7907525842668974, -23.39092370609808214)
    assert_close(_iers2010.ortho_eop(47100.0), expected, EOP_UAS)


def test_pmsdnut2_and_utlibr_libration_terms():
    assert_close(_iers2010.pmsdnut2(54335.0), (24.83144238273364834, -14.09240692041837661), ERFA_CONSTANTS_UAS)
    assert_close(_iers2010.utlibr(44239.1), (2.441143834386761746, -14.78971247349449492), ERFA_CONSTANTS_UAS)
    assert_close(_iers2010.utlibr(55227.4), (-2.655705844335680244, 27.39445826599846967), ERFA_CONSTANTS_UAS)


def test_fundarg_delaunay_arguments():
    expected = (
        2.291187512612069099,
        6.212931111003726414,
        3.658025792050572989,
        4.554139562402433228,
        -0.5167379217231804489,
    )
    assert_close(_iers2010.fundarg(0.07995893223819302), expected, ERFA_CONSTANTS_RAD)


@pytest.mark.parametrize(
    ("xsta", "xsun", "xmon", "date", "expected"),
    [
        (
            (4075578.385, 931852.890, 4801570.154),
            (137859926952.015, 54228127881.4350, 23509422341.6960),
            (-179996231.920342, -312468450.131567, -169288918.592160),
            (2009, 4, 13, 0.0),
            (0.7700420357108125891e-1, 0.6304056321824967613e-1, 0.5516568152597246810e-1),
        ),
        (
            (1112189.660, -4842955.026, 3985352.284),
            (-54537460436.2357, 130244288385.279, 56463429031.5996),
            (300396716.912, 243238281.451, 120548075.939),
            (2012, 7, 13, 0.0),
            (-0.2036831479592075833e-1, 0.5658254776225972449e-1, -0.7597679676871742227e-1),
        ),
        (
            (1112200.5696, -4842957.8511, 3985345.9122),
            (100210282451.6279, 103055630398.3160, 56855096480.4475),
            (369817604.4348, 1897917.5258, 120804980.8284),
            (2015, 7, 15, 0.0),
            (0.00509570869172363845, 0.0828663025983528700, -0.0636634925404189617),
        ),
        # The header's 2017 output is a copy of the 2015 case although its Sun is
        # only 14.5 Gm away; the value below is this case's own computed result.
        (
            (1112152.8166, -4842857.5435, 3985496.1783),
            (8382471154.1312895, 10512408445.356153, -5360583240.3763866),
            (380934092.93550891, 2871428.1904491195, 79015680.553570181),
            (2017, 1, 15, 0.0),
            (-18.217357581922339, -23.505348376537949, 12.097611382175685),
        ),
    ],
)
def test_dehanttideinel_solid_earth_tide(xsta, xsun, xmon, date, expected):
    assert_close(_iers2010.dehanttideinel(xsta, *date, xsun, xmon), expected, DISPLACEMENT_M)
