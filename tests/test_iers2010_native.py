from __future__ import annotations

import importlib.resources
import os
import shutil
import subprocess
import sys
from importlib import import_module
from typing import Any, cast

import numpy as np
import pytest

_iers2010 = cast(Any, import_module("lunarops._iers2010"))


_FCUL_ZD_EXPECTED_M = np.array(
    [
        1.935225924846803114,
        1.932992176591644462,
        0.002233748255158703871,
    ]
)

_HARDISP_ONSALA_AMP = np.array(
    [
        [0.00352, 0.00123, 0.00080, 0.00032, 0.00187, 0.00112, 0.00063, 0.00003, 0.00082, 0.00044, 0.00037],
        [0.00144, 0.00035, 0.00035, 0.00008, 0.00053, 0.00049, 0.00018, 0.00009, 0.00012, 0.00005, 0.00006],
        [0.00086, 0.00023, 0.00023, 0.00006, 0.00029, 0.00028, 0.00010, 0.00007, 0.00004, 0.00002, 0.00001],
    ]
)
_HARDISP_ONSALA_PHASE = np.array(
    [
        [-64.7, -52.0, -96.2, -55.2, -58.8, -151.4, -65.6, -138.1, 8.4, 5.2, 2.1],
        [85.5, 114.5, 56.5, 113.6, 99.4, 19.1, 94.1, -10.4, -167.4, -170.0, -177.7],
        [109.5, 147.0, 92.7, 148.8, 50.5, -55.1, 36.4, -170.4, -15.0, 2.3, 5.2],
    ]
)
_HARDISP_ONSALA_EXPECTED = np.array(
    [
        [0.003094, -0.001538, -0.000895],
        [0.001812, -0.000950, -0.000193],
        [0.000218, -0.000248, 0.000421],
        [-0.001104, 0.000404, 0.000741],
        [-0.001668, 0.000863, 0.000646],
        [-0.001209, 0.001042, 0.000137],
        [0.000235, 0.000926, -0.000667],
        [0.002337, 0.000580, -0.001555],
        [0.004554, 0.000125, -0.002278],
        [0.006271, -0.000291, -0.002615],
        [0.006955, -0.000537, -0.002430],
        [0.006299, -0.000526, -0.001706],
        [0.004305, -0.000244, -0.000559],
        [0.001294, 0.000245, 0.000793],
        [-0.002163, 0.000819, 0.002075],
        [-0.005375, 0.001326, 0.003024],
        [-0.007695, 0.001622, 0.003448],
        [-0.008669, 0.001610, 0.003272],
        [-0.008143, 0.001262, 0.002557],
        [-0.006290, 0.000633, 0.001477],
        [-0.003566, -0.000155, 0.000282],
        [-0.000593, -0.000941, -0.000766],
        [0.001992, -0.001561, -0.001457],
        [0.003689, -0.001889, -0.001680],
    ]
)

_HARDISP_UTC_REGRESSION_CASES = (
    (
        (2016, 12, 31, 23, 59, 59),
        np.array([0.004732082132250071, -0.0005435922648757696, -0.0012600324116647243]),
    ),
    (
        (2017, 1, 1, 0, 0, 0),
        np.array([0.00473177433013916, -0.0005435359198600054, -0.0012598361354321241]),
    ),
    (
        (2009, 6, 25, 1, 10, 45),
        np.array([0.003094081999734044, -0.0015382986748591065, -0.0008953595533967018]),
    ),
    (
        (2024, 1, 1, 0, 0, 0),
        np.array([0.0042690373957157135, -0.00022759537387173623, -0.0016440704930573702]),
    ),
)


def test_fcul_a_matches_iers_reference_case():
    mapping = _iers2010.fcul_a(30.67166667, 2075.0, 300.15, 15.0)

    assert mapping == pytest.approx(3.800243667312344087, rel=0.0, abs=1.0e-15)


def test_fculzd_hpa_matches_iers_reference_outputs():
    # The v1.3.0 source header says 2010.344 m, but its three reference
    # outputs are exactly reproduced with 2003.344 m. The upstream source is
    # preserved unchanged and the discrepancy is recorded in the build notes.
    actual = _iers2010.fculzd_hpa(
        30.67166667,
        2003.344,
        798.4188,
        14.322,
        0.532,
    )

    np.testing.assert_allclose(actual, _FCUL_ZD_EXPECTED_M, rtol=0.0, atol=1.0e-15)


def test_ortho_eop_matches_iers_reference_case():
    actual = _iers2010.ortho_eop(47100.0)
    np.testing.assert_allclose(
        actual,
        (-162.8386373279636530, 117.7907525842668974, -23.39092370609808214),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_pmsdnut2_matches_iers_reference_case():
    actual = _iers2010.pmsdnut2(54335.0)
    np.testing.assert_allclose(
        actual,
        (24.83144238273364834, -14.09240692041837661),
        rtol=0.0,
        # ERFA's IAU 2003 fundamental-argument constants differ from the
        # printed IERS routine constants in their final digits.
        atol=5.0e-10,
    )


def test_utlibr_matches_iers_reference_cases():
    np.testing.assert_allclose(
        _iers2010.utlibr(44239.1),
        (2.441143834386761746, -14.78971247349449492),
        rtol=0.0,
        atol=5.0e-10,
    )
    np.testing.assert_allclose(
        _iers2010.utlibr(55227.4),
        (-2.655705844335680244, 27.39445826599846967),
        rtol=0.0,
        atol=5.0e-10,
    )


def test_fundarg_matches_iers_reference_case():
    actual = _iers2010.fundarg(0.07995893223819302)
    np.testing.assert_allclose(
        actual,
        (
            2.291187512612069099,
            6.212931111003726414,
            3.658025792050572989,
            4.554139562402433228,
            -0.5167379217231804489,
        ),
        rtol=0.0,
        atol=2.0e-11,
    )


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
        (
            (1112152.8166, -4842857.5435, 3985496.1783),
            (8382471154.1312895, 10512408445.356153, -5360583240.3763866),
            (380934092.93550891, 2871428.1904491195, 79015680.553570181),
            (2017, 1, 15, 0.0),
            (-18.217357581922339, -23.505348376537949, 12.097611382175685),
        ),
    ],
)
def test_dehanttideinel_matches_iers_reference_and_source_cases(xsta, xsun, xmon, date, expected):
    # The v1.3.0 header's 2017 expected output is copied from the preceding
    # 2015 case despite a solar vector only 14.5 Gm long. Its source-input
    # calculation is retained as the regression value; see IERS_NATIVE_BUILD.
    actual = _iers2010.dehanttideinel(
        xsta,
        date[0],
        date[1],
        date[2],
        date[3],
        xsun,
        xmon,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_hardisp_matches_official_onsala_case():
    assert not hasattr(_iers2010, "lunarops_hardisp")

    actual = np.column_stack(
        _iers2010.hardisp(
            2009,
            6,
            25,
            1,
            10,
            45,
            24,
            3600.0,
            _HARDISP_ONSALA_AMP,
            _HARDISP_ONSALA_PHASE,
        )
    )
    np.testing.assert_allclose(actual, _HARDISP_ONSALA_EXPECTED, rtol=0.0, atol=6.0e-7)


@pytest.mark.parametrize(("calendar", "expected"), _HARDISP_UTC_REGRESSION_CASES)
def test_hardisp_utc_leap_boundary_and_nonzero_time(calendar, expected):
    actual = np.asarray(
        _iers2010.hardisp(
            *calendar,
            1,
            1.0,
            _HARDISP_ONSALA_AMP,
            _HARDISP_ONSALA_PHASE,
        ),
        dtype=float,
    ).reshape(3)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-9)


def test_hardisp_rejects_exact_utc_leap_second_label():
    with pytest.raises(ValueError, match="exact UTC leap-second label"):
        _iers2010.hardisp(
            2016,
            12,
            31,
            23,
            59,
            60,
            1,
            1.0,
            _HARDISP_ONSALA_AMP,
            _HARDISP_ONSALA_PHASE,
        )


def test_hardisp_rejects_regular_series_across_utc_offset_transition():
    with pytest.raises(ValueError, match="must not cross a UTC offset transition"):
        _iers2010.hardisp(
            2016,
            12,
            31,
            23,
            59,
            59,
            3,
            1.0,
            _HARDISP_ONSALA_AMP,
            _HARDISP_ONSALA_PHASE,
        )


def test_hardisp_regular_series_matches_individual_epochs():
    series = np.column_stack(
        _iers2010.hardisp(
            2009,
            6,
            25,
            1,
            10,
            45,
            4,
            900.0,
            _HARDISP_ONSALA_AMP,
            _HARDISP_ONSALA_PHASE,
        )
    )
    scalar = np.array(
        [
            np.asarray(
                _iers2010.hardisp(
                    2009,
                    6,
                    25,
                    1,
                    minute,
                    45,
                    1,
                    1.0,
                    _HARDISP_ONSALA_AMP,
                    _HARDISP_ONSALA_PHASE,
                ),
                dtype=float,
            ).reshape(3)
            for minute in (10, 25, 40, 55)
        ]
    )
    np.testing.assert_allclose(series, scalar, rtol=0.0, atol=5.0e-9)


def test_installed_iers_cython_sources_are_present():
    root = importlib.resources.files("lunarops")
    assert "Cython implementation" in root.joinpath("_iers2010_core.pyx").read_text(encoding="utf-8")
    assert "Generated mechanically" in root.joinpath("_iers2010_tables.pxi").read_text(encoding="utf-8")
    assert not root.joinpath("_external", "iers2010", "src").is_dir()
    assert not root.joinpath("_external", "iers2010", "bindings").is_dir()


def test_native_extension_imports_in_mpi_workers():
    mpi_runner = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpi_runner is None:
        pytest.skip("MPI launcher is not installed")
    pytest.importorskip("mpi4py")
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1:
        pytest.skip("do not start a nested MPI job")

    worker_code = """
from mpi4py import MPI
from lunarops import _iers2010

value = _iers2010.fcul_a(30.67166667, 2075.0, 300.15, 15.0)
values = MPI.COMM_WORLD.allgather(value)
assert len(values) == 2
assert all(abs(item - 3.800243667312344087) < 1.0e-15 for item in values)
"""
    env = os.environ.copy()
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    subprocess.run(
        [mpi_runner, "-n", "2", sys.executable, "-c", worker_code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
