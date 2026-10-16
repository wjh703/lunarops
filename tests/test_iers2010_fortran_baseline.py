"""Differential regression against outputs frozen before Fortran removal."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from lunarops import _iers2010

_BASELINE = Path(__file__).with_name("data") / "iers_fortran_baseline.npz"
_BASELINE_SHA256 = "26ac513ae09027ef391ce2d58928e017ee688e1f07fa30677c2da92b496feb0e"


def test_cython_backend_matches_frozen_fortran_grid():
    assert hashlib.sha256(_BASELINE.read_bytes()).hexdigest() == _BASELINE_SHA256

    with np.load(_BASELINE) as data:
        fcul = np.asarray([_iers2010.fcul_a(*row) for row in data["fcul_inputs"]])
        zd = np.asarray([_iers2010.fculzd_hpa(*row) for row in data["zd_inputs"]])
        eop = np.asarray([_iers2010.ortho_eop(mjd) for mjd in data["rmjd"]])
        pms = np.asarray([_iers2010.pmsdnut2(mjd) for mjd in data["rmjd"]])
        ut = np.asarray([_iers2010.utlibr(mjd) for mjd in data["rmjd"]])
        fund = np.asarray([_iers2010.fundarg((mjd - 51_544.5) / 36_525.0) for mjd in data["rmjd"]])
        dehant = np.asarray(
            [
                _iers2010.dehanttideinel(
                    row[:3],
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    row[6],
                    row[7:10],
                    row[10:13],
                )
                for row in data["deh_inputs"]
            ]
        )
        hardisp = np.asarray(
            [
                _iers2010.hardisp(
                    *(int(value) for value in row[:6]),
                    1,
                    900.0,
                    row[6:39].reshape(3, 11),
                    row[39:72].reshape(3, 11),
                )
                for row in data["hard_inputs"]
            ]
        ).reshape(-1, 3)

        np.testing.assert_allclose(fcul, data["fcul_out"], rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(zd, data["zd_out"], rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(eop, data["eop"], rtol=0.0, atol=2.0e-11)
        np.testing.assert_allclose(pms, data["pms"], rtol=0.0, atol=5.0e-10)
        np.testing.assert_allclose(ut, data["ut"], rtol=0.0, atol=5.0e-10)
        np.testing.assert_allclose(fund, data["fund"], rtol=0.0, atol=2.0e-11)
        np.testing.assert_allclose(dehant, data["deh_out"], rtol=0.0, atol=1.0e-13)
        np.testing.assert_allclose(hardisp, data["hard_scalar"][:, [0, 2, 1]], rtol=0.0, atol=5.0e-7)
