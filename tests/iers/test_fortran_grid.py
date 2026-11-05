"""Compiled IERS routines against frozen outputs of the IERS v1.3.0 reference software.

The baseline grid and its provenance are described in docs/developer/building.md.
"""

import hashlib

import numpy as np
import pytest

from lunarops import _iers2010
from tests.conftest import DATA_DIR
from tests.tolerances import DISPLACEMENT_M, EOP_UAS, REL, Tolerance, assert_close

BASELINE = DATA_DIR / "iers_fortran_baseline.npz"
BASELINE_SHA256 = "26ac513ae09027ef391ce2d58928e017ee688e1f07fa30677c2da92b496feb0e"
# ERFA's fundamental-argument constants differ from the Fortran ones in their final digits.
ERFA_CONSTANTS_UAS = Tolerance("ERFA_CONSTANTS_UAS", atol=2e-9, unit="uas")
ERFA_CONSTANTS_RAD = Tolerance("ERFA_CONSTANTS_RAD", atol=2e-11, unit="rad")
# The Fortran ETUTC table had a leap-second error that shifts HARDISP by up to 2.7e-7 m.
FORTRAN_ETUTC_M = Tolerance("FORTRAN_ETUTC_M", atol=5e-7, unit="m")


@pytest.fixture(scope="module")
def baseline():
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA256
    with np.load(BASELINE) as data:
        yield {key: data[key] for key in data.files}


def test_delay_mapping_and_zenith_delays(baseline):
    assert_close([_iers2010.fcul_a(*row) for row in baseline["fcul_inputs"]], baseline["fcul_out"], REL)
    assert_close([_iers2010.fculzd_hpa(*row) for row in baseline["zd_inputs"]], baseline["zd_out"], DISPLACEMENT_M)


def test_high_frequency_eop_series(baseline):
    mjd = baseline["rmjd"]
    assert_close([_iers2010.ortho_eop(value) for value in mjd], baseline["eop"], EOP_UAS)
    assert_close([_iers2010.pmsdnut2(value) for value in mjd], baseline["pms"], ERFA_CONSTANTS_UAS)
    assert_close([_iers2010.utlibr(value) for value in mjd], baseline["ut"], ERFA_CONSTANTS_UAS)
    fund = [_iers2010.fundarg((value - 51_544.5) / 36_525.0) for value in mjd]
    assert_close(fund, baseline["fund"], ERFA_CONSTANTS_RAD)


def test_solid_earth_tide(baseline):
    actual = [
        _iers2010.dehanttideinel(row[:3], int(row[3]), int(row[4]), int(row[5]), row[6], row[7:10], row[10:13])
        for row in baseline["deh_inputs"]
    ]
    assert_close(actual, baseline["deh_out"], DISPLACEMENT_M)


def test_ocean_loading(baseline):
    actual = np.asarray([
        _iers2010.hardisp(*(int(value) for value in row[:6]), 1, 900.0, row[6:39].reshape(3, 11),
                          row[39:72].reshape(3, 11))
        for row in baseline["hard_inputs"]
    ]).reshape(-1, 3)
    # The Fortran baseline stores (up, south, west).
    assert_close(actual, baseline["hard_scalar"][:, [0, 2, 1]], FORTRAN_ETUTC_M)
