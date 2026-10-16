"""Independent JPL pole formulas, ERFA component checks and time conventions."""

import erfa
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lunarops.classes.dynamics.earth_gravity_orientation import (
    de430_earth_fixed2inertial_matrix,
    de440_earth_fixed2inertial_matrix,
    de440_earth_pole_vector_inertial,
)
from lunarops.classes.time import Epoch

AS2R = np.pi / (180 * 3600)


def paper_pole(year, model):
    """Direct published pole expression and independent active SciPy rotations."""
    t = (year - 2000) / 100
    rate, a, b = (6962890.549, -17.206262, 9.205348) if model == "de430" else (6962890.539, -17.1996, 9.2025)
    omega = (450160.280 - rate*t + 7.455*t**2 + .008*t**3) * AS2R
    dp, de = a*AS2R*np.sin(omega), b*AS2R*np.cos(omega)
    eps = (84381.448 - 46.815*t - .00059*t**2 + .001813*t**3) * AS2R
    pole = np.array([np.sin(dp)*np.sin(eps+de),
                     np.cos(dp)*np.sin(eps+de)*np.cos(eps)-np.cos(eps+de)*np.sin(eps),
                     np.cos(dp)*np.sin(eps+de)*np.sin(eps)+np.cos(eps+de)*np.cos(eps)])
    if model == "de430":
        fx = (.005675420332289347 + .0002768991557448355*(year-2000))*AS2R
        fy = (-.01702265691498953 - .001211859121655924*(year-2000))*AS2R
        # R passive(-angle) equals SciPy active(+angle); rightmost applied first.
        pole = Rotation.from_rotvec([0, fy, 0]).apply(pole)
        pole = Rotation.from_rotvec([fx, 0, 0]).apply(pole)
        zeta = (2306.2181*t+.30188*t**2+.017998*t**3)*AS2R
        theta = (2004.3109*t-.42665*t**2-.041833*t**3)*AS2R
        z = (2306.2181*t+1.09468*t**2+.018203*t**3)*AS2R
        pole = Rotation.from_rotvec([0, 0, -z]).apply(pole)
        pole = Rotation.from_rotvec([0, theta, 0]).apply(pole)
        return Rotation.from_rotvec([0, 0, -zeta]).apply(pole)
    pole = Rotation.from_rotvec([0, -.017128824840317532*AS2R, 0]).apply(pole)
    pole = Rotation.from_rotvec([.005343991604414805*AS2R, 0, 0]).apply(pole)
    # Park Eq. (26), independently reconstruct using long-term pole services.
    n, k = erfa.ltpequ(year), erfa.ltpecl(year)
    x = np.cross(n, k); x /= np.linalg.norm(x)
    y = np.cross(n, x); y /= np.linalg.norm(y)
    return np.column_stack((x, y, n)) @ pole


@pytest.mark.parametrize("model", ["de430", "de440"])
@pytest.mark.parametrize("year", [1550., 1800., 1969.5, 2000., 2026., 2150., 2650.])
def test_pole_matches_independent_published_expression(model, year):
    epoch = Epoch(2451545., (year - 2000)*365.25, "tdb")
    matrix = (de430_earth_fixed2inertial_matrix if model == "de430" else de440_earth_fixed2inertial_matrix)(epoch)
    np.testing.assert_allclose(matrix[:, 2], paper_pole(year, model), rtol=0, atol=8e-16)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=7e-16)
    assert np.linalg.det(matrix) == pytest.approx(1., abs=7e-16)
    assert not matrix.flags.writeable
    if model == "de440":
        np.testing.assert_allclose(matrix[:, 2], de440_earth_pole_vector_inertial(epoch), rtol=0, atol=2e-16)


@pytest.mark.parametrize("provider", [de430_earth_fixed2inertial_matrix, de440_earth_fixed2inertial_matrix])
def test_split_jd_precision_and_tdb_requirement(provider):
    # Equivalent splits, including a fractional day that disappears when summing JD.
    first = Epoch(2451545., -10000.+1e-10, "tdb")
    second = Epoch(2441545., 1e-10, "tdb")
    np.testing.assert_array_equal(provider(first), provider(second))
    with pytest.raises(ValueError):
        provider(Epoch(2451545., 0., "tt"))


def test_full_iau_nutation_and_standard_frame_bias_are_different_models():
    epoch = Epoch(2451545., 0., "tdb")
    full_nutation = erfa.numat(erfa.obl80(epoch.jd1, epoch.jd2), *erfa.nut80(epoch.jd1, epoch.jd2))
    assert np.linalg.norm(full_nutation[2] - paper_pole(2000., "de440")) > 1e-6
    assert np.linalg.norm(erfa.ltp(2000.)[2] - erfa.ltpb(2000.)[2]) > 1e-8
