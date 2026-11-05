"""Earth figure orientation: DE430/DE440 pole formulas and the IAU 2006/2000A CIP."""

import erfa
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lunarops.classes.dynamics.assembly import build_lunar_propagation_problem
from lunarops.classes.dynamics.earth_figure_orientation import (
    de430_earth_fixed2inertial_matrix,
    de440_earth_fixed2inertial_matrix,
    de440_earth_pole_vector_inertial,
    iau2006_earth_fixed2inertial_matrix,
    iau2006_earth_pole_vector_inertial,
)
from lunarops.classes.time import Epoch, TimeScaleConverter
from tests.support.dynamics import orbit_program_config
from tests.tolerances import ANGLE_RAD, REL, TIME_S, assert_close

AS2R = np.pi / (180 * 3600)


def published_pole(year, model):
    """JPL DE430/DE440 pole expressions with independent active SciPy rotations."""
    t = (year - 2000) / 100
    rate, a, b = (6962890.549, -17.206262, 9.205348) if model == "de430" else (6962890.539, -17.1996, 9.2025)
    omega = (450160.280 - rate * t + 7.455 * t**2 + .008 * t**3) * AS2R
    dp, de = a * AS2R * np.sin(omega), b * AS2R * np.cos(omega)
    eps = (84381.448 - 46.815 * t - .00059 * t**2 + .001813 * t**3) * AS2R
    pole = np.array([np.sin(dp) * np.sin(eps + de),
                     np.cos(dp) * np.sin(eps + de) * np.cos(eps) - np.cos(eps + de) * np.sin(eps),
                     np.cos(dp) * np.sin(eps + de) * np.sin(eps) + np.cos(eps + de) * np.cos(eps)])
    if model == "de430":
        fx = (.005675420332289347 + .0002768991557448355 * (year - 2000)) * AS2R
        fy = (-.01702265691498953 - .001211859121655924 * (year - 2000)) * AS2R
        # A passive rotation by -angle equals SciPy's active rotation by +angle; rightmost first.
        pole = Rotation.from_rotvec([0, fy, 0]).apply(pole)
        pole = Rotation.from_rotvec([fx, 0, 0]).apply(pole)
        zeta = (2306.2181 * t + .30188 * t**2 + .017998 * t**3) * AS2R
        theta = (2004.3109 * t - .42665 * t**2 - .041833 * t**3) * AS2R
        z = (2306.2181 * t + 1.09468 * t**2 + .018203 * t**3) * AS2R
        pole = Rotation.from_rotvec([0, 0, -z]).apply(pole)
        pole = Rotation.from_rotvec([0, theta, 0]).apply(pole)
        return Rotation.from_rotvec([0, 0, -zeta]).apply(pole)
    pole = Rotation.from_rotvec([0, -.017128824840317532 * AS2R, 0]).apply(pole)
    pole = Rotation.from_rotvec([.005343991604414805 * AS2R, 0, 0]).apply(pole)
    # Park et al. Eq. (26), reconstructed with the long-term precession services.
    n, k = erfa.ltpequ(year), erfa.ltpecl(year)
    x = np.cross(n, k) / np.linalg.norm(np.cross(n, k))
    y = np.cross(n, x) / np.linalg.norm(np.cross(n, x))
    return np.column_stack((x, y, n)) @ pole


@pytest.mark.parametrize("model", ["de430", "de440"])
@pytest.mark.parametrize("year", [1550., 1800., 1969.5, 2000., 2026., 2150., 2650.])
def test_jpl_pole_matches_the_published_expression(model, year):
    epoch = Epoch(2451545., (year - 2000) * 365.25, "tdb")
    matrix = (de430_earth_fixed2inertial_matrix if model == "de430" else de440_earth_fixed2inertial_matrix)(epoch)

    assert_close(matrix[:, 2], published_pole(year, model), ANGLE_RAD)
    assert_close(matrix.T @ matrix, np.eye(3), ANGLE_RAD)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-14)
    if model == "de440":
        assert_close(matrix[:, 2], de440_earth_pole_vector_inertial(epoch), ANGLE_RAD)


@pytest.mark.parametrize("provider", [de430_earth_fixed2inertial_matrix, de440_earth_fixed2inertial_matrix,
                                      iau2006_earth_fixed2inertial_matrix])
def test_equivalent_jd_splits_agree_and_tdb_is_required(provider):
    first = Epoch(2451545., -10000. + 1e-10, "tdb")
    second = Epoch(2441545., 1e-10, "tdb")
    assert_close(provider(first), provider(second), ANGLE_RAD)
    with pytest.raises(ValueError):
        provider(Epoch(2451545., 0., "tt"))


@pytest.mark.parametrize("jd", [2440222.5, 2440400.5, 2451545.0, 2460676.5, 2462867.5])
def test_iau2006_pole_is_the_cip_of_xys06a_at_tt(jd):
    epoch = Epoch(jd, 0.123456789, "tdb")
    tt = TimeScaleConverter().tdb2tt(epoch)
    x, y, s = erfa.xys06a(tt.jd1, tt.jd2)
    expected = np.array([x, y, np.sqrt(1 - x * x - y * y)])
    pole = iau2006_earth_pole_vector_inertial(epoch)
    matrix = iau2006_earth_fixed2inertial_matrix(epoch)

    assert_close(pole, expected, ANGLE_RAD)
    assert_close(erfa.c2ixys(x, y, s) @ pole, [0, 0, 1], ANGLE_RAD)
    assert_close(matrix @ [0, 0, 1], pole, ANGLE_RAD)
    assert_close(matrix.T @ matrix, np.eye(3), ANGLE_RAD)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-14)


def test_iau2006_pole_converts_tdb_with_the_geocentric_dtdb():
    epoch = Epoch(2451545.0, 0.25, "tdb")
    tt = TimeScaleConverter().tdb2tt(epoch)
    assert_close(epoch.seconds_until(Epoch(tt.jd1, tt.jd2, "tdb")), -erfa.dtdb(epoch.jd1, epoch.jd2, 0, 0, 0, 0),
                 TIME_S)
    assert_close(iau2006_earth_pole_vector_inertial(epoch), erfa.pnm06a(tt.jd1, tt.jd2)[2], ANGLE_RAD)


def test_iau2006_frame_bias_is_applied_exactly_once():
    epoch = TimeScaleConverter().tt2tdb(Epoch(2451545.0, 0, "tt"))
    bias, precession, combined = erfa.bp06(2451545.0, 0)
    nutation = erfa.pn06a(2451545.0, 0)[6]
    pole = iau2006_earth_pole_vector_inertial(epoch)

    assert_close(pole, (nutation @ combined)[2], ANGLE_RAD)
    assert np.linalg.norm(pole - (nutation @ precession)[2]) > 1e-8
    assert np.linalg.norm(pole - (nutation @ combined @ bias)[2]) > 1e-8


def test_iau2006_pole_matches_the_erfa_validation_vector():
    # libERFA v2.0.1 src/t_erfa_c.c, t_pnm06a, TT MJD 50123.9999.
    epoch = TimeScaleConverter().tt2tdb(Epoch(2400000.5, 50123.9999, "tt"))
    assert_close(iau2006_earth_pole_vector_inertial(epoch),
                 [-0.3639337469629464969e-3, -0.4163377605910663999e-4, 0.9999999329094260057], ANGLE_RAD)


@pytest.mark.parametrize("model", ["ephemeris", "de440", "iau2006_2000a"])
def test_configured_pole_model_reaches_the_dynamics(tmp_path, model):
    config, context = orbit_program_config(tmp_path, eih=False)
    config["earthPoleModel"] = model
    problem = build_lunar_propagation_problem(config, context)
    provider = iau2006_earth_fixed2inertial_matrix if model == "iau2006_2000a" else de440_earth_fixed2inertial_matrix
    epoch = problem.initial_epoch

    assert_close(problem.dynamics.epoch_batch((epoch,)).earth_fixed2inertial_matrix[0], provider(epoch), ANGLE_RAD)
    assert_close(problem.initial_vector, [3.844e8, 0, 0, 0, 1000, 0], REL)
