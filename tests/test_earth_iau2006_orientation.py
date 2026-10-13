import erfa
import numpy as np
import pytest
from test_orbit_problem import prepare_run

from lunarops.classes.dynamics.earth_iau2006_orientation import (
    iau2006_earth_fixed2inertial_matrix,
    iau2006_earth_pole_vector_inertial,
)
from lunarops.classes.time import Epoch, TimeScaleConverter
from lunarops.programs import lunar_orbit
from lunarops.programs.registry import get_program


@pytest.mark.parametrize("jd", [2440222.5, 2440400.5, 2451545., 2460676.5, 2462867.5])
def test_cip_matches_independent_xys_and_coordinate_direction(jd):
    epoch = Epoch(jd, .123456789, "tdb")
    tt = TimeScaleConverter().tdb2tt(epoch)
    x, y, s = erfa.xys06a(tt.jd1, tt.jd2)
    expected = np.array([x, y, np.sqrt(1-x*x-y*y)])
    pole = iau2006_earth_pole_vector_inertial(epoch)
    R = iau2006_earth_fixed2inertial_matrix(epoch)
    np.testing.assert_allclose(pole, expected, rtol=0, atol=3e-16)
    np.testing.assert_allclose(erfa.c2ixys(x, y, s) @ pole, [0, 0, 1], atol=3e-16)
    np.testing.assert_allclose(R @ [0, 0, 1], pole, rtol=0, atol=3e-16)
    np.testing.assert_allclose(R.T @ pole, [0, 0, 1], atol=4e-16)
    np.testing.assert_allclose(R.T @ R, np.eye(3), rtol=0, atol=6e-16)
    assert np.linalg.det(R) == pytest.approx(1, abs=6e-16)
    assert not R.flags.writeable and not pole.flags.writeable


def test_time_conversion_uses_geocentric_dtdb_and_tt(monkeypatch):
    epoch = Epoch(2451545., .25, "tdb")
    tt = TimeScaleConverter().tdb2tt(epoch)
    delta = erfa.dtdb(epoch.jd1, epoch.jd2, 0, 0, 0, 0)
    assert epoch.seconds_until(Epoch(tt.jd1, tt.jd2, "tdb")) == pytest.approx(-delta, abs=2e-12)
    original = erfa.pnm06a
    called = []

    def record(jd1, jd2):
        called.append((jd1, jd2))
        return original(jd1, jd2)

    monkeypatch.setattr(erfa, "pnm06a", record)
    iau2006_earth_pole_vector_inertial(epoch)
    assert called == [(tt.jd1, tt.jd2)]
    with pytest.raises(ValueError):
        iau2006_earth_pole_vector_inertial(Epoch(2451545., 0, "tt"))


def test_frame_bias_present_once_at_j2000():
    epoch = TimeScaleConverter().tt2tdb(Epoch(2451545., 0, "tt"))
    bias, precession, combined = erfa.bp06(2451545., 0)
    nutation = erfa.pn06a(2451545., 0)[6]
    pole = iau2006_earth_pole_vector_inertial(epoch)
    np.testing.assert_allclose(pole, (nutation @ combined)[2], rtol=0, atol=3e-16)
    assert np.linalg.norm(pole-(nutation @ precession)[2]) > 1e-8
    assert np.linalg.norm(pole-(nutation @ combined @ bias)[2]) > 1e-8


def test_matches_erfa_published_validation_vector():
    # libERFA v2.0.1 src/t_erfa_c.c, t_pnm06a, TT MJD 50123.9999.
    epoch = TimeScaleConverter().tt2tdb(Epoch(2400000.5, 50123.9999, "tt"))
    np.testing.assert_allclose(iau2006_earth_pole_vector_inertial(epoch),
        [-0.3639337469629464969e-3, -0.4163377605910663999e-4, 0.9999999329094260057],
        rtol=0, atol=1e-14)


@pytest.mark.parametrize("model", ["ephemeris", "de440", "iau2006_2000a"])
def test_config_selection_reaches_dynamics_and_preserves_initial_state(tmp_path, model):
    cfg, context = prepare_run(tmp_path, eih=False)
    cfg["earthPoleModel"] = model
    problem = lunar_orbit.build_lunar_propagation_problem(cfg, context)
    epoch = problem.initial_epoch
    provider = (iau2006_earth_fixed2inertial_matrix if model == "iau2006_2000a"
                else lunar_orbit.de440_earth_fixed2inertial_matrix)
    np.testing.assert_array_equal(problem.dynamics.build_epoch_data(epoch).earth_fixed2inertial_matrix, provider(epoch))
    assert get_program("LunarOrbitPropagation").spec.schema.resolve(cfg)["earthPoleModel"] == model
    np.testing.assert_array_equal(problem.initial_vector, [3.844e8, 0, 0, 0, 1000, 0])
