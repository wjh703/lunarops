"""LunarOrbitPropagation end to end on an analytic ephemeris."""

import numpy as np
import pytest

from lunarops.classes.dynamics.assembly import build_lunar_propagation_problem, build_rotation
from lunarops.classes.dynamics.history import EphemerisHistory
from lunarops.fileio.numeric_table import read_numeric_table
from lunarops.fileio.structured_reports import read_lunar_orbit_metadata
from lunarops.programs import lunar_orbit
from lunarops.programs.registry import get_program
from tests.support.dynamics import (
    EARTH_MU,
    EPOCH,
    MOON_MU,
    AnalyticRunContext,
    FunctionEphemeris,
    circular_moon_ephemeris,
    orbit_program_config,
)
from tests.tolerances import REL, Tolerance, assert_close, assert_vector_close


def _resolve(config):
    return get_program("LunarOrbitPropagation")[0].schema.resolve(config)


@pytest.mark.parametrize("eih", [False, True])
@pytest.mark.parametrize("direction", [-1, 1])
def test_orbit_program_writes_consistent_products_in_both_directions(tmp_path, eih, direction):
    config, context = orbit_program_config(tmp_path, eih=eih, direction=direction)
    orbit, diagnostics, metadata_path = lunar_orbit.lunar_orbit_propagation(config, context)
    metadata = read_lunar_orbit_metadata(metadata_path)
    _, _, states, _ = read_numeric_table(orbit)
    _, _, accelerations, _ = read_numeric_table(diagnostics)

    assert metadata["timeScale"] == "TDB" and metadata["stepSeconds"] == 60
    assert metadata_path.read_text().splitlines()[0] == "lunarops lunarOrbitMetadata"
    np.testing.assert_array_equal(states[:, 0], direction * np.arange(0, 1201, 60))
    np.testing.assert_array_equal(accelerations[:, 0], states[:, 0])
    # The analytic ephemeris starts the integrated and ephemeris Moon together.
    assert_close(states[0, -6:], np.zeros(6), REL)
    earth, moon, emb, relative = states[:, 1:7], states[:, 7:13], states[:, 13:19], states[:, 19:25]
    # Earth and Moon sit about the ephemeris barycentre and differ by the integrated relative state.
    scale = Tolerance("EMB_PLACEMENT_M", atol=1e-6, unit="m")
    assert_vector_close(moon - earth, relative, scale)
    assert_vector_close((EARTH_MU * earth + MOON_MU * moon) / (EARTH_MU + MOON_MU), emb, scale)
    assert_close(emb[0], np.r_[3.844e8, 0.0, 0.0, 0.0, 1000.0, 0.0] * MOON_MU / (EARTH_MU + MOON_MU), REL)
    assert np.all(np.isfinite(states)) and np.all(np.isfinite(accelerations))
    assert np.linalg.norm(states[-1, -6:]) > 0


def test_orbit_program_writes_a_two_sided_propagation_in_time_order(tmp_path):
    config, context = orbit_program_config(tmp_path, eih=False, direction=1)
    config["backwardDurationSeconds"] = 600.0
    orbit, diagnostics, _ = lunar_orbit.lunar_orbit_propagation(config, context)
    _, _, states, _ = read_numeric_table(orbit)
    _, _, accelerations, _ = read_numeric_table(diagnostics)

    np.testing.assert_array_equal(states[:, 0], np.arange(-600, 1201, 60))
    np.testing.assert_array_equal(accelerations[:, 0], states[:, 0])
    assert_close(states[10, -6:], np.zeros(6), REL)


@pytest.mark.parametrize("core", [False, True])
@pytest.mark.parametrize("figure", [False, True])
def test_orbit_program_integrates_rotation_and_writes_torque_diagnostics(tmp_path, core, figure):
    config, context = orbit_program_config(tmp_path, eih=True)
    config["lunarRotation"] = {
        "inertiaMoments": [0.393, 0.3932, 0.3935],
        "radiusM": 1738000.0,
        "initialEulerAnglesRad": [0.1, 0.4, 1.2],
        "initialAngularVelocityRadS": [1e-7, -2e-7, 2.7e-6],
        "initialCoreAngularVelocityRadS": [2e-7, 0.0, 2.6e-6] if core else None,
        "sources": ["EARTH", "SUN"],
        "gravityTorque": "quadrupole",
        "earthFigureTorque": False,
        "geodeticPrecession": True,
        "core": {"polarMomentFraction": 0.0007, "flattening": 0.000246, "viscosityPerS": 7e-14} if core else None,
    }
    config["outputFileRotation"] = "rotation.dat"
    if figure:
        for name, radius, c20 in (("earth", 6378136.6, -4.8e-4), ("moon", 1738000.0, -9e-5)):
            (tmp_path / f"{name}.gfc").write_text(
                "begin_of_head\nproduct_type gravity_field\nmodelname rotation_test\n"
                f"gravity_constant 4.9e12\nradius {radius}\nmax_degree 2\n"
                "norm fully_normalized\ntide_system zero_tide\nend_of_head\n"
                f"gfc 0 0 1 0\ngfc 2 0 {c20} 0\ngfc 2 1 0 0\ngfc 2 2 3.4e-5 0\n"
            )
        config["gravityFields"] = {name: {"path": f"{name}.gfc"} for name in ("earth", "moon")}
        config["moonFigurePartners"] = ["EARTH", "SUN"]
        config["lunarRotation"]["gravityTorque"] = "figure"
        config["lunarRotation"]["earthFigureTorque"] = True
    config = _resolve(config)

    assert build_lunar_propagation_problem(config, context).initial_vector.shape == ((15,) if core else (12,))
    orbit, _, _, rotation = lunar_orbit.lunar_orbit_propagation(config, context)
    _, _, positions, _ = read_numeric_table(orbit)
    _, columns, states, _ = read_numeric_table(rotation, "lunarRotation")

    np.testing.assert_array_equal(states[:, 0], positions[:, 0])
    assert np.all(np.isfinite(states))
    assert list(columns[1:4]) == ["euler_phi_rad", "euler_theta_rad", "euler_psi_rad"]
    assert np.linalg.norm(states[-1, 1:4] - states[0, 1:4]) > 0
    assert f"{'figure' if figure else 'quadrupole'}:SUN_x_per_s2" in columns
    assert ("core_omega_pa_x_rad_s" in columns) == core


class _RelativeStateEphemeris(FunctionEphemeris):
    """The analytic ephemeris with a Moon-minus-Earth state 1 mm, 1 µm/s off its barycentric difference."""

    OFFSET = np.array([1e-3, -1e-3, 2e-3, 1e-6, 0.0, -1e-6])

    def __init__(self):
        super().__init__(circular_moon_ephemeris()._function)

    def relative_state_matrix(self, target, center, epoch_tdb):
        assert (target, center) in {("MOON", "EARTH"), ("EARTH", "MOON")}
        sign = 1.0 if target == "MOON" else -1.0
        return super().relative_state_matrix(target, center, epoch_tdb) + sign * self.OFFSET

    def relative_state_matrices(self, target, center, epochs):
        return np.asarray([self.relative_state_matrix(target, center, epoch) for epoch in epochs])


class _RelativeStateContext(AnalyticRunContext):
    def create_class(self, *args, **kwargs):
        return _RelativeStateEphemeris()


def test_initial_state_and_reference_orbit_use_the_ephemeris_relative_state(tmp_path):
    config, _ = orbit_program_config(tmp_path, eih=False)
    context = _RelativeStateContext(working_dir=tmp_path)
    ephemeris = _RelativeStateEphemeris()
    expected = ephemeris.relative_state_matrix("MOON", "EARTH", EPOCH)

    problem = build_lunar_propagation_problem(config, context)
    orbit, *_ = lunar_orbit.lunar_orbit_propagation(config, context)
    _, columns, values, _ = read_numeric_table(orbit)
    start = values[np.flatnonzero(values[:, 0] == 0.0)[0]]

    np.testing.assert_array_equal(problem.initial_vector[:6], expected)
    np.testing.assert_array_equal(EphemerisHistory(ephemeris, EARTH_MU, MOON_MU).relative_state(EPOCH), expected)
    # The reference orbit is the same relative state, so the propagation starts on it.
    for axis in ("r", "t", "n"):
        assert start[list(columns).index(f"reference_difference_rtn_{axis}_m")] == 0.0


def test_rotation_cannot_silently_reuse_the_prescribed_attitude_deformation(tmp_path):
    config, _ = orbit_program_config(tmp_path, eih=False)
    config["lunarRotation"] = {}
    config["lunarDegree2Gravity"] = {}
    with pytest.raises(ValueError, match="lunarRotation.deformation"):
        build_rotation(config, None, 4.9e12, 3.986e14)


@pytest.mark.parametrize("direction", [-1, 1])
def test_orbit_program_adds_the_ring_and_uses_the_epoch_solar_gm(tmp_path, direction):
    config, context = orbit_program_config(tmp_path, eih=True, direction=direction)
    config["solarGmVariation"] = {"fractionalRatePerYear": -9e-14, "referenceJdTdb": 2451545.0}
    config["kuiperBeltRing"] = {"pointGmM3S2": 2.4768842184e11, "referenceJdTdb": 2451545.0}
    config = _resolve(config)
    problem = build_lunar_propagation_problem(config, context)
    data = problem.dynamics.epoch_batch((EPOCH.shifted(direction * 1200),))

    assert len(problem.dynamics.body_names) == 3 + problem.dynamics.kuiper_belt_ring.point_count
    assert problem.dynamics.prescribed_body_names == ("EARTH MOON BARYCENTER", "SUN")
    assert_close(
        data.gravitational_parameters_m3_s2[0, 2],
        1.32712440018e20 * (1 - 9e-14 * direction * 1200 / (365.25 * 86400)),
        REL,
    )
    orbit, _, _ = lunar_orbit.lunar_orbit_propagation(config, context)
    assert np.isfinite(read_numeric_table(orbit)[2]).all()
