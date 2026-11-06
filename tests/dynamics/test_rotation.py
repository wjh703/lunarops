"""Rigid mantle and fluid-core rotation: kinematics, torques, conservation laws and integration."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from lunarops.base.constants import C
from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    GravitatingBody,
    IntegratorSettings,
    LunarCoreParameters,
    LunarDynamics,
    LunarRotationState,
    LunarTorqueGroup,
)
from lunarops.classes.dynamics import LunarRotationDynamics as NativeRotation
from lunarops.classes.dynamics.gravity import GravityField
from lunarops.classes.dynamics.history import DelayedHistory, LunarNodeBuffer
from lunarops.classes.dynamics.rotation import quaternion_matrix
from lunarops.classes.dynamics.trajectory import LunarTrajectory
from tests.support import reference_dynamics as oracle
from tests.support.dynamics import (
    EARTH_MU,
    EPOCH,
    MOON_MU,
    MOON_RADIUS,
    RELATIVE,
    K,
    Q,
    circular_moon_ephemeris,
    make_system,
    oracle_inputs,
)
from tests.tolerances import ANGLE_RAD, REL, Tolerance, assert_close, assert_vector_close

SUN_GM = 1.32712440018e20
CORE = LunarCoreParameters(0.0007, 2.46e-4, 7e-14)
# Euler-equation terms are about 3e-16 rad/s^2; 5e-27 leaves 1e-11 of them for round off.
ANGULAR_MOMENTUM_RATE = Tolerance("ANGULAR_MOMENTUM_RATE", atol=5e-27, unit="rad/s^2")
# Central differences of rotation matrices with dt = 0.01 s.
FINITE_DIFFERENCE = Tolerance("FINITE_DIFFERENCE", atol=2e-14)


def test_hamilton_kinematics_match_active_rotation_and_euler_rates():
    angles, rates = np.array([0.37, 0.42, 1.2]), np.array([2e-5, -3e-6, 2.7e-6])
    _, theta, psi = angles
    phi_dot, theta_dot, psi_dot = rates
    quaternion = Rotation.from_euler("ZXZ", angles).as_quat(scalar_first=True)
    omega = [phi_dot * np.sin(theta) * np.sin(psi) + theta_dot * np.cos(psi),
             phi_dot * np.sin(theta) * np.cos(psi) - theta_dot * np.sin(psi), psi_dot + phi_dot * np.cos(theta)]
    matrix = Rotation.from_euler("ZXZ", angles).as_matrix()
    dt = 0.01
    plus = Rotation.from_euler("ZXZ", angles + dt * rates).as_matrix()
    minus = Rotation.from_euler("ZXZ", angles - dt * rates).as_matrix()
    skew = matrix.T @ ((plus - minus) / (2 * dt))
    dq = oracle.quaternion_derivative(quaternion, omega)

    assert_close(quaternion_matrix(quaternion), matrix, ANGLE_RAD)
    assert_close(quaternion_matrix(-quaternion), matrix, ANGLE_RAD)
    assert_close(omega, [skew[2, 1], skew[0, 2], skew[1, 0]], FINITE_DIFFERENCE)
    assert_close((quaternion_matrix(quaternion + dt * dq) - quaternion_matrix(quaternion - dt * dq)) / (2 * dt),
                 (plus - minus) / (2 * dt), FINITE_DIFFERENCE)


def test_quadrupole_torque_matches_the_harmonic_reaction_and_the_potential_gradient():
    field = GravityField(oracle.inertia_gravity(K, gravitational_parameter_m3_s2=MOON_MU, radius_m=MOON_RADIUS))
    torque = oracle.PointMassGravityTorque("EARTH", K)
    system = make_system(rotation=oracle.LunarRotationDynamics(K, MOON_RADIUS, LunarTorqueGroup([torque])))
    _, inputs = oracle_inputs(system, np.concatenate((Q, [1e-6, -2e-7, 2.7e-6])))
    expected = torque.torque(inputs)

    assert_vector_close(oracle.LunarFigureTorque("EARTH", field).torque(inputs), expected, REL)
    # Quadrupole potential per M R^2; its derivative for a body rotation is minus the torque.
    direction = np.array([0.3, -0.4, 0.5]) / np.linalg.norm([0.3, -0.4, 0.5])
    r = inputs.positions_m[0] - inputs.positions_m[1]
    distance = np.linalg.norm(r)

    def potential(angle):
        u = (inputs.moon_fixed2inertial_matrix @ Rotation.from_rotvec(angle * direction).as_matrix()).T @ r / distance
        return 1.5 * EARTH_MU / distance**3 * (u @ K @ u)

    h = 1e-4
    assert (potential(h) - potential(-h)) / (2 * h) == pytest.approx(-expected @ direction, rel=2e-7)


@pytest.mark.parametrize("implementation", ["oracle", "native"])
def test_core_mantle_exchange_conserves_angular_momentum_and_dissipates_energy(implementation):
    rotation_class = oracle.LunarRotationDynamics if implementation == "oracle" else NativeRotation
    rotation = rotation_class(K, MOON_RADIUS, LunarTorqueGroup(), core=CORE)
    system = make_system(rotation=rotation)
    rng = np.random.default_rng(430)
    for _ in range(20):
        wm, wc = rng.normal(size=3) * 2.7e-6, rng.normal(size=3) * 2.7e-6
        if implementation == "oracle":
            _, inputs = oracle_inputs(system, np.concatenate((Q, wm, wc)))
            derivative = rotation.evaluate(inputs, np.concatenate((Q, wm, wc))).derivative
        else:
            derivative = system.derivatives(EPOCH, np.concatenate((RELATIVE, Q, wm, wc)))[6:]
        dm, dc = derivative[4:7], derivative[7:10]
        momentum = rotation.mantle_inertia @ wm + rotation.core_inertia @ wc

        assert_close(rotation.mantle_inertia @ dm + rotation.core_inertia @ dc + np.cross(wm, momentum), 0.0,
                     ANGULAR_MOMENTUM_RATE)
        energy_rate = wm @ rotation.mantle_inertia @ dm + wc @ rotation.core_inertia @ dc
        assert energy_rate == pytest.approx(-CORE.viscosity_per_s * np.sum((wc - wm) ** 2), abs=1e-31)


def test_zero_viscosity_keeps_only_the_pressure_coupling():
    rotation = oracle.LunarRotationDynamics(K, MOON_RADIUS, LunarTorqueGroup(),
                                            core=LunarCoreParameters(0.0007, 0.001, 0.0))
    state = np.concatenate((Q, [0.0, 0.0, 2.7e-6], [2e-7, -1e-7, 2.6e-6]))
    _, inputs = oracle_inputs(make_system(rotation=rotation), state)
    result = rotation.evaluate(inputs, state)

    np.testing.assert_array_equal(result.viscous_torque, 0.0)
    assert np.linalg.norm(result.pressure_torque) > 0


def test_de_sitter_rate_is_the_sum_of_earth_and_sun_terms_added_once_to_both_bodies():
    core = LunarCoreParameters(0.0007, 0.001, 0.0)
    plain = oracle.LunarRotationDynamics(K, MOON_RADIUS, LunarTorqueGroup(), core=core)
    relativistic = oracle.LunarRotationDynamics(K, MOON_RADIUS, LunarTorqueGroup(), core=core, geodetic_precession=True)
    system = LunarDynamics(GravitatingBody("EARTH", EARTH_MU), GravitatingBody("MOON", MOON_MU),
                           (GravitatingBody("SUN", SUN_GM),), ephemeris=circular_moon_ephemeris(), rotation=relativistic)
    state = np.concatenate((Q, [1e-7, 0.0, 2.7e-6], [2e-7, -1e-7, 2.6e-6]))
    _, inputs = oracle_inputs(system, state)
    a, b = plain.evaluate(inputs, state), relativistic.evaluate(inputs, state)
    moon, sun = inputs.body_indices["MOON"], inputs.body_indices["SUN"]
    r_sm = inputs.positions_m[moon] - inputs.positions_m[sun]
    v_sm = inputs.velocities_mps[moon] - inputs.velocities_mps[sun]
    # nu = 3/(2c^2) sum_j GM_j (r_jM x v_jM)/r_jM^3 with Moon-minus-source states.
    nu = quaternion_matrix(Q).T @ (1.5 / C**2 * (
        EARTH_MU * np.cross(RELATIVE[:3], RELATIVE[3:]) / np.linalg.norm(RELATIVE[:3]) ** 3
        + SUN_GM * np.cross(r_sm, v_sm) / np.linalg.norm(r_sm) ** 3))

    assert_vector_close(b.geodetic_rate_pa_rad_s, nu, REL)
    np.testing.assert_array_equal(b.derivative[:4], a.derivative[:4])
    for section, inertia in [(slice(4, 7), relativistic.mantle_inertia), (slice(7, 10), relativistic.core_inertia)]:
        assert_vector_close(b.derivative[section] - a.derivative[section],
                            np.linalg.solve(inertia, np.cross(nu, inertia @ state[section])),
                            Tolerance("DE_SITTER_DIFFERENCE", rtol=2e-6))


def test_de_sitter_solar_term_uses_the_heliocentric_moon_velocity():
    """Moving the Sun at constant Moon-relative geometry changes only the solar term."""
    sun_position = np.array([-1.49e11, -2e10, 0.0])

    def rate(sun_velocity):
        return oracle.LunarRotationDynamics.geodetic_rate(SimpleNamespace(
            body_indices={"EARTH": 0, "MOON": 1, "SUN": 2},
            positions_m=np.array([np.zeros(3), RELATIVE[:3], sun_position]),
            velocities_mps=np.array([np.zeros(3), RELATIVE[3:], sun_velocity]),
            gravitational_parameters_m3_s2=np.array([EARTH_MU, MOON_MU, SUN_GM]),
            moon_fixed2inertial_matrix=np.eye(3),
        ))

    sun_velocity = np.array([0.0, -2.98e4, 0.0])
    r_sm = RELATIVE[:3] - sun_position
    expected = -1.5 / C**2 * SUN_GM * np.cross(r_sm, sun_velocity) / np.linalg.norm(r_sm) ** 3

    assert_vector_close(rate(sun_velocity) - rate(np.zeros(3)), expected, REL)
    assert expected[2] > 0  # prograde, normal to the ecliptic here


def test_earth_figure_torque_is_rotation_covariant_and_vanishes_without_j2():
    model = oracle.EarthFigureTorque(K, 6378136.6, 0.0010826)
    system = make_system(rotation=oracle.LunarRotationDynamics(K, MOON_RADIUS, LunarTorqueGroup()))
    _, inputs = oracle_inputs(system, np.concatenate((Q, [0, 0, 2.7e-6])))
    expected = model.torque(inputs)
    rotation = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    batch = system.epoch_batch((EPOCH,))
    rotated = replace(batch, positions_m=batch.positions_m @ rotation.T, velocities_mps=batch.velocities_mps @ rotation.T,
                      earth_fixed2inertial_matrix=rotation @ batch.earth_fixed2inertial_matrix)
    other = oracle.DynamicsEvaluationContext(system.body_names, system.gravitational_parameters_m3_s2)
    other.load_evaluation(rotated, 0, EPOCH, np.r_[rotation @ RELATIVE[:3], rotation @ RELATIVE[3:]], None,
                          moon_matrix=rotation @ inputs.moon_fixed2inertial_matrix)

    assert np.linalg.norm(expected) > 0
    assert_vector_close(model.torque(other), expected, REL)
    np.testing.assert_array_equal(oracle.EarthFigureTorque(K, 6378136.6, 0.0).torque(inputs), 0.0)


@pytest.mark.slow
@pytest.mark.parametrize("with_core", [False, True])
@pytest.mark.parametrize("direction", [-1, 1])
def test_abm_rotation_matches_dop853_and_conserves_free_rigid_invariants(with_core, direction):
    core = LunarCoreParameters(0.0007, 0.000246, 0.0) if with_core else None
    rotation = NativeRotation(K, MOON_RADIUS, LunarTorqueGroup(), core=core)
    system = make_system(rotation=rotation)
    state = LunarRotationState(Q, [1e-6, -3e-7, 2.7e-6], [2e-7, 1e-7, 2.6e-6] if with_core else None).vector()
    y = np.concatenate((RELATIVE, state))
    settings = IntegratorSettings(step_s=600.0, order=6, startup_step_s=75.0, trajectory_interpolation_order=6)
    trajectory = AdamsBashforthMoultonIntegrator(settings).integrate(system, EPOCH, y, direction * 86400.0)
    reference = solve_ivp(lambda t, s: system.derivatives(EPOCH.shifted(t), s), (0, direction * 86400.0), y,
                          method="DOP853", rtol=3e-13, t_eval=trajectory.times_s, max_step=900.0,
                          atol=np.concatenate((np.full(6, 1e-5), np.full(4, 1e-14), np.full(len(state) - 4, 1e-17))))

    # Both integrators are converged to about 1e-12 in the attitude over one day.
    assert_close(trajectory.states[6:], reference.y[6:], Tolerance("INTEGRATOR_AGREEMENT", atol=2e-12))
    momentum, energy = [], []
    for vector in trajectory.states.T:
        wm = vector[10:13]
        h = rotation.mantle_inertia @ wm
        e = 0.5 * wm @ h
        if with_core:
            wc = vector[13:16]
            h = h + rotation.core_inertia @ wc
            e += 0.5 * wc @ rotation.core_inertia @ wc
        momentum.append(quaternion_matrix(vector[6:10]) @ h)
        energy.append(e)
    assert_close(momentum, np.broadcast_to(momentum[0], np.shape(momentum)), Tolerance("INERTIAL_MOMENTUM", atol=2e-17))
    assert_close(energy, energy[0], Tolerance("ROTATIONAL_ENERGY", rtol=2e-11))
    assert np.max(abs(np.linalg.norm(trajectory.states[6:10], axis=0) - 1)) < 2e-12
    matrix = trajectory.matrix(EPOCH.shifted(direction * 12345.0))
    assert_close(matrix.T @ matrix, np.eye(3), ANGLE_RAD)


def test_history_reconstructs_translation_and_trajectory_ignores_quaternion_sign():
    system = make_system(rotation=NativeRotation(K, MOON_RADIUS, LunarTorqueGroup()))
    y = np.concatenate((RELATIVE, LunarRotationState(Q, [0.0, 0.0, 2.7e-6]).vector()))
    nodes = LunarNodeBuffer(2, len(y))
    nodes.append(0, y, np.zeros_like(y))
    nodes.mark_trajectory_start()
    flipped = y.copy()
    flipped[6:10] *= -1

    earth, moon = DelayedHistory(system, EPOCH, nodes, 1).states(("EARTH", "MOON"), EPOCH)
    assert_vector_close(moon - earth, RELATIVE, Tolerance("EMB_PLACEMENT", atol=1e-7))
    assert_close(moon, EARTH_MU / (EARTH_MU + MOON_MU) * RELATIVE, REL)
    trajectory = LunarTrajectory(EPOCH, [0, 1], np.column_stack((y, flipped)), 1)
    assert_close(trajectory.matrix(EPOCH.shifted(0.5)), quaternion_matrix(Q), ANGLE_RAD)
