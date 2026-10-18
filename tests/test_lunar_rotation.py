"""Independent physical and numerical checks for the rigid rotation stage."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from test_lunar_dynamics import EARTH_MU, MOON_MU, Provider

from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    DynamicsEvaluationContext,
    EarthFigureTorque,
    FigureForce,
    GravitatingBody,
    IntegratorSettings,
    LunarCoreParameters,
    LunarDynamics,
    LunarFigureTorque,
    LunarForceGroup,
    LunarRotationDynamics,
    LunarRotationState,
    LunarTorqueGroup,
    PointMassGravityTorque,
    degree2_gravity_coefficients_from_inertia,
)
from lunarops.classes.dynamics.gravity import GravityField
from lunarops.classes.dynamics.rotation import (
    quaternion_derivative,
    quaternion_matrix,
)
from lunarops.classes.dynamics.trajectory import LunarTrajectory
from lunarops.classes.time import Epoch, TimeScale

EPOCH = Epoch(2451545.0, 0.0, TimeScale.TDB)
K = np.diag([0.393, 0.3932, 0.3935])
RADIUS = 1738000.0
Q = np.array([0.7, 0.2, -0.4, 0.1])
Q /= np.linalg.norm(Q)
RELATIVE = np.array([3.84e8, 2e7, -1e7, 0.0, 1000.0, 0.0])


def make_system(*, rotation, forces=(), ephemeris=None):
    return LunarDynamics(
        GravitatingBody("EARTH", EARTH_MU),
        GravitatingBody("MOON", MOON_MU),
        perturbing_bodies=(),
        ephemeris=Provider() if ephemeris is None else ephemeris,
        force_group=LunarForceGroup(forces),
        rotation=rotation,
    )


def state_for(system, rotation_state):
    y = np.concatenate((RELATIVE, rotation_state))
    inputs = system._load_evaluation(system.build_epoch_data(EPOCH), y, None, None)
    return y, inputs


def test_hamilton_kinematics_matches_active_rotation_and_de_euler_rates():
    angles = [0.37, 0.42, 1.2]
    rates = [2e-5, -3e-6, 2.7e-6]
    _phi, theta, psi = angles
    phi_dot, theta_dot, psi_dot = rates
    quat = Rotation.from_euler("ZXZ", angles).as_quat(scalar_first=True)
    omega = [phi_dot*np.sin(theta)*np.sin(psi)+theta_dot*np.cos(psi),
             phi_dot*np.sin(theta)*np.cos(psi)-theta_dot*np.sin(psi), psi_dot+phi_dot*np.cos(theta)]
    state = np.r_[quat, omega]
    r = Rotation.from_euler("ZXZ", angles).as_matrix()
    np.testing.assert_allclose(quaternion_matrix(state[:4]), r, atol=3e-16)
    dt = 0.01
    plus = Rotation.from_euler("ZXZ", np.array(angles) + dt * np.array(rates)).as_matrix()
    minus = Rotation.from_euler("ZXZ", np.array(angles) - dt * np.array(rates)).as_matrix()
    skew = r.T @ ((plus - minus) / (2 * dt))
    np.testing.assert_allclose(state[4:], [skew[2, 1], skew[0, 2], skew[1, 0]], atol=2e-14)
    dq = quaternion_derivative(state[:4], state[4:])
    np.testing.assert_allclose(
        (quaternion_matrix(state[:4] + dt * dq) - quaternion_matrix(state[:4] - dt * dq)) / (2 * dt),
        (plus - minus) / (2 * dt),
        atol=2e-14,
    )
    np.testing.assert_array_equal(quaternion_matrix(-state[:4]), quaternion_matrix(state[:4]))


def test_quadrupole_matches_spherical_harmonic_reaction_and_potential_sign():
    coefficients = degree2_gravity_coefficients_from_inertia(K, gravitational_parameter_m3_s2=MOON_MU, radius_m=RADIUS)
    field = GravityField(coefficients)
    torque = PointMassGravityTorque("EARTH", K)
    rotation = LunarRotationDynamics(K, RADIUS, LunarTorqueGroup([torque]))
    system = make_system(rotation=rotation)
    _, inputs = state_for(system, np.concatenate((Q, [1e-6, -2e-7, 2.7e-6])))
    expected = torque.torque(inputs)
    actual = LunarFigureTorque("EARTH", field).torque(inputs)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-29)
    # Independent quadrupole potential per whole-Moon M R²; derivative
    # for a positive body rotation is minus the body torque.
    direction = np.array([0.3, -0.4, 0.5])
    direction /= np.linalg.norm(direction)
    r = inputs.positions_m[0] - inputs.positions_m[1]
    distance = np.linalg.norm(r)
    R = inputs.moon_fixed2inertial_matrix

    def potential(angle):
        u = (R @ Rotation.from_rotvec(angle * direction).as_matrix()).T @ r / distance
        return 1.5 * EARTH_MU / distance**3 * (u @ K @ u)

    h = 1e-4
    du = (potential(h) - potential(-h)) / (2 * h)
    assert du == pytest.approx(-expected @ direction, rel=2e-7, abs=1e-25)


@pytest.mark.parametrize("pressure_spin", ["core", "mantle"])
def test_core_exchange_conserves_inertial_angular_momentum(pressure_spin):
    core = LunarCoreParameters(0.0007, 2.46e-4, 7e-14, pressure_spin)
    rotation = LunarRotationDynamics(K, RADIUS, LunarTorqueGroup(), core=core)
    system = make_system(rotation=rotation)
    rng = np.random.default_rng(430)
    for _ in range(20):
        wm = rng.normal(size=3) * 2.7e-6
        wc = rng.normal(size=3) * 2.7e-6
        _, inputs = state_for(system, np.concatenate((Q, wm, wc)))
        evaluation = rotation.evaluate(inputs, np.concatenate((Q, wm, wc)))
        dm, dc = evaluation.derivative[4:7], evaluation.derivative[7:10]
        H = rotation.mantle_inertia @ wm + rotation.core_inertia @ wc
        Hd = rotation.mantle_inertia @ dm + rotation.core_inertia @ dc + np.cross(wm, H)
        np.testing.assert_allclose(Hd, 0, atol=5e-27)
        if pressure_spin == "core":
            energy_dot = wm @ rotation.mantle_inertia @ dm + wc @ rotation.core_inertia @ dc
            assert energy_dot == pytest.approx(-core.viscosity_per_s * np.sum((wc - wm) ** 2), abs=1e-31)


def test_zero_viscosity_keeps_pressure_coupling_and_core_state():
    rotation = LunarRotationDynamics(
        K, RADIUS, LunarTorqueGroup(), core=LunarCoreParameters(0.0007, 0.001, 0.0, "core")
    )
    state = np.concatenate((Q, [0.0, 0.0, 2.7e-6], [2e-7, -1e-7, 2.6e-6]))
    _, inputs = state_for(make_system(rotation=rotation), state)
    result = rotation.evaluate(inputs, state)
    np.testing.assert_array_equal(result.viscous_torque, 0.0)
    assert np.linalg.norm(result.pressure_torque) > 0
    assert result.derivative.shape == (10,)


def test_diagonal_inertia_fast_path_matches_general_solve():
    core = LunarCoreParameters(0.0007, 0.001, 7e-14, "mantle")
    rotation = LunarRotationDynamics(K, RADIUS, LunarTorqueGroup(), core=core)
    system = make_system(rotation=rotation)
    state = np.concatenate((Q, [1.1e-7, -2.3e-8, 2.7e-6], [2.0e-7, -1.0e-7, 2.6e-6]))
    _, inputs = state_for(system, state)
    result = rotation.evaluate(inputs, state)

    wm, wc = state[4:7], state[7:10]
    mantle_rhs = (
        result.torques.get("total", np.zeros(3))
        + result.viscous_torque
        + result.pressure_torque
        - np.cross(wm - result.geodetic_rate_pa_rad_s, rotation.mantle_inertia @ wm)
    )
    core_rhs = (
        -result.viscous_torque
        - result.pressure_torque
        - np.cross(wm - result.geodetic_rate_pa_rad_s, rotation.core_inertia @ wc)
    )
    np.testing.assert_allclose(result.derivative[4:7], np.linalg.solve(rotation.mantle_inertia, mantle_rhs), rtol=0, atol=1e-30)
    np.testing.assert_allclose(result.derivative[7:10], np.linalg.solve(rotation.core_inertia, core_rhs), rtol=0, atol=1e-30)


def test_geodetic_transport_is_added_to_both_equations_once():
    # Include a real Sun state; compare pure transport differences, not a
    # trajectory fit which could absorb them into the initial conditions.
    from test_orbit_problem import AnalyticEphemeris

    core = LunarCoreParameters(0.0007, 0.001, 0.0, "core")
    plain = LunarRotationDynamics(K, RADIUS, LunarTorqueGroup(), core=core)
    rel = LunarRotationDynamics(K, RADIUS, LunarTorqueGroup(), core=core, geodetic_precession=True)
    system = LunarDynamics(
        GravitatingBody("EARTH", EARTH_MU),
        GravitatingBody("MOON", MOON_MU),
        (GravitatingBody("SUN", 1.32712440018e20),),
        ephemeris=AnalyticEphemeris(),
        rotation=rel,
    )
    state = np.concatenate((Q, [1e-7, 0.0, 2.7e-6], [2e-7, -1e-7, 2.6e-6]))
    _, inputs = state_for(system, state)
    a, b = plain.evaluate(inputs, state), rel.evaluate(inputs, state)
    r_em = RELATIVE[:3]
    v_em = RELATIVE[3:]
    r_sm = r_em - np.array([1.49e11, 2e10, 0.0])
    nu = quaternion_matrix(Q).T @ (
        1.5
        / 299792458.0**2
        * np.cross(
            v_em, EARTH_MU * r_em / np.linalg.norm(r_em) ** 3 + 1.32712440018e20 * r_sm / np.linalg.norm(r_sm) ** 3
        )
    )
    np.testing.assert_allclose(b.geodetic_rate_pa_rad_s, nu, rtol=1e-15)
    np.testing.assert_array_equal(b.derivative[:4], a.derivative[:4])
    for section, inertia in [(slice(4, 7), rel.mantle_inertia), (slice(7, 10), rel.core_inertia)]:
        np.testing.assert_allclose(
            b.derivative[section] - a.derivative[section],
            np.linalg.solve(inertia, np.cross(nu, inertia @ state[section])),
            rtol=2e-6,
            atol=1e-28,
        )


def test_earth_figure_torque_covariance_and_j2_zero():
    model = EarthFigureTorque(K, 6378136.6, 0.0010826)
    system = make_system(rotation=LunarRotationDynamics(K, RADIUS, LunarTorqueGroup()))
    _, inputs = state_for(system, np.concatenate((Q, [0, 0, 2.7e-6])))
    expected = model.torque(inputs)
    assert np.linalg.norm(expected) > 0
    zero = EarthFigureTorque(K, 6378136.6, 0.0)
    np.testing.assert_array_equal(zero.torque(inputs), 0.0)
    from dataclasses import replace

    C = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    rotated = replace(
        inputs.epoch_data,
        positions_m=inputs.epoch_data.positions_m @ C.T,
        velocities_mps=inputs.epoch_data.velocities_mps @ C.T,
        earth_fixed2inertial_matrix=C @ inputs.epoch_data.earth_fixed2inertial_matrix,
    )
    y = np.concatenate((C @ RELATIVE[:3], C @ RELATIVE[3:]))
    other = DynamicsEvaluationContext(system.body_names, system.gravitational_parameters_m3_s2)
    other.load_evaluation(rotated, y, None, None, moon_matrix=C @ inputs.moon_fixed2inertial_matrix)
    np.testing.assert_allclose(model.torque(other), expected, rtol=2e-14, atol=1e-30)


def test_shared_field_cache_is_invalidated_for_each_corrector_candidate():
    coefficients = degree2_gravity_coefficients_from_inertia(K, gravitational_parameter_m3_s2=MOON_MU, radius_m=RADIUS)
    field = GravityField(coefficients)
    force = FigureForce({"MOON": field}, {"MOON": ("EARTH",)})
    torque = LunarFigureTorque("EARTH", field)
    system = make_system(rotation=LunarRotationDynamics(K, RADIUS, LunarTorqueGroup([torque])), forces=[force])
    calls = []
    original = field.nonspherical_accelerations

    def counted(r):
        calls.append(r.copy())
        return original(r)

    field.nonspherical_accelerations = counted
    y, _ = state_for(system, np.concatenate((Q, [0, 0, 2.7e-6])))
    data = system.build_epoch_data(EPOCH)
    first = system.derivatives(EPOCH, y, data)
    assert len(calls) == 1
    y[6:10] = [1, 0, 0, 0]
    second = system.derivatives(EPOCH, y, data)
    assert len(calls) == 2
    assert not np.array_equal(first[3:6], second[3:6])
    assert not np.array_equal(first[10:13], second[10:13])


@pytest.mark.parametrize("with_core", [False, True])
@pytest.mark.parametrize("direction", [-1, 1])
def test_joint_abm_matches_dop853_and_conserves_free_rigid_invariants(with_core, direction):
    core = LunarCoreParameters(0.0007, 0.000246, 0.0, "core") if with_core else None
    rotation = LunarRotationDynamics(K, RADIUS, LunarTorqueGroup(), core=core)
    system = make_system(rotation=rotation)
    state = LunarRotationState(Q, [1e-6, -3e-7, 2.7e-6], [2e-7, 1e-7, 2.6e-6] if with_core else None).vector()
    y = np.concatenate((RELATIVE, state))
    settings = IntegratorSettings(step_s=600.0, order=6, startup_step_s=75.0, trajectory_interpolation_order=6)
    trajectory = AdamsBashforthMoultonIntegrator(settings).integrate(system, EPOCH, y, direction * 86400.0)

    def rhs(t, s):
        return system.derivatives(EPOCH.shifted(t), s)

    reference = solve_ivp(
        rhs,
        (0, direction * 86400.0),
        y,
        method="DOP853",
        rtol=3e-13,
        atol=np.concatenate((np.full(6, 1e-5), np.full(4, 1e-14), np.full(len(state) - 4, 1e-17))),
        t_eval=trajectory.times_s,
        max_step=900.0,
    )
    np.testing.assert_allclose(trajectory.states[6:], reference.y[6:], rtol=0, atol=2e-12)
    h = []
    energy = []
    for vector in trajectory.states.T:
        R = quaternion_matrix(vector[6:10])
        wm = vector[10:13]
        H = rotation.mantle_inertia @ wm
        E = 0.5 * wm @ H
        if with_core:
            wc = vector[13:16]
            H = H + rotation.core_inertia @ wc
            E += 0.5 * wc @ rotation.core_inertia @ wc
        h.append(R @ H)
        energy.append(E)
    np.testing.assert_allclose(np.asarray(h), np.broadcast_to(h[0], np.shape(h)), rtol=0, atol=2e-17)
    np.testing.assert_allclose(energy, energy[0], rtol=2e-11, atol=1e-28)
    assert np.max(abs(np.linalg.norm(trajectory.states[6:10], axis=0) - 1)) < 2e-12
    midpoint = EPOCH.shifted(direction * 12345.0)
    matrix = trajectory.matrix(midpoint)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=5e-16)


def test_joint_history_reconstructs_only_translation_and_initial_arrays_are_readonly():
    rotation = LunarRotationDynamics(K, RADIUS, LunarTorqueGroup())
    system = make_system(rotation=rotation)
    state = LunarRotationState(Q, [0.0, 0.0, 2.7e-6])
    assert not state.quaternion.flags.writeable
    y = np.concatenate((RELATIVE, state.vector()))
    history = system.build_history_provider(lambda epoch: y, system.ephemeris.body_state_matrix_bcrs)
    np.testing.assert_array_equal(history(("MOON",), EPOCH)[0], RELATIVE)
    trajectory = LunarTrajectory(EPOCH, [0, 1], np.column_stack((y, y)), 1)
    sign_flipped = y.copy()
    sign_flipped[6:10] *= -1
    trajectory = LunarTrajectory(EPOCH, [0, 1], np.column_stack((y, sign_flipped)), 1)
    np.testing.assert_allclose(trajectory.matrix(EPOCH.shifted(0.5)), quaternion_matrix(Q), atol=3e-16)
    np.testing.assert_array_equal(sign_flipped[6:10], -Q)
