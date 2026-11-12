"""Lunar tidal and rotational deformation, static degree-2 offsets and the delayed (DDE) propagation."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    IntegratorSettings,
    LunarCoreParameters,
    LunarFigureTorque,
    LunarTorqueGroup,
    MoonFigureForce,
)
from lunarops.classes.dynamics import LunarRotationDynamics as NativeRotation
from lunarops.classes.dynamics.deformation import (
    LunarDeformation,
    LunarDeformationParameters,
    LunarDegree2GravityModel,
    LunarDegree2GravityParameters,
    LunarInertiaEvaluation,
    deformation_tensors,
)
from lunarops.classes.dynamics.forces import EarthTideParameters
from lunarops.classes.dynamics.gravity import GravityCoefficients, GravityField
from lunarops.classes.dynamics.history import DelayedHistory, LunarNodeBuffer
from tests.support import reference_dynamics as oracle
from tests.support.dynamics import (
    ANGLES,
    EARTH_MU,
    EPOCH,
    MOON_MU,
    MOON_RADIUS,
    RELATIVE,
    K,
    make_system,
    oracle_inputs,
)
from tests.tolerances import REL, Tolerance, assert_close, assert_vector_close

P = LunarDeformationParameters(0.024059, 900.0, 2.6616995e-6)
TENSOR_KWARGS = {"radius_m": MOON_RADIUS, "earth_mu": EARTH_MU, "moon_mu": MOON_MU}
ANGULAR_MOMENTUM_RATE = Tolerance("ANGULAR_MOMENTUM_RATE", atol=5e-27, unit="rad/s^2")


def _symmetric_traceless(tensor):
    assert_close(tensor, tensor.T, Tolerance("SYMMETRY", atol=1e-15 * np.max(np.abs(tensor)) + 1e-300))
    assert abs(np.trace(tensor)) <= 1e-12 * np.max(np.abs(tensor)) + 1e-300


def test_reference_mean_motion_shifts_only_the_constant_axisymmetric_inertia():
    alternate = replace(P, mean_motion_rad_s=2.6653246342740375e-6)
    expected = P.love_k2 * MOON_RADIUS**3 / (9 * MOON_MU) * (alternate.mean_motion_rad_s**2 - P.mean_motion_rad_s**2)
    for w in (np.array([2e-8, -1e-8, 2.7e-6]), np.array([-3e-8, 4e-8, 2.6e-6])):
        args = (np.array([3.8e8, 2e7, -1e7]), np.array([100.0, -970.0, 24.0]), w, np.array([1e-13, -2e-14, 3e-15]))
        before = deformation_tensors(*args, parameters=P, **TENSOR_KWARGS)
        after = deformation_tensors(*args, parameters=alternate, **TENSOR_KWARGS)

        assert_close(
            after[1] - before[1], expected * np.diag([1.0, 1.0, -2.0]), Tolerance("SPIN_TENSOR", rtol=1e-12, atol=1e-25)
        )
        np.testing.assert_array_equal(after[0], before[0])  # the tidal part does not depend on the mean motion
        np.testing.assert_array_equal(after[2], before[2])


def test_analytic_inertia_derivative_matches_a_central_difference():
    r, rd = np.array([3.8e8, 2e7, -1e7]), np.array([130.0, -970.0, 24.0])
    w, wd = np.array([2e-8, -1e-8, 2.7e-6]), np.array([1e-13, -2e-14, 3e-15])

    def tensors(t):
        return deformation_tensors(r + t * rd, rd, w + t * wd, wd, parameters=P, **TENSOR_KWARGS)

    tidal, spin, derivative = tensors(0)
    plus, minus = tensors(0.1), tensors(-0.1)
    assert_close(
        derivative,
        ((plus[0] + plus[1]) - (minus[0] + minus[1])) / 0.2,
        Tolerance("FINITE_DIFFERENCE", rtol=2e-8, atol=1e-21),
    )
    for tensor in (tidal, spin, derivative):
        _symmetric_traceless(tensor)


@pytest.mark.parametrize("tidal,spin", [(False, False), (True, False), (False, True)])
def test_deformation_switches_control_both_inertia_and_its_derivative(tidal, spin):
    args = (
        np.array([3.8e8, 2e7, -1e7]),
        np.array([100, 20, 10]),
        np.array([1e-7, 2e-8, 2.7e-6]),
        np.array([1e-13, 2e-14, 0]),
    )
    full = deformation_tensors(*args, parameters=P, **TENSOR_KWARGS)
    selected = deformation_tensors(
        *args,
        parameters=LunarDeformationParameters(P.love_k2, P.delay_s, P.mean_motion_rad_s, tidal, spin),
        **TENSOR_KWARGS,
    )

    np.testing.assert_array_equal(selected[0], full[0] if tidal else np.zeros((3, 3)))
    np.testing.assert_array_equal(selected[1], full[1] if spin else np.zeros((3, 3)))
    if not tidal and not spin:
        np.testing.assert_array_equal(selected[2], 0)


def test_time_varying_mantle_inertia_conserves_angular_momentum():
    rotation = oracle.LunarRotationDynamics(
        K, MOON_RADIUS, LunarTorqueGroup(), core=LunarCoreParameters(0.0007, 0.000246, 7e-14)
    )
    wm, wc = np.array([2e-7, -3e-8, 2.7e-6]), np.array([1e-7, 1e-8, 2.6e-6])
    state = np.concatenate((ANGLES, wm, wc))
    _, inputs = oracle_inputs(make_system(rotation=rotation), state)
    delta = np.array([[2e-7, 1e-8, -3e-9], [1e-8, -1e-7, 2e-9], [-3e-9, 2e-9, -1e-7]])
    rate = np.array([[2e-13, -3e-14, 1e-14], [-3e-14, -1e-13, 2e-14], [1e-14, 2e-14, -1e-13]])
    inputs.lunar_inertia = LunarInertiaEvaluation(K + delta, rate, delta, np.zeros((3, 3)))
    result = rotation.evaluate(inputs, state)
    mantle = K + delta - rotation.core_inertia
    momentum = mantle @ wm + rotation.core_inertia @ wc

    # d/dt (I_m w_m + I_c w_c) in the rotating mantle frame, including dI_m/dt.
    momentum_rate = (
        mantle @ result.derivative[3:6]
        + rate @ wm
        + rotation.core_inertia @ result.derivative[6:]
        + np.cross(wm, momentum)
    )
    assert_close(momentum_rate, 0.0, ANGULAR_MOMENTUM_RATE)


C21 = 2.2779569194479859e-10
S21 = 1.3343450358392620e-9


@pytest.mark.parametrize("c,s,s22", [(C21, 0, 0), (0, S21, 0), (C21, S21, 0), (-C21, -S21, 0), (0, 0, 3e-9)])
def test_static_degree2_offsets_give_the_cartesian_quadrupole_torque(c, s, s22):
    moon_mu, source_mu = 4.9028e12, 3.986e14
    inertia = np.diag([0.39276, 0.39285, 0.39301])
    field = GravityField(oracle.inertia_gravity(inertia, gravitational_parameter_m3_s2=moon_mu, radius_m=MOON_RADIUS))
    location = np.array([3.7e8, -1.5e8, 2.2e7])

    def rigid_degree2(c21, s21, s22_):
        parameters = LunarDeformationParameters(
            0.02419, 12000, 2.66e-6, c21_static=c21, s21_static=s21, s22_static=s22_
        )
        return LunarDeformation(parameters, inertia, MOON_RADIUS, source_mu, moon_mu, field).evaluate(
            None, None, rigid=True
        )

    result = rigid_degree2(c, s, s22)
    before = field.nonspherical_accelerations(
        location[None], field.coefficients_with_degree2(rigid_degree2(0, 0, 0).degree2)
    )[0]
    after = field.nonspherical_accelerations(location[None], field.coefficients_with_degree2(result.degree2))[0]
    torque = np.cross(location, -source_mu / moon_mu * (after - before)) / MOON_RADIUS**2
    delta = np.array([[0, -2 * s22, -c], [-2 * s22, 0, -s], [-c, -s, 0]])
    unit = location / np.linalg.norm(location)
    # MacCullagh torque of the offset tensor; higher multipoles are below 2e-9 of it at lunar distance.
    expected = 3 * source_mu / np.linalg.norm(location) ** 3 * np.cross(unit, delta @ unit)
    assert_vector_close(torque, expected, Tolerance("QUADRUPOLE_TRUNCATION", rtol=2e-9))
    np.testing.assert_array_equal(result.total, inertia)  # offsets do not alter the mechanical inertia
    # The offsets replace the static degree-2 coefficients instead of accumulating onto them.
    np.testing.assert_array_equal(rigid_degree2(c, s, s22).degree2, result.degree2)


def test_pavlov_inertia_follows_eq15_with_tidal_and_rotational_terms():
    p = LunarDegree2GravityParameters(include_tidal_deformation=True, include_rotational_deformation=True)
    earth_minus_moon = -np.array([3.84e8, 1.2e7, -2.0e6])
    attitude = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    w = np.array([1.1e-8, -2.3e-8, 2.66e-6])
    model = LunarDegree2GravityModel(
        3.986004418e14,
        4.9048695e12,
        lambda t: earth_minus_moon,
        lambda t: attitude,
        angular_velocity_provider=lambda t: w,
        parameters=p,
    )
    r = attitude.T @ earth_minus_moon
    u = r / np.linalg.norm(r)
    tide = (
        -p.love_k2
        * (model.earth_gravitational_parameter_m3_s2 / model.moon_gravitational_parameter_m3_s2)
        * (p.radius_m / np.linalg.norm(r)) ** 3
        * (np.outer(u, u) - np.eye(3) / 3)
    )
    spin_tensor = np.outer(w, w) - np.eye(3) * (w @ w - p.mean_motion_rad_s**2) / 3
    spin_tensor[2, 2] -= p.mean_motion_rad_s**2
    spin = p.love_k2 * p.radius_m**3 / (3 * model.moon_gravitational_parameter_m3_s2) * spin_tensor

    # The 6e-8 deformation terms sit on a 0.4 undistorted tensor whose ulp is 5.6e-17.
    assert_close(
        model.inertia(EPOCH), p.undistorted_inertia() + tide + spin, Tolerance("INERTIA_ROUND_OFF", atol=2e-16)
    )


def test_pavlov_static_offsets_change_the_field_but_not_the_inertia():
    p = LunarDegree2GravityParameters(c21_static=C21, s21_static=S21, s22_static=4e-9)

    def model(parameters):
        return LunarDegree2GravityModel(
            3.986e14,
            4.9028e12,
            lambda t: np.array([3.8e8, 2e7, -1e7]),
            lambda t: np.eye(3),
            angular_velocity_provider=lambda t: np.zeros(3),
            parameters=parameters,
        )

    with_offsets, without = model(p), model(LunarDegree2GravityParameters())
    inertia = with_offsets.inertia(EPOCH)
    np.testing.assert_array_equal(inertia, without.inertia(EPOCH))
    coefficients = oracle.inertia_gravity(
        inertia,
        gravitational_parameter_m3_s2=4.9028e12,
        radius_m=p.radius_m,
        c21_static=p.c21_static,
        s21_static=p.s21_static,
        s22_static=p.s22_static,
    ).coeffs
    plain = oracle.inertia_gravity(inertia, gravitational_parameter_m3_s2=4.9028e12, radius_m=p.radius_m).coeffs
    offsets = np.array(
        [
            [0.0, p.c21_static / np.sqrt(5 / 3), 0.0],
            [0.0, p.s21_static / np.sqrt(5 / 3), p.s22_static / np.sqrt(5 / 12)],
        ]
    )
    # C20 and C22 are differences of 0.4 inertia moments, whose ulp is 5.6e-17.
    round_off = Tolerance("COEFFICIENT_ROUND_OFF", atol=2e-16)
    assert_close(coefficients[:, 2, :3], plain[:, 2, :3] + offsets, round_off)

    # Applying the model replaces degree 2 of a file field and keeps higher degrees.
    static = np.zeros((2, 4, 4))
    static[:, :3, :3] = coefficients
    static[0, 2, 1] += 2e-9
    static[0, 3, 1] = 1e-5
    field = GravityField(GravityCoefficients(static, 4.9028e12, p.radius_m))
    applied = field.coefficients_with_degree2(with_offsets.degree2_coefficients(EPOCH))
    assert_close(applied[:, 2, :3], coefficients[:, 2, :3], round_off)
    np.testing.assert_array_equal(applied[:, 3], static[:, 3])
    np.testing.assert_array_equal(field.coefficients.coeffs, static)


def test_earth_tide_k20_term_follows_jpl_eq7_for_one_raiser():
    p = EarthTideParameters(k21=0.0, k22=0.0, tau_orb_days=(0.0, 0.0, 0.0), tau_rot_days=(0.0, 0.0, 0.0))
    earth, moon = np.zeros(3), np.array([3.84e8, 2.1e7, 1.1e7])
    states = {"EARTH": earth, "MOON": moon}
    history = SimpleNamespace(
        states=lambda names, t: np.array([[*states[n], 0, 0, 0] for n in names], float),
        geocentric_positions=lambda names, epochs: np.array([[states[n] - earth for n in names] for _ in epochs]),
    )
    model = oracle.EarthTideModel(
        3.986004418e14, 4.9048695e12, tide_parameters=p, tide_raisers=("MOON",), tide_raiser_gm={"MOON": 4.9048695e12}
    )

    got = model.relative_acceleration(EPOCH, earth, moon, np.eye(3), history)

    r = moon - earth
    radius = np.linalg.norm(r)
    rho, z = np.array((r[0], r[1], 0.0)), r[2]
    t0 = (
        2 * z * z**2 * np.array((0.0, 0.0, 1.0))
        + rho @ rho * rho
        - 5 * ((z * z) ** 2 + 0.5 * (rho @ rho) ** 2) * r / radius**2
        + radius**2 * r
    ) / radius**5
    factor = (
        1.5
        * (model.earth_gravitational_parameter_m3_s2 + model.moon_gravitational_parameter_m3_s2)
        / (model.earth_gravitational_parameter_m3_s2)
        * 4.9048695e12
        * model.earth_radius_m**5
        / radius**5
    )
    assert_vector_close(got, factor * p.k20 * t0, REL)


def delayed_system(*, love=P.love_k2, core=True, deformation=True):
    field = GravityField(oracle.inertia_gravity(K, gravitational_parameter_m3_s2=MOON_MU, radius_m=MOON_RADIUS))
    model = (
        LunarDeformation(
            LunarDeformationParameters(love, P.delay_s, P.mean_motion_rad_s), K, MOON_RADIUS, EARTH_MU, MOON_MU, field
        )
        if deformation
        else None
    )
    rotation = NativeRotation(
        K,
        MOON_RADIUS,
        LunarTorqueGroup([LunarFigureTorque("EARTH", field)]),
        deformation=model,
        core=LunarCoreParameters(0.0007, 0.000246, 7e-14) if core else None,
    )
    system = make_system(rotation=rotation, forces=[MoonFigureForce(field, ("EARTH",))])
    initial = np.concatenate((RELATIVE, ANGLES, [1e-7, -2e-8, 2.7e-6], [2e-8, 1e-8, 2.6e-6] if core else []))
    return system, initial


def test_delayed_history_interpolates_old_nodes_without_extrapolating_the_newest():
    system, _ = delayed_system(core=False)
    nodes = LunarNodeBuffer(30, 13)
    for index, t in enumerate(range(-10, 20)):
        nodes.append(t, np.arange(13) * (t**5 - 2 * t**3 + 7), np.arange(13) * (5 * t**4 - 6 * t**2))
        if index == 10:
            nodes.mark_trajectory_start()
    history = DelayedHistory(system, EPOCH, nodes, 7)
    interpolation = Tolerance("HERMITE_INTERPOLATION", rtol=2e-10, atol=2e-8)

    for t in (-7.25, 1.75, 8.5, 19.0):
        state, derivative = history.integration_sample(EPOCH.shifted(t))
        assert_close(state, np.arange(13) * (t**5 - 2 * t**3 + 7), interpolation)
        assert_close(derivative, np.arange(13) * (5 * t**4 - 6 * t**2), interpolation)
    with pytest.raises(ValueError, match="outside"):
        history.integration_sample(EPOCH.shifted(19.1))
    # Translation may be advanced slightly beyond the last accepted node. The Moon sits at
    # GM_E / (GM_E + GM_M) of the relative vector from the (resting) barycentre.
    assert_close(
        history.states(("MOON",), EPOCH.shifted(19.1))[0],
        EARTH_MU / (EARTH_MU + MOON_MU) * np.arange(6) * (19.1**5 - 2 * 19.1**3 + 7),
        Tolerance("EXTRAPOLATION", rtol=2e-11),
    )


@pytest.mark.slow
@pytest.mark.parametrize("core", [False, True])
def test_delayed_propagation_converges_with_step_size(core):
    trajectories = []
    for step in (300.0, 150.0, 75.0):
        system, state = delayed_system(core=core)
        integrator = AdamsBashforthMoultonIntegrator(
            IntegratorSettings(step_s=step, order=6, history_interpolation_order=7, startup_step_s=step / 8)
        )
        trajectories.append(integrator.integrate(system, EPOCH, state, 7200.0, seed_duration_s=3600.0))
    coarse, medium, fine = [t.states[6:, -1] for t in trajectories]

    assert np.linalg.norm(medium[:3] - fine[:3]) < 2e-10
    assert np.linalg.norm(coarse[:3] - fine[:3]) < 2e-9
    for t in trajectories:
        past = t.history.integration_sample(EPOCH.shifted(900.0))[0]
        assert_close(past, t.vector(EPOCH.shifted(900.0)), Tolerance("HISTORY_VS_TRAJECTORY", rtol=1e-14, atol=1e-12))


def test_zero_love_number_reduces_to_the_rigid_propagation():
    settings = IntegratorSettings(step_s=300.0, order=6, startup_step_s=37.5, history_interpolation_order=7)
    integrator = AdamsBashforthMoultonIntegrator(settings)
    deforming, state = delayed_system(love=0.0, core=False)
    rigid, _ = delayed_system(core=False, deformation=False)

    delayed = integrator.integrate(deforming, EPOCH, state, 7200.0, seed_duration_s=3600.0)
    reference = integrator.integrate(rigid, EPOCH, state, 7200.0)

    assert_close(delayed.states[6:], reference.states[6:], Tolerance("RIGID_LIMIT", atol=3e-13))


def test_delayed_propagation_rejects_unsupported_schemes():
    system, state = delayed_system(core=False)
    integrator = AdamsBashforthMoultonIntegrator(
        IntegratorSettings(step_s=300.0, order=6, startup_step_s=37.5, history_interpolation_order=7)
    )
    with pytest.raises(ValueError, match="forward"):
        integrator.integrate(system, EPOCH, state, -7200.0, seed_duration_s=3600.0)
    with pytest.raises(ValueError, match="Explicit rigid seed"):
        integrator.integrate(system, EPOCH, state, 7200.0)
    with pytest.raises(ValueError, match="step <="):
        AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=1200.0)).integrate(
            system, EPOCH, state, 14400.0, seed_duration_s=24000.0
        )


@pytest.mark.slow
def test_abm_delayed_propagation_matches_an_independent_dop853_method_of_steps():
    system, state = delayed_system()
    settings = IntegratorSettings(step_s=150.0, order=6, startup_step_s=18.75, history_interpolation_order=7)
    trajectory = AdamsBashforthMoultonIntegrator(settings).integrate(
        system, EPOCH, state, 7200.0, seed_duration_s=3600.0
    )
    tolerances = np.concatenate((np.full(3, 1e-6), np.full(3, 1e-10), np.full(3, 1e-14), np.full(6, 1e-18)))
    seed = solve_ivp(
        lambda t, y: system.derivatives(EPOCH.shifted(t), y, rigid=True),
        (0, -3600.0),
        state,
        method="DOP853",
        rtol=3e-13,
        atol=tolerances,
        max_step=75.0,
        dense_output=True,
    )
    segments = []

    def sample(epoch):
        """History y(t) from the dense outputs, with y'(t) = f(t, y(t), y(t - delay)) from the RHS itself."""
        t = float(EPOCH.seconds_until(epoch))
        if t <= 0:
            y = state if t == 0 else seed.sol(t)
            return y, system.derivatives(epoch, y, rigid=True) if t < 0 else initial_derivative
        for end, solution in segments:
            if t <= end + 1e-7:
                y = solution(t)
                return y, system.derivatives(epoch, y, history=history)
        raise AssertionError("independent DDE queried an unaccepted future state")

    history = SimpleNamespace(integration_sample=sample)
    initial_derivative = system.derivatives(
        EPOCH,
        state,
        history=SimpleNamespace(
            integration_sample=lambda epoch: (
                seed.sol(EPOCH.seconds_until(epoch)),
                system.derivatives(epoch, seed.sol(EPOCH.seconds_until(epoch)), rigid=True),
            )
        ),
    )
    current = state
    for start in np.arange(0, 7200.0, P.delay_s):
        solution = solve_ivp(
            lambda t, y: system.derivatives(EPOCH.shifted(t), y, history=history),
            (start, start + P.delay_s),
            current,
            method="DOP853",
            rtol=3e-13,
            atol=tolerances,
            max_step=75.0,
            dense_output=True,
        )
        assert solution.success
        segments.append((start + P.delay_s, solution.sol))
        current = solution.y[:, -1]

    assert_close(trajectory.states[6:, -1], current[6:], Tolerance("DDE_ROTATION", atol=3e-12))
    assert_close(trajectory.states[:3, -1], current[:3], Tolerance("DDE_POSITION_M", atol=2e-6, unit="m"))
