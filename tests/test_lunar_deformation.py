"""Independent tensor identities and forward delayed propagation checks."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from test_lunar_dynamics import EARTH_MU, MOON_MU
from test_lunar_rotation import EPOCH, RADIUS, RELATIVE, K, Q, make_system

from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    FigureForce,
    IntegratorSettings,
    LunarCoreParameters,
    LunarFigureTorque,
    LunarRotationDynamics,
    LunarTorqueGroup,
    degree2_gravity_coefficients_from_inertia,
)
from lunarops.classes.dynamics.deformation import (
    LunarDeformation,
    LunarDeformationParameters,
    LunarInertiaEvaluation,
    deformation_tensors,
)
from lunarops.classes.dynamics.gravity import GravityField
from lunarops.classes.dynamics.history import DelayedHistory, LunarNodeBuffer

P = LunarDeformationParameters(0.024059, 900.0, 2.6616995e-6)


def test_analytic_inertia_derivative_matches_central_difference():
    r = np.array([3.8e8, 2e7, -1e7])
    rd = np.array([130.0, -970.0, 24.0])
    w = np.array([2e-8, -1e-8, 2.7e-6])
    wd = np.array([1e-13, -2e-14, 3e-15])

    def tensors(t):
        return deformation_tensors(r + t * rd, rd, w + t * wd, wd,
                                  parameters=P, radius_m=RADIUS, earth_mu=EARTH_MU, moon_mu=MOON_MU)

    tidal, spin, derivative = tensors(0)
    plus, minus = tensors(0.1), tensors(-0.1)
    fd = ((plus[0] + plus[1]) - (minus[0] + minus[1])) / 0.2
    np.testing.assert_allclose(derivative, fd, rtol=2e-8, atol=1e-21)
    for tensor in (tidal, spin, derivative):
        np.testing.assert_array_equal(tensor, tensor.T)
        assert abs(np.trace(tensor)) < 1e-20


@pytest.mark.parametrize("tidal,spin", [(False, False), (True, False), (False, True)])
def test_deformation_switches_control_both_inertia_and_its_derivative(tidal, spin):
    args = (np.array([3.8e8, 2e7, -1e7]), np.array([100, 20, 10]), np.array([1e-7, 2e-8, 2.7e-6]), np.array([1e-13, 2e-14, 0]))
    kwargs = {"radius_m": RADIUS, "earth_mu": EARTH_MU, "moon_mu": MOON_MU}
    full = deformation_tensors(*args, parameters=P, **kwargs)
    selected = deformation_tensors(
        *args, parameters=LunarDeformationParameters(P.love_k2, P.delay_s, P.mean_motion_rad_s, tidal, spin), **kwargs
    )
    np.testing.assert_array_equal(selected[0], full[0] if tidal else np.zeros((3, 3)))
    np.testing.assert_array_equal(selected[1], full[1] if spin else np.zeros((3, 3)))
    if not tidal and not spin:
        np.testing.assert_array_equal(selected[2], 0)


def test_joint_hermite_history_uses_old_stencil_and_one_polynomial_derivative():
    system = make_system(rotation=LunarRotationDynamics(K, RADIUS, LunarTorqueGroup()))
    # Enable the joint-history view without invoking a physical RHS.
    system.rotation.deformation = SimpleNamespace()
    nodes = LunarNodeBuffer(30, 13, 1)
    for index, t in enumerate(range(-10, 20)):
        state = np.arange(13) * (t**5 - 2 * t**3 + 7)
        derivative = np.arange(13) * (5 * t**4 - 6 * t**2)
        nodes.append(t, state, derivative, np.zeros((1, 6)))
        if index == 10:
            nodes.mark_trajectory_start()
    history = DelayedHistory(system, EPOCH, nodes, 7)
    # A far older epoch must NOT extrapolate the last few accepted nodes.
    for t in (-7.25, 1.75, 8.5, 19.0):
        state, derivative = history.integration_sample(EPOCH.shifted(t))
        np.testing.assert_allclose(state, np.arange(13) * (t**5 - 2 * t**3 + 7), rtol=2e-10, atol=2e-8)
        np.testing.assert_allclose(derivative, np.arange(13) * (5 * t**4 - 6 * t**2), rtol=2e-10, atol=2e-8)
    with pytest.raises(ValueError, match="outside"):
        history.integration_sample(EPOCH.shifted(19.1))
    advanced = history(("MOON",), EPOCH.shifted(19.1))
    np.testing.assert_allclose(advanced[0], np.arange(6) * (19.1**5 - 2 * 19.1**3 + 7), rtol=2e-11)
    assert history.advanced_translation_queries == 1
    assert history.max_translation_extrapolation_s == pytest.approx(.1)


def test_variable_mantle_chain_rule_conserves_inertial_angular_momentum():
    rotation = LunarRotationDynamics(K, RADIUS, LunarTorqueGroup(), core=LunarCoreParameters(.0007, .000246, 7e-14, "core"))
    system = make_system(rotation=rotation)
    wm, wc = np.array([2e-7, -3e-8, 2.7e-6]), np.array([1e-7, 1e-8, 2.6e-6])
    state = np.concatenate((Q, wm, wc))
    inputs = system._load_evaluation(system.build_epoch_data(EPOCH), np.concatenate((RELATIVE, state)), None, None)
    delta = np.array([[2e-7, 1e-8, -3e-9], [1e-8, -1e-7, 2e-9], [-3e-9, 2e-9, -1e-7]])
    kd = np.array([[2e-13, -3e-14, 1e-14], [-3e-14, -1e-13, 2e-14], [1e-14, 2e-14, -1e-13]])
    inputs.lunar_inertia = LunarInertiaEvaluation(K + delta, kd, delta, np.zeros((3, 3)))
    result = rotation.evaluate(inputs, state)
    km = K + delta - rotation.core_inertia
    H = km @ wm + rotation.core_inertia @ wc
    Hd = km @ result.derivative[4:7] + kd @ wm + rotation.core_inertia @ result.derivative[7:] + np.cross(wm, H)
    np.testing.assert_allclose(Hd, 0, atol=5e-27)


def delayed_system(*, love=P.love_k2, core=True):
    field = GravityField(degree2_gravity_coefficients_from_inertia(K, gravitational_parameter_m3_s2=MOON_MU, radius_m=RADIUS))
    deformation = LunarDeformation(LunarDeformationParameters(love, P.delay_s, P.mean_motion_rad_s), K, RADIUS,
                                   EARTH_MU, MOON_MU, field)
    rotation = LunarRotationDynamics(K, RADIUS, LunarTorqueGroup([LunarFigureTorque("EARTH", field)]),
                                    deformation=deformation,
                                    core=LunarCoreParameters(.0007, .000246, 7e-14, "core") if core else None)
    force = FigureForce({"MOON": field}, {"MOON": ("EARTH",)})
    system = make_system(rotation=rotation, forces=[force])
    initial = np.concatenate((RELATIVE, Q, [1e-7, -2e-8, 2.7e-6], [2e-8, 1e-8, 2.6e-6] if core else []))
    return system, initial


def test_force_and_torque_share_updated_field_and_delayed_candidate_cache():
    system, state = delayed_system()
    calls = []
    history = SimpleNamespace(integration_sample=lambda epoch: (state.copy(), np.zeros(16)))
    field = system.rotation.deformation.field
    original = field.nonspherical_accelerations

    def counted(r):
        calls.append(r.copy())
        return original(r)

    field.nonspherical_accelerations = counted
    before = field.coefficients.coeffs.copy()
    data = system.build_epoch_data(EPOCH)
    first = system.derivatives(EPOCH, state, data, history=history)
    assert len(calls) == 1
    assert not np.array_equal(before[:, 2], field.coefficients.coeffs[:, 2])
    # Change delayed omega at the SAME epoch: no stale field/state cache.
    later = state.copy()
    later[10] *= 10
    history.integration_sample = lambda epoch: (later, np.zeros(16))
    second = system.derivatives(EPOCH, state, data, history=history)
    assert len(calls) == 2
    assert not np.array_equal(first[3:6], second[3:6])
    assert not np.array_equal(first[10:13], second[10:13])


@pytest.mark.parametrize("core", [False, True])
def test_delayed_propagation_step_convergence_and_explicit_seed(core):
    trajectories = []
    for step in (300., 150., 75.):
        system, state = delayed_system(core=core)
        integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(
            step_s=step, order=6, history_interpolation_order=7, startup_step_s=step/8,
        ))
        trajectories.append(integrator.integrate(system, EPOCH, state, 7200., seed_duration_s=3600.))
    coarse, medium, fine = [t.states[6:, -1] for t in trajectories]
    assert np.linalg.norm(medium[:4] - fine[:4]) < 2e-10
    assert np.linalg.norm(coarse[:4] - fine[:4]) < 2e-9
    assert np.max(abs(np.linalg.norm(trajectories[0].states[6:10], axis=0) - 1)) < 2e-12
    for t in trajectories:
        past = t.history.integration_sample(EPOCH.shifted(900.))
        np.testing.assert_allclose(past[0], t.vector(EPOCH.shifted(900.)), rtol=3e-16, atol=1e-12)


def test_zero_love_reduces_to_constant_inertia_and_invalid_delay_scheme_fails():
    system, state = delayed_system(love=0., core=False)
    settings = IntegratorSettings(step_s=300., order=6, startup_step_s=37.5, history_interpolation_order=7)
    integrator = AdamsBashforthMoultonIntegrator(settings)
    delayed = integrator.integrate(system, EPOCH, state, 7200., seed_duration_s=3600.)
    system.rotation.deformation = None
    rigid = integrator.integrate(system, EPOCH, state, 7200.)
    np.testing.assert_allclose(delayed.states[6:], rigid.states[6:], atol=3e-13, rtol=0)
    system.rotation.deformation = SimpleNamespace(parameters=P)
    for duration, seed, message in [(-7200., 3600., "forward"), (7200., None, "Explicit rigid seed")]:
        with pytest.raises(ValueError, match=message):
            integrator.integrate(system, EPOCH, state, duration, seed_duration_s=seed)
    with pytest.raises(ValueError, match="step <="):
        AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=1200.)).integrate(
            system, EPOCH, state, 7200., seed_duration_s=24000.)


def dense_sample(solution, t):
    """Differentiate SciPy's DOP853 dense polynomial independently of LunarOps."""
    direction = np.sign(solution.ts[-1] - solution.ts[0])
    i = max(0, min(np.searchsorted(direction * solution.ts, direction * t, side="right") - 1,
                   len(solution.interpolants) - 1))
    segment = solution.interpolants[i]
    x = (t - segment.t_old) / segment.h
    y, derivative = np.zeros_like(segment.y_old), np.zeros_like(segment.y_old)
    for index, f in enumerate(reversed(segment.F)):
        y = y + f
        if index % 2 == 0:
            derivative, y = derivative * x + y, y * x
        else:
            derivative, y = derivative * (1 - x) - y, y * (1 - x)
    return y + segment.y_old, derivative / segment.h


def test_abm_delayed_rhs_matches_independent_dop853_method_of_steps():
    system, state = delayed_system()
    settings = IntegratorSettings(step_s=150., order=6, startup_step_s=18.75, history_interpolation_order=7)
    trajectory = AdamsBashforthMoultonIntegrator(settings).integrate(system, EPOCH, state, 7200., seed_duration_s=3600.)
    tolerances = np.concatenate((np.full(3, 1e-6), np.full(3, 1e-10), np.full(4, 1e-14), np.full(6, 1e-18)))
    seed = solve_ivp(lambda t, y: system.derivatives(EPOCH.shifted(t), y, rigid=True), (0, -3600.), state,
                     method="DOP853", rtol=3e-13, atol=tolerances, max_step=75., dense_output=True)
    segments = []
    initial_derivative = system.derivatives(EPOCH, state, history=SimpleNamespace(
        integration_sample=lambda epoch: dense_sample(seed.sol, float(EPOCH.seconds_until(epoch)))
    ))

    def sample(epoch):
        t = float(EPOCH.seconds_until(epoch))
        if t < 0:
            return dense_sample(seed.sol, t)
        if t == 0:
            return state, initial_derivative
        for end, solution in segments:
            if t <= end + 1e-7:
                return dense_sample(solution, t)
        raise AssertionError("Independent DDE queried unaccepted future state")

    history = SimpleNamespace(integration_sample=sample)
    current = state
    for start in np.arange(0, 7200., P.delay_s):
        end = start + P.delay_s
        solution = solve_ivp(
            lambda t, y: system.derivatives(EPOCH.shifted(t), y, history=history), (start, end), current,
            method="DOP853", rtol=3e-13, atol=tolerances, max_step=75., dense_output=True,
        )
        assert solution.success
        segments.append((end, solution.sol))
        current = solution.y[:, -1]
    np.testing.assert_allclose(trajectory.states[6:, -1], current[6:], rtol=0, atol=3e-12)
    np.testing.assert_allclose(trajectory.states[:3, -1], current[:3], rtol=0, atol=2e-6)
