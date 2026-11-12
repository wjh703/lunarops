"""Double-double accumulation of the ABM integrator and two-sided propagation."""

from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
import pytest

from lunarops._dynamics_core import adams_accumulate
from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, IntegratorSettings
from lunarops.classes.dynamics import integrators as integrators_module
from lunarops.classes.dynamics.integrators import _integral_weights
from tests.dynamics.test_deformation import delayed_system
from tests.dynamics.test_translation import CIRCULAR, MEAN_MOTION, RADIUS, TEN_DAYS, _two_body
from tests.support.dynamics import EPOCH, build
from tests.tolerances import Tolerance, assert_close, assert_vector_close

DAY = 86400.0
getcontext().prec = 60


def _value(high, low):
    return Fraction(float(high)) + Fraction(float(low))


def test_adams_weights_are_exact_double_doubles():
    order = 13
    high, low = _integral_weights(range(0, -order, -1))
    assert abs(sum(_value(a, b) for a, b in zip(high, low, strict=True)) - 1) < Fraction(1, 10**28)
    # Fourth-order Adams-Bashforth weights are known in closed form.
    high, low = _integral_weights(range(0, -4, -1))
    for hi, lo, exact in zip(
        high, low, (Fraction(55, 24), Fraction(-59, 24), Fraction(37, 24), Fraction(-9, 24)), strict=True
    ):
        assert abs(_value(hi, lo) - exact) < Fraction(1, 10**30)


def test_adams_accumulate_is_accurate_to_double_double():
    step = 5400.0
    state, state_low = np.array([3.84e8, -1.0]), np.array([2.0e-9, 1e-17])
    derivative = np.array([[1.0e3, -7.3e2, 3.1e2], [2.7e-3, 1.1e-3, -9.0e-4]])
    derivative_low = np.array([[1e-14, -3e-13, 2e-13], [4e-20, -1e-19, 3e-20]])
    weight, weight_low = (
        np.array([2.2916666666666665, -2.4583333333333335, 1.5416666666666667]),
        np.array([1e-16, -1e-17, 4e-17]),
    )
    high, low = adams_accumulate(step, state, state_low, derivative, derivative_low, weight, weight_low)
    for row in range(2):
        exact = _value(state[row], state_low[row]) + Fraction(step) * sum(
            _value(weight[j], weight_low[j]) * _value(derivative[row, j], derivative_low[row, j]) for j in range(3)
        )
        assert abs(float(_value(high[row], low[row]) - exact)) < 1e-26 * abs(float(exact))
        assert abs(low[row]) <= np.spacing(abs(high[row]))


def test_two_body_acceleration_carries_the_low_state_part():
    system, state = _two_body(), CIRCULAR
    kernel = system._fused.kernel
    batch = system.epoch_batch((EPOCH,))
    system._fused._load(batch, 0, system.prepare_step(EPOCH, batch, None))
    low = np.zeros_like(state)
    low[:3] = (3e-9, -2e-9, 4e-9)
    high, low_part = kernel.split_derivative(state, low)
    base_high, base_low = kernel.split_derivative(state, np.zeros_like(state))
    gm = batch.gravitational_parameters_m3_s2[0]
    mu = Decimal(float(gm[0])) + Decimal(float(gm[1]))
    position = [Decimal(float(value)) for value in state[:3]]
    shifted = [c + Decimal(float(d)) for c, d in zip(position, low[:3], strict=True)]

    def acceleration(x):
        r2 = sum(c * c for c in x)
        return [-mu * c / (r2 * r2.sqrt()) for c in x]

    expected = [float(p - q) for p, q in zip(acceleration(shifted), acceleration(position), strict=True)]
    actual = [float(_value(high[3 + k], low_part[3 + k]) - _value(base_high[3 + k], base_low[3 + k])) for k in range(3)]
    assert np.all(np.abs(expected) > 0)
    assert_close(np.asarray(actual), np.asarray(expected), Tolerance("LOW_PART_GRADIENT", rtol=1e-6))


def test_two_sided_propagation_matches_separate_arcs():
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=1800.0, order=8))
    two_sided = integrator.integrate(_two_body(), EPOCH, CIRCULAR, TEN_DAYS, backward_duration_s=TEN_DAYS)
    forward = integrator.integrate(_two_body(), EPOCH, CIRCULAR, TEN_DAYS)
    backward = integrator.integrate(_two_body(), EPOCH, CIRCULAR, -TEN_DAYS)

    assert two_sided.times_s[0] == -TEN_DAYS and two_sided.times_s[-1] == TEN_DAYS
    assert np.all(np.diff(two_sided.times_s) == 1800.0)
    np.testing.assert_array_equal(two_sided.states[:, two_sided.times_s >= 0], forward.states)
    assert_vector_close(
        two_sided.states[:3, 0], backward.states[:3, -1], Tolerance("TWO_SIDED_BACKWARD", atol=2e-5, unit="m")
    )
    assert_close(two_sided.vector(EPOCH.shifted(0.0)), CIRCULAR, Tolerance("TWO_SIDED_START", atol=1e-12))


@pytest.mark.parametrize("arguments", [(1800.0 * 8, -1.0), (-1800.0 * 8, 1800.0)])
def test_two_sided_propagation_rejects_invalid_requests(arguments):
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=1800.0, order=8))
    duration, backward = arguments
    with pytest.raises(ValueError, match="backward_duration_s|two-sided"):
        integrator.integrate(_two_body(), EPOCH, CIRCULAR, duration, backward_duration_s=backward)


@pytest.mark.parametrize("core", [False, True])
def test_two_sided_delayed_propagation_closes_with_the_forward_arc(core):
    system, state = delayed_system(core=core)
    settings = IntegratorSettings(step_s=300.0, order=6, history_interpolation_order=7, startup_step_s=37.5)
    integrator = AdamsBashforthMoultonIntegrator(settings)
    forward = integrator.integrate(system, EPOCH, state, DAY, seed_duration_s=3600.0)

    middle_epoch = EPOCH.shifted(DAY / 2)
    system, _ = delayed_system(core=core)
    two_sided = integrator.integrate(
        system, middle_epoch, forward.vector(middle_epoch), DAY / 2, backward_duration_s=DAY / 2, seed_duration_s=3600.0
    )

    assert np.all(np.diff(two_sided.times_s) == 300.0)
    # Backward closure onto the epoch where the forward arc started.
    back_end = two_sided.vector(EPOCH)
    assert_vector_close(back_end[:3], state[:3], Tolerance("DELAY_CLOSURE_M", atol=2e-3, unit="m"))
    assert_vector_close(back_end[6:9], state[6:9], Tolerance("DELAY_CLOSURE_ANGLES", atol=1e-9))
    # Between nodes, so the Hermite history and the Lagrange trajectory are two different interpolants.
    between = EPOCH.shifted(DAY / 4 + 123.4)
    past = two_sided.history.integration_sample(between)[0]
    assert_close(past[:3], two_sided.vector(between)[:3], Tolerance("DELAY_HISTORY", atol=1e-6, unit="m"))


def test_adams_moulton_weights_are_exact():
    # Fourth-order Adams-Moulton weights on the nodes 1, 0, -1, -2.
    high, low = _integral_weights([1, 0, -1, -2])
    for hi, lo, exact in zip(
        high, low, (Fraction(9, 24), Fraction(19, 24), Fraction(-5, 24), Fraction(1, 24)), strict=True
    ):
        assert abs(_value(hi, lo) - exact) < Fraction(1, 10**30)


def test_double_double_accumulation_removes_the_along_track_random_walk(monkeypatch):
    """One year of a circular Kepler orbit at the production step and order, with and without the low parts."""
    year = 5844 * 5400.0
    settings = IntegratorSettings(step_s=5400.0, order=13)

    def along_track_error(trajectory):
        times, states = trajectory.times_s, trajectory.states
        angle = np.arctan2(states[1], states[0]) - np.mod(MEAN_MOTION * times + np.pi, 2 * np.pi) + np.pi
        return np.max(np.abs(RADIUS * np.angle(np.exp(1j * angle))))

    double_double = along_track_error(
        AdamsBashforthMoultonIntegrator(settings).integrate(_two_body(), EPOCH, CIRCULAR, year)
    )
    accumulate = integrators_module.adams_accumulate

    def high_only(*arguments):
        high, low = accumulate(*arguments)
        return high, np.zeros_like(low)

    monkeypatch.setattr(integrators_module, "adams_accumulate", high_only)
    binary64 = along_track_error(
        AdamsBashforthMoultonIntegrator(settings).integrate(_two_body(), EPOCH, CIRCULAR, year)
    )

    assert double_double < 2e-5
    assert binary64 > 10 * double_double, (binary64, double_double)


def test_backward_arc_shorter_than_the_start_up_is_seeded_by_the_forward_nodes():
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=1800.0, order=8))
    two_sided = integrator.integrate(_two_body(), EPOCH, CIRCULAR, TEN_DAYS, backward_duration_s=1800.0)

    assert two_sided.times_s[0] == -1800.0 and two_sided.times_s[1] == 0.0
    expected = [RADIUS * np.cos(-MEAN_MOTION * 1800.0), RADIUS * np.sin(-MEAN_MOTION * 1800.0), 0.0]
    assert_vector_close(two_sided.states[:3, 0], expected, Tolerance("SHORT_BACKWARD_ARC", atol=1e-5, unit="m"))


def test_a_backward_duration_off_the_grid_is_rejected_before_integrating(monkeypatch):
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=1800.0, order=13))
    calls = []
    monkeypatch.setattr(integrator, "_main_grid", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="multiple"):
        integrator.integrate(_two_body(), EPOCH, CIRCULAR, TEN_DAYS, backward_duration_s=1000.0)
    assert not calls


def test_a_forward_arc_too_short_to_cover_the_advanced_tide_is_rejected():
    # At 600 s the mirrored nodes must reach 0.21 d (31 steps) ahead for the semidiurnal tide argument.
    system, state = build(fused=True, rotation=False)
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=600.0, order=4, startup_step_s=75.0))
    with pytest.raises(ValueError, match="seed the backward arc"):
        integrator.integrate(system, EPOCH, state, 10 * 600.0, backward_duration_s=600.0)


def test_two_sided_delayed_deformation_stays_stable_at_small_steps():
    """Lunar delay tau = 0.145 d at a 2700 s step: before the widened extrapolation stencil the backward arc
    diverged within 20 days. It must follow a one-sided run started from its far end."""
    tau, step = 0.14519388 * DAY, 2700.0
    settings = IntegratorSettings(step_s=step, order=13, history_interpolation_order=13, startup_step_s=step / 8)
    integrator = AdamsBashforthMoultonIntegrator(settings)
    system, state = delayed_system(delay_s=tau)
    back = 20 * DAY
    two_sided = integrator.integrate(system, EPOCH, state, DAY, backward_duration_s=back, seed_duration_s=DAY)

    assert np.all(np.isfinite(two_sided.states))
    start = EPOCH.shifted(-back)
    system, _ = delayed_system(delay_s=tau)
    one_sided = integrator.integrate(system, start, two_sided.states[:, 0], back + DAY, seed_duration_s=DAY)
    difference = two_sided.states - one_sided.states
    assert np.max(np.abs(difference[:3])) < 1e-3
    assert np.max(np.abs(difference[6:9])) < 1e-12

    # Re-evaluating the dynamics uses the history of the arc that integrated the epoch.
    assert two_sided.history_at(EPOCH.shifted(step)) is two_sided.history._forward
    assert two_sided.history_at(EPOCH.shifted(-step)) is two_sided.history._backward
