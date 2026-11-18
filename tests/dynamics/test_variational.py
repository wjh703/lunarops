"""Variational equations: the partials against global differences, and the state they must not change."""

from functools import partial

import numpy as np
import pytest

from lunarops.classes.dynamics import AdamsBashforthMoultonIntegrator, IntegratorSettings
from lunarops.classes.dynamics.arc_process import BackwardArcProcess
from lunarops.classes.dynamics.variational import DifferenceSteps, LocalJacobian, VariationalEquations, _PartialStore
from tests.dynamics import arc_cases
from tests.parallel import probe_handlers
from tests.support.dynamics import ALL_FORCES, EPOCH, History, build, kernel_difference

DAY = 86400.0
NO_TIDE = tuple(name for name in ALL_FORCES if name != "tide")
CASES = {
    # Instantaneous dynamics: backward DOP853 warm-up, no delayed argument.
    "rigid": {"deformation": False, "core": False, "forces": NO_TIDE},
    # Earth tide: delayed and advanced relative positions, read from the warm-up and the accepted nodes.
    "tide": {"deformation": False, "core": True},
    # Tide and delayed deformation: rigid seed arc, Hermite samples of the state and its derivative.
    "full": {"deformation": True, "core": True},
}
# Scales that make the partials dimensionless: position, velocity, Euler angle, angular velocity.
SCALES = np.r_[[3.84e8] * 3, [1.0e3] * 3, [1.0] * 3, [2.7e-6] * 6]
# Increments of the global differences; the response is linear in them to well below the tolerance.
INCREMENTS = np.r_[[1.0e3] * 3, [1.0e-2] * 3, [1.0e-4] * 3, [1.0e-9] * 6]


def _setup(case, step=2700.0):
    system, initial = build(fused=True, **CASES[case])
    settings = IntegratorSettings(step_s=step, order=13, corrector_iterations=1, startup_step_s=step / 8)
    options = {}
    if CASES[case].get("forces") is None:
        options["external_history_provider"] = History(initial)
    if CASES[case]["deformation"]:
        options["seed_duration_s"] = DAY
    return system, initial, settings, options


def _node_states(trajectory, times):
    return np.asarray([trajectory.vector(EPOCH.shifted(float(offset))) for offset in times]).T


@pytest.mark.parametrize("case", list(CASES))
def test_partials_match_global_differences(case):
    system, initial, settings, options = _setup(case)
    integrator = AdamsBashforthMoultonIntegrator(settings)
    duration = 3 * DAY
    variational = VariationalEquations(system, settings)
    integrator.integrate(system, EPOCH, initial, duration, variational=variational, **options)
    times, partials = variational.times, variational.partials
    dimension = len(initial)
    assert partials.shape == (len(times), dimension, dimension)
    np.testing.assert_array_equal(partials[0], np.eye(dimension))

    scales = SCALES[:dimension]
    worst = 0.0
    for column in range(dimension):
        plus, minus = initial.copy(), initial.copy()
        plus[column] += INCREMENTS[column]
        minus[column] -= INCREMENTS[column]
        difference = (
            _node_states(integrator.integrate(system, EPOCH, plus, duration, **options), times)
            - _node_states(integrator.integrate(system, EPOCH, minus, duration, **options), times)
        ) / (2 * INCREMENTS[column])
        error = (partials[:, :, column].T - difference) * scales[column] / scales[:, None]
        worst = max(worst, float(np.abs(error).max()))
    # Dimensionless partials are of order one; the differences themselves are good to about 1e-8.
    assert worst < 2.0e-7


def test_state_and_history_counters_are_unchanged():
    system, initial, settings, options = _setup("full")
    integrator = AdamsBashforthMoultonIntegrator(settings)
    duration = 2 * DAY
    plain = integrator.integrate(system, EPOCH, initial, duration, **options)
    variational = VariationalEquations(system, settings)
    augmented = integrator.integrate(system, EPOCH, initial, duration, variational=variational, **options)
    times = variational.times
    np.testing.assert_array_equal(_node_states(plain, times), _node_states(augmented, times))
    history, reference = augmented.history_at(EPOCH), plain.history_at(EPOCH)
    assert history.extrapolated_translation_queries == reference.extrapolated_translation_queries
    assert history.max_translation_extrapolation_s == reference.max_translation_extrapolation_s
    assert reference.extrapolated_translation_queries > 0


def test_output_stride_keeps_every_nth_node():
    system, initial, settings, options = _setup("rigid")
    integrator = AdamsBashforthMoultonIntegrator(settings)
    every = VariationalEquations(system, settings)
    integrator.integrate(system, EPOCH, initial, DAY, variational=every, **options)
    thinned = VariationalEquations(system, settings, output_stride=4)
    integrator.integrate(system, EPOCH, initial, DAY, variational=thinned, **options)
    np.testing.assert_array_equal(thinned.times, every.times[::4])
    np.testing.assert_array_equal(thinned.partials, every.partials[::4])
    with pytest.raises(ValueError, match="output_stride"):
        VariationalEquations(system, settings, output_stride=0)


@pytest.mark.parametrize("case", ["tide", "full"])
def test_two_sided_partials_match_global_differences(case):
    # The backward arc starts from the partials of the mirrored forward nodes; with deformation its delayed
    # argument lies ahead of the integration and is extrapolated.
    system, initial, settings, options = _setup(case)
    integrator = AdamsBashforthMoultonIntegrator(settings)
    spans = {"backward_duration_s": 3 * DAY}
    duration = 2 * DAY
    variational = VariationalEquations(system, settings, output_stride=4)
    integrator.integrate(system, EPOCH, initial, duration, variational=variational, **spans, **options)
    times, partials = variational.times, variational.partials
    assert times[0] == -3 * DAY and times[-1] == duration and np.all(np.diff(times) == 4 * settings.step_s)
    dimension = len(initial)
    scales = SCALES[:dimension]
    worst = 0.0
    for column in range(dimension):
        plus, minus = initial.copy(), initial.copy()
        plus[column] += INCREMENTS[column]
        minus[column] -= INCREMENTS[column]
        difference = (
            _node_states(integrator.integrate(system, EPOCH, plus, duration, **spans, **options), times)
            - _node_states(integrator.integrate(system, EPOCH, minus, duration, **spans, **options), times)
        ) / (2 * INCREMENTS[column])
        error = (partials[:, :, column].T - difference) * scales[column] / scales[:, None]
        worst = max(worst, float(np.abs(error).max()))
    assert worst < 2.0e-7


@pytest.mark.parametrize("case", ["tide-deformation", "advanced-tide"])
def test_a_parallel_backward_arc_gives_the_same_partials(case):
    settings, system, initial, spans, options = arc_cases.CASES[case]()
    integrator = AdamsBashforthMoultonIntegrator(settings)
    sequential = VariationalEquations(system, settings, output_stride=3)
    reference = integrator.integrate(
        system, EPOCH, initial, spans["duration"], backward_duration_s=spans["back"], variational=sequential, **options
    )
    parallel = VariationalEquations(system, settings, output_stride=3)
    factory = partial(arc_cases.setup, case)
    with BackwardArcProcess(factory, initializer=probe_handlers.register_test_factories) as arc:
        trajectory = integrator.integrate(
            system,
            EPOCH,
            initial,
            spans["duration"],
            backward_duration_s=spans["back"],
            backward_arc=arc,
            variational=parallel,
            **options,
        )
    np.testing.assert_array_equal(parallel.times, sequential.times)
    np.testing.assert_array_equal(parallel.partials, sequential.partials)
    assert parallel.times[0] < -0.9 * spans["back"] and parallel.times[-1] > 0.9 * spans["duration"]
    np.testing.assert_array_equal(_node_states(trajectory, parallel.times), _node_states(reference, sequential.times))


def test_euler_angle_block_matches_the_analytic_kinematics():
    # 3-1-3 kinematics (Folkner et al. 2014, Eq. 14): e' = E^-1 w, with E the map from Euler-angle rates to
    # the body angular velocity.
    system, y, _, _ = _setup("rigid")
    epoch_batch = system.epoch_batch((EPOCH,))
    step = system.prepare_step(EPOCH, epoch_batch, None)
    jacobian = LocalJacobian(system).state_jacobian(epoch_batch, 0, step, np.ascontiguousarray(y))
    theta, psi = y[7], y[8]
    wx, wy = y[9], y[10]
    p = wx * np.sin(psi) + wy * np.cos(psi)
    d = wx * np.cos(psi) - wy * np.sin(psi)
    sin, cos = np.sin(theta), np.cos(theta)
    angles = np.array([[0.0, -p * cos / sin**2, d / sin], [0.0, 0.0, -p], [0.0, p / sin**2, -d * cos / sin]])
    rates = np.array(
        [
            [np.sin(psi) / sin, np.cos(psi) / sin, 0.0],
            [np.cos(psi), -np.sin(psi), 0.0],
            [-cos * np.sin(psi) / sin, -cos * np.cos(psi) / sin, 1.0],
        ]
    )
    np.testing.assert_allclose(jacobian[6:9, 6:9], angles, rtol=0, atol=1e-7 * np.abs(angles).max())
    np.testing.assert_allclose(jacobian[6:9, 9:12], rates, rtol=1e-9, atol=1e-9)
    # The position rate is the velocity; the Euler angles do not depend on the orbit.
    np.testing.assert_allclose(jacobian[0:3, 3:6], np.eye(3), rtol=0, atol=1e-9)
    assert not jacobian[0:3, [0, 1, 2, 6, 7, 8, 9, 10, 11]].any()
    assert not jacobian[6:9, 0:6].any()


def test_difference_steps_follow_the_state_layout():
    steps = DifferenceSteps()
    assert len(steps.state(6)) == 6 and len(steps.state(12)) == 12 and len(steps.state(15)) == 15
    assert steps.state(15)[6] == steps.angle_rad and steps.state(15)[14] == steps.angular_velocity_rad_s
    assert steps.derivative(15)[9] == steps.angular_acceleration_rad_s2


def test_partial_store_slides_and_addresses_nodes_by_index():
    store = _PartialStore(2, window=3)
    for node in range(40):
        store.append(np.full(2, float(node)), np.full(2, -float(node)))
    assert store.size == 40
    assert store.start >= 40 - 12 and store.size - store.start >= 3
    np.testing.assert_array_equal(store.values(slice(37, 40))[0], [37.0, 38.0, 39.0])
    np.testing.assert_array_equal(store.derivatives(slice(37, 40, 2))[1], [-37.0, -39.0])
    np.testing.assert_array_equal(store.derivative_window(2)[0], [-38.0, -39.0])
    np.testing.assert_array_equal(store.latest(), [39.0, 39.0])
    with pytest.raises(RuntimeError, match="before the stored window"):
        store.values(slice(0, 3))


def _capture_deformation_arguments(monkeypatch, call=200):
    """Propagate the full case and return the LocalJacobian and the arguments of one delayed-deformation call."""
    from lunarops.classes.dynamics import variational as module

    system, initial, settings, options = _setup("full")
    original = module.LocalJacobian.deformation_jacobians
    captured = {}

    def capture(self, batch, index, step, y, state, state_dot):
        result = original(self, batch, index, step, y, state, state_dot)
        captured["calls"] = captured.get("calls", 0) + 1
        if captured["calls"] == call:
            captured["args"] = (self, batch, index, step, y.copy(), state.copy(), state_dot.copy())
        return result

    monkeypatch.setattr(module.LocalJacobian, "deformation_jacobians", capture)
    variational = VariationalEquations(system, settings)
    AdamsBashforthMoultonIntegrator(settings).integrate(
        system, EPOCH, initial, 3 * DAY, variational=variational, **options
    )
    monkeypatch.setattr(module.LocalJacobian, "deformation_jacobians", original)
    return system, captured["args"]


def test_compiled_deformation_differences_match_differences_of_samples(monkeypatch):
    _, (jacobian, *_, state, state_dot) = _capture_deformation_arguments(monkeypatch)
    deformation = jacobian.deformation
    components = np.asarray([0, 4, 7, 10, 9])
    of_derivative = np.asarray([False, False, False, False, True])
    increments = np.asarray([1.0e3, 1.0e-2, 3.0e-4, 1.0e-8, 1.0e-10])
    change, rate = deformation.differences(state, state_dot, components, of_derivative, increments)
    for row, (component, derivative, increment) in enumerate(zip(components, of_derivative, increments, strict=True)):
        samples = []
        for sign in (1.0, -1.0):
            shifted_state, shifted_dot = state.copy(), state_dot.copy()
            (shifted_dot if derivative else shifted_state)[component] += sign * increment
            samples.append(deformation.evaluate_sample(shifted_state, shifted_dot))
        plus, minus = samples
        expected = ((plus.tidal + plus.rotational) - (minus.tidal + minus.rotational)) / (2 * increment)
        np.testing.assert_allclose(change[row], expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())
        expected = (plus.derivative - minus.derivative) / (2 * increment)
        np.testing.assert_allclose(rate[row], expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def test_deformation_jacobians_match_differences_of_the_total_inertia(monkeypatch):
    from lunarops._native.dynamics import KERNEL_EVAL_FIGURE
    from lunarops.classes.dynamics.context import StepData

    _, (jacobian, batch, index, step, y, state, state_dot) = _capture_deformation_arguments(monkeypatch)
    partial_state, partial_dot = jacobian.deformation_jacobians(batch, index, step, y, state, state_dot)
    active_state, active_dot = jacobian._active_deformation_inputs(state, state_dot)
    assert len(active_state) == 12 and list(active_dot) == [9, 10, 11]
    for partials, active, increments, of_state in (
        (partial_state, active_state, jacobian._state_steps, True),
        (partial_dot, active_dot, jacobian._derivative_steps, False),
    ):
        for component in active:
            # The previous form: kernel differences of the whole inertia of two shifted deformation samples, ten
            # times the increment so that its rounding (a change of 1e-13 on 0.4) stays below 1e-5.
            increment = 10 * increments[component]
            samples = []
            for sign in (1.0, -1.0):
                shifted_state, shifted_dot = state.copy(), state_dot.copy()
                (shifted_state if of_state else shifted_dot)[component] += sign * increment
                samples.append(
                    StepData(step.force_payloads, jacobian.deformation.evaluate_sample(shifted_state, shifted_dot))
                )
            reference = kernel_difference(
                jacobian._fused, batch, *samples, y, y, increment, KERNEL_EVAL_FIGURE, index=index
            )
            for block in (slice(3, 6), slice(9, 12), slice(12, 15)):
                scale = np.abs(reference[block]).max()
                if scale > 0:
                    # Truncation of the angle columns at ten times the increment, (3e-3)^2.
                    assert np.abs(partials[block, component] - reference[block]).max() < 2.0e-5 * scale, (
                        component,
                        block,
                    )
