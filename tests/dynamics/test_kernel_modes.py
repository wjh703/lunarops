"""Partial evaluation modes of the compiled right-hand side, and the Jacobians built on them."""

import numpy as np

from lunarops._dynamics_core import KERNEL_EVAL_FIGURE, KERNEL_EVAL_FULL, KERNEL_EVAL_TIDE
from lunarops.classes.dynamics.context import StepData
from lunarops.classes.dynamics.variational import DifferenceSteps
from tests.support.dynamics import EPOCH, History, build


def _setup():
    system, initial = build(fused=True, deformation=False)
    history = History(initial)
    epoch = EPOCH.shifted(3600.0)
    batch = system.epoch_batch((epoch,))
    step = system.prepare_step(epoch, batch, history)
    return system, np.ascontiguousarray(initial, dtype=float), batch, step


def _difference(fused, batch, step_plus, step_minus, plus, minus, increment, mode=KERNEL_EVAL_FULL):
    zero = np.zeros_like(plus)
    high_plus, low_plus = fused.split_derivative(batch, 0, step_plus, plus, zero, mode)
    high_minus, low_minus = fused.split_derivative(batch, 0, step_minus, minus, zero, mode)
    return ((high_plus - high_minus) + (low_plus - low_minus)) / (2.0 * increment)


def test_full_mode_is_the_default_derivative():
    system, y, batch, step = _setup()
    fused, zero = system._fused, np.zeros_like(y)
    default = fused.split_derivative(batch, 0, step, y, zero)
    full = fused.split_derivative(batch, 0, step, y, zero, KERNEL_EVAL_FULL)
    for left, right in zip(default, full, strict=True):
        np.testing.assert_array_equal(left, right)


def test_figure_mode_keeps_the_terms_that_depend_on_the_rotation_state():
    system, y, batch, step = _setup()
    fused, zero = system._fused, np.zeros_like(y)
    full, _ = fused.split_derivative(batch, 0, step, y, zero)
    figure, _ = fused.split_derivative(batch, 0, step, y, zero, KERNEL_EVAL_FIGURE)
    terms = fused.terms(batch, 0, step, y)
    # The rotation rows are evaluated in full; the acceleration holds the Moon figure force alone.
    np.testing.assert_array_equal(figure[6:], full[6:])
    np.testing.assert_array_equal(figure[:3], full[:3])
    moon_figure = terms["forces"]["figure_moon"] - terms["earth_forces"]["figure_moon"]
    np.testing.assert_allclose(figure[3:6], moon_figure, rtol=1e-12, atol=0)


def test_tide_mode_keeps_the_tide_force_alone():
    system, y, batch, step = _setup()
    fused, zero = system._fused, np.zeros_like(y)
    tide, _ = fused.split_derivative(batch, 0, step, y, zero, KERNEL_EVAL_TIDE)
    terms = fused.terms(batch, 0, step, y)
    np.testing.assert_allclose(tide[3:6], terms["forces"]["tide"] - terms["earth_forces"]["tide"], rtol=1e-12, atol=0)
    assert not tide[6:].any()


def test_kernel_state_jacobian_matches_full_differences():
    system, y, batch, step = _setup()
    fused = system._fused
    increments = DifferenceSteps().state(len(y))
    jacobian = fused.state_jacobian(batch, 0, step, y, increments)
    for column, increment in enumerate(increments):
        plus, minus = y.copy(), y.copy()
        plus[column] += increment
        minus[column] -= increment
        reference = _difference(fused, batch, step, step, plus, minus, increment)
        if column < 6:
            # Orbit columns: the same evaluations in the same order.
            np.testing.assert_array_equal(jacobian[:, column], reference)
        else:
            # Rotation columns: the terms left out are equal on both sides and cancel up to their rounding.
            floor = np.spacing(np.abs(fused.split_derivative(batch, 0, step, y, np.zeros_like(y))[0])) / increment
            np.testing.assert_array_less(np.abs(jacobian[:, column] - reference), 4 * floor + 1e-300)


def test_kernel_tide_jacobian_matches_shifted_payloads():
    system, y, batch, step = _setup()
    fused = system._fused
    vectors, raiser_gm = step.force_payloads["tide"]
    before = fused.split_derivative(batch, 0, step, y, np.zeros_like(y))[0]
    rng = np.random.default_rng(1)
    shifts = rng.normal(scale=1.0e3, size=(3, 3, vectors.size))
    increment = 1.0e3
    partials = fused.tide_jacobian(batch, 0, step, y, shifts, increment)
    for order in range(3):
        for axis in range(3):
            shift = shifts[order, axis].reshape(vectors.shape)
            plus = StepData({**step.force_payloads, "tide": (vectors + shift, raiser_gm)}, step.lunar_inertia)
            minus = StepData({**step.force_payloads, "tide": (vectors - shift, raiser_gm)}, step.lunar_inertia)
            reference = _difference(fused, batch, plus, minus, y, y, increment)
            floor = np.spacing(np.abs(fused.split_derivative(batch, 0, step, y, np.zeros_like(y))[0])) / increment
            np.testing.assert_array_less(np.abs(partials[order, :, axis] - reference), 4 * floor + 1e-300)
    # The loaded tide vectors are restored.
    np.testing.assert_array_equal(fused.split_derivative(batch, 0, step, y, np.zeros_like(y))[0], before)


def _segment(fused, name):
    offset, size = next((offset, size) for segment, offset, size in fused.kernel.input_layout if segment == name)
    return slice(offset, offset + size)


def test_input_vector_follows_the_kernel_input_layout():
    system, _, batch, step = _setup()
    fused = system._fused
    vector = fused.input_vector(step, batch.gravitational_parameters_m3_s2[0])
    assert vector.size == fused.input_size
    vectors, raiser_gm = step.force_payloads["tide"]
    np.testing.assert_array_equal(vector[_segment(fused, "tide_vectors")], vectors.ravel())
    np.testing.assert_array_equal(vector[_segment(fused, "raiser_gm")], raiser_gm)
    np.testing.assert_array_equal(
        vector[_segment(fused, "earth_moon_gm")][:2], batch.gravitational_parameters_m3_s2[0, :2]
    )


def test_input_differences_match_shifted_payloads_and_restore_the_inputs():
    system, y, batch, step = _setup()
    fused, zero = system._fused, np.zeros_like(y)
    vectors, raiser_gm = step.force_payloads["tide"]
    before = fused.split_derivative(batch, 0, step, y, zero)[0]
    rng = np.random.default_rng(2)
    shift = rng.normal(scale=1.0e3, size=vectors.shape)
    plus, minus = np.zeros((2, fused.input_size)), np.zeros((2, fused.input_size))
    # Row 0: tide vectors, unevenly on either side; row 1: the Earth and Moon GM of the epoch.
    plus[0, _segment(fused, "tide_vectors")] = shift.ravel()
    minus[0, _segment(fused, "tide_vectors")] = -0.5 * shift.ravel()
    gm = batch.gravitational_parameters_m3_s2[0, :2]
    gm_segment = _segment(fused, "earth_moon_gm")
    plus[1, gm_segment.start : gm_segment.start + 2] = 1.0e-6 * gm
    minus[1, gm_segment.start : gm_segment.start + 2] = -1.0e-6 * gm
    increments = np.asarray([1.0e3, 4.0e8])
    modes = np.asarray([KERNEL_EVAL_TIDE, KERNEL_EVAL_FULL], dtype=np.intc)
    partials = fused.input_differences(batch, 0, step, y, plus, minus, increments, modes)
    floor = np.spacing(np.abs(before)) / increments[:, None]

    tide_plus = StepData({**step.force_payloads, "tide": (vectors + shift, raiser_gm)}, step.lunar_inertia)
    tide_minus = StepData({**step.force_payloads, "tide": (vectors - 0.5 * shift, raiser_gm)}, step.lunar_inertia)
    reference = _difference(fused, batch, tide_plus, tide_minus, y, y, increments[0])
    np.testing.assert_array_less(np.abs(partials[:, 0] - reference), 4 * floor[0] + 1e-300)

    from lunarops.classes.dynamics.parameters import batch_with_earth_moon_gm

    high_plus, low_plus = fused.split_derivative(batch_with_earth_moon_gm(batch, *(gm * (1 + 1e-6))), 0, step, y, zero)
    high_minus, low_minus = fused.split_derivative(
        batch_with_earth_moon_gm(batch, *(gm * (1 - 1e-6))), 0, step, y, zero
    )
    reference = ((high_plus - high_minus) + (low_plus - low_minus)) / (2.0 * increments[1])
    np.testing.assert_array_less(np.abs(partials[:, 1] - reference), 4 * floor[1] + 1e-300)
    np.testing.assert_array_equal(fused.split_derivative(batch, 0, step, y, zero)[0], before)
