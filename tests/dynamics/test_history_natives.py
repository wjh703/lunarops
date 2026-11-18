"""The compiled history and variational helpers reproduce the numpy formulas they replace, bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from lunarops._native.dynamics import (
    hermite_evaluate_native,
    lagrange_column_sample,
    tide_lag_shifts,
    tide_vector_shifts,
)
from lunarops.classes.dynamics.history import interpolate_state_derivative
from lunarops.classes.dynamics.trajectory import interpolate_regular


def _bits(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=float).view(np.uint64)


@pytest.mark.parametrize("seed", range(4))
def test_lagrange_column_sample_matches_interpolate_regular(seed):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        count = int(rng.integers(1, 40))
        step = float(rng.choice([5400.0, -5400.0, 2700.0]))
        nodes = float(rng.uniform(-1e6, 1e6)) + step * np.arange(count)
        rows = int(rng.choice([3, 15, 81]))
        values = rng.standard_normal((rows, count)) * 10 ** rng.uniform(-5, 9, (rows, 1))
        x = float(nodes[0] + step * rng.uniform(-3, count + 3))
        if rng.random() < 0.1:
            x = float(nodes[int(rng.integers(count))])
        order = int(rng.choice([0, 6, 13]))
        for extrapolate in (True, False):
            # check_grid=True keeps the numpy implementation; the grid is regular, so it does not raise.
            try:
                expected = interpolate_regular(nodes, values, x, order, extrapolate=extrapolate, check_grid=True)
            except ValueError:
                with pytest.raises(ValueError):
                    lagrange_column_sample(nodes, values, x, order + 1, extrapolate)
                continue
            # A strided view, as the history and the partial store pass it.
            padded = np.zeros((rows + 2, count + 3))
            padded[1 : rows + 1, :count] = values
            result = lagrange_column_sample(nodes, padded[1 : rows + 1, :count], x, order + 1, extrapolate)
            np.testing.assert_array_equal(_bits(result), _bits(expected))


def test_hermite_evaluate_native_matches_python_interpolant():
    rng = np.random.default_rng(3)
    grid = 5400.0 * np.arange(7) - 2.0e5
    states = rng.standard_normal((15, 7)) * 1e8
    derivatives = rng.standard_normal((15, 7))
    for offset in (grid[0] + 1234.5, grid[3], grid[-1] + 6000.0):
        state, derivative = hermite_evaluate_native(grid, states, derivatives, offset)
        expected_state, expected_derivative = interpolate_state_derivative(grid, states, derivatives, offset)
        # The compiled sums are ordered differently from the numpy products: a few ulp.
        np.testing.assert_allclose(state, expected_state, rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-10, atol=1e-12)


def test_tide_vector_shifts_match_numpy():
    rng = np.random.default_rng(4)
    raisers = 5
    share = rng.random(raisers)
    lagged = [rng.standard_normal((3, 3)) for _ in range(3)]
    increment = np.float64(1.0e3)
    vectors = np.zeros((raisers, 3, 3))
    expected = np.empty((3, 3, vectors.size))
    for order in range(3):
        for axis in range(3):
            shift = np.zeros_like(vectors)
            shift[:, order] = increment * share[:, None] * lagged[order][:, axis][None, :]
            expected[order, axis] = shift.ravel()
    result = tide_vector_shifts(float(increment), share, lagged, vectors.size)
    np.testing.assert_array_equal(_bits(result), _bits(expected))


def test_tide_lag_shifts_match_numpy():
    rng = np.random.default_rng(5)
    payload = rng.standard_normal((4, 3, 3)) * 1e8
    size, start = 60, 7
    angle = 1.0e-4
    for order in range(3):
        expected = []
        for value in (angle, -angle):
            c, s = np.cos(value), np.sin(value)
            vectors = payload[:, order]
            shift = np.zeros(payload.shape)
            shift[:, order, 0] = (c - 1.0) * vectors[:, 0] - s * vectors[:, 1]
            shift[:, order, 1] = s * vectors[:, 0] + (c - 1.0) * vectors[:, 1]
            full = np.zeros(size)
            full[start : start + payload.size] = shift.ravel()
            expected.append(full)
        trig = [(float(np.cos(value)), float(np.sin(value))) for value in (angle, -angle)]
        plus, minus = tide_lag_shifts(payload, order, *trig[0], *trig[1], size, start)
        np.testing.assert_array_equal(_bits(plus), _bits(expected[0]))
        np.testing.assert_array_equal(_bits(minus), _bits(expected[1]))
