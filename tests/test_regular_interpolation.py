"""The one regular-grid Lagrange interpolation against polynomials and the three implementations it replaced."""

from math import comb

import numpy as np
import pytest

from lunarops.base.interpolation import interpolate_regular_grid, regular_stencil_start
from lunarops.classes.dynamics.trajectory import interpolate_regular
from lunarops.classes.parametrization.lunar_dynamics import interpolate_regular_rows


def _old_table(nodes, values, x, count):
    """``integrated_lunar._RegularTable`` before the merge: floor stencil, normalised barycentric weights."""
    coordinate = (x - nodes[0]) / (nodes[1] - nodes[0])
    start = int(np.clip(np.floor(coordinate) - (count - 1) // 2, 0, len(nodes) - count))
    signs = np.array([(-1.0) ** j * comb(count - 1, j) for j in range(count)]) / (coordinate - start - np.arange(count))
    stencil = values[start : start + count]
    return stencil[0] + (signs / signs.sum()) @ (stencil - stencil[0])


def _old_trajectory(nodes, values, x, count):
    """``trajectory.interpolate_regular`` before the merge: searchsorted stencil, barycentric sum then divide."""
    start = max(0, min(int(np.searchsorted(nodes, x)) - count // 2, len(nodes) - count))
    coordinate = (x - nodes[start]) / (nodes[start + 1] - nodes[start])
    factors = np.array([(-1.0) ** j * comb(count - 1, j) for j in range(count)]) / (coordinate - np.arange(count))
    stencil = values[start : start + count]
    return stencil[0] + (factors @ (stencil - stencil[0])) / factors.sum()


def _old_rows(nodes, values, x, count):
    """``lunar_dynamics.interpolate_regular_rows`` before the merge: searchsorted stencil, product weights."""
    step = (nodes[-1] - nodes[0]) / (len(nodes) - 1)
    start = max(0, min(int(np.searchsorted(nodes, x)) - count // 2, len(nodes) - count))
    coordinate = (x - nodes[start]) / step
    weights = np.ones(count)
    for k in range(count):
        others = np.arange(count) != k
        weights[others] *= (coordinate - k) / (np.arange(count)[others] - k)
    stencil = values[start : start + count]
    return stencil[0] + weights[1:] @ (stencil[1:] - stencil[0])


@pytest.mark.parametrize("count", [1, 2, 7, 10, 11, 14])
@pytest.mark.parametrize("step", [3600.0, -5400.0])
def test_polynomials_below_the_stencil_degree_are_reproduced(count, step):
    rng = np.random.default_rng(count)
    nodes = 1.0e5 + step * np.arange(40)
    scale = abs(step) * 40
    coefficients = rng.normal(size=(count, 3))

    def poly(t):
        return np.polynomial.polynomial.polyval((t - nodes[0]) / scale, coefficients).T

    # Interior, near and at both ends, exactly on and a hair off nodes, and a little beyond the ends.
    points = np.r_[
        nodes[0] + step * rng.uniform(-0.5, 39.5, 200),
        nodes[[0, 1, 17, 38, 39]],
        nodes[17] + step * 1e-15,
        nodes[0] - 0.3 * step,
        nodes[-1] + 0.3 * step,
    ]
    result = interpolate_regular_grid(nodes, poly(nodes), points, count, check_grid=True)

    assert result.shape == (len(points), 3)
    np.testing.assert_allclose(result, poly(points), rtol=0.0, atol=1e-12 * np.abs(coefficients).sum())
    # Node hits return the stored sample bit for bit.
    np.testing.assert_array_equal(result[200:205], poly(nodes)[[0, 1, 17, 38, 39]])


def test_stencil_is_centred_for_even_and_odd_counts():
    # Even count: count / 2 nodes on either side; odd count: centred on the nearest node; clipped at the ends.
    assert regular_stencil_start([10.2, 10.8], 4, 100).tolist() == [9, 9]
    assert regular_stencil_start([10.2, 10.8], 5, 100).tolist() == [8, 9]
    assert regular_stencil_start([0.3, 98.7, -2.0, 120.0], 6, 100).tolist() == [0, 94, 0, 94]


@pytest.mark.parametrize("count", [10, 14])
def test_smooth_function_agrees_with_the_replaced_implementations(count):
    """For even counts all three old stencils are the centred one; the formulas then agree to round-off."""
    nodes = 3600.0 * np.arange(200)
    period = 27.3 * 86400.0
    values = np.column_stack(
        [3.8e8 * np.cos(2 * np.pi * nodes / period), 1.5 + 2.66e-6 * nodes + 0.1 * np.sin(2 * np.pi * nodes / period)]
    )
    rng = np.random.default_rng(7)
    points = nodes[count] + rng.uniform(0.0, 1.0, 300) * (nodes[-count - 1] - nodes[count])
    new = interpolate_regular_grid(nodes, values, points, count)

    # Relative to the amplitude of each column: near a zero of the cosine the element-wise relative error of
    # any formula is unbounded.
    scale = np.max(np.abs(values), axis=0)
    for old in (_old_table, _old_trajectory, _old_rows):
        reference = np.array([old(nodes, values, x, count) for x in points])
        assert np.max(np.abs(new - reference) / scale) < 2e-15


def test_wrappers_keep_their_layouts_and_range_policies():
    nodes = -3600.0 * np.arange(30)  # a backward arc
    values = np.column_stack([np.sin(nodes / 4e4), np.cos(nodes / 4e4)])
    x = -3600.0 * 11.4

    expected = interpolate_regular_grid(nodes, values, [x], 14)[0]
    np.testing.assert_array_equal(interpolate_regular(nodes, values.T, x, 13, extrapolate=False), expected)
    np.testing.assert_array_equal(interpolate_regular(nodes, values.T, 1e-10, 13, extrapolate=False), values[0])
    with pytest.raises(ValueError, match="outside"):
        interpolate_regular(nodes, values.T, 10.0, 13, extrapolate=False)
    ascending = nodes[::-1]
    np.testing.assert_allclose(
        interpolate_regular_rows(ascending, values[::-1], [x], 13)[0], expected, rtol=1e-15, atol=1e-16
    )
