"""Compiled delayed-history kernels against the Python formulas they replace."""

import numpy as np
import pytest

from lunarops._dynamics_core import hermite_sample, lunar_deformation
from lunarops.classes.dynamics.deformation import (
    LunarDeformationParameters,
    _degree2_coefficients_from_inertia,
    deformation_tensors,
)
from lunarops.classes.dynamics.history import _hermite_weights, interpolate_state_derivative
from lunarops.classes.dynamics.rotation import euler_matrix

EARTH_MU, MOON_MU, RADIUS = 3.986004418e14, 4.9028e12, 1738000.0


def _nodes(rng, dimension, count, step):
    """Smooth node data with lunar-orbit magnitudes: a random cubic in time plus a large constant."""
    times = rng.uniform(-1e8, 1e8) + step * np.arange(count)
    coefficients = rng.normal(size=(dimension, 4)) * np.array([3.8e8, 1e3, 1e-3, 1e-9])
    tau = times - times[0]
    states = coefficients @ np.vstack((np.ones_like(tau), tau, tau**2, tau**3))
    derivatives = coefficients[:, 1:] @ np.vstack((np.ones_like(tau), 2 * tau, 3 * tau**2))
    return times, np.ascontiguousarray(states), np.ascontiguousarray(derivatives)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("where", ["inside", "beyond", "node"])
def test_hermite_sample_matches_the_numpy_interpolant(count, where):
    rng = np.random.default_rng(count * 7 + len(where))
    for _ in range(50):
        step = rng.choice([337.5, 2700.0, 5400.0])
        times, states, derivatives = _nodes(rng, 15, count, step)
        offset = {
            "inside": times[0] + rng.uniform(0, max(count - 1, 1)) * step,
            "beyond": times[-1] + rng.uniform(0.1, 6.0) * step,
            "node": times[rng.integers(count)],
        }[where]
        expected_state, expected_derivative = interpolate_state_derivative(times, states, derivatives, offset)
        state, derivative = np.empty(15), np.empty(15)
        exact = hermite_sample(states, derivatives, times[0], step, offset, state, derivative)

        if where == "node" or count == 1 and abs(offset - times[0]) < 1e-13 * step:
            assert exact >= 0
            np.testing.assert_array_equal(state, expected_state)
            np.testing.assert_array_equal(derivative, expected_derivative)
            continue
        # The sums are reordered, so the results differ by a few ulp of each term; extrapolated weights
        # grow to thousands, so the bound scales with the summed weight magnitudes (Lebesgue constant).
        _, (h, g, h_dot, g_dot) = _hermite_weights(count, float((offset - times[0]) / (step if count > 1 else 1.0)))
        values, rates = np.abs(states).max(axis=1), np.abs(derivatives).max(axis=1)
        state_bound = 8 * np.spacing(values * (1 + np.abs(h).sum()) + step * rates * np.abs(g).sum())
        rate_bound = 8 * np.spacing(values * np.abs(h_dot).sum() / step + rates * np.abs(g_dot).sum())
        assert np.all(np.abs(state - expected_state) <= state_bound)
        assert np.all(np.abs(derivative - expected_derivative) <= rate_bound)


def test_hermite_sample_accepts_strided_node_windows():
    rng = np.random.default_rng(3)
    times, states, derivatives = _nodes(rng, 15, 12, 2700.0)
    window = slice(3, 10)  # node buffers are (d, capacity) arrays, so windows are non-contiguous views
    offset = times[5] + 1234.5
    expected = interpolate_state_derivative(times[window], states[:, window], derivatives[:, window], offset)
    state, derivative = np.empty(15), np.empty(15)
    hermite_sample(states[:, window], derivatives[:, window], times[3], 2700.0, offset, state, derivative)
    np.testing.assert_allclose(state, expected[0], rtol=1e-15, atol=1e-6)
    np.testing.assert_allclose(derivative, expected[1], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(("tidal", "rotational"), [(True, True), (True, False), (False, True), (False, False)])
def test_lunar_deformation_matches_the_python_tensors(tidal, rotational):
    rng = np.random.default_rng(11 + 2 * tidal + rotational)
    parameters = LunarDeformationParameters(
        love_k2=0.02419,
        delay_s=12544.75,
        mean_motion_rad_s=2.6616995e-06,
        include_tidal=tidal,
        include_rotational=rotational,
        c21_static=2.28e-10,
        s21_static=1.33e-9,
        s22_static=3e-11,
    )
    undistorted = np.diag([0.39276, 0.39285, 0.39301]) + 1e-9
    for _ in range(100):
        position = rng.normal(size=3) * 2.2e8 + np.array([3.6e8, 0, 0])
        state = np.concatenate(
            (
                position,
                rng.normal(size=3) * 1e3,
                rng.uniform(-np.pi, np.pi, 3),
                np.array([1e-9, -2e-10, 2.66e-6]) + rng.normal(size=3) * 1e-10,
            )
        )
        state_dot = np.concatenate((rng.normal(size=9), rng.normal(size=3) * 1e-15))
        R = euler_matrix(state[6:9])
        omega = state[9:12]
        r = -R.T @ state[:3]
        r_dot = -R.T @ state[3:6] - np.cross(omega, r)
        expected_tidal, expected_rotational, expected_derivative = deformation_tensors(
            r, r_dot, omega, state_dot[9:12], parameters=parameters, radius_m=RADIUS, earth_mu=EARTH_MU, moon_mu=MOON_MU
        )
        expected_total = undistorted + expected_tidal + expected_rotational
        expected_degree2 = _degree2_coefficients_from_inertia(
            expected_total, c21_static=2.28e-10, s21_static=1.33e-9, s22_static=3e-11
        )
        out = {name: np.empty((3, 3)) for name in ("total", "derivative", "tidal", "rotational")}
        degree2 = np.empty((2, 3))
        lunar_deformation(
            state, state_dot, np.ascontiguousarray(undistorted), 0.02419, 2.6616995e-06, tidal, rotational,
            RADIUS, EARTH_MU, MOON_MU, 2.28e-10, 1.33e-9, 3e-11,
            out["total"], out["derivative"], out["tidal"], out["rotational"], degree2,
        )  # fmt: skip

        for name, expected in (
            ("tidal", expected_tidal),
            ("rotational", expected_rotational),
            ("derivative", expected_derivative),
            ("total", expected_total),
        ):
            np.testing.assert_allclose(
                out[name], expected, rtol=0, atol=16 * np.spacing(np.abs(expected).max() + 1e-300), err_msg=name
            )
        np.testing.assert_allclose(degree2, expected_degree2, rtol=0, atol=8 * np.spacing(np.abs(expected_total).max()))
