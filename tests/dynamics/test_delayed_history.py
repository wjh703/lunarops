"""Delayed-history queries: geocentric positions, extrapolation beyond the latest node, deformation."""

from types import SimpleNamespace

import numpy as np
import pytest

from lunarops.classes.dynamics.bodies import EMB, GravitatingBody
from lunarops.classes.dynamics.deformation import (
    LunarDeformation,
    LunarDeformationParameters,
    _degree2_coefficients_from_inertia,
    deformation_tensors,
)
from lunarops.classes.dynamics.history import (
    _TRANSLATION_EXTRAPOLATION_NODES,
    DelayedHistory,
    EphemerisHistory,
    LunarNodeBuffer,
    TwoSidedHistory,
)
from lunarops.classes.dynamics.rotation import euler_matrix
from lunarops.classes.dynamics.trajectory import interpolate_regular
from tests.dynamics.test_delayed_kernels import deformation_rounding_bounds
from tests.support.dynamics import EARTH_MU, EPOCH, MOON_MU, FunctionEphemeris

BODIES = {
    EMB: (np.array([-2.6e10, 1.33e11, 5.77e10]), np.array([-2.98e4, -4.9e3, -2.1e3])),
    "SUN": (np.array([-1.1e9, -4.0e8, -1.4e8]), np.array([5.0, -12.0, -5.0])),
    "MOON": (np.array([-2.6e10 + 3.2e8, 1.33e11 - 1.1e8, 5.77e10 - 4.0e7]), np.array([-2.98e4, -4.0e3, -2.5e3])),
}
OFFSET = np.array([1e-3, -2e-3, 3e-3, 0.0, 0.0, 0.0])


class _RelativeEphemeris(FunctionEphemeris):
    """Linear motion; the Moon-Earth state is offset from the barycentric difference by 1-3 mm."""

    def __init__(self):
        super().__init__(lambda name, t: (BODIES[_moon_or(name)][0] + BODIES[_moon_or(name)][1] * t,
                                          BODIES[_moon_or(name)][1]))  # fmt: skip

    def relative_state_matrix(self, target, center, epoch_tdb):
        return super().relative_state_matrix(target, center, epoch_tdb) + OFFSET


def _moon_or(name):
    return EMB if name == "EARTH" else name


def test_geocentric_positions_take_the_moon_from_the_relative_state_and_the_rest_from_one_batch():
    ephemeris = _RelativeEphemeris()
    history = EphemerisHistory(ephemeris, EARTH_MU, MOON_MU)
    epochs = [EPOCH.shifted(3600.0 * k) for k in range(3)]

    positions = history.geocentric_positions(("MOON", "SUN", "EARTH"), epochs)

    assert positions.shape == (3, 3, 3)
    for index, epoch in enumerate(epochs):
        relative = ephemeris.relative_state_matrix("MOON", "EARTH", epoch)[:3]
        np.testing.assert_array_equal(positions[index, 0], relative)  # not a barycentric difference
        earth = ephemeris.body_state_matrix_bcrs((EMB,), epoch)[0, :3] - MOON_MU / (EARTH_MU + MOON_MU) * relative
        sun = ephemeris.body_state_matrix_bcrs(("SUN",), epoch)[0, :3]
        np.testing.assert_allclose(positions[index, 1], sun - earth, rtol=0, atol=1e-4)
        np.testing.assert_array_equal(positions[index, 2], np.zeros(3))
    # The Sun row agrees with the existing state query up to rounding.
    states = history.states(("EARTH", "SUN"), epochs[1])
    np.testing.assert_allclose(positions[1, 1], states[1, :3] - states[0, :3], rtol=0, atol=1e-4)


def test_two_sided_history_routes_each_epoch_to_its_arc():
    def arc(label):
        return SimpleNamespace(geocentric_positions=lambda names, epochs: np.full((len(epochs), len(names), 3), label))

    history = TwoSidedHistory(EPOCH, arc(1.0), arc(-1.0))
    positions = history.geocentric_positions(("MOON",), [EPOCH.shifted(-60.0), EPOCH, EPOCH.shifted(60.0)])

    np.testing.assert_array_equal(positions[:, 0, 0], [-1.0, 1.0, 1.0])


def _history(*, joint, step, count=14):
    """A DelayedHistory over ``count`` nodes of a degree-7 polynomial, so 7 and 14 nodes extrapolate differently."""
    rng = np.random.default_rng(5)
    coefficients = rng.normal(size=(6, 8)) * np.logspace(8, -20, 8)[None, :]
    times = step * np.arange(count, dtype=float)
    states = coefficients @ np.vstack([times**power for power in range(8)])
    nodes = LunarNodeBuffer(count + 1, 6)
    for index, offset in enumerate(times):
        nodes.append(offset, states[:, index], np.zeros(6))
    rotation = SimpleNamespace(deformation=object()) if joint else None
    system = SimpleNamespace(
        rotation=rotation,
        ephemeris=FunctionEphemeris(),
        earth_body=GravitatingBody("EARTH", EARTH_MU),
        moon_body=GravitatingBody("MOON", MOON_MU),
    )
    return DelayedHistory(system, EPOCH, nodes, 13), times, states


@pytest.mark.parametrize("joint", [True, False])
@pytest.mark.parametrize("step", [2700.0, -2700.0])
def test_queries_beyond_the_latest_node_extrapolate_through_the_latest_seven_nodes(joint, step):
    history, times, states = _history(joint=joint, step=step)
    ahead = times[-1] + 20828.75 * np.sign(step)

    actual = history.relative_state(EPOCH.shifted(ahead))

    nodes = _TRANSLATION_EXTRAPOLATION_NODES
    expected = interpolate_regular(times[-nodes:], states[:, -nodes:], float(EPOCH.seconds_until(EPOCH.shifted(ahead))),
                                   nodes - 1, extrapolate=True, check_grid=False)  # fmt: skip
    full = interpolate_regular(times, states, ahead, 13, extrapolate=True, check_grid=False)
    np.testing.assert_array_equal(actual, expected)
    assert not np.allclose(actual, full, rtol=1e-12)  # the full stencil would give another value
    if joint:
        assert history.advanced_translation_queries == 1
        assert history.max_translation_extrapolation_s == pytest.approx(20828.75)


def test_orbit_only_queries_inside_the_history_keep_the_full_stencil():
    history, times, states = _history(joint=False, step=2700.0)
    inside = times[-3] + 1234.5

    actual = history.relative_state(EPOCH.shifted(inside))

    expected = interpolate_regular(times, states, float(EPOCH.seconds_until(EPOCH.shifted(inside))), 13,
                                   extrapolate=True, check_grid=False)  # fmt: skip
    np.testing.assert_array_equal(actual, expected)


def test_lunar_deformation_evaluate_matches_the_python_composition():
    rng = np.random.default_rng(9)
    parameters = LunarDeformationParameters(
        love_k2=0.02419, delay_s=12544.75, mean_motion_rad_s=2.6616995e-06,
        c21_static=2.28e-10, s21_static=1.33e-9, s22_static=0.0,
    )  # fmt: skip
    inertia = np.diag([0.3927622575, 0.3928517809, 0.3930102336])
    state = np.concatenate(
        ([3.4e8, -1.2e8, 0.6e8], rng.normal(size=3) * 1e3, [0.005, 0.38, 1.29], [5e-10, -2.6e-11, 2.6614e-6])
    )
    state_dot = np.concatenate((rng.normal(size=9), [1e-16, -3e-16, 2e-17]))
    queried = []

    def integration_sample(epoch):
        queried.append(epoch)
        return state, state_dot

    model = LunarDeformation(parameters, inertia, 1738000.0, EARTH_MU, MOON_MU, field=None)
    result = model.evaluate(EPOCH, SimpleNamespace(integration_sample=integration_sample))

    assert queried == [EPOCH.shifted(-parameters.delay_s)]
    R, omega = euler_matrix(state[6:9]), state[9:12]
    r = -R.T @ state[:3]
    tidal, rotational, derivative = deformation_tensors(
        r, -R.T @ state[3:6] - np.cross(omega, r), omega, state_dot[9:12],
        parameters=parameters, radius_m=1738000.0, earth_mu=EARTH_MU, moon_mu=MOON_MU,
    )  # fmt: skip
    total = inertia + tidal + rotational
    degree2 = _degree2_coefficients_from_inertia(total, c21_static=2.28e-10, s21_static=1.33e-9, s22_static=0.0)
    bounds = deformation_rounding_bounds(
        r, -R.T @ state[3:6] - np.cross(omega, r), omega, state_dot[9:12], love_k2=0.02419,
        mean_motion=2.6616995e-06, radius=1738000.0, earth_mu=EARTH_MU, moon_mu=MOON_MU,
    )  # fmt: skip
    # The total and C20 cancel inertia terms of ~0.39, so their rounding scales with the inertia.
    inertia_bound = 16 * np.spacing(np.abs(total).max()) + bounds["tidal"] + bounds["rotational"]
    for actual, expected, bound in (
        (result.total, total, inertia_bound),
        (result.tidal, tidal, bounds["tidal"]),
        (result.rotational, rotational, bounds["rotational"]),
        (result.derivative, derivative, bounds["derivative"]),
        (result.degree2, degree2, inertia_bound),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=bound)
    rigid = model.evaluate(EPOCH, None, rigid=True)
    np.testing.assert_array_equal(rigid.total, inertia)
    np.testing.assert_array_equal(rigid.derivative, np.zeros((3, 3)))
