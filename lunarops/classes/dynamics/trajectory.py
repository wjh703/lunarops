"""Regular-grid trajectory storage and interpolation."""

from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np

from lunarops.classes.time import Epoch

from .rotation import LunarOrientationState, quaternion_matrix


def regular_times(duration_s: float, step_s: float) -> np.ndarray:
    ratio = abs(float(duration_s)) / step_s
    count = round(ratio)
    if not np.isclose(ratio, count, rtol=0.0, atol=1e-12):
        raise ValueError("duration_s must be an integer multiple of the fixed integration step")
    direction = np.copysign(1.0, duration_s)
    return direction * step_s * np.arange(count + 1, dtype=float)


@lru_cache(maxsize=32)
def _regular_barycentric_weights(count: int) -> np.ndarray:
    weights = np.asarray([(-1.0) ** index * comb(count - 1, index) for index in range(count)], dtype=float)
    weights.setflags(write=False)
    return weights


def interpolate_regular(nodes, values, x: float, order: int, *, extrapolate: bool) -> np.ndarray:
    node_array = np.asarray(nodes, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if node_array.ndim != 1 or value_array.ndim != 2 or value_array.shape[1] != len(node_array):
        raise ValueError("Invalid regular interpolation arrays")
    if not len(node_array):
        raise ValueError("Regular interpolation requires at least one node")
    if node_array[0] > node_array[-1]:
        node_array = node_array[::-1]
        value_array = value_array[:, ::-1]
    if not extrapolate and x < node_array[0] - 1e-9:
        raise ValueError("Epoch is outside propagated trajectory coverage")
    if not extrapolate and x > node_array[-1] + 1e-9:
        raise ValueError("Epoch is outside propagated trajectory coverage")
    if not extrapolate and x <= node_array[0]:
        return value_array[:, 0].copy()
    if not extrapolate and x >= node_array[-1]:
        return value_array[:, -1].copy()
    right = int(np.searchsorted(node_array, x, side="left"))
    count = min(order + 1, len(node_array))
    start = max(0, min(right - count // 2, len(node_array) - count))
    stencil = node_array[start : start + count]
    exact = np.flatnonzero(stencil == x)
    if len(exact):
        return value_array[:, start + int(exact[0])].copy()
    if count == 1:
        return value_array[:, start].copy()
    step = stencil[1] - stencil[0]
    if step == 0 or not np.allclose(np.diff(stencil), step, rtol=0.0, atol=abs(step) * 1e-12):
        raise ValueError("Interpolation nodes must form a regular grid")
    coordinate = (x - stencil[0]) / step
    factors = _regular_barycentric_weights(count) / (coordinate - np.arange(count))
    return (value_array[:, start : start + count] @ factors) / np.sum(factors)


class LunarTrajectory:
    """Output trajectory independent from the integrator's history buffer."""

    def __init__(self, initial_epoch: Epoch, times_s, states, trajectory_interpolation_order: int) -> None:
        self.initial_epoch = initial_epoch
        self.history = None
        self.times_s = np.array(times_s, dtype=float, copy=True)
        self.states = np.array(states, dtype=float, copy=True)
        if self.times_s.ndim != 1 or self.states.ndim != 2 or self.states.shape[1] != len(self.times_s):
            raise ValueError("Invalid trajectory arrays")
        if self.states.shape[0] in (13, 16):
            # q and -q represent one orientation. Align the output copy only;
            # the accepted ABM state/derivative history remains untouched.
            for index in range(1, self.states.shape[1]):
                if self.states[6:10, index - 1] @ self.states[6:10, index] < 0:
                    self.states[6:10, index] *= -1
        self.times_s.setflags(write=False)
        self.states.setflags(write=False)
        self.trajectory_interpolation_order = int(trajectory_interpolation_order)

    def orientation_state(self, epoch: Epoch) -> LunarOrientationState:
        if self.states.shape[0] not in (13, 16):
            raise ValueError("Trajectory does not contain an integrated lunar rotation")
        state = self.vector(epoch)
        return LunarOrientationState(quaternion_matrix(state[6:10]), state[10:13])

    def matrix(self, epoch: Epoch) -> np.ndarray:
        return self.orientation_state(epoch).pa_to_lcrs

    def matrices(self, epochs) -> np.ndarray:
        return np.asarray([self.matrix(epoch) for epoch in epochs]).reshape(-1, 3, 3)

    def orientation_states(self, epochs) -> tuple[LunarOrientationState, ...]:
        return tuple(self.orientation_state(epoch) for epoch in epochs)

    def vector(self, epoch: Epoch, *, extrapolate: bool = False) -> np.ndarray:
        offset = float(self.initial_epoch.seconds_until(epoch))
        return interpolate_regular(
            self.times_s,
            self.states,
            offset,
            self.trajectory_interpolation_order,
            extrapolate=extrapolate,
        )
