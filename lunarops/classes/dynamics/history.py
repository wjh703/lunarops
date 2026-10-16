"""Integration-node storage and delayed force history."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from lunarops.classes.time import Epoch

from .trajectory import interpolate_regular


@lru_cache(maxsize=16)
def _hermite_grid(count):
    grid = np.arange(count, dtype=float)
    differences = grid[:, None] - grid[None, :]
    np.fill_diagonal(differences, 1)
    denominators = np.prod(differences, axis=1)
    inverse = 1 / differences
    np.fill_diagonal(inverse, 0)
    return grid, denominators, np.sum(inverse, axis=1)


def interpolate_state_derivative(times, states, derivatives, offset):
    """Hermite cardinal basis on a regular stencil, derivative of same polynomial."""
    count = len(times)
    step = 1.0 if count == 1 else times[1] - times[0]
    grid, denominators, slopes = _hermite_grid(count)
    distances = (offset - times[0]) / step - grid
    exact = np.flatnonzero(abs(distances) < 1e-13)
    if len(exact):
        return states[:, exact[0]].copy(), derivatives[:, exact[0]].copy()
    factors = np.broadcast_to(distances, (count, count)).copy()
    np.fill_diagonal(factors, 1)
    lagrange = np.prod(factors, axis=1) / denominators
    inverse = 1 / factors
    np.fill_diagonal(inverse, 0)
    lagrange_dot = lagrange * np.sum(inverse, axis=1)
    squared = lagrange**2
    square_dot = 2 * lagrange * lagrange_dot
    h = (1 - 2 * distances * slopes) * squared
    g = distances * squared
    h_dot = -2 * slopes * squared + (1 - 2 * distances * slopes) * square_dot
    g_dot = squared + distances * square_dot
    # Subtract a constant before differentiating large position values.
    reference = states[:, count // 2]
    centered = states - reference[:, None]
    return (
        reference + centered @ h + step * (derivatives @ g),
        centered @ h_dot / step + derivatives @ g_dot,
    )


class LunarNodeBuffer:
    """Append-only node and derivative storage used by ABM."""

    def __init__(self, node_capacity: int, state_dimension: int, external_state_count: int) -> None:
        self._times = np.empty(node_capacity, dtype=float)
        self._states = np.empty((state_dimension, node_capacity), dtype=float)
        self._derivatives = np.empty_like(self._states)
        self._external_states = np.empty((external_state_count, 6, node_capacity), dtype=float)
        self._size = 0
        self._trajectory_start = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def trajectory_start(self) -> int:
        return self._trajectory_start

    def append(self, offset_s: float, state, derivative, external_state) -> None:
        if self._size == len(self._times):
            raise IndexError("Lunar node buffer is full")
        self._external_states[:, :, self._size] = external_state
        self._times[self._size] = float(offset_s)
        self._states[:, self._size] = state
        self._derivatives[:, self._size] = derivative
        self._size += 1

    def state(self, index: int = -1) -> np.ndarray:
        if not self._size:
            raise IndexError("Lunar node buffer is empty")
        return self._states[:, index if index >= 0 else self._size + index]

    def derivative_window(self, count: int) -> np.ndarray:
        if count < 1 or self._size < count:
            raise ValueError("Lunar node buffer does not contain enough derivative nodes")
        return self._derivatives[:, self._size - count : self._size]

    def mark_trajectory_start(self) -> None:
        if not self._size:
            raise IndexError("Lunar node buffer is empty")
        self._trajectory_start = self._size - 1

    def trajectory_data(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._times[self._trajectory_start : self._size],
            self._states[:, self._trajectory_start : self._size],
        )

    def integration_data(self):
        return self._times[:self._size], self._states[:, :self._size], self._derivatives[:, :self._size]

    def replace_last_derivative(self, derivative):
        self._derivatives[:, self._size - 1] = derivative

    def external_data(self):
        return self._times[:self._size], self._external_states[:, :, :self._size]

    def recent_state_data(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        if count < 1 or self._size < count:
            raise ValueError("Lunar node buffer does not contain enough state nodes")
        start = self._size - count
        return self._times[start : self._size], self._states[:, start : self._size]

    def recent_external_state_data(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        if count < 1 or self._size < count:
            raise ValueError("Lunar node buffer does not contain enough external state nodes")
        start = self._size - count
        return self._times[start : self._size], self._external_states[:, :, start : self._size]


class DelayedHistory:
    """Shared accepted-node history for delayed translation and rotation."""

    def __init__(
        self,
        system,
        initial_epoch: Epoch,
        nodes: LunarNodeBuffer,
        history_interpolation_order: int,
    ) -> None:
        self._system = system
        self._initial_epoch = initial_epoch
        self._nodes = nodes
        self._history_interpolation_order = history_interpolation_order
        self._capacity = history_interpolation_order + 1
        self._joint_delay = system.rotation is not None and system.rotation.deformation is not None
        self.advanced_translation_queries = 0
        self.max_translation_extrapolation_s = 0.0
        self._provider = system.build_history_provider(self._state, self._external_state)

    def _state(self, epoch: Epoch) -> np.ndarray:
        if self._joint_delay:
            offset = float(self._initial_epoch.seconds_until(epoch))
            times, states, _ = self._nodes.integration_data()
            if offset > times[-1] + 1e-9:
                # Earth orbital tides intentionally contain negative delays.
                # Extrapolate translation explicitly; rotational deformation
                # remains strictly causal through integration_sample().
                self.advanced_translation_queries += 1
                self.max_translation_extrapolation_s = max(self.max_translation_extrapolation_s, offset - times[-1])
                return interpolate_regular(times, states, offset, self._history_interpolation_order, extrapolate=True)
            return self.integration_sample(epoch)[0]
        offset = float(self._initial_epoch.seconds_until(epoch))
        nodes, states = self._nodes.recent_state_data(min(self._capacity, self._nodes.size))
        return interpolate_regular(
            nodes,
            states,
            offset,
            self._history_interpolation_order,
            extrapolate=True,
        )

    def _external_state(self, body_names, epoch: Epoch) -> np.ndarray:
        offset = float(self._initial_epoch.seconds_until(epoch))
        nodes, states = (
            self._nodes.external_data() if self._joint_delay else
            self._nodes.recent_external_state_data(min(self._capacity, self._nodes.size))
        )
        if self._joint_delay:
            # Select the local stencil BEFORE fancy indexing. Copying all
            # external history for every tide query would cost O(n²) per arc.
            count = min(self._capacity, len(nodes))
            right = int(np.searchsorted(nodes, offset))
            start = max(0, min(right - count // 2, len(nodes) - count))
            nodes, states = nodes[start:start + count], states[:, :, start:start + count]
        indices = [self._system.history_body_names.index(name) for name in body_names]
        selected = states[indices]
        values = selected.reshape(selected.shape[0] * selected.shape[1], selected.shape[2])
        interpolated = interpolate_regular(
            nodes,
            values,
            offset,
            self._history_interpolation_order,
            extrapolate=True,
        )
        return interpolated.reshape(selected.shape[0], selected.shape[1])

    def integration_sample(self, epoch: Epoch):
        """State and derivative of ONE Hermite polynomial over accepted nodes.

        Use the stencil around the requested epoch, including old accepted
        nodes. Never substitute reference rotation. Keep the explicit seed
        arc and the forward delayed solution on separate stencils.
        """
        offset = float(self._initial_epoch.seconds_until(epoch))
        times, states, derivatives = self._nodes.integration_data()
        if offset < times[0] - 1e-9 or offset > times[-1] + 1e-9:
            raise ValueError("Delayed integration sample is outside the accepted/seed history")
        split = self._nodes.trajectory_start
        if offset < 0:
            times, states, derivatives = times[:split], states[:, :split], derivatives[:, :split]
        else:
            times, states, derivatives = times[split:], states[:, split:], derivatives[:, split:]
        count = min((self._history_interpolation_order + 1) // 2, len(times))
        right = int(np.searchsorted(times, offset))
        start = max(0, min(right - count // 2, len(times) - count))
        selected = slice(start, start + count)
        grid = times[selected]
        return interpolate_state_derivative(grid, states[:, selected], derivatives[:, selected], offset)

    def __call__(self, body_names: Sequence[str], epoch: Epoch) -> np.ndarray:
        return self._provider(body_names, epoch)
