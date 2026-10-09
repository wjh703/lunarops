"""Integration-node storage and delayed force history."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunarops.classes.time import Epoch

from .context import StateHistoryProvider
from .trajectory import interpolate_regular


class LunarNodeBuffer:
    """Append-only node and derivative storage used by ABM."""

    def __init__(self, node_capacity: int, state_dimension: int, external_state_count: int = 0) -> None:
        self._times = np.empty(node_capacity, dtype=float)
        self._states = np.empty((state_dimension, node_capacity), dtype=float)
        self._derivatives = np.empty_like(self._states)
        if external_state_count < 0:
            raise ValueError("external_state_count must be nonnegative")
        self._external_states = (
            None if external_state_count == 0 else np.empty((external_state_count, 6, node_capacity), dtype=float)
        )
        self._size = 0
        self._trajectory_start = 0

    @property
    def size(self) -> int:
        return self._size

    def append(self, offset_s: float, state, derivative, external_state=None) -> None:
        if self._size == len(self._times):
            raise IndexError("Lunar node buffer is full")
        if self._external_states is None:
            if external_state is not None:
                raise ValueError("This node buffer does not store external states")
        else:
            if external_state is None:
                raise ValueError("An external state matrix is required for this node buffer")
            external_state = np.asarray(external_state, dtype=float)
            if external_state.shape != self._external_states.shape[:2] or not np.all(np.isfinite(external_state)):
                raise ValueError("Invalid external state matrix")
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

    def recent_state_data(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        if count < 1 or self._size < count:
            raise ValueError("Lunar node buffer does not contain enough state nodes")
        start = self._size - count
        return self._times[start : self._size], self._states[:, start : self._size]

    def recent_external_state_data(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        if self._external_states is None:
            raise ValueError("This node buffer does not store external states")
        if count < 1 or self._size < count:
            raise ValueError("Lunar node buffer does not contain enough external state nodes")
        start = self._size - count
        return self._times[start : self._size], self._external_states[:, :, start : self._size]


class DelayedHistory:
    """Interpolate recent lunar nodes and delegate body mixing to dynamics."""

    def __init__(
        self,
        system,
        initial_epoch: Epoch,
        nodes: LunarNodeBuffer,
        history_interpolation_order: int,
        external_provider: StateHistoryProvider | None,
    ) -> None:
        self._system = system
        self._initial_epoch = initial_epoch
        self._nodes = nodes
        self._history_interpolation_order = history_interpolation_order
        self._capacity = history_interpolation_order + 1
        self._provider = system.build_history_provider(self._state, external_provider, history_state_at=self._external_state)

    def _state(self, epoch: Epoch) -> np.ndarray:
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
        nodes, states = self._nodes.recent_external_state_data(min(self._capacity, self._nodes.size))
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

    def __call__(self, body_names: Sequence[str], epoch: Epoch) -> np.ndarray:
        return self._provider(body_names, epoch)
