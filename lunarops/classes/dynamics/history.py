"""Accepted integration nodes and the state history read by delayed forces.

:class:`LunarNodeBuffer` stores the accepted states and derivatives of a
propagation. :class:`EphemerisHistory` answers state queries from the
ephemeris alone, and :class:`DelayedHistory` replaces the Moon by the
integrated solution so that delayed tides and deformation see the
propagated orbit and attitude.
"""

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
    """Evaluate a Hermite interpolant and its derivative on a regular stencil.

    The interpolant matches both the states and the derivatives at every
    node; the returned derivative is the derivative of the same polynomial.

    Parameters
    ----------
    times : numpy.ndarray
        Equally spaced, increasing node offsets in s, shape ``(n,)``.
    states : numpy.ndarray
        States at the nodes, shape ``(d, n)``.
    derivatives : numpy.ndarray
        Time derivatives of the states at the nodes, shape ``(d, n)``.
    offset : float
        Evaluation offset in s.

    Returns
    -------
    state : numpy.ndarray
        Interpolated state, shape ``(d,)``.
    derivative : numpy.ndarray
        Its time derivative, shape ``(d,)``.
    """
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
    """Fixed-capacity store of accepted integration nodes.

    Nodes are appended in integration order. Nodes before
    :attr:`trajectory_start` belong to the warm-up or seed arc; the
    trajectory output starts at the initial epoch.

    Parameters
    ----------
    node_capacity : int
        Maximum number of nodes.
    state_dimension : int
        Length of the state vector.
    """

    def __init__(self, node_capacity: int, state_dimension: int) -> None:
        self._times = np.empty(node_capacity, dtype=float)
        self._states = np.empty((state_dimension, node_capacity), dtype=float)
        self._derivatives = np.empty_like(self._states)
        self._size = 0
        self._trajectory_start = 0

    @property
    def size(self) -> int:
        """int: Number of stored nodes."""
        return self._size

    @property
    def trajectory_start(self) -> int:
        """int: Index of the node at the initial epoch."""
        return self._trajectory_start

    def append(self, offset_s: float, state, derivative) -> None:
        """Append one accepted node.

        Parameters
        ----------
        offset_s : float
            Node offset from the initial epoch in TDB seconds.
        state : array_like
            Accepted state, shape ``(d,)``.
        derivative : array_like
            Derivative of the accepted state, shape ``(d,)``.

        Raises
        ------
        IndexError
            If the buffer is full.
        """
        if self._size == len(self._times):
            raise IndexError("Lunar node buffer is full")
        self._times[self._size] = float(offset_s)
        self._states[:, self._size] = state
        self._derivatives[:, self._size] = derivative
        self._size += 1

    def state(self, index: int = -1) -> np.ndarray:
        """Return one stored state.

        Parameters
        ----------
        index : int, default -1
            Node index; negative values count from the latest node.

        Returns
        -------
        numpy.ndarray
            View of the state, shape ``(d,)``.

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        if not self._size:
            raise IndexError("Lunar node buffer is empty")
        return self._states[:, index if index >= 0 else self._size + index]

    def derivative_window(self, count: int) -> np.ndarray:
        """Return the latest derivatives in chronological order.

        Parameters
        ----------
        count : int
            Number of nodes.

        Returns
        -------
        numpy.ndarray
            View of shape ``(d, count)``, oldest node first.

        Raises
        ------
        ValueError
            If ``count < 1`` or fewer than ``count`` nodes are stored.
        """
        if count < 1 or self._size < count:
            raise ValueError("Lunar node buffer does not contain enough derivative nodes")
        return self._derivatives[:, self._size - count : self._size]

    def mark_trajectory_start(self) -> None:
        """Mark the latest node as the initial-epoch node.

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        if not self._size:
            raise IndexError("Lunar node buffer is empty")
        self._trajectory_start = self._size - 1

    def trajectory_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the nodes from the initial epoch onwards.

        Returns
        -------
        times : numpy.ndarray
            Offsets in s, shape ``(n,)``.
        states : numpy.ndarray
            States, shape ``(d, n)``.
        """
        return (
            self._times[self._trajectory_start : self._size],
            self._states[:, self._trajectory_start : self._size],
        )

    def integration_data(self):
        """Return all stored nodes, including the warm-up or seed arc.

        Returns
        -------
        times : numpy.ndarray
            Offsets in s, shape ``(n,)``.
        states : numpy.ndarray
            States, shape ``(d, n)``.
        derivatives : numpy.ndarray
            Derivatives, shape ``(d, n)``.
        """
        return self._times[:self._size], self._states[:, :self._size], self._derivatives[:, :self._size]

    def replace_last_derivative(self, derivative):
        """Overwrite the derivative of the latest node.

        Parameters
        ----------
        derivative : array_like
            New derivative, shape ``(d,)``.
        """
        self._derivatives[:, self._size - 1] = derivative

    def recent_state_data(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the latest nodes.

        Parameters
        ----------
        count : int
            Number of nodes.

        Returns
        -------
        times : numpy.ndarray
            Offsets in s, shape ``(count,)``.
        states : numpy.ndarray
            States, shape ``(d, count)``.

        Raises
        ------
        ValueError
            If ``count < 1`` or fewer than ``count`` nodes are stored.
        """
        if count < 1 or self._size < count:
            raise ValueError("Lunar node buffer does not contain enough state nodes")
        start = self._size - count
        return self._times[start : self._size], self._states[:, start : self._size]


class EphemerisHistory:
    """State history taken entirely from the ephemeris.

    Parameters
    ----------
    ephemeris : Ephemeris
        Ephemeris providing BCRS body states.
    """

    def __init__(self, ephemeris):
        self._ephemeris = ephemeris

    def relative_state(self, epoch: Epoch) -> np.ndarray:
        """Return the ephemeris Moon-minus-Earth state.

        Parameters
        ----------
        epoch : Epoch
            TDB epoch.

        Returns
        -------
        numpy.ndarray
            Relative position in m and velocity in m s⁻¹ with ICRF axes,
            shape ``(6,)``.
        """
        states = self._ephemeris.body_state_matrix_bcrs(("EARTH", "MOON"), epoch)
        return states[1] - states[0]

    def states(self, body_names: Sequence[str], epoch: Epoch) -> np.ndarray:
        """Return BCRS states of several bodies at one epoch.

        The Moon state is the ephemeris Earth state plus
        :meth:`relative_state`, so subclasses that override the relative
        state also change the Moon.

        Parameters
        ----------
        body_names : sequence of str
            Canonical body identifiers.
        epoch : Epoch
            TDB epoch.

        Returns
        -------
        numpy.ndarray
            BCRS positions in m and velocities in m s⁻¹ with ICRF axes,
            shape ``(len(body_names), 6)``.
        """
        names = tuple(body_names)
        externals = tuple(name for name in names if name != "MOON")
        cached_names = tuple(dict.fromkeys(("EARTH", *externals) if "MOON" in names else externals))
        cached = self._ephemeris.body_state_matrix_bcrs(cached_names, epoch)
        rows = {name: cached[index] for index, name in enumerate(cached_names)}
        return np.asarray([rows["EARTH"] + self.relative_state(epoch)[:6] if name == "MOON" else rows[name]
                           for name in names])


class DelayedHistory(EphemerisHistory):
    """State history in which the Moon follows the accepted integration nodes.

    Every other body comes from the ephemeris. Only accepted nodes are used;
    trial states of the current step never enter the history.

    Parameters
    ----------
    system : LunarDynamics
        Dynamics being propagated.
    initial_epoch : Epoch
        TDB epoch of offset 0.
    nodes : LunarNodeBuffer
        Accepted nodes of the propagation.
    history_interpolation_order : int
        Interpolation order of history queries.

    Attributes
    ----------
    advanced_translation_queries : int
        Number of translation queries beyond the latest accepted node, which
        arise from the negative Earth-tide delays.
    max_translation_extrapolation_s : float
        Largest such extrapolation interval in s.
    """

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
        super().__init__(system.ephemeris)

    def relative_state(self, epoch: Epoch) -> np.ndarray:
        """Return the integrated Moon-minus-Earth state at one epoch.

        With integrated rotation and deformation the state is a Hermite
        interpolant of the accepted nodes, and epochs beyond the latest node
        are extrapolated with a Lagrange polynomial (translation only).
        Otherwise a Lagrange polynomial through the latest nodes is
        evaluated, extrapolating where needed.

        Parameters
        ----------
        epoch : Epoch
            TDB epoch.

        Returns
        -------
        numpy.ndarray
            Integrated state; the first six entries are the relative
            position in m and velocity in m s⁻¹ with ICRF axes.

        Raises
        ------
        ValueError
            If a Hermite query lies before the first stored node.
        """
        if self._joint_delay:
            offset = float(self._initial_epoch.seconds_until(epoch))
            times, states, _ = self._nodes.integration_data()
            if offset > times[-1] + 1e-9:
                # Earth orbital tides intentionally contain negative delays.
                # Extrapolate translation explicitly; rotational deformation
                # remains strictly causal through integration_sample().
                self.advanced_translation_queries += 1
                self.max_translation_extrapolation_s = max(self.max_translation_extrapolation_s, offset - times[-1])
                return interpolate_regular(times, states, offset, self._history_interpolation_order, extrapolate=True, check_grid=False)
            return self.integration_sample(epoch)[0]
        offset = float(self._initial_epoch.seconds_until(epoch))
        nodes, states = self._nodes.recent_state_data(min(self._capacity, self._nodes.size))
        return interpolate_regular(
            nodes,
            states,
            offset,
            self._history_interpolation_order,
            extrapolate=True,
            check_grid=False,
        )

    def integration_sample(self, epoch: Epoch):
        """Return the state and derivative of one Hermite interpolant.

        The stencil of ``(order + 1) // 2`` nodes is centred on the requested
        epoch and may use any accepted node. The seed arc (negative offsets)
        and the forward solution are interpolated on separate stencils, and
        no reference rotation is ever substituted.

        Parameters
        ----------
        epoch : Epoch
            TDB epoch inside the accepted history.

        Returns
        -------
        state : numpy.ndarray
            Interpolated integration state, shape ``(d,)``.
        derivative : numpy.ndarray
            Time derivative of the same interpolant, shape ``(d,)``.

        Raises
        ------
        ValueError
            If ``epoch`` lies outside the accepted or seed history.
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
