"""Accepted integration nodes and the state history read by delayed forces.

:class:`LunarNodeBuffer` stores the accepted states and derivatives of a
propagation. :class:`EphemerisHistory` answers state queries from the
ephemeris alone, and :class:`DelayedHistory` replaces the relative state by the
integrated solution so that delayed tides and deformation see the
propagated orbit and attitude.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from lunarops.classes.time import Epoch

from .bodies import EMB, earth_moon_about_barycenter
from .trajectory import interpolate_regular

_EXTRAPOLATION_NODES = 4


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
        self._states_low = np.zeros_like(self._states)
        self._derivatives_low = np.zeros_like(self._states)
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

    def append(self, offset_s: float, state, derivative, state_low=None, derivative_low=None) -> None:
        """Append one accepted node.

        Parameters
        ----------
        offset_s : float
            Node offset from the initial epoch in TDB seconds.
        state : array_like
            Accepted state, shape ``(d,)``.
        derivative : array_like
            Derivative of the accepted state, shape ``(d,)``.
        state_low, derivative_low : array_like or None, optional
            Low parts of the double-double state and derivative, shape
            ``(d,)``; zero when ``None``.

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
        if state_low is not None:
            self._states_low[:, self._size] = state_low
        if derivative_low is not None:
            self._derivatives_low[:, self._size] = derivative_low
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

    def state_low(self) -> np.ndarray:
        """Return the low part of the latest state, shape ``(d,)``."""
        return self._states_low[:, self._size - 1]

    def node(self, index: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return one node as the positional arguments of :meth:`append`."""
        return (
            float(self._times[index]),
            self._states[:, index],
            self._derivatives[:, index],
            self._states_low[:, index],
            self._derivatives_low[:, index],
        )

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

    def derivative_low_window(self, count: int) -> np.ndarray:
        """Return the low parts matching :meth:`derivative_window`, shape ``(d, count)``."""
        return self._derivatives_low[:, self._size - count : self._size]

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
        return self._times[: self._size], self._states[:, : self._size], self._derivatives[:, : self._size]

    def replace_last_derivative(self, derivative, derivative_low=0.0):
        """Overwrite the derivative of the latest node.

        Parameters
        ----------
        derivative : array_like
            New derivative, shape ``(d,)``.
        derivative_low : array_like, default 0.0
            Low part of the double-double derivative.
        """
        self._derivatives[:, self._size - 1] = derivative
        self._derivatives_low[:, self._size - 1] = derivative_low

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

    The Earth and the Moon are placed about the ephemeris Earth-Moon
    barycentre from :meth:`relative_state`, as in the native right-hand side.

    Parameters
    ----------
    ephemeris : Ephemeris
        Ephemeris providing BCRS body states.
    earth_gm_m3_s2, moon_gm_m3_s2 : float
        Earth and Moon GM in m³ s⁻² of the dynamics; their ratio places the
        two bodies about the barycentre.
    """

    def __init__(self, ephemeris, earth_gm_m3_s2: float, moon_gm_m3_s2: float):
        self._ephemeris = ephemeris
        self._earth_gm = float(earth_gm_m3_s2)
        self._moon_gm = float(moon_gm_m3_s2)

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
        return self._ephemeris.relative_state_matrix("MOON", "EARTH", epoch)

    def states(self, body_names: Sequence[str], epoch: Epoch) -> np.ndarray:
        """Return BCRS states of several bodies at one epoch.

        The Earth and the Moon are placed about the ephemeris barycentre from
        :meth:`relative_state`, so subclasses that override the relative state
        move both bodies.

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
        pair = "EARTH" in names or "MOON" in names
        externals = tuple(name for name in names if name not in ("EARTH", "MOON"))
        cached_names = tuple(dict.fromkeys(((EMB,) if pair else ()) + externals))
        cached = self._ephemeris.body_state_matrix_bcrs(cached_names, epoch)
        rows = {name: cached[index] for index, name in enumerate(cached_names)}
        if pair:
            rows["EARTH"], rows["MOON"] = earth_moon_about_barycenter(
                rows[EMB], self.relative_state(epoch)[:6], self._earth_gm, self._moon_gm
            )
        return np.asarray([rows[name] for name in names])


class DelayedHistory(EphemerisHistory):
    """State history in which the Earth-Moon relative state follows the accepted nodes.

    The Earth and the Moon are placed about the ephemeris barycentre; every
    other body comes from the ephemeris. Only accepted nodes are used;
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
        super().__init__(
            system.ephemeris,
            system.earth_body.gravitational_parameter_m3_s2,
            system.moon_body.gravitational_parameter_m3_s2,
        )

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
            ahead = (offset - times[-1]) if times[-1] >= times[0] else (times[-1] - offset)
            if ahead > 1e-9:
                # Earth orbital tides intentionally contain negative delays.
                # Extrapolate translation explicitly; rotational deformation
                # remains strictly causal through integration_sample().
                self.advanced_translation_queries += 1
                self.max_translation_extrapolation_s = max(self.max_translation_extrapolation_s, ahead)
                return interpolate_regular(
                    times, states, offset, self._history_interpolation_order, extrapolate=True, check_grid=False
                )
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
        no reference rotation is ever substituted. On a backward arc, where
        ``t - tau`` lies ahead of the latest node, the interpolant is
        evaluated by extrapolation.

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
            If a forward-arc ``epoch`` lies outside the accepted or seed history.
        """
        offset = float(self._initial_epoch.seconds_until(epoch))
        times, states, derivatives = self._nodes.integration_data()
        if times[-1] >= times[0]:
            if offset < times[0] - 1e-9 or offset > times[-1] + 1e-9:
                raise ValueError("Delayed integration sample is outside the accepted/seed history")
            split = self._nodes.trajectory_start
            if offset < 0:
                times, states, derivatives = times[:split], states[:, :split], derivatives[:, :split]
            else:
                times, states, derivatives = times[split:], states[:, split:], derivatives[:, split:]
        count = min((self._history_interpolation_order + 1) // 2, len(times))
        step = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 1.0
        if (offset - times[-1]) * step > 1e-9:
            # High-degree extrapolation amplifies the node noise; the delayed
            # response is small enough for a degree-7 Hermite polynomial.
            count = min(_EXTRAPOLATION_NODES, count)
        right = int(np.ceil((offset - times[0]) / step))
        start = max(0, min(right - count // 2, len(times) - count))
        selected = slice(start, start + count)
        grid = times[selected]
        return interpolate_state_derivative(grid, states[:, selected], derivatives[:, selected], offset)


class TwoSidedHistory:
    """Accepted history of a two-sided propagation.

    Queries at non-negative offsets from ``initial_epoch`` are answered by the
    forward arc and the others by the backward arc.

    Parameters
    ----------
    initial_epoch : Epoch
        TDB epoch of offset 0.
    forward, backward : DelayedHistory
        Histories of the two arcs.
    """

    def __init__(self, initial_epoch: Epoch, forward: DelayedHistory, backward: DelayedHistory) -> None:
        self._initial_epoch = initial_epoch
        self._forward = forward
        self._backward = backward

    @property
    def advanced_translation_queries(self) -> int:
        """int: Translation queries ahead of the latest node on either arc."""
        return self._forward.advanced_translation_queries + self._backward.advanced_translation_queries

    @property
    def max_translation_extrapolation_s(self) -> float:
        """float: Largest translation extrapolation interval on either arc, in s."""
        return max(self._forward.max_translation_extrapolation_s, self._backward.max_translation_extrapolation_s)

    def _arc(self, epoch: Epoch) -> DelayedHistory:
        return self._forward if float(self._initial_epoch.seconds_until(epoch)) >= 0 else self._backward

    def relative_state(self, epoch: Epoch) -> np.ndarray:
        """Return the integrated state at one epoch; see :meth:`DelayedHistory.relative_state`."""
        return self._arc(epoch).relative_state(epoch)

    def states(self, body_names: Sequence[str], epoch: Epoch) -> np.ndarray:
        """Return BCRS body states at one epoch; see :meth:`EphemerisHistory.states`."""
        return self._arc(epoch).states(body_names, epoch)

    def integration_sample(self, epoch: Epoch):
        """Return the Hermite sample at one epoch; see :meth:`DelayedHistory.integration_sample`."""
        return self._arc(epoch).integration_sample(epoch)
