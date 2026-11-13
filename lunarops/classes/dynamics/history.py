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

from lunarops._dynamics_core import hermite_sample  # pyright: ignore[reportMissingModuleSource]
from lunarops.classes.time import Epoch

from .bodies import EMB, earth_moon_about_barycenter
from .trajectory import interpolate_regular

_EXTRAPOLATION_NODES = 4
# Largest extrapolation distance of the Hermite sample, in stencil spacings. On a backward arc the delayed
# deformation reads t - tau_M, tau_M + h ahead of the latest node. That is 3.3 spacings at the 1/16 d step
# of Aksim & Pavlov (2019) and stable; at 5.6 spacings (h = 2700 s) the extrapolated node noise feeds back
# through the deformation and grows exponentially. Smaller steps therefore take every k-th node.
_MAX_EXTRAPOLATION_SPACINGS = 3.5
# Nodes of the Lagrange extrapolation of the relative state beyond the latest node (advanced arguments
# of the Earth tide); see DelayedHistory._extrapolate.
_TRANSLATION_EXTRAPOLATION_NODES = 7
# Stencil limit of the compiled Hermite sample; longer stencils use interpolate_state_derivative.
_MAX_COMPILED_HERMITE_NODES = 32


def _is_behind(times: np.ndarray, offset: float) -> bool:
    """Return whether ``offset`` lies behind the oldest node of ``times`` (stored in integration order)."""
    direction = 1.0 if times[-1] >= times[0] else -1.0
    return (times[0] - offset) * direction > 1e-9


@lru_cache(maxsize=16)
def _hermite_grid(count):
    grid = np.arange(count, dtype=float)
    differences = grid[:, None] - grid[None, :]
    np.fill_diagonal(differences, 1)
    denominators = np.prod(differences, axis=1)
    inverse = 1 / differences
    np.fill_diagonal(inverse, 0)
    return grid, denominators, np.sum(inverse, axis=1)


@lru_cache(maxsize=1024)
def _hermite_weights(count: int, coordinate: float):
    """Return ``(exact_index, weights)`` of the Hermite interpolant at a grid coordinate.

    ``coordinate`` is ``(offset - times[0]) / step``. With a fixed step and fixed delays the same float
    usually recurs from step to step, so caching on it returns the weights the full computation would give,
    bit for bit. Far from the initial epoch the rounding of the offsets varies it in the last bits, and the
    cache then only misses. ``exact_index`` is the node within 1e-13 of the coordinate (``weights`` is then
    None), otherwise -1.
    """
    grid, denominators, slopes = _hermite_grid(count)
    distances = coordinate - grid
    exact = np.flatnonzero(abs(distances) < 1e-13)
    if len(exact):
        return int(exact[0]), None
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
    for array in (h, g, h_dot, g_dot):
        array.setflags(write=False)
    return -1, (h, g, h_dot, g_dot)


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
    exact, weights = _hermite_weights(count, float((offset - times[0]) / step))
    if exact >= 0:
        return states[:, exact].copy(), derivatives[:, exact].copy()
    h, g, h_dot, g_dot = weights
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
    def state_dimension(self) -> int:
        """int: Length of the state vector."""
        return self._states.shape[0]

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

    def extend(self, block) -> None:
        """Append the nodes of a :meth:`block`, in its column order.

        Parameters
        ----------
        block : tuple of numpy.ndarray
            Times, states, derivatives and their low parts, one column per node, as :meth:`block` returns.

        Raises
        ------
        IndexError
            If the nodes do not fit in the buffer.
        """
        times, states, derivatives, states_low, derivatives_low = block
        stop = self._size + len(times)
        if stop > len(self._times):
            raise IndexError("Lunar node buffer is full")
        columns = slice(self._size, stop)
        self._times[columns] = times
        self._states[:, columns] = states
        self._derivatives[:, columns] = derivatives
        self._states_low[:, columns] = states_low
        self._derivatives_low[:, columns] = derivatives_low
        self._size = stop

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

    def block(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return copies of the nodes ``start:stop``, one column per node, in the order of :meth:`node`."""
        columns = slice(start, stop)
        return (
            self._times[columns].copy(),
            self._states[:, columns].copy(),
            self._derivatives[:, columns].copy(),
            self._states_low[:, columns].copy(),
            self._derivatives_low[:, columns].copy(),
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
        self._state_plans: dict[tuple[str, ...], tuple[bool, tuple[str, ...]]] = {}

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
        plan = self._state_plans.get(names)
        if plan is None:
            pair = "EARTH" in names or "MOON" in names
            externals = tuple(name for name in names if name not in ("EARTH", "MOON"))
            cached_names = tuple(dict.fromkeys(((EMB,) if pair else ()) + externals))
            plan = self._state_plans[names] = (pair, cached_names)
        pair, cached_names = plan
        cached = self._ephemeris.body_state_matrix_bcrs(cached_names, epoch)
        rows = {name: cached[index] for index, name in enumerate(cached_names)}
        if pair:
            rows["EARTH"], rows["MOON"] = earth_moon_about_barycenter(
                rows[EMB], self.relative_state(epoch)[:6], self._earth_gm, self._moon_gm
            )
        return np.asarray([rows[name] for name in names])

    def geocentric_positions(self, body_names: Sequence[str], epochs: Sequence[Epoch]) -> np.ndarray:
        """Return body-minus-Earth positions at several epochs, shape ``(len(epochs), len(body_names), 3)``.

        The Moon's row is :meth:`relative_state`; every other body is taken from one batched ephemeris call
        and referred to the Earth placed about the ephemeris barycentre, as in :meth:`states`.
        """
        names, epochs = tuple(body_names), tuple(epochs)
        relative = np.asarray([self.relative_state(epoch)[:3] for epoch in epochs], dtype=float).reshape(-1, 3)
        others = tuple(name for name in names if name not in ("EARTH", "MOON"))
        result = np.zeros((len(epochs), len(names), 3))
        if others:
            positions = self._ephemeris.body_state_matrices_bcrs((EMB, *others), epochs)[:, :, :3]
            earth = positions[:, 0] - (self._moon_gm / (self._earth_gm + self._moon_gm)) * relative
        for column, name in enumerate(names):
            if name == "MOON":
                result[:, column] = relative
            elif name != "EARTH":
                result[:, column] = positions[:, 1 + others.index(name)] - earth
        return result


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
    extrapolated_translation_queries : int
        Number of translation queries beyond the latest accepted node in the
        direction of integration. On a forward arc they come from the
        negative Earth-tide delays, on a backward arc from the positive ones.
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
        self._joint_delay = system.rotation is not None and system.rotation.deformation is not None
        self.extrapolated_translation_queries = 0
        self.max_translation_extrapolation_s = 0.0
        super().__init__(
            system.ephemeris,
            system.earth_body.gravitational_parameter_m3_s2,
            system.moon_body.gravitational_parameter_m3_s2,
        )

    def relative_state(self, epoch: Epoch) -> np.ndarray:
        """Return the integrated Moon-minus-Earth state at one epoch.

        With integrated rotation and deformation the state is a Hermite
        interpolant of the accepted nodes. Otherwise a Lagrange polynomial on
        a stencil of accepted nodes centred on the epoch is evaluated. In both
        cases epochs beyond the latest node in the direction of integration
        are extrapolated with a Lagrange polynomial through the latest seven
        nodes (translation only). On a forward arc these come from the negative
        Earth-tide delays, on a backward arc from the positive ones.

        Parameters
        ----------
        epoch : Epoch
            TDB epoch.

        Returns
        -------
        numpy.ndarray
            Integrated state; the first six entries are the relative
            position in m and velocity in m s⁻¹ with ICRF axes.

        Before the oldest node (the start of a propagation with a positive
        tidal delay at a small step) the ephemeris state is returned, as for
        the delayed forces during the DOP853 warm-up.
        """
        offset = float(self._initial_epoch.seconds_until(epoch))
        times, states, _ = self._nodes.integration_data()
        ahead = (offset - times[-1]) if times[-1] >= times[0] else (times[-1] - offset)
        if ahead > 1e-9:
            # The Earth orbital tides contain advanced arguments; rotational deformation stays causal on a
            # forward arc through integration_sample().
            self.extrapolated_translation_queries += 1
            self.max_translation_extrapolation_s = max(self.max_translation_extrapolation_s, ahead)
            return self._extrapolate(times, states, offset)
        if self._joint_delay:
            return self.integration_sample(epoch)[0]
        if _is_behind(times, offset):
            # Before the oldest node the history is the ephemeris, as for the delayed forces of the warm-up.
            return super().relative_state(epoch)
        return interpolate_regular(
            times, states, offset, self._history_interpolation_order, extrapolate=False, check_grid=False
        )

    def _extrapolate(self, times, states, offset: float) -> np.ndarray:
        """Lagrange extrapolation beyond the latest node, as in Aksim & Pavlov (2019) for advanced arguments.

        The stencil is the latest ``_TRANSLATION_EXTRAPOLATION_NODES`` nodes. Extrapolating with the full
        history stencil (14 nodes at order 13) amplifies the truncation of the node values: for the Earth
        tide's order-2 delay of -0.2098 d, i.e. 20,800 s past the latest node at a 2700 s step, it misses the
        DE440 Moon by up to 0.5 km. Seven nodes keep the error below 0.4 m for steps of 1350-5400 s, the
        range this was checked over; at 675-900 s six nodes would be better (0.2 m against 0.9-2 m). Either
        error is about 1e-9 of the tide vector, which moves the propagated Moon by well under 1 um per year.
        """
        order = min(self._history_interpolation_order, _TRANSLATION_EXTRAPOLATION_NODES - 1)
        return interpolate_regular(times, states, offset, order, extrapolate=True, check_grid=False)

    def integration_sample(self, epoch: Epoch):
        """Return the state and derivative of one Hermite interpolant.

        The stencil of ``(order + 1) // 2`` nodes is centred on the requested
        epoch and may use any accepted node. The seed arc (negative offsets)
        and the forward solution are interpolated on separate stencils, and
        no reference rotation is ever substituted. The seed stencil ends at the
        last seed node before the initial epoch: the initial node carries the
        deformed derivative, which does not belong to the rigid seed solution.

        On a backward arc ``t - tau`` lies ahead of the latest node, and the
        interpolant is extrapolated from the latest ``_EXTRAPOLATION_NODES``
        nodes. When the step is small, the stencil takes every k-th node, so
        that the extrapolation spans at most ``_MAX_EXTRAPOLATION_SPACINGS``
        stencil spacings.

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
            If ``epoch`` lies behind the oldest accepted node, or, on a forward
            arc, beyond the latest one.
        """
        offset = float(self._initial_epoch.seconds_until(epoch))
        times, states, derivatives = self._nodes.integration_data()
        if _is_behind(times, offset):
            raise ValueError("Delayed integration sample is outside the accepted history (before the oldest node)")
        if times[-1] >= times[0]:
            if offset > times[-1] + 1e-9:
                raise ValueError("Delayed integration sample is outside the accepted history (beyond the latest node)")
            split = self._nodes.trajectory_start
            if offset < 0:
                times, states, derivatives = times[:split], states[:, :split], derivatives[:, :split]
            else:
                times, states, derivatives = times[split:], states[:, split:], derivatives[:, split:]
        count = min((self._history_interpolation_order + 1) // 2, len(times))
        step = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 1.0
        if (offset - times[-1]) * step > 1e-9:
            # High-degree extrapolation amplifies the node noise; the delayed response is small enough for a
            # degree-7 Hermite polynomial, on a stencil wide enough to keep the extrapolation stable.
            ahead_steps = (offset - times[-1]) / step
            stride = max(1, int(np.ceil(ahead_steps / _MAX_EXTRAPOLATION_SPACINGS)))
            count = min(_EXTRAPOLATION_NODES, (len(times) - 1) // stride + 1)
            selected = slice(len(times) - 1 - stride * (count - 1), len(times), stride)
        else:
            right = int(np.ceil((offset - times[0]) / step))
            start = max(0, min(right - count // 2, len(times) - count))
            selected = slice(start, start + count)
        grid = times[selected]
        if count > _MAX_COMPILED_HERMITE_NODES:
            return interpolate_state_derivative(grid, states[:, selected], derivatives[:, selected], offset)
        state, derivative = np.empty(len(states)), np.empty(len(states))
        # Compiled form of interpolate_state_derivative (same interpolant; sums reordered, a few ulp).
        hermite_sample(
            states[:, selected],
            derivatives[:, selected],
            float(grid[0]),
            float(grid[1] - grid[0]) if count > 1 else 1.0,
            offset,
            state,
            derivative,
        )
        return state, derivative


class TwoSidedHistory:
    """Accepted history of a two-sided propagation.

    Queries at non-negative offsets from ``initial_epoch`` are answered by the
    forward arc and the others by the backward arc.

    The two arcs are not one delayed solution: the forward arc started from a
    rigid seed before the initial epoch and never sees the backward arc. To
    re-evaluate the dynamics at an epoch as the integrator did, use the
    history of the arc that integrated that epoch, :meth:`for_evaluation`.

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

    def for_evaluation(self, epoch: Epoch) -> DelayedHistory:
        """Return the history the integrator used to evaluate the dynamics at ``epoch``."""
        return self._arc(epoch)

    @property
    def extrapolated_translation_queries(self) -> int:
        """int: Translation queries beyond the latest node on either arc."""
        return self._forward.extrapolated_translation_queries + self._backward.extrapolated_translation_queries

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

    def geocentric_positions(self, body_names: Sequence[str], epochs: Sequence[Epoch]) -> np.ndarray:
        """Return body-minus-Earth positions from the arc of each epoch; see :meth:`DelayedHistory.geocentric_positions`."""
        return np.concatenate([self._arc(epoch).geocentric_positions(body_names, (epoch,)) for epoch in epochs])

    def integration_sample(self, epoch: Epoch):
        """Return the Hermite sample at one epoch; see :meth:`DelayedHistory.integration_sample`."""
        return self._arc(epoch).integration_sample(epoch)
