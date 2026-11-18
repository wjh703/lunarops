"""Propagated lunar trajectory on a regular grid and its interpolation.

:class:`LunarTrajectory` stores the accepted integration nodes of one
propagation and evaluates the state at arbitrary TDB epochs with barycentric
Lagrange interpolation on a centred stencil of equally spaced nodes
(:func:`lunarops.base.interpolation.interpolate_regular_grid`).
"""

from __future__ import annotations

import numpy as np

from lunarops.base.interpolation import interpolate_regular_grid
from lunarops.classes.time import Epoch

from .rotation import euler_matrix

OFFSET_TOLERANCE_S = 1e-9
"""Tolerance in s of a time offset beyond the first or latest node (rounding of epoch round trips)."""


def regular_times(duration_s: float, step_s: float) -> np.ndarray:
    """Return the regular grid of time offsets covering one propagation.

    Parameters
    ----------
    duration_s : float
        Signed propagation duration in TDB seconds; negative for backward
        propagation.
    step_s : float
        Positive grid step in s.

    Returns
    -------
    numpy.ndarray
        Offsets ``0, ±step_s, ..., duration_s`` in s, shape ``(count + 1,)``
        with ``count = |duration_s| / step_s``.

    Raises
    ------
    ValueError
        If ``|duration_s|`` is not an integer multiple of ``step_s``.
    """
    ratio = abs(float(duration_s)) / step_s
    count = round(ratio)
    if not np.isclose(ratio, count, rtol=0.0, atol=1e-12):
        raise ValueError("duration_s must be an integer multiple of the fixed integration step")
    direction = np.copysign(1.0, duration_s)
    return direction * step_s * np.arange(count + 1, dtype=float)


def interpolate_regular(
    nodes, values, x: float, order: int, *, extrapolate: bool, check_grid: bool = True
) -> np.ndarray:
    """Interpolate column-stored samples on a regular grid.

    A stencil of ``order + 1`` nodes centred on ``x`` (shifted inwards at the
    ends) is evaluated with the barycentric form of Lagrange interpolation
    (:func:`lunarops.base.interpolation.interpolate_regular_grid`). Node hits
    (within :data:`NODE_TOLERANCE` grid steps) return the stored column unchanged.

    Parameters
    ----------
    nodes : array_like
        Equally spaced abscissae, increasing or decreasing, shape ``(n,)``.
    values : array_like
        Samples with one column per node, shape ``(m, n)``.
    x : float
        Evaluation abscissa in the units of ``nodes``.
    order : int
        Polynomial order; the stencil holds ``min(order + 1, n)`` nodes.
    extrapolate : bool
        If false, ``x`` outside the node range (beyond :data:`OFFSET_TOLERANCE_S`)
        raises and ``x`` at or just beyond an end node returns that node.
    check_grid : bool, default True
        Verify that the stencil spacing is uniform to 1e-12 relative.

    Returns
    -------
    numpy.ndarray
        Interpolated values of shape ``(m,)``.

    Raises
    ------
    ValueError
        If the arrays have inconsistent shapes or no nodes, if ``x`` lies
        outside the nodes while ``extrapolate`` is false, or if the stencil
        is not a regular grid.
    """
    node_array = np.asarray(nodes, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if node_array.ndim != 1 or value_array.ndim != 2 or value_array.shape[1] != len(node_array):
        raise ValueError("Invalid regular interpolation arrays")
    if not len(node_array):
        raise ValueError("Regular interpolation requires at least one node")
    if not extrapolate:
        low, high = (0, -1) if node_array[0] <= node_array[-1] else (-1, 0)
        if x < node_array[low] - OFFSET_TOLERANCE_S or x > node_array[high] + OFFSET_TOLERANCE_S:
            raise ValueError("Epoch is outside propagated trajectory coverage")
        if x <= node_array[low]:
            return value_array[:, low].copy()
        if x >= node_array[high]:
            return value_array[:, high].copy()
    return interpolate_regular_grid(node_array, value_array.T, x, order + 1, check_grid=check_grid)[0]


class LunarTrajectory:
    """Propagated lunar state on a regular TDB grid.

    The trajectory keeps its own copy of the accepted nodes, independent of
    the integrator's history buffer. For a two-sided propagation the offsets
    run from a negative value through 0.

    Parameters
    ----------
    initial_epoch : Epoch
        TDB epoch of offset 0.
    times_s : array_like
        Node offsets from ``initial_epoch`` in TDB seconds, shape ``(n,)``.
    states : array_like
        States with one column per node, shape ``(d, n)``. Rows 0–2 are the
        Moon-minus-Earth position in m and rows 3–5 its velocity in m s⁻¹,
        both with BCRS (ICRF) axes. With integrated rotation (``d`` = 12 or
        15), rows 6–8 hold the PA-to-inertial 3-1-3 Euler angles (φ, θ, ψ) in
        rad, continuous in time, rows 9–11 the mantle angular velocity in PA
        axes in rad s⁻¹, and rows 12–14 the core angular velocity in PA axes
        in rad s⁻¹.
    trajectory_interpolation_order : int
        Lagrange interpolation order used by :meth:`vector`.
    history : DelayedHistory, TwoSidedHistory or None, optional
        Accepted-node history of a delayed propagation, kept for diagnostics.

    Raises
    ------
    ValueError
        If ``times_s`` and ``states`` have inconsistent shapes.
    """

    def __init__(
        self, initial_epoch: Epoch, times_s, states, trajectory_interpolation_order: int, history=None
    ) -> None:
        self.initial_epoch = initial_epoch
        self.history = history
        self.times_s = np.asarray(times_s, dtype=float)
        self.states = np.asarray(states, dtype=float)
        if self.times_s.ndim != 1 or self.states.ndim != 2 or self.states.shape[1] != len(self.times_s):
            raise ValueError("Invalid trajectory arrays")
        self.trajectory_interpolation_order = trajectory_interpolation_order

    def history_at(self, epoch: Epoch):
        """Return the history the integrator used for the dynamics at ``epoch``, or None.

        For a two-sided propagation this is the history of the arc that
        integrated ``epoch``, so that re-evaluated forces and torques match the
        integration (see :class:`~lunarops.classes.dynamics.history.TwoSidedHistory`).
        """
        return None if self.history is None else self.history.for_evaluation(epoch)

    def matrix(self, epoch: Epoch) -> np.ndarray:
        """Return the integrated PA-to-inertial rotation matrix.

        Parameters
        ----------
        epoch : Epoch
            Evaluation epoch in TDB, inside the trajectory coverage.

        Returns
        -------
        numpy.ndarray
            Rotation matrix ``R`` with ``v_inertial = R @ v_pa``, shape
            ``(3, 3)``.

        Raises
        ------
        ValueError
            If ``epoch`` is outside the propagated interval.
        """
        return euler_matrix(self.vector(epoch)[6:9])

    def vector(self, epoch: Epoch) -> np.ndarray:
        """Return the interpolated state vector at one epoch.

        Parameters
        ----------
        epoch : Epoch
            Evaluation epoch in TDB, inside the trajectory coverage.

        Returns
        -------
        numpy.ndarray
            State of shape ``(d,)`` with the row layout described in the
            class docstring.

        Raises
        ------
        ValueError
            If ``epoch`` is outside the propagated interval.
        """
        offset = float(self.initial_epoch.seconds_until(epoch))
        return interpolate_regular(
            self.times_s,
            self.states,
            offset,
            self.trajectory_interpolation_order,
            extrapolate=False,
        )
