"""Lagrange interpolation of samples on a regular grid.

One routine serves the propagated trajectory and its history, the ``lunarOrbit``/``lunarRotation`` tables and the
tabulated partials of the lunar dynamics. It evaluates many abscissae at once:

* Stencil: the ``count`` consecutive nodes whose midpoint is closest to the abscissa, shifted inwards at the ends
  of the grid. For an even ``count`` these are ``count / 2`` nodes on either side of the abscissa; for an odd
  ``count`` the stencil is centred on the nearest node. A centred stencil minimises the node polynomial
  ``prod_k (u - k)`` that multiplies the interpolation error, and its Lebesgue constant is the smallest of the
  stencils of that size.
* Weights: the barycentric form ``p(u) = sum_j f_j w_j / (u - j) / sum_j w_j / (u - j)`` with
  ``w_j = (-1)^j C(count - 1, j)`` on the unit grid ``0 .. count - 1`` (Berrut & Trefethen 2004, SIAM Rev. 46,
  501-517, Sec. 5). It costs ``O(count)`` per abscissa and is forward stable on the small equispaced stencils
  used here (Higham 2004, IMA J. Numer. Anal. 24, 547-556); within :data:`NODE_TOLERANCE` grid steps of a node,
  where ``w_j / (u - j)`` would overflow, the node value is returned unchanged.
* Each stencil is interpolated about its first sample, so large values (unwrapped angles, which grow by about
  85 rad per year) keep their digits.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np

NODE_TOLERANCE = 1e-13
"""Distance, in grid steps, within which an abscissa is taken to be the node itself."""


@lru_cache(maxsize=32)
def barycentric_weights(count: int) -> np.ndarray:
    """Return the barycentric weights ``(-1)^j C(count - 1, j)`` of ``count`` unit-spaced nodes (read-only)."""
    weights = np.asarray([(-1.0) ** index * comb(count - 1, index) for index in range(count)], dtype=float)
    weights.setflags(write=False)
    return weights


def regular_stencil_start(coordinate, count: int, rows: int) -> np.ndarray:
    """Return the first node of the centred stencil of ``count`` nodes at each grid coordinate.

    Parameters
    ----------
    coordinate : array_like, shape (p,)
        Abscissae in grid steps from the first node.
    count : int
        Stencil size, ``1 <= count <= rows``.
    rows : int
        Number of grid nodes.

    Returns
    -------
    ndarray of intp, shape (p,)
        ``ceil(coordinate - count / 2)`` clipped to ``0 .. rows - count``.
    """
    start = np.ceil(np.asarray(coordinate, dtype=float) - 0.5 * count)
    return np.clip(start, 0, rows - count).astype(np.intp)


def interpolate_regular_grid(nodes, values, x, count: int, *, check_grid: bool = False) -> np.ndarray:
    """Interpolate samples on a regular grid at many abscissae with a centred Lagrange stencil.

    Abscissae beyond the ends are extrapolated with the end stencil; callers that must not extrapolate check the
    range themselves.

    Parameters
    ----------
    nodes : array_like, shape (n,)
        Equally spaced abscissae, increasing or decreasing.
    values : array_like, shape (n, ...)
        Samples, one leading row per node.
    x : array_like, shape (p,) or scalar
        Evaluation abscissae, in the units of ``nodes``.
    count : int
        Stencil size; ``min(count, n)`` nodes are used.
    check_grid : bool, default False
        Verify that the spacing of every stencil used is uniform to 1e-12 of the step.

    Returns
    -------
    ndarray, shape (p, ...)
        Interpolated values.

    Raises
    ------
    ValueError
        If the arrays are inconsistent or empty, an abscissa is not finite, ``count < 1``, or (with
        ``check_grid``) a stencil is not regular.
    """
    node_array = np.asarray(nodes, dtype=float)
    value_array = np.asarray(values, dtype=float)
    points = np.asarray(x, dtype=float).reshape(-1)
    rows = len(node_array)
    if node_array.ndim != 1 or not rows or value_array.shape[:1] != (rows,) or count < 1:
        raise ValueError("Invalid regular interpolation arrays")
    if not np.all(np.isfinite(points)):
        raise ValueError("Interpolation abscissae must be finite")
    count = min(int(count), rows)
    if count == 1:
        step = (node_array[-1] - node_array[0]) / (rows - 1) if rows > 1 else 1.0
        nearest = np.clip(np.rint((points - node_array[0]) / step), 0, rows - 1).astype(np.intp)
        return value_array[nearest].copy()
    step = (node_array[-1] - node_array[0]) / (rows - 1)
    if step == 0.0 or not np.isfinite(step):
        raise ValueError("Interpolation nodes must form a regular grid")
    start = regular_stencil_start((points - node_array[0]) / step, count, rows)
    offsets = np.arange(count)
    index = start[:, None] + offsets
    stencil_nodes = node_array[index]
    local_step = (stencil_nodes[:, -1] - stencil_nodes[:, 0]) / (count - 1)
    if check_grid and not np.all(
        np.abs(np.diff(stencil_nodes, axis=1) - local_step[:, None]) <= 1e-12 * np.abs(local_step)[:, None]
    ):
        raise ValueError("Interpolation nodes must form a regular grid")
    local = (points - stencil_nodes[:, 0]) / local_step
    nearest = np.rint(local)
    hit = (np.abs(local - nearest) < NODE_TOLERANCE) & (nearest >= 0) & (nearest < count)
    hit_rows = np.flatnonzero(hit)
    hit_nodes = start[hit_rows] + nearest[hit_rows].astype(np.intp)
    local[hit_rows] = 0.5  # any non-node coordinate: these rows are replaced by the node values below
    factors = barycentric_weights(count) / (local[:, None] - offsets)
    stencil = value_array[index]
    reference = stencil[:, 0]
    centred = (stencil - reference[:, None]).reshape(len(points), count, -1)
    # Weighted sum by matmul, then one division by the sum of the factors: of the orderings tried this rounds
    # least (2e-6 m against 2.3e-6 m with normalised weights, at the end stencils of a daily lunarOrbit table).
    weighted = np.matmul(factors[:, None, :], centred)[:, 0] / factors.sum(axis=1)[:, None]
    result = reference + weighted.reshape(reference.shape)
    result[hit_rows] = value_array[hit_nodes]
    return result


__all__ = ["NODE_TOLERANCE", "barycentric_weights", "interpolate_regular_grid", "regular_stencil_start"]
