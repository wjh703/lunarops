"""Parametrization of the initial values and dynamical parameters of the integrated lunar orbit and rotation.

The lunar orbit and rotation are integrated together with the partials of the state with respect to the
initial values and the dynamical parameters (the variational equations; Montenbruck & Gill 2000, Satellite
Orbits, Section 7.2). The ``lunarDynamicsPartials`` table of that integration gives, on a regular TDB grid, the
state ``y`` and ``dy/dx`` for the estimated quantities ``x``. The partial of the one-way range of a normal point
with respect to ``x`` is the chain rule at its bounce epoch ``t_b``,

    d(rho)/dx = d(rho)/dr . dr/dx  +  d(rho)/dw . E(e) . de/dx,

where ``r`` is the Moon-minus-Earth position (state rows 0-2), ``e = (phi, theta, psi)`` are the 3-1-3 Euler
angles of the PA-to-inertial rotation ``R = Rz(phi) Rx(theta) Rz(psi)`` (state rows 6-8), ``d(rho)/dr`` is the
``moon_position_bcrs`` partial of the observation equations and ``d(rho)/dw`` their ``lunar_orientation_pa``
partial with respect to a small rotation ``w`` of the PA frame, ``R -> R (I + [w]x)``. ``E`` maps increments of
the Euler angles to that rotation, ``dw = E de``,

    E = [[sin(theta) sin(psi),  cos(psi), 0],
         [sin(theta) cos(psi), -sin(psi), 0],
         [cos(theta),           0,        1]],

which is also the matrix from Euler-angle rates to the body-frame angular velocity of the 3-1-3 sequence
(Goldstein, Poole & Safko 2002, Classical Mechanics, 3rd ed., Eq. 4.87). The table is interpolated to ``t_b`` by
Lagrange interpolation on the regular grid.

The dependence of the light-time solution (the event epochs) on ``x`` is neglected, a relative error of the
order ``v/c`` of the design columns, as for the other partials of the observation equations. The dependence of
the solid tides, the gravitational delay and the relativistic frame transformation on the lunar position is
neglected as well.

The corrections enter the observation equations linearly: the forward model keeps the ephemeris of the
nominal integration, and ``design @ correction`` is subtracted from O-C. The estimated parameters are the
corrections to the nominal values, in the units of the quantities. After an adjustment the updated values
(nominal plus correction) are written as a ``lunarDynamicsParameters`` file, from which the orbit and rotation
are integrated again; integration and adjustment are iterated until the corrections vanish.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from lunarops.base.parameter_name import LUNAR_DYNAMICS_TYPE_PREFIX, ParameterName
from lunarops.classes.observation.equations import EquationBatch
from lunarops.config.schema import ConfigSchema, integer, path, sequence

from .base import Parametrization

_SECONDS_PER_DAY = 86400.0
_GRID_TOLERANCE = 1.0e-9
_POSITION_ROWS = slice(0, 3)
_EULER_ROWS = slice(6, 9)


def euler_to_pa_rotation(euler_rad) -> np.ndarray:
    """Return the matrix ``E`` from Euler-angle increments to the small PA-frame rotation they produce.

    For the PA-to-inertial rotation ``R = Rz(phi) Rx(theta) Rz(psi)``, increments ``de = (dphi, dtheta, dpsi)``
    change it to ``R (I + [w]x)`` with ``w = E de`` to first order.

    Parameters
    ----------
    euler_rad : array_like, shape (..., 3)
        Euler angles ``(phi, theta, psi)``, in radians; ``phi`` does not enter.

    Returns
    -------
    ndarray, shape (..., 3, 3)
        ``E``, dimensionless.

    References
    ----------
    .. [1] Goldstein, H., Poole, C., & Safko, J. (2002). Classical Mechanics, 3rd ed., Eq. (4.87).
    """
    euler = np.asarray(euler_rad, dtype=float)
    theta, psi = euler[..., 1], euler[..., 2]
    matrix = np.zeros((*euler.shape[:-1], 3, 3))
    matrix[..., 0, 0] = np.sin(theta) * np.sin(psi)
    matrix[..., 0, 1] = np.cos(psi)
    matrix[..., 1, 0] = np.sin(theta) * np.cos(psi)
    matrix[..., 1, 1] = -np.sin(psi)
    matrix[..., 2, 0] = np.cos(theta)
    matrix[..., 2, 2] = 1.0
    return matrix


def interpolate_regular_rows(nodes, values, x, order: int) -> np.ndarray:
    """Interpolate samples on a regular grid at many abscissae by Lagrange interpolation.

    The vectorised counterpart of :func:`lunarops.classes.dynamics.trajectory.interpolate_regular` with the same
    stencil: ``order + 1`` nodes centred on each abscissa and shifted inwards at the ends. Each stencil is
    interpolated about its first sample, so that large values (unwrapped angles) keep their digits.

    Parameters
    ----------
    nodes : array_like, shape (n,)
        Equally spaced, increasing abscissae.
    values : array_like, shape (n, ...)
        Samples, one leading row per node.
    x : array_like, shape (m,)
        Evaluation abscissae, in the units of ``nodes``.
    order : int
        Polynomial order; the stencil holds ``min(order + 1, n)`` nodes.

    Returns
    -------
    ndarray, shape (m, ...)
        Interpolated values.

    Raises
    ------
    ValueError
        If the nodes are not a regular increasing grid, or an abscissa lies outside them by more than 1e-9 of
        the grid step.
    """
    node_array = np.asarray(nodes, dtype=float)
    value_array = np.asarray(values, dtype=float)
    points = np.asarray(x, dtype=float).reshape(-1)
    count = min(int(order) + 1, len(node_array))
    if node_array.ndim != 1 or not len(node_array) or len(value_array) != len(node_array) or count < 1:
        raise ValueError("Invalid regular interpolation arrays")
    if len(node_array) == 1:
        if np.any(points != node_array[0]):
            raise ValueError("Epoch is outside the interpolation nodes")
        return np.repeat(value_array[:1], len(points), axis=0)
    step = (node_array[-1] - node_array[0]) / (len(node_array) - 1)
    if not step > 0.0 or np.max(np.abs(np.diff(node_array) - step)) > _GRID_TOLERANCE * step:
        raise ValueError("Interpolation nodes must form a regular increasing grid")
    tolerance = _GRID_TOLERANCE * step
    if np.any(points < node_array[0] - tolerance) or np.any(points > node_array[-1] + tolerance):
        raise ValueError("Epoch is outside the interpolation nodes")
    points = np.clip(points, node_array[0], node_array[-1])
    right = np.searchsorted(node_array, points, side="left")
    start = np.clip(right - count // 2, 0, len(node_array) - count)
    coordinate = (points - node_array[start]) / step
    # Lagrange basis on the integer stencil 0..count-1: w_j(u) = prod_{k != j} (u - k) / (j - k).
    index = np.arange(count)
    weights = np.ones((len(points), count))
    for k in range(count):
        others = index != k
        weights[:, others] *= (coordinate[:, None] - k) / (index[others] - k)
    trailing = (slice(None), *(None,) * (value_array.ndim - 1))
    reference = value_array[start]
    result = np.zeros_like(reference)
    for j in range(1, count):
        result += weights[:, j][trailing] * (value_array[start + j] - reference)
    return reference + result


class LunarDynamicsParametrization(Parametrization):
    """Corrections to the initial values and dynamical parameters of the integrated lunar orbit and rotation.

    Parameter names are ``moon:dynamics.<quantity>``, in the unit of the quantity (see
    :data:`lunarops.classes.dynamics.parameters.PARAMETER_DEFINITIONS`). The parameters are corrections to the
    nominal values of the integration, starting at zero.

    Parameters
    ----------
    partials : LunarDynamicsPartials
        Table of the state and its partials of the nominal integration.
    parameters : LunarDynamicsParameters, optional
        Nominal values the integration used. Without it the nominal initial values are the state of the table
        at offset 0 and the nominal dynamical parameters are unknown.
    quantities : Sequence[str], optional
        Estimated quantities, in column order; all quantities of the table when ``None``.
    interpolation_order : int, default 9
        Order of the Lagrange interpolation of the table (10 nodes).

    Raises
    ------
    ValueError
        If a quantity is not in the table or is repeated, the table grid is not regular, an orientation quantity
        is requested from a table without rotation, the epoch of ``parameters`` differs from the initial epoch
        of the table, or an initial value of ``parameters`` differs from the state of the table at offset 0.
    """

    TYPE = "lunarDynamics"
    SCHEMA = ConfigSchema(
        fields=(
            path(
                "inputFilePartials",
                required=True,
                non_empty=True,
                description=(
                    "`lunarDynamicsPartials` table of the nominal integration (`outputFilePartials` of "
                    "`LunarOrbitPropagation`): the state and its partials with respect to the initial values and "
                    "the dynamical parameters. It must cover the bounce epochs of all normal points."
                ),
            ),
            path(
                "inputFileLunarDynamicsParameters",
                non_empty=True,
                description=(
                    "`lunarDynamicsParameters` file with the nominal values the integration used. Needed to "
                    "report updated values of dynamical parameters and to write "
                    "`outputFileLunarDynamicsParameters`; without it only the initial values, read from the table, "
                    "have nominal values."
                ),
            ),
            sequence(
                "quantities",
                item_kind="string",
                min_items=1,
                unique=True,
                description=(
                    "Estimated quantities, a subset of the quantity names of the table (for example `r0_x`, "
                    "`gm_emb`, `beta`), in column order. All quantities of the table if omitted."
                ),
            ),
            integer(
                "interpolationOrder",
                default=9,
                ge=1,
                description="Order of the Lagrange interpolation of the table to the bounce epoch (9: ten nodes).",
            ),
        ),
        type_name="lunarDynamics",
    )

    def __init__(
        self,
        *,
        partials,
        parameters=None,
        quantities: Sequence[str] | None = None,
        interpolation_order: int = 9,
    ) -> None:
        from lunarops.classes.dynamics.parameters import INITIAL_VALUE_NAMES, parameter_definition

        table_names = list(partials.quantity_names)
        self.quantities = table_names if quantities is None else [str(name) for name in quantities]
        unknown = [name for name in self.quantities if name not in table_names]
        if unknown:
            raise ValueError(f"lunarDynamics quantities not in the partials table: {unknown}; it has {table_names}.")
        if len(set(self.quantities)) != len(self.quantities) or not self.quantities:
            raise ValueError("lunarDynamics quantities must be distinct and not empty.")
        self._definitions = [parameter_definition(name) for name in self.quantities]
        columns = [table_names.index(name) for name in self.quantities]
        self.interpolation_order = int(interpolation_order)
        self.epoch_tdb_jd1 = float(partials.epoch_tdb_jd1)
        self.epoch_tdb_jd2 = float(partials.epoch_tdb_jd2)
        self.producer = str(partials.producer)
        self._offsets = np.array(partials.offsets_s, dtype=float)
        if len(self._offsets) > 1:
            step = (self._offsets[-1] - self._offsets[0]) / (len(self._offsets) - 1)
            if np.max(np.abs(np.diff(self._offsets) - step)) > _GRID_TOLERANCE * step:
                raise ValueError("The lunarDynamicsPartials table must be on a regular time grid.")
        # Only the position and Euler-angle rows of the estimated quantities are kept (the table of a century
        # on a 6 h grid has about 146 000 epochs).
        self._position_partials = np.ascontiguousarray(partials.partials[:, _POSITION_ROWS, :][:, :, columns])
        self._has_rotation = partials.dimension >= 9
        self._euler = self._euler_partials = None
        if self._has_rotation:
            self._euler = np.ascontiguousarray(partials.states[:, _EULER_ROWS])
            self._euler_partials = np.ascontiguousarray(partials.partials[:, _EULER_ROWS, :][:, :, columns])
        else:
            rotational = [name for name in self.quantities if name in INITIAL_VALUE_NAMES[6:]]
            if rotational:
                raise ValueError(f"The partials table has no rotation state for the quantities {rotational}.")

        initial = np.flatnonzero(self._offsets == 0.0)
        table_initial = (
            dict(zip(INITIAL_VALUE_NAMES[: partials.dimension], partials.states[initial[0]], strict=True))
            if len(initial)
            else {}
        )
        self.parameters = parameters
        if parameters is not None:
            if not parameters.is_at(self.epoch_tdb_jd1, self.epoch_tdb_jd2):
                raise ValueError(
                    "The epoch of the lunarDynamicsParameters file differs from the initial epoch of the "
                    "lunarDynamicsPartials table."
                )
            for name, value in table_initial.items():
                tolerance = 1.0e-3 * parameter_definition(name).step
                if name in parameters and abs(parameters[name] - value) > tolerance:
                    raise ValueError(
                        f"lunarDynamicsParameters {name} = {parameters[name]!r} differs from the initial state "
                        f"{float(value)!r} of the lunarDynamicsPartials table; the files are not of one integration."
                    )
        self.nominal: dict[str, float | None] = {}
        for name in self.quantities:
            if parameters is not None and name in parameters:
                self.nominal[name] = float(parameters[name])
            else:
                self.nominal[name] = float(table_initial[name]) if name in table_initial else None

        self.names = [ParameterName("moon", f"{LUNAR_DYNAMICS_TYPE_PREFIX}{name}") for name in self.quantities]
        # Corrections to the nominal values, by quantity name, in the units of the quantities.
        self.values = dict.fromkeys(self.quantities, 0.0)
        self._column_rms: np.ndarray | None = None
        self._gram: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict, context) -> LunarDynamicsParametrization:
        """Create the parametrization from a validated configuration, reading its input files."""
        from lunarops.classes.dynamics.parameters import read_lunar_dynamics_parameters
        from lunarops.fileio.lunar_partials import read_lunar_dynamics_partials

        parameters_file = config.get("inputFileLunarDynamicsParameters")
        return cls(
            partials=read_lunar_dynamics_partials(context.resolve_path(config["inputFilePartials"])),
            parameters=(
                None
                if parameters_file is None
                else read_lunar_dynamics_parameters(context.resolve_path(parameters_file))
            ),
            quantities=config.get("quantities"),
            interpolation_order=config["interpolationOrder"],
        )

    def _offsets_of(self, batch: EquationBatch) -> np.ndarray:
        """Return the bounce epochs of ``batch`` as TDB seconds from the initial epoch of the table."""
        offsets = (
            (batch.bounce_tdb_jd1 - self.epoch_tdb_jd1) + (batch.bounce_tdb_jd2 - self.epoch_tdb_jd2)
        ) * _SECONDS_PER_DAY
        if not np.all(np.isfinite(offsets)):
            raise ValueError("lunarDynamics needs the TDB bounce epochs of the observation equations.")
        tolerance = _GRID_TOLERANCE * abs(self._offsets[-1] - self._offsets[0]) / max(len(self._offsets) - 1, 1)
        outside = (offsets < self._offsets[0] - tolerance) | (offsets > self._offsets[-1] + tolerance)
        if np.any(outside):
            start, end = (
                self.epoch_tdb_jd1 + self.epoch_tdb_jd2 + offset / _SECONDS_PER_DAY
                for offset in (self._offsets[0], self._offsets[-1])
            )
            observed = self.epoch_tdb_jd1 + self.epoch_tdb_jd2 + offsets[outside] / _SECONDS_PER_DAY
            raise ValueError(
                f"{int(np.count_nonzero(outside))} normal point(s) have bounce epochs (TDB JD {observed.min():.5f} "
                f"to {observed.max():.5f}) outside the lunarDynamicsPartials table (TDB JD {start:.5f} to "
                f"{end:.5f})."
            )
        return offsets

    def _design(self, batch: EquationBatch) -> np.ndarray:
        """Return the design block, shape ``(n, len(names))``, cached on the batch."""
        cache = batch.linear_design_cache
        if self in cache:
            return cache[self]
        count = len(batch)
        if batch.moon_position_bcrs.shape != (count, 3):
            raise ValueError("lunarDynamics needs the Moon-position partials of the observation equations.")
        if self._has_rotation and batch.lunar_orientation_pa.shape != (count, 3):
            raise ValueError("lunarDynamics needs the lunar-orientation partials of the observation equations.")
        offsets = self._offsets_of(batch)
        order = self.interpolation_order
        position = interpolate_regular_rows(self._offsets, self._position_partials, offsets, order)
        design = np.einsum("ni,niq->nq", batch.moon_position_bcrs, position)
        if self._has_rotation:
            euler = interpolate_regular_rows(self._offsets, self._euler, offsets, order)
            euler_partials = interpolate_regular_rows(self._offsets, self._euler_partials, offsets, order)
            # d(rho)/d(e) = d(rho)/d(w) . E(e), then the chain rule with de/dx.
            range_per_euler = np.einsum("ni,nij->nj", batch.lunar_orientation_pa, euler_to_pa_rotation(euler))
            design += np.einsum("nj,njq->nq", range_per_euler, euler_partials)
        design.setflags(write=False)
        cache[self] = design
        if count:
            self._column_rms = np.sqrt(np.mean(design**2, axis=0))
            self._gram = design.T @ design / len(design)
        return design

    def setup(self, batch: EquationBatch) -> list[ParameterName]:
        """Check that the table covers ``batch`` and return the parameter names."""
        self._design(batch)
        return self.names

    def x0(self) -> np.ndarray:
        """Return the current corrections to the nominal values, in the units of the quantities."""
        return np.asarray([self.values[name] for name in self.quantities], dtype=float)

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        """Add the partials of the one-way range with respect to the quantities, in metres per quantity unit.

        Raises
        ------
        ValueError
            If the batch lacks the Moon-position or lunar-orientation partials or the bounce epochs, or a bounce
            epoch lies outside the table.
        """
        out += self._design(batch)

    def _contributions(self, batch: EquationBatch):
        yield self._design(batch) @ self.x0()

    def update(self, delta: np.ndarray) -> None:
        """Add the corrections ``delta``, in the units of the quantities."""
        for name, value in zip(self.quantities, np.asarray(delta, dtype=float), strict=True):
            self.values[name] += float(value)

    def max_update_norm(self, delta: np.ndarray) -> float:
        """Return the root-mean-square effect of the corrections on the one-way range, in metres.

        The quantities have different units and are strongly correlated: single corrections can move by
        much more than their joint effect on the range. The norm is therefore that of the whole update,
        ``sqrt(mean((D delta)^2))`` with ``D`` the design of the last linearization; the convergence
        threshold of this block is then a range in metres, as for the position blocks.
        """
        delta = np.asarray(delta, dtype=float)
        if not len(delta):
            return 0.0
        if self._gram is None:
            return float(np.max(np.abs(delta)))
        return float(np.sqrt(max(delta @ self._gram @ delta, 0.0)))

    def updated_parameters(self):
        """Return the nominal parameter set with the corrections added.

        Returns
        -------
        LunarDynamicsParameters
            The values of the nominal file (or, without one, the nominal values of the estimated quantities)
            with every estimated quantity replaced by nominal plus correction, at the initial epoch of the table.

        Raises
        ------
        ValueError
            If the nominal value of an estimated quantity is unknown.
        """
        from lunarops.classes.dynamics.parameters import LunarDynamicsParameters

        missing = [name for name in self.quantities if self.nominal[name] is None]
        if missing:
            raise ValueError(
                f"The nominal values of {missing} are unknown; give inputFileLunarDynamicsParameters of the "
                "lunarDynamics parametrization."
            )
        updated = {name: cast(float, self.nominal[name]) + self.values[name] for name in self.quantities}
        if self.parameters is not None:
            return self.parameters.with_values(updated)
        return LunarDynamicsParameters(updated, self.epoch_tdb_jd1, self.epoch_tdb_jd2)

    def state(self) -> dict[str, object]:
        """Return the corrections and, per quantity, its unit, nominal value, correction and updated value."""
        return {
            "epochTdbJd1": self.epoch_tdb_jd1,
            "epochTdbJd2": self.epoch_tdb_jd2,
            "producer": self.producer,
            "values": {name: float(self.values[name]) for name in self.quantities},
            "quantities": {
                name: {
                    "unit": definition.unit,
                    "nominal": self.nominal[name],
                    "correction": float(self.values[name]),
                    "updated": None if (nominal := self.nominal[name]) is None else nominal + self.values[name],
                }
                for name, definition in zip(self.quantities, self._definitions, strict=True)
            },
        }


__all__ = ["LunarDynamicsParametrization", "euler_to_pa_rotation", "interpolate_regular_rows"]
