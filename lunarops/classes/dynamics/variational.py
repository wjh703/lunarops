"""Variational equations of the lunar dynamics.

The integration state ``y`` (dimension ``d``) obeys a delay differential
equation: the right-hand side ``f`` depends on the current state, on the
state at the delayed (or advanced) epochs of the Earth tide and on the state
and its derivative at the delayed epoch of the lunar deformation. The partials
``W = [Φ | S]`` of the state with respect to the initial state
(``Φ = ∂y/∂y₀``, ``d x d``) and to dynamical parameters (``S = ∂y/∂p``)
then obey

    W' = A W + D,
    D  = Σ_j B_j W(t - τ_j) + B_M W(t - τ_M) + C_M W'(t - τ_M) + [0 | ∂f/∂p],

with ``A = ∂f/∂y`` at fixed step payloads, ``B`` and ``C`` the partials of
``f`` with respect to the delayed states and derivatives (through the step
payloads), ``W(t₀) = [I | 0]``.

:class:`VariationalEquations` advances ``W`` beside the state, on the same
grid and with the same schemes: DOP853 on the seed arc and the start-up
steps, the predictor-corrector Adams formulas afterwards, the same Hermite
and Lagrange stencils for the delayed arguments. It only reads the state and
the accepted history, so a propagation with it produces the same state as one
without. :class:`LocalJacobian` evaluates ``A``, ``B`` and ``C`` by central
differences of the native right-hand side at one epoch.

References
----------
Montenbruck & Gill (2000), *Satellite Orbits*, ch. 7; Tapley, Schutz & Born
(2004), *Statistical Orbit Determination*, ch. 4.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from lunarops._dynamics_core import KERNEL_EVAL_FIGURE, KERNEL_EVAL_FULL, KERNEL_EVAL_TIDE

from .context import StepData
from .history import (
    _TRANSLATION_EXTRAPOLATION_NODES,
    LunarNodeBuffer,
    _is_behind,
    hermite_evaluate,
    hermite_stencil,
)
from .trajectory import interpolate_regular

if TYPE_CHECKING:
    from lunarops.classes.time import Epoch

    from .integrators import IntegratorSettings
    from .lunar_dynamics import LunarDynamics


@dataclass(frozen=True, slots=True)
class DifferenceSteps:
    """Increments of the central differences of the right-hand side.

    They depend only on the scale over which the right-hand side varies with
    each kind of argument, not on the arc length.

    Attributes
    ----------
    position_m : float
        Increment of a relative position component, in m.
    velocity_mps : float
        Increment of a relative velocity component, in m s⁻¹.
    angle_rad : float
        Increment of an Euler angle, in rad.
    angular_velocity_rad_s : float
        Increment of an angular velocity component, in rad s⁻¹.
    acceleration_mps2 : float
        Increment of a delayed relative acceleration component, in m s⁻².
    angular_acceleration_rad_s2 : float
        Increment of a delayed angular acceleration component, in rad s⁻².
    """

    position_m: float = 1.0e3
    velocity_mps: float = 1.0e-2
    angle_rad: float = 3.0e-4
    angular_velocity_rad_s: float = 1.0e-8
    acceleration_mps2: float = 1.0e-6
    angular_acceleration_rad_s2: float = 1.0e-10

    def state(self, dimension: int) -> np.ndarray:
        """Return the increment of every state component, shape ``(dimension,)``."""
        steps = [self.position_m] * 3 + [self.velocity_mps] * 3
        if dimension > 6:
            steps += [self.angle_rad] * 3 + [self.angular_velocity_rad_s] * (dimension - 9)
        return np.asarray(steps, dtype=float)

    def derivative(self, dimension: int) -> np.ndarray:
        """Return the increment of every state-derivative component, shape ``(dimension,)``."""
        steps = [self.velocity_mps] * 3 + [self.acceleration_mps2] * 3
        if dimension > 6:
            steps += [self.angular_velocity_rad_s] * 3 + [self.angular_acceleration_rad_s2] * (dimension - 9)
        return np.asarray(steps, dtype=float)


@dataclass(frozen=True)
class ParameterVariation:
    """A dynamical parameter of the variational equations, as two dynamics built on either side of it.

    The partial ``∂f/∂p`` is the central difference of the right-hand sides of ``plus`` and ``minus`` at the
    state and the accepted history of the propagated (nominal) dynamics. Every path of the parameter into
    the right-hand side, the step payloads included, is then differenced the way the models implement it.

    Attributes
    ----------
    name : str
        Name of the parameter.
    increment : float
        Half-width of the difference, in the unit of the parameter.
    plus, minus : LunarDynamics
        Dynamics with the parameter at its nominal value plus and minus ``increment``; everything else as in
        the nominal dynamics.
    """

    name: str
    increment: float
    plus: LunarDynamics
    minus: LunarDynamics


class _Sibling:
    """One of the two dynamics of a :class:`ParameterVariation`, evaluated on the nominal state and history.

    The parts of the step payload that do not depend on the parameter are taken from the nominal step
    instead of being evaluated again; which parts do is found by comparing a full evaluation with the
    nominal one, separately for the rigid seed model, and checked again from time to time. In the same way,
    a parameter that changes only the Moon figure force and the rotation, or only the Earth tide force, is
    evaluated with those terms alone: the others are equal in the two dynamics and drop out of the difference.
    """

    _RECHECK = 4096

    def __init__(self, system: LunarDynamics, nominal: LunarDynamics) -> None:
        self.system = system
        own = np.asarray(system.gravitational_parameters_m3_s2[:2], dtype=float)
        self._gm = None if np.array_equal(own, nominal.gravitational_parameters_m3_s2[:2]) else own
        self._batch = None
        self._batch_view = None
        self._zero = np.zeros(int(system._fused.kernel.state_dimension))
        self._reuse: dict[bool, tuple[bool, bool]] = {}
        self._modes: dict[bool, int] = {}
        self._nominal = nominal._fused
        self._calls = 0

    def _view(self, batch):
        if self._gm is None:
            return batch
        if batch is not self._batch:
            gm = batch.gravitational_parameters_m3_s2.copy()
            gm[:, :2] = self._gm
            self._batch_view = dataclasses.replace(batch, gravitational_parameters_m3_s2=gm)
            self._batch = batch
        return self._batch_view

    @staticmethod
    def _same(left, right) -> bool:
        if left is None or right is None:
            return left is right
        if isinstance(left, tuple):
            return len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right, strict=True))
        if isinstance(left, np.ndarray):
            return np.array_equal(left, right)
        return all(
            np.array_equal(getattr(left, name), getattr(right, name)) for name in ("total", "derivative", "degree2")
        )

    def rate(self, epoch, batch, index: int, step: StepData, y, history, rigid: bool):
        """Return the right-hand side of this dynamics as a double-double pair."""
        system = self.system
        view = self._view(batch)
        self._calls += 1
        reuse = self._reuse.get(rigid)
        if reuse is None or self._calls % self._RECHECK == 0:
            with _unchanged_counters(history):
                own = system.prepare_step(epoch, view, history, index=index, rigid=rigid)
            payloads_same = own.force_payloads.keys() == step.force_payloads.keys() and all(
                self._same(own.force_payloads[name], step.force_payloads[name]) for name in own.force_payloads
            )
            found = (payloads_same, self._same(own.lunar_inertia, step.lunar_inertia))
            if reuse is not None and ((reuse[0] and not found[0]) or (reuse[1] and not found[1])):
                raise RuntimeError("A parameter began to change a step payload it did not change before")
            self._reuse[rigid] = found if reuse is None else (reuse[0] and found[0], reuse[1] and found[1])
            mode, previous = self._term_mode(view, batch, index, own, step, y), self._modes.get(rigid)
            if previous is not None and previous != KERNEL_EVAL_FULL and mode != previous:
                raise RuntimeError("A parameter began to change a force term it did not change before")
            self._modes[rigid] = mode if previous is None else previous
        else:
            payloads, inertia = step.force_payloads, step.lunar_inertia
            with _unchanged_counters(history):
                if not reuse[0]:
                    payloads = system._fused.prepare_step(epoch, view, index, history)
                if not reuse[1]:
                    assert system.rotation is not None
                    inertia = system.rotation.deformation.evaluate(epoch, history, rigid=rigid)
            own = StepData(payloads, inertia)
        return system._fused.split_derivative(view, index, own, y, self._zero, self._modes[rigid])

    def _term_mode(self, view, batch, index: int, own: StepData, step: StepData, y) -> int:
        """Return the kernel mode that evaluates every term in which this dynamics differs from the nominal."""
        if self._gm is not None:
            return KERNEL_EVAL_FULL
        mine, nominal = self.system._fused, self._nominal
        mine._load(view, index, own)
        ours = {key: np.array(value) for key, value in mine.kernel.terms(np.ascontiguousarray(y)).items()}
        nominal._load(batch, index, step)
        theirs = nominal.kernel.terms(np.ascontiguousarray(y))

        def forces_equal_except(row: int) -> bool:
            return all(
                np.array_equal(np.delete(ours[key], row, axis=0), np.delete(theirs[key], row, axis=0))
                for key in ("forces", "earth_forces")
            )

        # Kernel rows: 4 is the Moon figure force, 5 the Earth tide force.
        if forces_equal_except(4):
            return KERNEL_EVAL_FIGURE
        rotation_equal = all(np.array_equal(ours[key], theirs[key]) for key in ("torques", "rotation", "inertia"))
        if (
            forces_equal_except(5)
            and rotation_equal
            and np.array_equal(ours["derivative"][6:], theirs["derivative"][6:])
        ):
            return KERNEL_EVAL_TIDE
        return KERNEL_EVAL_FULL


class LocalJacobian:
    """Partials of the right-hand side at one epoch, by central differences.

    Parameters
    ----------
    system : LunarDynamics
        Dynamics whose native right-hand side is differenced.
    steps : DifferenceSteps or None, optional
        Increments; the defaults when ``None``.
    """

    def __init__(self, system: LunarDynamics, steps: DifferenceSteps | None = None) -> None:
        self._system = system
        self._fused = system._fused
        self.dimension = int(self._fused.kernel.state_dimension)
        steps = DifferenceSteps() if steps is None else steps
        self._state_steps = steps.state(self.dimension)
        self._derivative_steps = steps.derivative(self.dimension)
        self._zero = np.zeros(self.dimension)
        tides = [model for model in system.force_group.active_models if hasattr(model, "tide_model")]
        self._tide_name = tides[0].name if tides else None
        self.tide = tides[0].tide_model if tides else None
        if self.tide is not None:
            earth = system.earth_body.gravitational_parameter_m3_s2
            moon = system.moon_body.gravitational_parameter_m3_s2
            # A raiser other than the Moon is referred to the Earth at EMB - GM_M/(GM_E+GM_M) r, so its
            # geocentric position moves by that fraction of a change of the delayed relative position r.
            self._raiser_share = np.asarray(
                [
                    1.0 if name == "MOON" else 0.0 if name == "EARTH" else moon / (earth + moon)
                    for name in self.tide.tide_raisers
                ]
            )
        rotation = system.rotation
        self.deformation = None if rotation is None else rotation.deformation
        self._deformation_inputs: tuple[np.ndarray, np.ndarray] | None = None

    def rate(self, batch, index: int, step: StepData, y: np.ndarray) -> np.ndarray:
        """Return the state derivative ``f(y)`` at fixed step payloads, shape ``(d,)``."""
        high, low = self._fused.split_derivative(batch, index, step, y, self._zero)
        return high + low

    def _difference(
        self,
        batch,
        index: int,
        plus: StepData,
        minus: StepData,
        y_plus,
        y_minus,
        step: float,
        mode: int = KERNEL_EVAL_FULL,
    ):
        high_plus, low_plus = self._fused.split_derivative(batch, index, plus, y_plus, self._zero, mode)
        high_minus, low_minus = self._fused.split_derivative(batch, index, minus, y_minus, self._zero, mode)
        return ((high_plus - high_minus) + (low_plus - low_minus)) / (2.0 * step)

    def state_jacobian(self, batch, index: int, step: StepData, y: np.ndarray) -> np.ndarray:
        """Return ``A = ∂f/∂y`` at fixed step payloads.

        Parameters
        ----------
        batch : EpochBatch
            Prescribed inputs.
        index : int
            Row of the epoch in ``batch``.
        step : StepData
            Step payloads, held fixed.
        y : numpy.ndarray
            Integration state, shape ``(d,)``.

        Returns
        -------
        numpy.ndarray
            Jacobian of shape ``(d, d)``.
        """
        return self._fused.state_jacobian(batch, index, step, y, self._state_steps)

    def tide_jacobians(self, batch, index: int, step: StepData, y: np.ndarray) -> np.ndarray:
        """Return the partials of ``f`` with respect to the delayed relative positions of the Earth tide.

        Parameters
        ----------
        batch, index, step, y
            As for :meth:`state_jacobian`.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(3, d, 3)``: for each tidal order ``j``, ``∂f/∂r(t - τ_orb,j)`` with the delayed
            position in ICRF axes. The raiser vectors are linear in it (the lagged frame rotation; the other
            raisers through the Earth's place about the barycentre), so only the dependence of ``f`` on the
            vectors is differenced.
        """
        assert self.tide is not None and self._tide_name is not None
        vectors, _ = cast(tuple, step.force_payloads[self._tide_name])
        inertial2fixed = np.ascontiguousarray(batch.earth_fixed2inertial_matrix[index].T)
        lagged = self.tide.lagged_frames(inertial2fixed)
        increment = self._state_steps[0]
        shifts = np.empty((3, 3, vectors.size))
        for order in range(3):
            for axis in range(3):
                shift = np.zeros_like(vectors)
                shift[:, order] = increment * self._raiser_share[:, None] * lagged[order][:, axis][None, :]
                shifts[order, axis] = shift.ravel()
        return self._fused.tide_jacobian(batch, index, step, y, shifts, increment)

    def _active_deformation_inputs(self, state: np.ndarray, state_dot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the components of the delayed state and of its derivative that the deformation depends on."""
        if self._deformation_inputs is None:
            deformation = self.deformation
            assert deformation is not None
            reference = deformation.evaluate_sample(state, state_dot)

            def changes(sample_state, sample_dot) -> bool:
                value = deformation.evaluate_sample(sample_state, sample_dot)
                return not (
                    np.array_equal(value.total, reference.total)
                    and np.array_equal(value.derivative, reference.derivative)
                    and np.array_equal(value.degree2, reference.degree2)
                )

            active_state, active_dot = [], []
            for component in range(self.dimension):
                shifted = state.copy()
                shifted[component] += self._state_steps[component]
                if changes(shifted, state_dot):
                    active_state.append(component)
                shifted = state_dot.copy()
                shifted[component] += self._derivative_steps[component]
                if changes(state, shifted):
                    active_dot.append(component)
            self._deformation_inputs = (np.asarray(active_state, dtype=int), np.asarray(active_dot, dtype=int))
        return self._deformation_inputs

    def deformation_jacobians(self, batch, index: int, step: StepData, y: np.ndarray, state, state_dot):
        """Return the partials of ``f`` with respect to the delayed state and derivative of the deformation.

        Parameters
        ----------
        batch, index, step, y
            As for :meth:`state_jacobian`.
        state, state_dot : numpy.ndarray
            Integration state and its derivative at ``t - τ_M``, shape ``(d,)``, from which the lunar inertia
            of ``step`` was evaluated.

        Returns
        -------
        delayed_state, delayed_derivative : numpy.ndarray
            ``B_M = ∂f/∂y(t - τ_M)`` and ``C_M = ∂f/∂y'(t - τ_M)``, each of shape ``(d, d)``; columns of
            components the deformation does not depend on are zero.
        """
        assert self.deformation is not None
        active_state, active_dot = self._active_deformation_inputs(state, state_dot)
        partial_state = np.zeros((self.dimension, self.dimension))
        partial_dot = np.zeros((self.dimension, self.dimension))
        for target, active, increments, vary_state in (
            (partial_state, active_state, self._state_steps, True),
            (partial_dot, active_dot, self._derivative_steps, False),
        ):
            base = state if vary_state else state_dot
            for component in active:
                increment = increments[component]
                plus, minus = base.copy(), base.copy()
                plus[component] += increment
                minus[component] -= increment
                if vary_state:
                    inertia_plus = self.deformation.evaluate_sample(plus, state_dot)
                    inertia_minus = self.deformation.evaluate_sample(minus, state_dot)
                else:
                    inertia_plus = self.deformation.evaluate_sample(state, plus)
                    inertia_minus = self.deformation.evaluate_sample(state, minus)
                target[:, component] = self._difference(
                    batch,
                    index,
                    StepData(step.force_payloads, inertia_plus),
                    StepData(step.force_payloads, inertia_minus),
                    y,
                    y,
                    increment,
                    KERNEL_EVAL_FIGURE,
                )
        return partial_state, partial_dot


class _PartialStore:
    """Sliding window of the partials and their derivatives at the latest nodes.

    Columns are addressed by the index of the node in the node buffer of the arc. Only the latest ``window``
    nodes are kept: a century of nodes would take gigabytes.
    """

    def __init__(self, rows: int, window: int) -> None:
        self._window = window
        self._values = np.empty((rows, 4 * window))
        self._derivatives = np.empty_like(self._values)
        self._base = 0
        self._count = 0

    @property
    def size(self) -> int:
        """int: Number of nodes appended so far (the index of the next node)."""
        return self._base + self._count

    @property
    def start(self) -> int:
        """int: Index of the oldest node still held."""
        return self._base

    def append(self, values: np.ndarray, derivatives: np.ndarray) -> None:
        if self._count == self._values.shape[1]:
            keep = self._window
            self._values[:, :keep] = self._values[:, self._count - keep : self._count]
            self._derivatives[:, :keep] = self._derivatives[:, self._count - keep : self._count]
            self._base += self._count - keep
            self._count = keep
        self._values[:, self._count] = values
        self._derivatives[:, self._count] = derivatives
        self._count += 1

    def replace_last_derivative(self, derivatives: np.ndarray) -> None:
        self._derivatives[:, self._count - 1] = derivatives

    def _local(self, nodes: slice) -> slice:
        start, stop, stride = nodes.indices(self.size)
        if start < self._base:
            raise RuntimeError("A delayed argument of the variational equations reaches before the stored window")
        return slice(start - self._base, stop - self._base, stride)

    def values(self, nodes: slice, rows: slice = slice(None)) -> np.ndarray:
        return self._values[rows, self._local(nodes)]

    def derivatives(self, nodes: slice, rows: slice = slice(None)) -> np.ndarray:
        return self._derivatives[rows, self._local(nodes)]

    def latest(self) -> np.ndarray:
        return self._values[:, self._count - 1]

    def derivative_window(self, count: int) -> np.ndarray:
        if self._count < count:
            raise RuntimeError("The variational equations hold too few derivative nodes")
        return self._derivatives[:, self._count - count : self._count]


@contextmanager
def _unchanged_counters(history):
    """Restore the query counters of ``history`` afterwards: the partials must not alter what a run reports."""
    names = ("extrapolated_translation_queries", "max_translation_extrapolation_s")
    saved = {name: getattr(history, name) for name in names if hasattr(history, name)}
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(history, name, value)


class _SharedGeocentric:
    """An accepted history whose geocentric positions are computed once for the siblings of one epoch.

    The siblings are evaluated on the nominal history, so their delayed raiser positions are those of the
    nominal step unless a parameter moves the delayed epochs; a parameter that changes only the rotational
    lag of the tide (τR1, τR2) or the GM then needs no history query of its own. Every other attribute,
    assignments included, is that of the wrapped history.
    """

    def __init__(self, history) -> None:
        object.__setattr__(self, "_history", history)
        object.__setattr__(self, "_memo", {})

    def geocentric_positions(self, body_names, epochs) -> np.ndarray:
        names, epochs = tuple(body_names), tuple(epochs)
        key = (names, tuple((epoch.jd1, epoch.jd2, epoch.scale) for epoch in epochs))
        positions = self._memo.get(key)
        if positions is None:
            positions = self._memo[key] = self._history.geocentric_positions(names, epochs)
        return positions.copy()

    def __getattr__(self, name):
        return getattr(self._history, name)

    def __setattr__(self, name, value) -> None:
        setattr(self._history, name, value)


class VariationalEquations:
    """Partials of the propagated state, advanced beside it by the integrator.

    Pass an instance to
    :meth:`~lunarops.classes.dynamics.integrators.AdamsBashforthMoultonIntegrator.integrate` as
    ``variational``; after the propagation :attr:`times` and :attr:`partials` hold the result.

    Parameters
    ----------
    system : LunarDynamics
        Dynamics being propagated.
    settings : IntegratorSettings
        Settings of the integrator.
    steps : DifferenceSteps or None, optional
        Increments of the local central differences; the defaults when ``None``.
    parameters : sequence of ParameterVariation, optional
        Dynamical parameters whose partials ``S = ∂y/∂p`` are propagated, in column order.
    output_stride : int, default 1
        Keep the partials of every ``output_stride``-th node, counted from the initial epoch.

    Attributes
    ----------
    dimension : int
        State dimension ``d``.
    columns : int
        Number of columns of ``W``: ``d`` initial-state partials followed by the parameter partials.
    parameter_names : tuple of str
        Names of the parameter columns.
    """

    def __init__(
        self,
        system: LunarDynamics,
        settings: IntegratorSettings,
        *,
        steps: DifferenceSteps | None = None,
        parameters: Sequence[ParameterVariation] = (),
        output_stride: int = 1,
    ) -> None:
        if output_stride < 1:
            raise ValueError("output_stride must be positive")
        self._settings = settings
        self._steps = steps
        self.parameter_names = tuple(variation.name for variation in parameters)
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("Parameter variations must have distinct names")
        self._increments = tuple(float(variation.increment) for variation in parameters)
        if any(increment <= 0 for increment in self._increments):
            raise ValueError("Parameter increments must be positive")
        self._variations: Sequence[ParameterVariation] | None = tuple(parameters)
        self._output_stride = int(output_stride)
        self._initial_epoch: Epoch | None = None
        self._output_times: list[float] = []
        self._output: list[np.ndarray] = []
        self._backward_times: list[float] = []
        self._backward_output: list[np.ndarray] = []
        self._node_index = 0
        self._initial_node = 0
        self._mirror: tuple[np.ndarray, np.ndarray] | None = None
        self.bind(system)

    def bind(self, system: LunarDynamics, parameters: Sequence[ParameterVariation] | None = None) -> None:
        """Evaluate the right-hand side with ``system`` (the compiled dynamics do not pickle).

        Parameters
        ----------
        system : LunarDynamics
            Dynamics being propagated.
        parameters : sequence of ParameterVariation or None, optional
            Parameter variations built for ``system``; those given at construction when ``None``.

        Raises
        ------
        ValueError
            If the variations are not those of the parameter columns (names and increments).
        """
        settings = self._settings
        variations = self._variations if parameters is None else tuple(parameters)
        if variations is None:
            if self.parameter_names:
                raise ValueError("The partials have parameter columns; bind() needs their parameter variations")
            variations = ()
        if (
            tuple(variation.name for variation in variations) != self.parameter_names
            or tuple(float(variation.increment) for variation in variations) != self._increments
        ):
            raise ValueError("The parameter variations differ from the parameter columns of the partials")
        self._variations = variations
        self._siblings = [(_Sibling(v.plus, system), _Sibling(v.minus, system)) for v in variations]
        self._system = system
        self._jacobian = LocalJacobian(system, self._steps)
        self.dimension = self._jacobian.dimension
        self.columns = self.dimension + len(variations)
        delays = [0.0]
        if self._jacobian.tide is not None:
            delays += [abs(delay) for delay in self._jacobian.tide.delay_seconds]
        if self._jacobian.deformation is not None:
            delays.append(self._jacobian.deformation.parameters.delay_s)
        # Nodes a delayed argument can reach back to: the delay, the stencil around it, and on a backward
        # arc the widened extrapolation stencil of the deformation.
        reach = 4 * int(np.ceil(max(delays) / settings.step_s)) + settings.history_interpolation_order + 8
        self._store = _PartialStore(self.dimension * self.columns, max(64, settings.order + 1, reach))
        self._joint_delay = self._jacobian.deformation is not None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Rebuilt by bind() in the receiver: the compiled dynamics do not pickle.
        for name in ("_system", "_jacobian", "_store", "_variations", "_siblings"):
            state[name] = None
        return state

    def for_backward_arc(self, mirrored: int) -> VariationalEquations:
        """Return the partials of the backward arc of a two-sided propagation, seeded by the forward nodes.

        Call it once the first ``mirrored`` forward nodes, from the initial epoch on, are accepted. The
        result carries their partials and no dynamics: the process that propagates the backward arc calls
        :meth:`bind` and :meth:`start_backward`.

        Parameters
        ----------
        mirrored : int
            Number of forward nodes that seed the backward arc.

        Returns
        -------
        VariationalEquations
            Unbound partials of the backward arc.
        """
        nodes = slice(self._initial_node, self._initial_node + mirrored)
        other = object.__new__(VariationalEquations)
        other.__dict__.update(self.__getstate__())
        other._variations = self._variations  # used when the backward arc runs in this process
        other._output_times, other._output = [], []
        other._backward_times, other._backward_output = [], []
        other._mirror = (self._store.values(nodes).copy(), self._store.derivatives(nodes).copy())
        return other

    def start_backward(self) -> None:
        """Load the mirrored forward nodes, newest first, as the backward node buffer stores them."""
        assert self._mirror is not None, "for_backward_arc() makes the partials of a backward arc"
        values, derivatives = self._mirror
        for column in range(values.shape[1] - 1, -1, -1):
            self._store.append(values[:, column], derivatives[:, column])
        self._initial_node = self._store.size - 1
        self._node_index = 1  # the initial node is recorded by the forward arc

    def merge_backward(self, other: VariationalEquations) -> None:
        """Take over the partials ``other`` recorded on the backward arc."""
        self._backward_times = other._output_times
        self._backward_output = other._output

    # ----------------------------------------------------------------- results

    @property
    def times(self) -> np.ndarray:
        """numpy.ndarray: Offsets from the initial epoch of the stored partials, in s, increasing."""
        return np.asarray(self._backward_times[::-1] + self._output_times)

    @property
    def partials(self) -> np.ndarray:
        """numpy.ndarray: ``W`` at :attr:`times`, shape ``(n, d, columns)``."""
        output = self._backward_output[::-1] + self._output
        return np.asarray(output).reshape(len(output), self.dimension, self.columns)

    # ---------------------------------------------------------------- forcing

    def _parameter_forcing(self, epoch, batch, index, step, y, history, rigid) -> np.ndarray | None:
        """Return ``∂f/∂p`` for the parameter columns, shape ``(d, columns - d)``; ``None`` without parameters."""
        if not self._siblings:
            return None
        forcing = np.empty((self.dimension, len(self._siblings)))
        history = _SharedGeocentric(history)
        for column, ((plus, minus), increment) in enumerate(zip(self._siblings, self._increments, strict=True)):
            high_plus, low_plus = plus.rate(epoch, batch, index, step, y, history, rigid)
            high_minus, low_minus = minus.rate(epoch, batch, index, step, y, history, rigid)
            forcing[:, column] = ((high_plus - high_minus) + (low_plus - low_minus)) / (2.0 * increment)
        return forcing

    def _delayed_position_partials(self, nodes: LunarNodeBuffer, offset: float) -> np.ndarray | None:
        """Return the partials of the relative position the history gives at ``offset``, shape ``(3, columns)``.

        The branches are those of :meth:`DelayedHistory.relative_state`; ``None`` where the history answers
        from the ephemeris, which does not depend on the initial state or the parameters.
        """
        times, _, _ = nodes.integration_data()
        rows = slice(0, 3 * self.columns)
        order = self._settings.history_interpolation_order
        ahead = (offset - times[-1]) if times[-1] >= times[0] else (times[-1] - offset)
        if ahead > 1e-9 or not self._joint_delay:
            if ahead <= 1e-9 and _is_behind(times, offset):
                return None
            window = slice(self._store.start, len(times))
            if ahead > 1e-9:
                order = min(order, _TRANSLATION_EXTRAPOLATION_NODES - 1)
            sample = interpolate_regular(
                times[window],
                self._store.values(window, rows),
                offset,
                order,
                extrapolate=ahead > 1e-9,
                check_grid=False,
            )
        else:
            selected = hermite_stencil(times, nodes.trajectory_start, offset, order)
            sample = hermite_evaluate(
                times[selected], self._store.values(selected, rows), self._store.derivatives(selected, rows), offset
            )[0]
        return sample.reshape(3, self.columns)

    def _forcing(self, epoch, batch, index, step: StepData, y, nodes: LunarNodeBuffer | None, history, rigid: bool):
        """Return ``A`` and ``D`` at one epoch.

        ``nodes`` is the node buffer of the accepted history that ``step`` was evaluated from, or ``None``
        when the delayed arguments were read from an external history (the seed and warm-up arcs), which
        does not depend on the initial state.
        """
        jacobian = self._jacobian
        state_jacobian = jacobian.state_jacobian(batch, index, step, y)
        forcing = np.zeros((self.dimension, self.columns))
        if nodes is not None:
            assert self._initial_epoch is not None
            if jacobian.tide is not None:
                partials = jacobian.tide_jacobians(batch, index, step, y)
                for order, delay in enumerate(jacobian.tide.delay_seconds):
                    offset = float(self._initial_epoch.seconds_until(epoch.shifted(delay)))
                    sample = self._delayed_position_partials(nodes, offset)
                    if sample is not None:
                        forcing += partials[order] @ sample
            if jacobian.deformation is not None and not rigid:
                delay = jacobian.deformation.parameters.delay_s
                offset = float(self._initial_epoch.seconds_until(epoch.shifted(-delay)))
                times, states, derivatives = nodes.integration_data()
                selected = hermite_stencil(
                    times, nodes.trajectory_start, offset, self._settings.history_interpolation_order
                )
                grid = times[selected]
                state, state_dot = hermite_evaluate(grid, states[:, selected], derivatives[:, selected], offset)
                partial_state, partial_dot = jacobian.deformation_jacobians(batch, index, step, y, state, state_dot)
                sample, sample_dot = hermite_evaluate(
                    grid, self._store.values(selected), self._store.derivatives(selected), offset
                )
                forcing += partial_state @ sample.reshape(self.dimension, self.columns)
                forcing += partial_dot @ sample_dot.reshape(self.dimension, self.columns)
        parameter = self._parameter_forcing(epoch, batch, index, step, y, history, rigid)
        if parameter is not None:
            forcing[:, self.dimension :] += parameter
        return state_jacobian, forcing

    # ------------------------------------------------------------- integration

    def _augmented_rhs(self, prefetch, history, nodes, *, rigid: bool, origin: float = 0.0):
        """Return the DOP853 right-hand side of ``[y, W]`` at ``origin + local offset``."""
        system, dimension, columns = self._system, self.dimension, self.columns
        initial_epoch = cast("Epoch", self._initial_epoch)

        def rhs(local_offset: float, vector: np.ndarray) -> np.ndarray:
            offset = origin + local_offset
            batch, index = prefetch.batch(offset)
            epoch = initial_epoch.shifted(offset)
            y = np.ascontiguousarray(vector[:dimension])
            with _unchanged_counters(history):
                step = system.prepare_step(epoch, batch, history, index=index, rigid=rigid)
            state_jacobian, forcing = self._forcing(epoch, batch, index, step, y, nodes, history, rigid)
            partials = vector[dimension:].reshape(dimension, columns)
            return np.concatenate(
                (self._jacobian.rate(batch, index, step, y), (state_jacobian @ partials + forcing).ravel())
            )

        return rhs

    def _record(self, offset: float, partials: np.ndarray) -> None:
        if self._node_index % self._output_stride == 0:
            self._output_times.append(float(offset))
            self._output.append(partials.copy())
        self._node_index += 1

    def seed(self, initial_epoch, nodes: LunarNodeBuffer, prefetch, external_history, history, *, rigid: bool) -> None:
        """Fill the partials of the seed or warm-up nodes, which end with the initial-epoch node.

        The augmented state ``[y, W]`` is integrated from the initial epoch, where ``W = [I | 0]``, over the
        nodes before it with the fixed-step DOP853 scheme of the state. The delayed arguments of these arcs
        come from ``external_history`` and carry no partials.

        Parameters
        ----------
        initial_epoch : Epoch
            TDB epoch of offset 0.
        nodes : LunarNodeBuffer
            Seed or warm-up nodes of the state, the initial-epoch node last.
        prefetch : object
            Epoch data of the stage offsets of the seed integration (``batch(offset)``).
        external_history : StateHistoryProvider or None
            History read by the delayed forces on the seed arc.
        history : StateHistoryProvider
            Accepted history of the main grid, over ``nodes``.
        rigid : bool
            Whether the nodes are a rigid seed arc: the seed model is undeformed, and the derivative of the
            initial node is that of the deformed model, read from the seed history.
        """
        from .integrators import dop853_startup

        self._initial_epoch = initial_epoch
        settings = self._settings
        dimension, columns = self.dimension, self.columns
        times, states, _ = nodes.integration_data()
        initial = np.zeros((dimension, columns))
        initial[:, :dimension] = np.eye(dimension)
        count = len(times)
        samples = initial.ravel()[:, None]
        if count > 1:
            vector = np.concatenate((states[:, -1], initial.ravel()))
            samples, _ = dop853_startup(
                self._augmented_rhs(prefetch, external_history, None, rigid=rigid),
                vector,
                float(times[0]),
                step_s=cast(float, settings.startup_step_s),
                sample_step_s=settings.step_s,
                prefetch=prefetch,
            )
            # Samples run from the initial epoch outwards; the nodes are stored oldest first.
            samples = samples[dimension:, ::-1]
        for column in range(count):
            offset = float(times[column])
            epoch = initial_epoch.shifted(offset)
            batch, index = prefetch.batch(offset)
            y = np.ascontiguousarray(states[:, column])
            partials = samples[:, column].reshape(dimension, columns)
            last = column == count - 1
            if last:
                partials = initial
            if last and rigid:
                # The deformed model at the initial epoch reads the seed nodes, whose partials are stored.
                self._store.append(partials.ravel(), np.zeros(dimension * columns))
                with _unchanged_counters(history):
                    step = self._system.prepare_step(epoch, batch, history, index=index)
                state_jacobian, forcing = self._forcing(epoch, batch, index, step, y, nodes, history, False)
                self._store.replace_last_derivative((state_jacobian @ partials + forcing).ravel())
            else:
                step = self._system.prepare_step(epoch, batch, external_history, index=index, rigid=rigid)
                state_jacobian, forcing = self._forcing(epoch, batch, index, step, y, None, external_history, rigid)
                self._store.append(partials.ravel(), (state_jacobian @ partials + forcing).ravel())
        self._initial_node = self._store.size - 1
        self._record(0.0, initial)

    def dop853_step(self, prefetch, origin: float, duration: float, epoch, offset, batch, index, state, nodes, history):
        """Advance the partials over one DOP853 start-up step of the main grid.

        Called before the new node is appended to ``nodes``, so that the delayed arguments see the history
        the state saw.

        Parameters
        ----------
        prefetch : object
            Epoch data of the stage offsets of the step.
        origin : float
            Offset of the previous node, in s.
        duration : float
            Signed length of the step, in s.
        epoch : Epoch
            Epoch of the new node.
        offset : float
            Offset of the new node, in s.
        batch : EpochBatch
            Prescribed inputs containing ``epoch``.
        index : int
            Row of ``epoch`` in ``batch``.
        state : numpy.ndarray
            Accepted state of the new node, shape ``(d,)``.
        nodes : LunarNodeBuffer
            Accepted nodes, without the new one.
        history : StateHistoryProvider
            Accepted history over ``nodes``.
        """
        from .integrators import dop853_startup

        dimension, columns = self.dimension, self.columns
        vector = np.concatenate((nodes.state(), self._store.latest()))
        samples, _ = dop853_startup(
            self._augmented_rhs(prefetch, history, nodes, rigid=False, origin=origin),
            vector,
            duration,
            step_s=cast(float, self._settings.startup_step_s),
            sample_step_s=self._settings.step_s,
        )
        partials = samples[dimension:, -1].reshape(dimension, columns)
        y = np.ascontiguousarray(state)
        with _unchanged_counters(history):
            step = self._system.prepare_step(epoch, batch, history, index=index)
        state_jacobian, forcing = self._forcing(epoch, batch, index, step, y, nodes, history, False)
        self._store.append(partials.ravel(), (state_jacobian @ partials + forcing).ravel())
        self._record(offset, partials)

    def abm_step(self, epoch, offset, batch, index, step: StepData, state, nodes, history, step_s, weights) -> None:
        """Advance the partials over one Adams-Bashforth-Moulton step, with the scheme of the state.

        ``A`` and ``D`` are evaluated once, at the accepted state: ``D`` reads accepted history only, and the
        change of ``A`` between the predicted and the corrected state is of the order of the corrector
        change of the state.

        Parameters
        ----------
        epoch, offset, batch, index, state, nodes, history
            As for :meth:`dop853_step`.
        step : StepData
            Step payloads the state step used.
        step_s : float
            Signed step, in s.
        weights : tuple of numpy.ndarray
            Adams-Bashforth and Adams-Moulton weights of the integrator (high parts), newest node first.
        """
        dimension, columns = self.dimension, self.columns
        bashforth, moulton = weights
        y = np.ascontiguousarray(state)
        state_jacobian, forcing = self._forcing(epoch, batch, index, step, y, nodes, history, False)
        window = self._store.derivative_window(self._settings.order)[:, ::-1]
        current = self._store.latest()
        partials = (current + step_s * (window @ bashforth)).reshape(dimension, columns)
        base = current + step_s * (window @ moulton[1:])
        for _ in range(self._settings.corrector_iterations):
            derivative = state_jacobian @ partials + forcing
            partials = (base + step_s * moulton[0] * derivative.ravel()).reshape(dimension, columns)
        self._store.append(partials.ravel(), (state_jacobian @ partials + forcing).ravel())
        self._record(offset, partials)
