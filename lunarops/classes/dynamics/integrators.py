"""Fixed-step Adams-Bashforth-Moulton integration of the lunar dynamics.

The main integrator is a predictor-corrector Adams-Bashforth-Moulton (ABM)
method of fixed order on a regular grid of TDB offsets. The order-1
derivative nodes it needs before the first step come from a fixed-step
DOP853 warm-up: a backward arc for instantaneous dynamics, or an explicit
rigid backward seed arc followed by DOP853 steps for delayed deformation.

The propagated state and its derivative are kept as double-double pairs
(``high + low``). In ordinary double precision the rounding of the state
update and of the dominant Earth-Moon term random-walks into the velocity
and, through the mean motion, into an along-track error growing as
``t^(3/2)``; over 50 years it reaches about a metre. Carrying the low
parts keeps that error well below a millimetre without quadruple precision,
for a negligible extra cost compared with the right-hand side.

A propagation may be two-sided: from one initial epoch the solution is carried
forward and backward, which also halves the longest arc over which the
rounding error accumulates.

Attributes
----------
NodeDiagnosticCallback : type alias
    Signature ``(epoch, state, batch, index, history) -> None`` of the
    per-node diagnostic callback of
    :meth:`AdamsBashforthMoultonIntegrator.integrate`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np

from lunarops._dynamics_core import adams_accumulate
from lunarops.classes.time import Epoch

from .context import StateHistoryProvider
from .epoch_data import EpochBatch, SerialEpochSource
from .history import DelayedHistory, LunarNodeBuffer, TwoSidedHistory
from .trajectory import LunarTrajectory, regular_times

if TYPE_CHECKING:
    from .lunar_dynamics import LunarDynamics


@dataclass(frozen=True, slots=True)
class IntegratorSettings:
    """Settings of the fixed-step integrator.

    Attributes
    ----------
    step_s : float, default 5400.0
        Main grid step in TDB seconds.
    order : int, default 13
        Number of derivative nodes of the Adams-Bashforth predictor.
    corrector_iterations : int, default 2
        Number of Adams-Moulton corrector iterations per step.
    history_interpolation_order : int, default 13
        Interpolation order of delayed-history queries.
    trajectory_interpolation_order : int, default 13
        Interpolation order of the output trajectory.
    startup_step_s : float or None, default None
        DOP853 warm-up step in s; ``step_s / 8`` when ``None``.

    Raises
    ------
    ValueError
        If ``step_s`` is not an integer multiple of ``startup_step_s``.
    """

    step_s: float = 5400.0
    order: int = 13
    corrector_iterations: int = 2
    history_interpolation_order: int = 13
    trajectory_interpolation_order: int = 13
    startup_step_s: float | None = None

    def __post_init__(self) -> None:
        """Resolve the default warm-up step and check the step ratio."""
        startup_step_s = self.step_s / 8 if self.startup_step_s is None else float(self.startup_step_s)
        ratio = self.step_s / startup_step_s
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-12):
            raise ValueError("step_s must be an integer multiple of startup_step_s")
        object.__setattr__(self, "startup_step_s", startup_step_s)


NodeDiagnosticCallback = Callable[
    [Epoch, np.ndarray, EpochBatch, int, StateHistoryProvider | None],
    None,
]

_EPOCH_BATCH_SIZE = 64


class _EpochDataPrefetch:
    def __init__(self, system: LunarDynamics, initial_epoch: Epoch):
        self._system = system
        self._initial_epoch = initial_epoch
        self._data: dict[float, tuple[EpochBatch, int]] = {}

    def __call__(self, offsets) -> None:
        missing = [offset for offset in dict.fromkeys(offsets) if offset not in self._data]
        for start in range(0, len(missing), _EPOCH_BATCH_SIZE):
            chunk = missing[start : start + _EPOCH_BATCH_SIZE]
            epochs = tuple(self._initial_epoch.shifted(offset) for offset in chunk)
            batch = self._system.epoch_batch(epochs)
            self._data.update((offset, (batch, index)) for index, offset in enumerate(chunk))

    def clear(self) -> None:
        self._data.clear()

    def derivatives(self, offset: float, state: np.ndarray, **kwargs) -> np.ndarray:
        batch, index = self._data[offset]
        return self._system.derivatives(self._initial_epoch.shifted(offset), state, batch, index=index, **kwargs)


def dop853_startup(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    duration_s: float,
    *,
    step_s: float,
    sample_step_s: float,
    prefetch: Callable[[list[float]], None] | None = None,
    initial_low: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate with fixed-step DOP853 and sample the result on the ABM grid.

    The state update is compensated (Neumaier), so that the rounding of the
    many small start-up steps does not accumulate into the ABM initial state.

    Parameters
    ----------
    rhs : Callable[[float, numpy.ndarray], numpy.ndarray]
        Right-hand side ``f(offset_s, state)``.
    initial_state : numpy.ndarray
        State at offset 0, shape ``(d,)``.
    duration_s : float
        Signed integration interval in s; an integer multiple of ``step_s``.
    step_s : float
        DOP853 step in s.
    sample_step_s : float
        Output sampling interval in s; an integer multiple of ``step_s``.
    prefetch : Callable[[list of float], None] or None, optional
        Called once with every stage offset before integration, computed
        with the same floating-point expression as the right-hand-side
        calls, so that time-only inputs can be built in batches.
    initial_low : numpy.ndarray or None, optional
        Low part of the initial state; zero when ``None``.

    Returns
    -------
    states, lows : numpy.ndarray
        High and low parts of the states at offsets
        ``0, ±sample_step_s, ...``, each of shape ``(d, m)``.

    Raises
    ------
    ValueError
        If ``duration_s`` is not an integer multiple of ``step_s``.
    """
    from scipy.integrate import DOP853

    times = regular_times(duration_s, step_s)
    if prefetch is not None:
        prefetch(
            [
                float(times[index]) + float(DOP853.C[stage]) * float(times[index + 1] - times[index])
                for index in range(len(times) - 1)
                for stage in range(DOP853.n_stages)
            ]
        )
    sample_stride = round(abs(float(sample_step_s)) / step_s)
    states = np.empty((len(initial_state), len(times)), dtype=float)
    lows = np.zeros_like(states)
    states[:, 0] = initial_state
    if initial_low is not None:
        lows[:, 0] = initial_low
    stages = np.empty((DOP853.n_stages, len(initial_state)), dtype=float)
    for index in range(len(times) - 1):
        t = float(times[index])
        dt = float(times[index + 1] - times[index])
        state = states[:, index]
        for stage in range(DOP853.n_stages):
            stages[stage] = rhs(
                t + float(DOP853.C[stage]) * dt,
                state + dt * (stages[:stage].T @ DOP853.A[stage, :stage]),
            )
        increment = dt * (stages.T @ DOP853.B)
        total = state + increment
        carry = total - state
        low = lows[:, index] + ((state - (total - carry)) + (increment - carry))
        states[:, index + 1] = total + low
        lows[:, index + 1] = low - (states[:, index + 1] - total)
    return states[:, ::sample_stride], lows[:, ::sample_stride]


def _integral_weights(nodes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return exact weights of the Lagrange basis integrated over [0, 1] as double-double."""
    weights = []
    for index, node in enumerate(nodes):
        polynomial = [Fraction(1)]
        denominator = Fraction(1)
        for other_index, other in enumerate(nodes):
            if other_index == index:
                continue
            raised = [Fraction(0), *polynomial]
            polynomial = [a - other * b for a, b in zip(raised, [*polynomial, Fraction(0)], strict=True)]
            denominator *= node - other
        weights.append(sum(coefficient / (power + 1) for power, coefficient in enumerate(polynomial)) / denominator)
    high = [float(weight) for weight in weights]
    low = [float(weight - Fraction(part)) for weight, part in zip(weights, high, strict=True)]
    return np.asarray(high), np.asarray(low)


@lru_cache(maxsize=32)
def _adams_bashforth_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    return _integral_weights(range(0, -order, -1))


@lru_cache(maxsize=32)
def _adams_moulton_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    return _integral_weights([1, *range(0, -order, -1)])


class AdamsBashforthMoultonIntegrator:
    """Fixed-step Adams-Bashforth-Moulton predictor-corrector integrator.

    Each step predicts with the ``order``-node Adams-Bashforth formula and
    applies ``corrector_iterations`` Adams-Moulton corrections, followed by
    one final evaluation so that every stored derivative belongs to the
    accepted state (P(EC)ⁿE). Data that depend only on the epoch and
    accepted history are computed once per step. States, derivatives and
    Adams weights are double-double pairs (see the module docstring).

    Parameters
    ----------
    settings : IntegratorSettings or None, optional
        Integrator settings; the defaults when ``None``.

    Attributes
    ----------
    beta_ab : numpy.ndarray
        Adams-Bashforth coefficients, newest node first, shape ``(order,)``.
    beta_am : numpy.ndarray
        Adams-Moulton coefficients, implicit node first,
        shape ``(order + 1,)``.
    """

    def __init__(self, settings: IntegratorSettings | None = None):
        self.settings = IntegratorSettings() if settings is None else settings
        self._ab_weights = _adams_bashforth_weights(self.settings.order)
        self._am_weights = _adams_moulton_weights(self.settings.order)
        self.beta_ab = self._ab_weights[0]
        self.beta_am = self._am_weights[0]

    def _compute_abm_step(
        self,
        system: LunarDynamics,
        nodes: LunarNodeBuffer,
        history: StateHistoryProvider,
        epoch: Epoch,
        batch: EpochBatch,
        step_s: float,
        index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state, state_low = nodes.state(), nodes.state_low()
        window, window_low = (
            nodes.derivative_window(self.settings.order),
            nodes.derivative_low_window(self.settings.order),
        )
        window, window_low = window[:, ::-1], window_low[:, ::-1]
        corrected, corrected_low = adams_accumulate(step_s, state, state_low, window, window_low, *self._ab_weights)
        am_high, am_low = self._am_weights
        base, base_low = adams_accumulate(step_s, state, state_low, window, window_low, am_high[1:], am_low[1:])
        # Delayed tides and deformation read only accepted history, so they are
        # identical for every corrector iteration of this step.
        step = system.prepare_step(epoch, batch, history, index=index)
        for _ in range(self.settings.corrector_iterations):
            derivative, derivative_low = system.split_derivatives(
                epoch, corrected, corrected_low, batch, index=index, step=step
            )
            corrected, corrected_low = adams_accumulate(
                step_s, base, base_low, derivative[:, None], derivative_low[:, None], am_high[:1], am_low[:1]
            )
        # Every stored derivative must belong to the accepted state, so the
        # last evaluation uses the accepted candidate rather than the final iterate's input.
        derivative, derivative_low = system.split_derivatives(
            epoch, corrected, corrected_low, batch, index=index, step=step
        )
        return corrected, corrected_low, derivative, derivative_low

    @staticmethod
    def _epoch_batches(system: LunarDynamics, initial_epoch: Epoch, offsets: np.ndarray, source=None):
        batches = [
            [float(offset) for offset in offsets[start : start + _EPOCH_BATCH_SIZE]]
            for start in range(0, len(offsets), _EPOCH_BATCH_SIZE)
        ]
        if source is None:
            source = SerialEpochSource(system.epoch_inputs, system.ephemeris)
        for chunk, batch in zip(batches, source.stream(initial_epoch, batches), strict=True):
            for index, offset in enumerate(chunk):
                yield offset, batch, index

    def integrate(
        self,
        system: LunarDynamics,
        initial_epoch: Epoch,
        initial_state: np.ndarray,
        duration_s: float,
        *,
        backward_duration_s: float = 0.0,
        external_history_provider: StateHistoryProvider | None = None,
        node_diagnostic: NodeDiagnosticCallback | None = None,
        seed_duration_s: float | None = None,
        epoch_source=None,
    ) -> LunarTrajectory:
        """Propagate the lunar dynamics over a fixed regular grid.

        Parameters
        ----------
        system : LunarDynamics
            Dynamics to integrate.
        initial_epoch : Epoch
            TDB epoch of ``initial_state``.
        initial_state : numpy.ndarray
            Initial integration state, shape ``(d,)``: Moon-minus-Earth
            position (m) and velocity (m s⁻¹) with ICRF axes, followed by the
            rotational state when rotation is integrated.
        duration_s : float
            Signed propagation interval in TDB seconds; an integer multiple of
            ``settings.step_s``. Delayed deformation requires a positive value.
        backward_duration_s : float, default 0.0
            Non-negative length in s of an additional arc propagated backward
            from ``initial_epoch``; an integer multiple of ``settings.step_s``.
            Requires a positive ``duration_s``. The returned trajectory then
            covers ``-backward_duration_s`` to ``duration_s``.
        external_history_provider : StateHistoryProvider or None, optional
            History used by delayed forces during the warm-up or seed arc.
        node_diagnostic : NodeDiagnosticCallback or None, optional
            Called after every accepted main-grid node and at the initial
            epoch.
        seed_duration_s : float or None, optional
            Length of the rigid backward seed arc in s; required with delayed
            deformation and at least the delay plus
            ``history_interpolation_order`` steps.
        epoch_source : object, optional
            Producer of main-grid epoch batches with a ``stream`` method like
            :class:`~lunarops.classes.dynamics.epoch_data.SerialEpochSource`;
            built in-process when ``None``.

        Returns
        -------
        LunarTrajectory
            Accepted nodes over the propagated interval, in increasing time.

        Raises
        ------
        ValueError
            If the interval is shorter than ``order - 1`` steps or not a
            multiple of the step, if ``backward_duration_s`` is negative or
            combined with a non-positive ``duration_s``, or, with delayed
            deformation, if the propagation is backward, the step exceeds the
            delay, or the seed arc is too short.
        """
        times = regular_times(duration_s, self.settings.step_s)
        startup = self.settings.order - 1
        if len(times) - 1 < startup:
            raise ValueError(f"Propagation needs at least order-1 = {startup} steps; got {len(times) - 1}")
        if backward_duration_s < 0:
            raise ValueError("backward_duration_s must be non-negative")
        if backward_duration_s and duration_s <= 0:
            raise ValueError("A two-sided propagation needs a positive duration_s")
        prefetch = _EpochDataPrefetch(system, initial_epoch)
        seed = (
            self._seed_rigid
            if system.rotation is not None and system.rotation.deformation is not None
            else self._seed_backward
        )
        nodes, history, initial_batch, dop853_steps, trajectory_history = seed(
            system,
            initial_epoch,
            initial_state,
            duration_s,
            times,
            prefetch,
            seed_duration_s,
            external_history_provider,
        )
        if node_diagnostic is not None:
            node_diagnostic(initial_epoch, nodes.state(), initial_batch, 0, history)
        self._main_grid(
            system,
            initial_epoch,
            times,
            nodes,
            history,
            prefetch,
            epoch_source,
            node_diagnostic,
            dop853_steps=dop853_steps,
        )
        output_times, output_states = nodes.trajectory_data()
        if backward_duration_s:
            backward, backward_history = self._backward_arc(
                system, initial_epoch, nodes, backward_duration_s, epoch_source, node_diagnostic
            )
            reverse_times, reverse_states = backward.trajectory_data()
            output_times = np.concatenate((reverse_times[:0:-1], output_times))
            output_states = np.concatenate((reverse_states[:, :0:-1], output_states), axis=1)
            if trajectory_history is not None:
                trajectory_history = TwoSidedHistory(initial_epoch, trajectory_history, backward_history)
        return LunarTrajectory(
            initial_epoch, output_times, output_states, self.settings.trajectory_interpolation_order, trajectory_history
        )

    def _backward_arc(self, system, initial_epoch, forward, duration_s, epoch_source, node_diagnostic):
        """Propagate backward from the initial epoch, seeded by the first forward nodes.

        The mirrored forward nodes are the derivative window of the first
        backward step, so no extra start-up integration is needed. With
        delayed forces the epoch ``t - tau`` lies in the direction of
        integration; it is extrapolated from the latest accepted nodes, which
        is what the forward arc does for the negative Earth-tide delays.
        """
        settings = self.settings
        times = regular_times(-duration_s, settings.step_s)
        startup = settings.order - 1
        nodes = LunarNodeBuffer(startup + len(times), len(forward.state()))
        for index in range(forward.trajectory_start + startup, forward.trajectory_start - 1, -1):
            nodes.append(*forward.node(index))
        nodes.mark_trajectory_start()
        history = DelayedHistory(system, initial_epoch, nodes, settings.history_interpolation_order)
        self._main_grid(
            system, initial_epoch, times, nodes, history, None, epoch_source, node_diagnostic, dop853_steps=0
        )
        return nodes, history

    def _seed_backward(
        self,
        system,
        initial_epoch,
        initial_state,
        duration_s,
        times,
        prefetch,
        seed_duration_s,
        external_history_provider,
    ):
        count = len(times) - 1
        startup = self.settings.order - 1
        nodes = LunarNodeBuffer(count + self.settings.order, len(initial_state))

        initial_data = system.epoch_batch((initial_epoch,))
        initial_derivative, initial_derivative_low = system.split_derivatives(
            initial_epoch,
            initial_state,
            np.zeros_like(initial_state),
            initial_data,
            history=external_history_provider,
        )

        def startup_rhs(offset_s: float, state: np.ndarray) -> np.ndarray:
            return prefetch.derivatives(offset_s, state, history=external_history_provider)

        # Backward DOP853 warm-up supplies the order-1 derivative nodes before the start.
        direction = np.copysign(1.0, duration_s)
        warmup = dop853_startup(
            startup_rhs,
            initial_state,
            -direction * startup * self.settings.step_s,
            step_s=cast(float, self.settings.startup_step_s),
            sample_step_s=self.settings.step_s,
            prefetch=prefetch,
        )[0][:, ::-1]
        prefetch.clear()
        offsets = -direction * self.settings.step_s * np.arange(startup, 0, -1, dtype=float)
        for (offset, data, slot), state in zip(
            self._epoch_batches(system, initial_epoch, offsets), warmup[:, :-1].T, strict=True
        ):
            nodes.append(
                offset,
                state,
                system.derivatives(
                    initial_epoch.shifted(offset), state, data, index=slot, history=external_history_provider
                ),
            )
        nodes.append(times[0], initial_state, initial_derivative, derivative_low=initial_derivative_low)
        nodes.mark_trajectory_start()
        history = DelayedHistory(system, initial_epoch, nodes, self.settings.history_interpolation_order)
        return nodes, history, initial_data, 0, None

    def _seed_rigid(
        self,
        system,
        initial_epoch,
        initial_state,
        duration_s,
        times,
        prefetch,
        seed_duration_s,
        external_history_provider,
    ):
        settings = self.settings
        delay = system.rotation.deformation.parameters.delay_s
        if duration_s <= 0:
            raise ValueError("Retarded lunar dynamics require forward propagation")
        if settings.step_s > delay:
            raise ValueError("Delayed propagation requires integration step <= lunar deformation delay")
        if seed_duration_s is None or seed_duration_s < delay + settings.history_interpolation_order * settings.step_s:
            raise ValueError("Explicit rigid seed must cover delay plus the history interpolation stencil")
        seed_times = regular_times(-seed_duration_s, settings.step_s)[::-1]
        nodes = LunarNodeBuffer(len(seed_times) + len(times) - 1, len(initial_state))

        def seed_rhs(offset, state):
            return prefetch.derivatives(offset, state, history=external_history_provider, rigid=True)

        seed_states = dop853_startup(
            seed_rhs,
            initial_state,
            -seed_duration_s,
            step_s=cast(float, settings.startup_step_s),
            sample_step_s=settings.step_s,
            prefetch=prefetch,
        )[0][:, ::-1]
        prefetch.clear()
        seed_batch = system.epoch_batch(tuple(initial_epoch.shifted(float(offset)) for offset in seed_times))
        for index, (offset, state) in enumerate(zip(seed_times, seed_states.T, strict=True)):
            epoch = initial_epoch.shifted(float(offset))
            derivative = system.derivatives(
                epoch, state, seed_batch, index=index, history=external_history_provider, rigid=True
            )
            nodes.append(offset, state, derivative)
        nodes.mark_trajectory_start()
        history = DelayedHistory(system, initial_epoch, nodes, settings.history_interpolation_order)
        data = system.epoch_batch((initial_epoch,))
        nodes.replace_last_derivative(
            *system.split_derivatives(initial_epoch, initial_state, np.zeros_like(initial_state), data, history=history)
        )
        return nodes, history, data, self.settings.order - 1, history

    def _main_grid(
        self,
        system,
        initial_epoch,
        times,
        nodes,
        history,
        prefetch,
        epoch_source,
        node_diagnostic,
        *,
        dop853_steps,
    ):
        for index, (offset, batch, slot) in enumerate(
            self._epoch_batches(system, initial_epoch, times[1:], epoch_source), start=1
        ):
            offset = float(offset)
            epoch = initial_epoch.shifted(offset)
            if index <= dop853_steps:
                origin = float(times[index - 1])

                def rhs(local_offset, state, origin=origin):
                    return prefetch.derivatives(origin + local_offset, state, history=history)

                def prefetch_stages(local_offsets, origin=origin):
                    prefetch([origin + local_offset for local_offset in local_offsets])

                states, lows = dop853_startup(
                    rhs,
                    nodes.state(),
                    float(times[index] - times[index - 1]),
                    step_s=cast(float, self.settings.startup_step_s),
                    sample_step_s=self.settings.step_s,
                    prefetch=prefetch_stages,
                    initial_low=nodes.state_low(),
                )
                prefetch.clear()
                state, state_low = states[:, -1], lows[:, -1]
                derivative, derivative_low = system.split_derivatives(
                    epoch, state, state_low, batch, index=slot, history=history
                )
            else:
                state, state_low, derivative, derivative_low = self._compute_abm_step(
                    system, nodes, history, epoch, batch, float(times[index] - times[index - 1]), slot
                )
            nodes.append(offset, state, derivative, state_low, derivative_low)
            if node_diagnostic is not None:
                node_diagnostic(epoch, state, batch, slot, history)
