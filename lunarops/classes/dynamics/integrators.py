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
rounding error accumulates. The backward arc may run in a process of its own
(:mod:`~lunarops.classes.dynamics.arc_process`).

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

from .arc_process import ArcDiagnostic, system_digest
from .context import StateHistoryProvider
from .epoch_data import EpochBatch, SerialEpochSource
from .history import DelayedHistory, LunarNodeBuffer, TwoSidedHistory
from .trajectory import LunarTrajectory, regular_times

if TYPE_CHECKING:
    from .arc_process import BackwardArcProcess
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


def _epoch_chunks(offsets: Sequence[float] | np.ndarray) -> list[list[float]]:
    """Split main-grid offsets into the epoch batches the integrator requests, in order."""
    return [
        [float(offset) for offset in offsets[start : start + _EPOCH_BATCH_SIZE]]
        for start in range(0, len(offsets), _EPOCH_BATCH_SIZE)
    ]


class _EpochDataPrefetch:
    """Epoch data of the DOP853 start-up stages, built in batches before the start-up integration.

    With an epoch source (the runtime workers) the batches are built there; the workers form the epochs with
    the same ``initial_epoch.shifted(offset)`` and evaluate the same recipe, so the data are identical.
    """

    def __init__(self, system: LunarDynamics, initial_epoch: Epoch, source=None):
        self._system = system
        self._initial_epoch = initial_epoch
        self._source = source
        self._data: dict[float, tuple[EpochBatch, int]] = {}

    def __call__(self, offsets) -> None:
        missing = [float(offset) for offset in dict.fromkeys(offsets) if offset not in self._data]
        if not missing:
            # Opening even an empty runtime stream would reset the runtime's open-stream guard while the
            # main-grid stream is open.
            return
        chunks = [missing[start : start + _EPOCH_BATCH_SIZE] for start in range(0, len(missing), _EPOCH_BATCH_SIZE)]
        if self._source is None:
            batches = (
                self._system.epoch_batch(tuple(self._initial_epoch.shifted(offset) for offset in chunk))
                for chunk in chunks
            )
        else:
            batches = self._source.stream(self._initial_epoch, chunks)
        for chunk, batch in zip(chunks, batches, strict=True):
            self._data.update((offset, (batch, index)) for index, offset in enumerate(chunk))

    def clear(self) -> None:
        self._data.clear()

    def batch(self, offset: float) -> tuple[EpochBatch, int]:
        """Return the prefetched batch holding ``offset`` and the row of ``offset`` in it."""
        return self._data[offset]

    def derivatives(self, offset: float, state: np.ndarray, **kwargs) -> np.ndarray:
        batch, index = self._data[offset]
        return self._system.derivatives(self._initial_epoch.shifted(offset), state, batch, index=index, **kwargs)

    def split_derivatives(self, offset: float, state: np.ndarray, state_low: np.ndarray, **kwargs):
        batch, index = self._data[offset]
        return self._system.split_derivatives(
            self._initial_epoch.shifted(offset), state, state_low, batch, index=index, **kwargs
        )


@dataclass(frozen=True, slots=True)
class _Seed:
    """Nodes and history from which the main grid starts.

    Attributes
    ----------
    nodes : LunarNodeBuffer
        Warm-up or seed nodes, ending with the initial-epoch node.
    history : StateHistoryProvider
        History read by the delayed forces on the main grid.
    initial_batch : EpochBatch
        Epoch data containing the initial epoch.
    initial_index : int
        Row of the initial epoch in ``initial_batch``.
    dop853_steps : int
        Number of main-grid steps still to be taken with DOP853.
    trajectory_history : DelayedHistory or None
        History kept with the returned trajectory.
    """

    nodes: LunarNodeBuffer
    history: StateHistoryProvider
    initial_batch: EpochBatch
    initial_index: int
    dop853_steps: int
    trajectory_history: DelayedHistory | None


def dop853_stage_offsets(duration_s: float, step_s: float) -> list[float]:
    """Return the stage offsets of a fixed-step DOP853 integration over ``duration_s``.

    The expression is the one :func:`dop853_startup` evaluates the right-hand side at, so the offsets can
    key prefetched epoch data exactly.
    """
    from scipy.integrate import DOP853

    times = regular_times(duration_s, step_s)
    return [
        float(times[index]) + float(DOP853.C[stage]) * float(times[index + 1] - times[index])
        for index in range(len(times) - 1)
        for stage in range(DOP853.n_stages)
    ]


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

    The state update is compensated (TwoSum of state and increment, renormalised
    with FastTwoSum), so that the rounding of the many small start-up steps does
    not accumulate into the ABM initial state.

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
        prefetch(dop853_stage_offsets(duration_s, step_s))
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
    """

    def __init__(self, settings: IntegratorSettings | None = None):
        self.settings = IntegratorSettings() if settings is None else settings
        self._ab_weights = _adams_bashforth_weights(self.settings.order)
        self._am_weights = _adams_moulton_weights(self.settings.order)

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
        batches = _epoch_chunks(offsets)
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
        initial_state_low: np.ndarray | None = None,
        backward_duration_s: float = 0.0,
        external_history_provider: StateHistoryProvider | None = None,
        node_diagnostic: NodeDiagnosticCallback | None = None,
        seed_duration_s: float | None = None,
        epoch_source=None,
        backward_arc: BackwardArcProcess | None = None,
        progress: Callable[[int], None] | None = None,
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
        initial_state_low : numpy.ndarray or None, optional
            Low part of a double-double initial state, shape ``(d,)``, for
            example when restarting from saved nodes; zero when ``None``.
        backward_duration_s : float, default 0.0
            Non-negative length in s of an additional arc propagated backward
            from ``initial_epoch``; an integer multiple of ``settings.step_s``.
            Requires a positive ``duration_s``, long enough to seed the
            backward arc (``order - 1`` steps, more for long advanced
            Earth-tide delays at small steps). The returned trajectory then
            covers ``-backward_duration_s`` to ``duration_s``.
        external_history_provider : StateHistoryProvider or None, optional
            History used by delayed forces during the warm-up or seed arc.
        node_diagnostic : NodeDiagnosticCallback or None, optional
            Called at the initial epoch and after every accepted main-grid
            node: the forward nodes in increasing time and, for a two-sided
            propagation, then the backward nodes in decreasing time. Each call
            sees the history of its arc as it was when the node was accepted.
            With ``backward_arc`` it must be an
            :class:`~lunarops.classes.dynamics.arc_process.ArcDiagnostic`,
            whose backward calls happen in the arc process and are merged
            after the forward arc.
        seed_duration_s : float or None, optional
            Length of the rigid backward seed arc in s; required with delayed
            deformation and at least the delay plus
            ``history_interpolation_order`` steps.
        epoch_source : object, optional
            Producer of epoch batches with a ``stream`` method like
            :class:`~lunarops.classes.dynamics.epoch_data.SerialEpochSource`,
            used for every epoch of the propagation; built in-process when
            ``None``. A parallel backward arc builds its own epoch data.
        backward_arc : BackwardArcProcess or None, optional
            Process that propagates the backward arc of a two-sided run while
            this process continues the forward arc. It is seeded once the
            mirrored forward nodes are accepted and polled after every forward
            node. The result is bitwise identical to the sequential
            propagation, which runs the backward arc after the forward one
            when ``None``.
        progress : Callable[[int], None] or None, optional
            Called with the number of nodes accepted since the last call, on
            both arcs, the initial node included.

        Returns
        -------
        LunarTrajectory
            Accepted nodes over the propagated interval, in increasing time.

        Raises
        ------
        ValueError
            Before any integration, if the interval is shorter than
            ``order - 1`` steps or either duration is not a multiple of the
            step, if ``backward_duration_s`` is negative or the forward arc is
            too short to seed it, if ``backward_arc`` is given for a one-sided
            propagation, or, with delayed deformation, if the propagation is
            backward, the step exceeds the delay, or the seed arc is too short.
        TypeError
            If ``backward_arc`` is given with a node diagnostic that is not an
            :class:`~lunarops.classes.dynamics.arc_process.ArcDiagnostic`.
        RuntimeError
            With ``backward_arc``, if the arc process exits early, stalls or
            loses nodes. An exception raised in that process is raised here,
            with the remote traceback as its cause.
        """
        settings = self.settings
        times = regular_times(duration_s, settings.step_s)
        startup = settings.order - 1
        if len(times) - 1 < startup:
            raise ValueError(f"Propagation needs at least order-1 = {startup} steps; got {len(times) - 1}")
        if backward_duration_s < 0:
            raise ValueError("backward_duration_s must be non-negative")
        backward_times = None
        if backward_duration_s:
            if duration_s <= 0:
                raise ValueError("A two-sided propagation needs a positive duration_s")
            backward_times = regular_times(-backward_duration_s, settings.step_s)
            mirrored = self._mirrored_node_count(system)
            if len(times) < mirrored:
                raise ValueError(
                    f"A two-sided propagation needs a forward arc of at least {mirrored - 1} steps to seed the "
                    "backward arc"
                )
        if backward_arc is not None:
            if backward_times is None:
                raise ValueError("backward_arc needs a two-sided propagation")
            if node_diagnostic is not None and not isinstance(node_diagnostic, ArcDiagnostic):
                raise TypeError("With backward_arc the node diagnostic must be an ArcDiagnostic")
        initial_state = np.asarray(initial_state, dtype=float)
        initial_low = np.zeros_like(initial_state) if initial_state_low is None else np.asarray(initial_state_low)
        prefetch = _EpochDataPrefetch(system, initial_epoch, epoch_source)
        if system.rotation is not None and system.rotation.deformation is not None:
            seed = self._seed_rigid(
                system,
                initial_epoch,
                initial_state,
                initial_low,
                duration_s=duration_s,
                times=times,
                prefetch=prefetch,
                seed_duration_s=seed_duration_s,
                external_history_provider=external_history_provider,
            )
        else:
            seed = self._seed_backward(
                system,
                initial_epoch,
                initial_state,
                initial_low,
                duration_s=duration_s,
                times=times,
                prefetch=prefetch,
                external_history_provider=external_history_provider,
            )
        nodes, history = seed.nodes, seed.history
        if node_diagnostic is not None:
            node_diagnostic(initial_epoch, nodes.state(), seed.initial_batch, seed.initial_index, history)
        if progress is not None:
            progress(1)
        mirrored = self._mirrored_node_count(system)
        mirrored_block = None

        def seed_backward_arc(nodes) -> None:
            nonlocal mirrored_block
            assert backward_arc is not None
            mirrored_block = nodes.block(nodes.trajectory_start, nodes.trajectory_start + mirrored)
            backward_arc.seed(
                initial_epoch,
                backward_duration_s,
                mirrored,
                mirrored_block,
                system_digest(system, settings),
                None if node_diagnostic is None else node_diagnostic.for_backward_arc(),
            )

        def after_forward_node(nodes, index: int) -> None:
            if progress is not None:
                progress(1)
            if backward_arc is not None:
                if index == mirrored - 1:
                    seed_backward_arc(nodes)
                backward_arc.poll(progress)

        if backward_arc is not None and mirrored == 1:
            seed_backward_arc(nodes)
        self._main_grid(
            system,
            initial_epoch,
            times,
            nodes,
            history,
            prefetch=prefetch,
            epoch_source=epoch_source,
            node_diagnostic=node_diagnostic,
            dop853_steps=seed.dop853_steps,
            after_node=after_forward_node,
        )
        backward_result = None
        if backward_arc is not None:
            assert mirrored_block is not None and backward_times is not None
            arc = backward_arc.finish(progress)
            if arc.count != len(backward_times) - 1:
                raise RuntimeError("The backward-arc process did not return every node")
            # The buffer and history of a sequential backward arc, with the same nodes and counters.
            backward, backward_history = self._backward_start(system, initial_epoch, mirrored_block, backward_times)
            backward.extend(arc.nodes)
            backward_history.extrapolated_translation_queries = arc.extrapolated_translation_queries
            backward_history.max_translation_extrapolation_s = arc.max_translation_extrapolation_s
            if node_diagnostic is not None:
                node_diagnostic.merge_backward(arc.diagnostic)
            backward_result = backward, backward_history
        elif backward_times is not None:
            backward_result = self._backward_arc(
                system,
                initial_epoch,
                nodes.block(nodes.trajectory_start, nodes.trajectory_start + mirrored),
                backward_times,
                epoch_source=epoch_source,
                node_diagnostic=node_diagnostic,
                progress=progress,
            )
        output_times, output_states = nodes.trajectory_data()
        trajectory_history = seed.trajectory_history
        if backward_times is not None:
            assert backward_result is not None
            backward, backward_history = backward_result
            reverse_times, reverse_states = backward.trajectory_data()
            output_times = np.concatenate((reverse_times[:0:-1], output_times))
            output_states = np.concatenate((reverse_states[:, :0:-1], output_states), axis=1)
            if trajectory_history is not None:
                trajectory_history = TwoSidedHistory(initial_epoch, trajectory_history, backward_history)
        return LunarTrajectory(
            initial_epoch, output_times, output_states, settings.trajectory_interpolation_order, trajectory_history
        )

    def _mirrored_node_count(self, system: LunarDynamics) -> int:
        """Return the number of forward nodes, from the initial epoch on, that seed a backward arc.

        They are the derivative window of the first backward step (``order`` nodes) and must also cover the
        advanced Earth-tide arguments, which on the backward arc lie behind the integration: the first step,
        at ``-step``, queries the Moon at ``-step + advanced_delay_s``.
        """
        step = self.settings.step_s
        return max(self.settings.order, int(np.ceil(system.advanced_delay_s / step - 1e-9)) + 1)

    def _backward_arc(self, system, initial_epoch, block, times, *, epoch_source, node_diagnostic, progress=None):
        """Propagate backward from the initial epoch, seeded by the first forward nodes.

        The mirrored forward nodes are the derivative window of the first
        backward step, so no extra start-up integration is needed. With
        delayed deformation the epoch ``t - tau`` lies in the direction of
        integration and is extrapolated from the latest accepted nodes (see
        :meth:`DelayedHistory.integration_sample`); the advanced Earth-tide
        arguments lie behind the integration, inside the mirrored or accepted
        nodes.
        """
        nodes, history = self._backward_start(system, initial_epoch, block, times)
        self._main_grid(
            system,
            initial_epoch,
            times,
            nodes,
            history,
            prefetch=None,
            epoch_source=epoch_source,
            node_diagnostic=node_diagnostic,
            dop853_steps=0,
            after_node=None if progress is None else lambda nodes, index: progress(1),
        )
        return nodes, history

    def _backward_start(self, system, initial_epoch, block, times):
        """Return the backward-arc buffer, holding the mirrored forward nodes, and its history.

        ``block`` holds the mirrored forward nodes from the initial epoch on (see
        :meth:`LunarNodeBuffer.block`). The backward buffer stores them newest first, so that it ends at the
        initial epoch like a forward buffer ends at its latest node.
        """
        count = self._mirrored_node_count(system)
        if len(block[0]) != count:
            raise ValueError(f"The backward arc needs {count} mirrored forward nodes, got {len(block[0])}")
        nodes = LunarNodeBuffer(count - 1 + len(times), block[1].shape[0])
        nodes.extend(tuple(part[..., ::-1] for part in block))
        nodes.mark_trajectory_start()
        return nodes, DelayedHistory(system, initial_epoch, nodes, self.settings.history_interpolation_order)

    def _seed_backward(
        self,
        system,
        initial_epoch,
        initial_state,
        initial_low,
        *,
        duration_s,
        times,
        prefetch,
        external_history_provider,
    ) -> _Seed:
        """Fill the ``order - 1`` derivative nodes before the initial epoch with a backward DOP853 warm-up."""
        settings = self.settings
        startup = settings.order - 1
        nodes = LunarNodeBuffer(len(times) - 1 + settings.order, len(initial_state))
        direction = np.copysign(1.0, duration_s)
        offsets = [float(offset) for offset in -direction * settings.step_s * np.arange(startup, 0, -1, dtype=float)]
        prefetch([*offsets, 0.0])

        def startup_rhs(offset_s: float, state: np.ndarray) -> np.ndarray:
            return prefetch.derivatives(offset_s, state, history=external_history_provider)

        warmup, warmup_low = dop853_startup(
            startup_rhs,
            initial_state,
            -direction * startup * settings.step_s,
            step_s=cast(float, settings.startup_step_s),
            sample_step_s=settings.step_s,
            prefetch=prefetch,
            initial_low=initial_low,
        )
        # Samples run from the initial epoch outwards; the nodes are stored oldest first.
        warmup, warmup_low = warmup[:, :0:-1], warmup_low[:, :0:-1]
        for column, offset in enumerate(offsets):
            state, state_low = warmup[:, column], warmup_low[:, column]
            derivative, derivative_low = prefetch.split_derivatives(
                offset, state, state_low, history=external_history_provider
            )
            nodes.append(offset, state, derivative, state_low, derivative_low)
        initial_batch, initial_index = prefetch.batch(0.0)
        derivative, derivative_low = prefetch.split_derivatives(
            0.0, initial_state, initial_low, history=external_history_provider
        )
        prefetch.clear()
        nodes.append(times[0], initial_state, derivative, initial_low, derivative_low)
        nodes.mark_trajectory_start()
        history = DelayedHistory(system, initial_epoch, nodes, settings.history_interpolation_order)
        return _Seed(nodes, history, initial_batch, initial_index, 0, None)

    def _seed_rigid(
        self,
        system,
        initial_epoch,
        initial_state,
        initial_low,
        *,
        duration_s,
        times,
        prefetch,
        seed_duration_s,
        external_history_provider,
    ) -> _Seed:
        """Fill the delayed history before the initial epoch with a rigid (undeformed) backward seed arc."""
        settings = self.settings
        delay = system.rotation.deformation.parameters.delay_s
        if duration_s <= 0:
            raise ValueError("Retarded lunar dynamics require forward propagation")
        if settings.step_s > delay:
            raise ValueError("Delayed propagation requires integration step <= lunar deformation delay")
        if seed_duration_s is None or seed_duration_s < delay + settings.history_interpolation_order * settings.step_s:
            raise ValueError("Explicit rigid seed must cover delay plus the history interpolation stencil")
        seed_times = [float(offset) for offset in regular_times(-seed_duration_s, settings.step_s)[::-1]]
        nodes = LunarNodeBuffer(len(seed_times) + len(times) - 1, len(initial_state))
        prefetch(seed_times)

        def seed_rhs(offset, state):
            return prefetch.derivatives(offset, state, history=external_history_provider, rigid=True)

        seed_states, seed_lows = dop853_startup(
            seed_rhs,
            initial_state,
            -seed_duration_s,
            step_s=cast(float, settings.startup_step_s),
            sample_step_s=settings.step_s,
            prefetch=prefetch,
            initial_low=initial_low,
        )
        seed_states, seed_lows = seed_states[:, ::-1], seed_lows[:, ::-1]
        for column, offset in enumerate(seed_times):
            state, state_low = seed_states[:, column], seed_lows[:, column]
            derivative, derivative_low = prefetch.split_derivatives(
                offset, state, state_low, history=external_history_provider, rigid=True
            )
            nodes.append(offset, state, derivative, state_low, derivative_low)
        nodes.mark_trajectory_start()
        history = DelayedHistory(system, initial_epoch, nodes, settings.history_interpolation_order)
        initial_batch, initial_index = prefetch.batch(0.0)
        # The initial node takes the deformed derivative, read from the rigid seed history.
        nodes.replace_last_derivative(*prefetch.split_derivatives(0.0, initial_state, initial_low, history=history))
        prefetch.clear()
        return _Seed(nodes, history, initial_batch, initial_index, settings.order - 1, history)

    def _main_grid(
        self,
        system,
        initial_epoch,
        times,
        nodes,
        history,
        *,
        prefetch,
        epoch_source,
        node_diagnostic,
        dop853_steps,
        after_node=None,
    ):
        """Propagate over ``times[1:]``, appending every accepted node to ``nodes``.

        ``after_node(nodes, index)``, when given, is called after the node at ``times[index]`` is accepted and
        after ``node_diagnostic``.
        """
        if dop853_steps:
            # Build the stage data of every start-up step now: once the main-grid stream below is open
            # its tasks occupy the runtime workers, so no second stream may be opened inside the loop.
            prefetch(
                [
                    float(times[index - 1]) + local_offset
                    for index in range(1, dop853_steps + 1)
                    for local_offset in dop853_stage_offsets(
                        float(times[index] - times[index - 1]), cast(float, self.settings.startup_step_s)
                    )
                ]
            )
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
                if index == dop853_steps:
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
            if after_node is not None:
                after_node(nodes, index)
