"""Fixed-step GROOPS-style ABMD integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np

from lunarops.classes.time import Epoch

from .context import StateHistoryProvider
from .epoch_data import DynamicsEpochData, EpochInputs, SerialEpochSource, batch_epochs
from .history import DelayedHistory, LunarNodeBuffer
from .trajectory import LunarTrajectory, regular_times

if TYPE_CHECKING:
    from .lunar_dynamics import LunarDynamics


@dataclass(frozen=True, slots=True)
class IntegratorSettings:
    step_s: float = 5400.0
    order: int = 13
    corrector_iterations: int = 2
    history_interpolation_order: int = 13
    trajectory_interpolation_order: int = 13
    startup_step_s: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.step_s) or self.step_s <= 0:
            raise ValueError("step_s must be positive and finite")
        if not isinstance(self.order, int) or self.order < 2 or self.order > 13:
            raise ValueError("order must be an integer in [2, 13]")
        if not isinstance(self.corrector_iterations, int) or self.corrector_iterations < 1:
            raise ValueError("corrector_iterations must be a positive integer")
        for name in ("history_interpolation_order", "trajectory_interpolation_order"):
            value = getattr(self, name)
            if not isinstance(value, int) or not 1 <= value <= 13:
                raise ValueError(f"{name} must be an integer in [1, 13]")
        startup_step_s = self.step_s / 8 if self.startup_step_s is None else float(self.startup_step_s)
        if not np.isfinite(startup_step_s) or startup_step_s <= 0:
            raise ValueError("startup_step_s must be positive and finite")
        ratio = self.step_s / startup_step_s
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-12):
            raise ValueError("step_s must be an integer multiple of startup_step_s")
        object.__setattr__(self, "startup_step_s", startup_step_s)


NodeDiagnosticCallback = Callable[
    [Epoch, np.ndarray, DynamicsEpochData, StateHistoryProvider | None],
    None,
]

_EPOCH_BATCH_SIZE = 64


class _EpochDataPrefetch:
    """Epoch data for DOP853 stage times, built in ephemeris batches before the stages run."""

    def __init__(self, system: LunarDynamics, initial_epoch: Epoch):
        self._system = system
        self._initial_epoch = initial_epoch
        self._data: dict[float, DynamicsEpochData] = {}

    def __call__(self, offsets) -> None:
        missing = [offset for offset in dict.fromkeys(offsets) if offset not in self._data]
        for start in range(0, len(missing), _EPOCH_BATCH_SIZE):
            chunk = missing[start : start + _EPOCH_BATCH_SIZE]
            epochs = tuple(self._initial_epoch.shifted(offset) for offset in chunk)
            self._data.update(zip(chunk, self._system.build_epoch_data_batch(epochs), strict=True))

    def clear(self) -> None:
        self._data.clear()

    def derivatives(self, offset: float, state: np.ndarray, **kwargs) -> np.ndarray:
        data = self._data.get(offset)
        epoch = self._initial_epoch.shifted(offset) if data is None else data.epoch_tdb
        return self._system.derivatives(epoch, state, data, **kwargs)


def dop853_startup(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    duration_s: float,
    *,
    step_s: float,
    sample_step_s: float,
    prefetch: Callable[[list[float]], None] | None = None,
) -> np.ndarray:
    """Return DOP853 startup states sampled on the ABM grid.

    ``prefetch`` receives every stage time before integration starts, using the
    same floating-point expression as the RHS calls, so time-only data can be
    built in batches.
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
    states[:, 0] = initial_state
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
        states[:, index + 1] = state + dt * (stages.T @ DOP853.B)
    return states[:, ::sample_stride]


def _lagrange_integral_coefficients(nodes: tuple[float, ...]) -> np.ndarray:
    coefficients = np.empty(len(nodes), dtype=float)
    for index, node in enumerate(nodes):
        polynomial = np.array([1.0])
        denominator = 1.0
        for other_index, other in enumerate(nodes):
            if other_index == index:
                continue
            polynomial = np.polynomial.polynomial.polymul(polynomial, (-other, 1.0))
            denominator *= node - other
        coefficients[index] = np.sum(polynomial / np.arange(1, len(polynomial) + 1)) / denominator
    coefficients.setflags(write=False)
    return coefficients


@lru_cache(maxsize=32)
def _adams_bashforth_coefficients(order: int) -> np.ndarray:
    return _lagrange_integral_coefficients(tuple(float(-index) for index in range(order)))


@lru_cache(maxsize=32)
def _adams_moulton_coefficients(order: int) -> np.ndarray:
    return _lagrange_integral_coefficients((1.0, *(float(-index) for index in range(order))))


class AdamsBashforthMoultonIntegrator:
    """Modified ABMD/ABM PECEC integrator on a fixed regular grid."""

    def __init__(self, settings: IntegratorSettings | None = None):
        self.settings = IntegratorSettings() if settings is None else settings
        self.beta_ab = _adams_bashforth_coefficients(self.settings.order)
        self.beta_am = _adams_moulton_coefficients(self.settings.order)

    def _compute_abm_step(
        self,
        system: LunarDynamics,
        nodes: LunarNodeBuffer,
        history: StateHistoryProvider,
        epoch: Epoch,
        epoch_data: DynamicsEpochData,
        step_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        derivative_history = nodes.derivative_window(self.settings.order)[:, ::-1]
        predicted = nodes.state() + step_s * (derivative_history @ self.beta_ab)
        corrected = predicted
        correction_base = nodes.state() + step_s * (derivative_history @ self.beta_am[1:])
        # Delayed tides and deformation read only accepted history, so they are
        # identical for every corrector iteration of this step.
        step = system.prepare_step(epoch_data, history)
        for _ in range(self.settings.corrector_iterations):
            derivative = np.asarray(system.derivatives(epoch, corrected, epoch_data, history=history, step=step))
            corrected = correction_base + step_s * self.beta_am[0] * derivative
        # Every stored derivative must belong to the accepted state, so the
        # last evaluation uses the accepted candidate rather than the final iterate's input.
        derivative = system.derivatives(epoch, corrected, epoch_data, history=history, step=step)
        return corrected, derivative

    @staticmethod
    def _epoch_batches(system: LunarDynamics, initial_epoch: Epoch, offsets: np.ndarray, source=None):
        """Grid epoch data in fixed batches; ``source`` (for example MPI workers) may produce them ahead."""
        batches = [
            [float(offset) for offset in offsets[start : start + _EPOCH_BATCH_SIZE]]
            for start in range(0, len(offsets), _EPOCH_BATCH_SIZE)
        ]
        if source is None:
            source = SerialEpochSource(system.epoch_inputs, system.ephemeris)
        for batch, arrays in zip(batches, source.stream(initial_epoch, batches), strict=True):
            yield from zip(batch, EpochInputs.records(batch_epochs(initial_epoch, batch), arrays), strict=True)

    def integrate(
        self,
        system: LunarDynamics,
        initial_epoch: Epoch,
        initial_state: np.ndarray,
        duration_s: float,
        *,
        external_history_provider: StateHistoryProvider | None = None,
        node_diagnostic: NodeDiagnosticCallback | None = None,
        seed_duration_s: float | None = None,
        epoch_source=None,
    ) -> LunarTrajectory:
        """Integrate on the fixed grid.

        ``epoch_source`` produces the main-grid epoch data; by default it is built in-process.
        """
        expected = 6 if system.rotation is None else 6 + system.rotation.state_dimension
        if np.shape(initial_state) != (expected,):
            raise ValueError(f"Initial state must contain {expected} translational/rotational components")
        if system.rotation is not None and system.rotation.deformation is not None:
            return self._integrate_delayed(
                system,
                initial_epoch,
                initial_state,
                duration_s,
                seed_duration_s,
                external_history_provider,
                node_diagnostic,
                epoch_source,
            )
        times = regular_times(duration_s, self.settings.step_s)
        count = len(times) - 1
        startup = self.settings.order - 1
        if count < startup:
            raise ValueError(f"Propagation needs at least order-1 = {startup} steps; got {count}")
        nodes = LunarNodeBuffer(count + self.settings.order, len(initial_state))

        initial_data = system.build_epoch_data(initial_epoch)
        initial_derivative = system.derivatives(
            initial_epoch,
            initial_state,
            initial_data,
            history=external_history_provider,
        )

        prefetch = _EpochDataPrefetch(system, initial_epoch)

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
        )[:, ::-1]
        prefetch.clear()
        offsets = -direction * self.settings.step_s * np.arange(startup, 0, -1, dtype=float)
        for (offset, data), state in zip(
            self._epoch_batches(system, initial_epoch, offsets), warmup[:, :-1].T, strict=True
        ):
            nodes.append(offset, state, system.derivatives(data.epoch_tdb, state, data, history=external_history_provider))
        nodes.append(times[0], initial_state, initial_derivative)
        nodes.mark_trajectory_start()
        history = DelayedHistory(system, initial_epoch, nodes, self.settings.history_interpolation_order)
        if node_diagnostic is not None:
            node_diagnostic(initial_epoch, nodes.state(), initial_data, history)

        grid = self._epoch_batches(system, initial_epoch, times[1:], epoch_source)
        for index, (offset, data) in enumerate(grid, start=1):
            state, derivative = self._compute_abm_step(
                system, nodes, history, data.epoch_tdb, data, float(times[index] - times[index - 1])
            )
            nodes.append(offset, state, derivative)
            if node_diagnostic is not None:
                node_diagnostic(data.epoch_tdb, nodes.state(), data, history)

        output_times, output_states = nodes.trajectory_data()
        return LunarTrajectory(
            initial_epoch,
            output_times,
            output_states,
            self.settings.trajectory_interpolation_order,
        )

    def _integrate_delayed(
        self,
        system,
        initial_epoch,
        initial_state,
        duration_s,
        seed_duration_s,
        external_history_provider,
        node_diagnostic,
        epoch_source,
    ):
        """Forward method of steps, explicit rigid seed, shared accepted nodes.

        step <= delay ensures every RK stage/corrector reads accepted past
        states. A retarded DDE cannot use the rigid solver's backward warmup.
        """
        settings = self.settings
        delay = system.rotation.deformation.parameters.delay_s
        if duration_s <= 0:
            raise ValueError("Retarded lunar dynamics require forward propagation")
        if settings.step_s > delay:
            raise ValueError("Delayed propagation requires integration step <= lunar deformation delay")
        if seed_duration_s is None or seed_duration_s < delay + settings.history_interpolation_order * settings.step_s:
            raise ValueError("Explicit rigid seed must cover delay plus the history interpolation stencil")
        times = regular_times(duration_s, settings.step_s)
        seed_times = regular_times(-seed_duration_s, settings.step_s)[::-1]
        nodes = LunarNodeBuffer(len(seed_times) + len(times) - 1, len(initial_state))

        prefetch = _EpochDataPrefetch(system, initial_epoch)

        def seed_rhs(offset, state):
            return prefetch.derivatives(offset, state, history=external_history_provider, rigid=True)

        seed_states = dop853_startup(
            seed_rhs,
            initial_state,
            -seed_duration_s,
            step_s=cast(float, settings.startup_step_s),
            sample_step_s=settings.step_s,
            prefetch=prefetch,
        )[:, ::-1]
        prefetch.clear()
        for offset, state in zip(seed_times, seed_states.T, strict=True):
            data = system.build_epoch_data(initial_epoch.shifted(float(offset)))
            derivative = system.derivatives(data.epoch_tdb, state, data, history=external_history_provider, rigid=True)
            nodes.append(offset, state, derivative)
        nodes.mark_trajectory_start()
        history = DelayedHistory(system, initial_epoch, nodes, settings.history_interpolation_order)
        data = system.build_epoch_data(initial_epoch)
        nodes.replace_last_derivative(system.derivatives(initial_epoch, initial_state, data, history=history))
        if node_diagnostic is not None:
            node_diagnostic(initial_epoch, nodes.state(), data, history)

        startup = settings.order - 1
        if len(times) - 1 < startup:
            raise ValueError(f"Propagation needs at least order-1 = {startup} steps; got {len(times) - 1}")
        grid = self._epoch_batches(system, initial_epoch, times[1:], epoch_source)
        for index, (offset, data) in enumerate(grid, start=1):
            offset = float(offset)
            epoch = data.epoch_tdb
            if index <= startup:
                origin = float(times[index - 1])

                def rhs(local_offset, state, origin=origin):
                    return prefetch.derivatives(origin + local_offset, state, history=history)

                def prefetch_stages(local_offsets, origin=origin):
                    prefetch([origin + local_offset for local_offset in local_offsets])

                state = dop853_startup(
                    rhs,
                    nodes.state(),
                    settings.step_s,
                    step_s=cast(float, settings.startup_step_s),
                    sample_step_s=settings.step_s,
                    prefetch=prefetch_stages,
                )[:, -1]
                prefetch.clear()
                derivative = system.derivatives(epoch, state, data, history=history)
            else:
                state, derivative = self._compute_abm_step(system, nodes, history, epoch, data, settings.step_s)
            nodes.append(offset, state, derivative)
            if node_diagnostic is not None:
                node_diagnostic(epoch, state, data, history)
        output_times, output_states = nodes.trajectory_data()
        return LunarTrajectory(
            initial_epoch, output_times, output_states, settings.trajectory_interpolation_order, history
        )
