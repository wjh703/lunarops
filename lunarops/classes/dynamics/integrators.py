"""Fixed-step orbit integrators and regular-node interpolation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from math import comb
from typing import Protocol

import numpy as np
from scipy.integrate import DOP853

from lunarops.classes.ephemerides.body_ids import body_name

from .context import HistoryProvider

AcceptedStepObserver = Callable[[object, np.ndarray, object, HistoryProvider | None], None]


class DynamicsSystem(Protocol):
    integrated_body_names: tuple[str, ...]

    def prepare_epoch(self, epoch): ...
    def prepare_epochs(self, epochs, *, max_workers=1): ...
    def accelerations(self, epoch, state, prepared=None, *, history=None): ...
    def accelerations_batch(self, epoch, states, prepared=None, *, histories=None): ...
    def derivatives(self, epoch, state, prepared=None, *, history=None): ...
    def derivatives_batch(self, epoch, states, prepared=None, *, histories=None): ...
    def history_state_vectors(self, body_names, epoch, vector): ...
    def detect_events(self, epoch, state, prepared=None): ...


@dataclass(frozen=True, slots=True)
class IntegratorSettings:
    step_s: float = 5400.0
    order: int = 13
    corrector_iterations: int = 2
    interpolation_order: int = 13
    startup_step_s: float | None = None
    preparation_batch_size: int = 64
    preparation_workers: int = 4
    rebuild_differences_every: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.step_s) or self.step_s <= 0:
            raise ValueError("step_s must be positive and finite")
        if not isinstance(self.order, int) or self.order < 2 or self.order > 13:
            raise ValueError("order must be an integer in [2, 13]")
        if not isinstance(self.corrector_iterations, int) or self.corrector_iterations < 1:
            raise ValueError("corrector_iterations must be a positive integer")
        if not isinstance(self.interpolation_order, int) or self.interpolation_order < 1:
            raise ValueError("interpolation_order must be a positive integer")
        if not isinstance(self.preparation_batch_size, int) or self.preparation_batch_size < 1:
            raise ValueError("preparation_batch_size must be a positive integer")
        if not isinstance(self.preparation_workers, int) or self.preparation_workers < 1:
            raise ValueError("preparation_workers must be a positive integer")
        if not isinstance(self.rebuild_differences_every, int) or self.rebuild_differences_every < 0:
            raise ValueError("rebuild_differences_every must be a nonnegative integer")
        startup_step_s = self.step_s / 8 if self.startup_step_s is None else float(self.startup_step_s)
        if not np.isfinite(startup_step_s) or startup_step_s <= 0:
            raise ValueError("startup_step_s must be positive and finite")
        startup_ratio = self.step_s / startup_step_s
        if not np.isclose(startup_ratio, round(startup_ratio), rtol=0.0, atol=1e-12):
            raise ValueError("step_s must be an integer multiple of startup_step_s")
        object.__setattr__(self, "startup_step_s", startup_step_s)


class _PreparedEpochCache:
    """Bounded exact-epoch cache with batch preparation for prescribed dynamics data."""

    def __init__(self, system: DynamicsSystem, *, capacity: int, workers: int) -> None:
        self.system = system
        self.capacity = int(capacity)
        self.workers = int(workers)
        self._values: OrderedDict[object, object] = OrderedDict()

    @staticmethod
    def _key(epoch) -> tuple[float, float, object]:
        return float(epoch.jd1), float(epoch.jd2), epoch.scale

    def get_many(self, epochs: Sequence[object]) -> tuple[object, ...]:
        if len(epochs) > self.capacity:
            raise ValueError("Prepared epoch request exceeds cache capacity")
        keys = tuple(self._key(epoch) for epoch in epochs)
        missing_epochs = []
        missing_keys = []
        for key, epoch in zip(keys, epochs, strict=True):
            if key not in self._values and key not in missing_keys:
                missing_keys.append(key)
                missing_epochs.append(epoch)
        if missing_epochs:
            batch = getattr(self.system, "prepare_epochs", None)
            if callable(batch):
                prepared = tuple(batch(tuple(missing_epochs), max_workers=self.workers))
            else:
                prepared = tuple(self.system.prepare_epoch(epoch) for epoch in missing_epochs)
            if len(prepared) != len(missing_epochs):
                raise ValueError("DynamicsSystem.prepare_epochs returned an invalid batch")
            for key, value in zip(missing_keys, prepared, strict=True):
                self._values[key] = value
        result = tuple(self._values[key] for key in keys)
        for key in keys:
            self._values.move_to_end(key)
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)
        return result


def _prepared_epoch_sequence(system: DynamicsSystem, epochs, settings: IntegratorSettings):
    iterator = iter(epochs)
    first = tuple(islice(iterator, settings.preparation_batch_size))
    if not first:
        return

    def prepare(chunk):
        cache = _PreparedEpochCache(system, capacity=len(chunk), workers=settings.preparation_workers)
        return cache.get_many(chunk)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="lunarops-prefetch") as executor:
        pending: Future[tuple[object, ...]] | None = executor.submit(prepare, first)
        while pending is not None:
            current = pending.result()
            next_chunk = tuple(islice(iterator, settings.preparation_batch_size))
            pending = executor.submit(prepare, next_chunk) if next_chunk else None
            yield from current


@lru_cache(maxsize=32)
def _regular_barycentric_weights(count: int) -> np.ndarray:
    weights = np.asarray([(-1.0) ** index * comb(count - 1, index) for index in range(count)], dtype=float)
    weights.setflags(write=False)
    return weights


def _interpolate_regular(nodes, values, x: float, order: int, *, extrapolate: bool) -> np.ndarray:
    node_array = np.asarray(nodes, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if node_array.ndim != 1 or value_array.ndim != 2 or value_array.shape[1] != len(node_array):
        raise ValueError("Invalid regular interpolation arrays")
    if not len(node_array):
        raise ValueError("Regular interpolation requires at least one node")
    if node_array[0] > node_array[-1]:
        node_array = node_array[::-1]
        value_array = value_array[:, ::-1]
    if not extrapolate and x < node_array[0] - 1e-9:
        raise ValueError("Epoch is outside propagated trajectory coverage")
    if not extrapolate and x > node_array[-1] + 1e-9:
        raise ValueError("Epoch is outside propagated trajectory coverage")
    if not extrapolate and x <= node_array[0]:
        return value_array[:, 0].copy()
    if not extrapolate and x >= node_array[-1]:
        return value_array[:, -1].copy()
    right = int(np.searchsorted(node_array, x, side="left"))
    count = min(order + 1, len(node_array))
    start = max(0, min(right - count // 2, len(node_array) - count))
    stencil = node_array[start : start + count]
    exact = np.flatnonzero(stencil == x)
    if len(exact):
        return value_array[:, start + int(exact[0])].copy()
    if count == 1:
        return value_array[:, start].copy()
    step = stencil[1] - stencil[0]
    if step == 0 or not np.allclose(np.diff(stencil), step, rtol=0.0, atol=abs(step) * 1e-12):
        raise ValueError("Interpolation nodes must form a regular grid")
    coordinate = (x - stencil[0]) / step
    factors = _regular_barycentric_weights(count) / (coordinate - np.arange(count))
    return (value_array[:, start : start + count] @ factors) / np.sum(factors)


class RegularTrajectory:
    def __init__(self, initial_epoch, times_s, states, *, interpolation_order=8, function_evaluations=0):
        self.initial_epoch = initial_epoch
        self.times_s = np.asarray(times_s, dtype=float)
        self.states = np.asarray(states, dtype=float)
        if self.times_s.ndim != 1 or self.states.ndim != 2 or self.states.shape[1] != len(self.times_s):
            raise ValueError("Invalid trajectory arrays")
        self.interpolation_order = int(interpolation_order)
        self.function_evaluations = int(function_evaluations)
        self.history_buffer_capacity = 0
        self.history_buffer_max_nodes = 0

    def vector(self, epoch, *, extrapolate=False):
        offset = float(self.initial_epoch.seconds_until(epoch))
        return _interpolate_regular(
            self.times_s,
            self.states,
            offset,
            self.interpolation_order,
            extrapolate=extrapolate,
        )

    def state_history(self, system: DynamicsSystem, external_history: HistoryProvider | None) -> HistoryProvider:
        """Return a batched history view without copying trajectory prefixes."""

        def states(body_names: Sequence[str], epoch) -> np.ndarray:
            names = tuple(body_name(name) for name in body_names)
            result = np.empty((len(names), 6), dtype=float)
            integrated = set(system.integrated_body_names)
            integrated_slots = [index for index, name in enumerate(names) if name in integrated]
            external_slots = [index for index, name in enumerate(names) if name not in integrated]
            if integrated_slots:
                requested = tuple(names[index] for index in integrated_slots)
                result[integrated_slots] = system.history_state_vectors(
                    requested,
                    epoch,
                    self.vector(epoch, extrapolate=True),
                )
            if external_slots:
                if external_history is None:
                    raise KeyError(names[external_slots[0]])
                requested = tuple(names[index] for index in external_slots)
                result[external_slots] = external_history(requested, epoch)
            return result

        return states


class _AcceptedStateHistory:
    """Fixed-capacity regular-node history for delayed force queries."""

    def __init__(
        self,
        system: DynamicsSystem,
        initial_epoch,
        state_dimension: int,
        capacity: int,
        initial_history: HistoryProvider | None,
    ) -> None:
        self.system = system
        self.initial_epoch = initial_epoch
        self.capacity = int(capacity)
        self.initial_history = initial_history
        self._times = np.empty(self.capacity, dtype=float)
        self._states = np.empty((state_dimension, self.capacity), dtype=float)
        self._start = 0
        self._size = 0
        self._vector_cache: dict[float, np.ndarray] = {}
        self.max_stored_nodes = 0

    def append(self, offset_s: float, state) -> None:
        if self._size < self.capacity:
            slot = (self._start + self._size) % self.capacity
            self._size += 1
        else:
            slot = self._start
            self._start = (self._start + 1) % self.capacity
        self._times[slot] = float(offset_s)
        self._states[:, slot] = state
        self._vector_cache.clear()
        self.max_stored_nodes = max(self.max_stored_nodes, self._size)

    def _ordered(self) -> tuple[np.ndarray, np.ndarray]:
        indices = (self._start + np.arange(self._size)) % self.capacity
        return self._times[indices], self._states[:, indices]

    def __call__(self, body_names: Sequence[str], epoch) -> np.ndarray:
        names = tuple(body_name(name) for name in body_names)
        result = np.empty((len(names), 6), dtype=float)
        integrated = set(self.system.integrated_body_names)
        integrated_slots = [index for index, name in enumerate(names) if name in integrated]
        external_slots = [index for index, name in enumerate(names) if name not in integrated]
        if integrated_slots:
            offset = float(self.initial_epoch.seconds_until(epoch))
            vector = self._vector_cache.get(offset)
            if vector is None:
                times, states = self._ordered()
                vector = _interpolate_regular(
                    times,
                    states,
                    offset,
                    self.capacity - 1,
                    extrapolate=True,
                )
                self._vector_cache[offset] = vector
            requested = tuple(names[index] for index in integrated_slots)
            result[integrated_slots] = self.system.history_state_vectors(requested, epoch, vector)
        if external_slots:
            if self.initial_history is None:
                raise KeyError(names[external_slots[0]])
            requested = tuple(names[index] for index in external_slots)
            result[external_slots] = self.initial_history(requested, epoch)
        return result


class Integrator:
    def __init__(self, settings: IntegratorSettings | None = None):
        self.settings = IntegratorSettings() if settings is None else settings

    def integrate(
        self,
        system: DynamicsSystem,
        initial_epoch,
        initial_state,
        duration_s,
        *,
        initial_history=None,
        accepted_step_observer: AcceptedStepObserver | None = None,
    ):
        raise NotImplementedError

    def _regular_times(self, duration_s):
        ratio = abs(float(duration_s)) / self.settings.step_s
        count = round(ratio)
        if not np.isclose(ratio, count, rtol=0.0, atol=1e-12):
            raise ValueError("duration_s must be an integer multiple of the fixed integration step")
        direction = np.copysign(1.0, duration_s)
        return direction * self.settings.step_s * np.arange(count + 1, dtype=float)


class RungeKuttaIntegrator(Integrator):
    """Fixed-step classical RK4 startup/validation integrator."""

    def integrate(
        self,
        system,
        initial_epoch,
        initial_state,
        duration_s,
        *,
        initial_history=None,
        accepted_step_observer=None,
    ):
        times = self._regular_times(duration_s)
        states = np.empty((len(initial_state), len(times)), dtype=float)
        states[:, 0] = initial_state
        evaluations = 0
        for index in range(len(times) - 1):
            dt = times[index + 1] - times[index]
            epoch = initial_epoch.shifted(float(times[index]))
            midpoint = initial_epoch.shifted(float(times[index] + dt / 2))
            endpoint = initial_epoch.shifted(float(times[index + 1]))
            y = states[:, index]
            k1 = np.asarray(system.derivatives(epoch, y, system.prepare_epoch(epoch), history=initial_history))
            prepared_midpoint = system.prepare_epoch(midpoint)
            k2 = np.asarray(system.derivatives(midpoint, y + dt * k1 / 2, prepared_midpoint, history=initial_history))
            k3 = np.asarray(system.derivatives(midpoint, y + dt * k2 / 2, prepared_midpoint, history=initial_history))
            prepared_endpoint = system.prepare_epoch(endpoint)
            k4 = np.asarray(system.derivatives(endpoint, y + dt * k3, prepared_endpoint, history=initial_history))
            evaluations += 4
            states[:, index + 1] = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if accepted_step_observer is not None:
                accepted_step_observer(endpoint, states[:, index + 1], prepared_endpoint, initial_history)
        return RegularTrajectory(
            initial_epoch,
            times,
            states,
            interpolation_order=self.settings.interpolation_order,
            function_evaluations=evaluations,
        )


class _DormandPrince8StartupIntegrator(Integrator):
    """Fixed-step eighth-order Dormand-Prince integrator used by ABMD startup."""

    def integrate(
        self,
        system,
        initial_epoch,
        initial_state,
        duration_s,
        *,
        initial_history=None,
        accepted_step_observer=None,
    ):
        times = self._regular_times(duration_s)
        states = np.empty((len(initial_state), len(times)), dtype=float)
        states[:, 0] = initial_state
        stages = np.empty((DOP853.n_stages, len(initial_state)), dtype=float)
        evaluations = 0
        stage_epochs = (
            initial_epoch.shifted(
                float(times[index])
                + float(DOP853.C[stage]) * float(times[index + 1] - times[index])
            )
            for index in range(len(times) - 1)
            for stage in range(DOP853.n_stages)
        )
        prepared_stages = iter(_prepared_epoch_sequence(system, stage_epochs, self.settings))
        for index in range(len(times) - 1):
            t = float(times[index])
            dt = float(times[index + 1] - times[index])
            y = states[:, index]
            for stage in range(DOP853.n_stages):
                stage_offset = float(DOP853.C[stage]) * dt
                increment = dt * (stages[:stage].T @ DOP853.A[stage, :stage])
                stage_epoch = initial_epoch.shifted(t + stage_offset)
                prepared = next(prepared_stages)
                stages[stage] = system.derivatives(
                    stage_epoch,
                    y + increment,
                    prepared,
                    history=initial_history,
                )
                evaluations += 1
            states[:, index + 1] = y + dt * (stages.T @ DOP853.B)
            if accepted_step_observer is not None:
                endpoint = initial_epoch.shifted(float(times[index + 1]))
                accepted_step_observer(
                    endpoint,
                    states[:, index + 1],
                    system.prepare_epoch(endpoint),
                    initial_history,
                )
        return RegularTrajectory(
            initial_epoch,
            times,
            states,
            interpolation_order=self.settings.interpolation_order,
            function_evaluations=evaluations,
        )


def _adams_gamma(order):
    """Return the paper's c_j partial sums gamma_j, Eqs. (13)--(14)."""
    c = np.empty(order + 1, dtype=float)
    gamma = np.empty_like(c)
    c[0] = 1.0
    gamma[0] = 1.0
    for j in range(1, order + 1):
        c[j] = -sum(c[i] / (j + 1 - i) for i in range(j))
        gamma[j] = gamma[j - 1] + c[j]
    return gamma


def _backward_differences(history):
    """Return [f_n, nabla f_n, ...] from oldest-to-newest RHS values."""
    work = np.asarray(history, dtype=float)
    result = np.empty_like(work)
    for level in range(work.shape[1]):
        result[:, level] = work[:, -1]
        work = np.diff(work, axis=1)
    return result


def _updated_backward_differences(previous, new_value):
    """Append one RHS value to a backward-difference table in O(k*n)."""
    result = np.empty((previous.shape[0], previous.shape[1] + 1), dtype=float)
    result[:, 0] = new_value
    for level in range(1, result.shape[1]):
        result[:, level] = result[:, level - 1] - previous[:, level - 1]
    return result


class AdamsBashforthMoultonIntegrator(Integrator):
    """Modified ABMD/ABM PECEC integrator on a fixed regular grid."""

    def integrate(
        self,
        system,
        initial_epoch,
        initial_state,
        duration_s,
        *,
        initial_history=None,
        accepted_step_observer=None,
    ):
        times = self._regular_times(duration_s)
        count = len(times) - 1
        states = np.empty((len(initial_state), count + 1), dtype=float)
        derivatives = np.empty_like(states)
        states[:, 0] = initial_state
        initial_prepared = system.prepare_epoch(initial_epoch)
        derivatives[:, 0] = system.derivatives(
            initial_epoch,
            states[:, 0],
            initial_prepared,
            history=initial_history,
        )
        evaluations = 1
        startup = min(self.settings.order - 1, count)
        assert self.settings.startup_step_s is not None
        startup_substeps = round(self.settings.step_s / self.settings.startup_step_s)
        startup_integrator = _DormandPrince8StartupIntegrator(
            IntegratorSettings(
                step_s=self.settings.startup_step_s,
                order=self.settings.order,
                corrector_iterations=self.settings.corrector_iterations,
                interpolation_order=self.settings.interpolation_order,
                startup_step_s=self.settings.startup_step_s,
                preparation_batch_size=self.settings.preparation_batch_size,
                preparation_workers=self.settings.preparation_workers,
            )
        )
        history = _AcceptedStateHistory(
            system,
            initial_epoch,
            len(initial_state),
            self.settings.interpolation_order + 1,
            initial_history,
        )
        history.append(float(times[0]), states[:, 0])
        if accepted_step_observer is not None:
            accepted_step_observer(initial_epoch, states[:, 0], initial_prepared, initial_history)

        if startup:
            warm = startup_integrator.integrate(
                system,
                initial_epoch,
                initial_state,
                times[startup],
                initial_history=initial_history,
            )
            states[:, : startup + 1] = warm.states[:, ::startup_substeps]
            startup_epochs = tuple(initial_epoch.shifted(float(times[index])) for index in range(1, startup + 1))
            prepare_batch = getattr(system, "prepare_epochs", None)
            if callable(prepare_batch):
                startup_prepared = tuple(
                    prepare_batch(startup_epochs, max_workers=self.settings.preparation_workers)
                )
            else:
                startup_prepared = tuple(system.prepare_epoch(item) for item in startup_epochs)
            for index, (epoch, prepared) in enumerate(zip(startup_epochs, startup_prepared, strict=True), start=1):
                derivatives[:, index] = system.derivatives(
                    epoch,
                    states[:, index],
                    prepared,
                    history=initial_history,
                )
                if accepted_step_observer is not None:
                    accepted_step_observer(epoch, states[:, index], prepared, history)
                history.append(float(times[index]), states[:, index])
            evaluations += warm.function_evaluations + startup

        order = self.settings.order
        gamma = _adams_gamma(order)
        differences = _backward_differences(derivatives[:, startup - order + 1 : startup + 1])
        main_epochs = (
            initial_epoch.shifted(float(times[index + 1])) for index in range(startup, count)
        )
        prepared_main = iter(_prepared_epoch_sequence(system, main_epochs, self.settings))
        for index in range(startup, count):
            dt = times[index + 1] - times[index]
            next_epoch = initial_epoch.shifted(float(times[index + 1]))
            predicted_state = states[:, index] + dt * (differences @ gamma[:order])
            corrected_state = predicted_state
            updated_differences = None
            prepared = next(prepared_main)
            for _ in range(self.settings.corrector_iterations):
                next_derivative = np.asarray(system.derivatives(next_epoch, corrected_state, prepared, history=history))
                evaluations += 1
                updated_differences = _updated_backward_differences(differences, next_derivative)
                corrected_state = predicted_state + dt * gamma[order] * updated_differences[:, order]
            assert updated_differences is not None
            states[:, index + 1] = corrected_state
            derivatives[:, index + 1] = next_derivative
            rebuild_period = self.settings.rebuild_differences_every
            if rebuild_period and (index - startup + 1) % rebuild_period == 0:
                differences = _backward_differences(derivatives[:, index + 2 - order : index + 2])
            else:
                differences = updated_differences[:, :order]
            history.append(float(times[index + 1]), corrected_state)
            if accepted_step_observer is not None:
                accepted_step_observer(next_epoch, corrected_state, prepared, history)
        trajectory = RegularTrajectory(
            initial_epoch,
            times,
            states,
            interpolation_order=self.settings.interpolation_order,
            function_evaluations=evaluations,
        )
        trajectory.history_buffer_capacity = history.capacity
        trajectory.history_buffer_max_nodes = history.max_stored_nodes
        return trajectory


class RKNIntegrator(AdamsBashforthMoultonIntegrator):
    """Second-order extension point; currently uses the ABMD PECEC state form."""


class StoermerCowellIntegrator(AdamsBashforthMoultonIntegrator):
    """Fixed-step second-order propagation extension point."""


class GaussJacksonIntegrator(AdamsBashforthMoultonIntegrator):
    """High-order long-arc propagator with regular history nodes."""
