"""Fixed-step GROOPS-style ABMD integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import DOP853

from lunarops.classes.time import Epoch

from .context import DynamicsEpochData, StateHistoryProvider
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
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
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


def dop853_startup(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    duration_s: float,
    *,
    step_s: float,
    sample_step_s: float,
) -> np.ndarray:
    """Return DOP853 startup states sampled on the ABM grid."""
    times = regular_times(duration_s, step_s)
    sample_ratio = abs(float(sample_step_s)) / step_s
    sample_stride = round(sample_ratio)
    if not np.isclose(sample_ratio, sample_stride, rtol=0.0, atol=1e-12):
        raise ValueError("sample_step_s must be an integer multiple of the startup step")
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
        for _ in range(self.settings.corrector_iterations):
            derivative = np.asarray(system.derivatives(epoch, corrected, epoch_data, history=history))
            corrected = nodes.state() + step_s * (
                np.column_stack((derivative, derivative_history)) @ self.beta_am
            )
        return corrected, derivative

    def integrate(
        self,
        system: LunarDynamics,
        initial_epoch: Epoch,
        initial_state: np.ndarray,
        duration_s: float,
        *,
        external_history_provider: StateHistoryProvider | None = None,
        node_diagnostic: NodeDiagnosticCallback | None = None,
    ) -> LunarTrajectory:
        times = regular_times(duration_s, self.settings.step_s)
        count = len(times) - 1
        startup = min(self.settings.order - 1, count)
        nodes = LunarNodeBuffer(
            count + self.settings.order,
            len(initial_state),
            external_state_count=len(system.history_body_names),
        )

        initial_data = system.build_epoch_data(initial_epoch)
        initial_derivative = system.derivatives(
            initial_epoch,
            initial_state,
            initial_data,
            history=external_history_provider,
        )

        def startup_rhs(offset_s: float, state: np.ndarray) -> np.ndarray:
            epoch = initial_epoch.shifted(offset_s)
            return system.derivatives(
                epoch,
                state,
                system.build_epoch_data(epoch),
                history=external_history_provider,
            )

        if startup == self.settings.order - 1:
            direction = np.copysign(1.0, duration_s)
            warmup = dop853_startup(
                startup_rhs,
                initial_state,
                -direction * startup * self.settings.step_s,
                step_s=self.settings.startup_step_s,
                sample_step_s=self.settings.step_s,
            )[:, ::-1]
            offsets = -direction * self.settings.step_s * np.arange(startup, 0, -1, dtype=float)
            for offset, state in zip(offsets, warmup[:, :-1].T, strict=True):
                epoch = initial_epoch.shifted(float(offset))
                data = system.build_epoch_data(epoch)
                nodes.append(
                    offset,
                    state,
                    system.derivatives(epoch, state, data, history=external_history_provider),
                    system.history_state_matrix(data),
                )
            nodes.append(times[0], initial_state, initial_derivative, system.history_state_matrix(initial_data))
            nodes.mark_trajectory_start()
        elif count:
            nodes.append(times[0], initial_state, initial_derivative, system.history_state_matrix(initial_data))
            nodes.mark_trajectory_start()
            startup_states = dop853_startup(
                startup_rhs,
                initial_state,
                duration_s,
                step_s=self.settings.startup_step_s,
                sample_step_s=self.settings.step_s,
            )
            for index in range(1, count + 1):
                epoch = initial_epoch.shifted(float(times[index]))
                state = startup_states[:, index]
                data = system.build_epoch_data(epoch)
                nodes.append(
                    times[index],
                    state,
                    system.derivatives(epoch, state, data, history=external_history_provider),
                    system.history_state_matrix(data),
                )
        else:
            nodes.append(times[0], initial_state, initial_derivative, system.history_state_matrix(initial_data))
            nodes.mark_trajectory_start()
        history = DelayedHistory(
            system,
            initial_epoch,
            nodes,
            self.settings.history_interpolation_order,
            external_history_provider,
        )
        if node_diagnostic is not None:
            node_diagnostic(initial_epoch, nodes.state(), initial_data, history)

        if startup == self.settings.order - 1:
            for index in range(count):
                epoch = initial_epoch.shifted(float(times[index + 1]))
                data = system.build_epoch_data(epoch)
                state, derivative = self._compute_abm_step(
                    system,
                    nodes,
                    history,
                    epoch,
                    data,
                    float(times[index + 1] - times[index]),
                )
                nodes.append(times[index + 1], state, derivative, system.history_state_matrix(data))
                if node_diagnostic is not None:
                    node_diagnostic(epoch, nodes.state(), data, history)
        elif count and node_diagnostic is not None:
            for index in range(1, count + 1):
                epoch = initial_epoch.shifted(float(times[index]))
                node_diagnostic(epoch, nodes.state(index), system.build_epoch_data(epoch), history)

        output_times, output_states = nodes.trajectory_data()
        return LunarTrajectory(
            initial_epoch,
            output_times,
            output_states,
            self.settings.trajectory_interpolation_order,
        )
