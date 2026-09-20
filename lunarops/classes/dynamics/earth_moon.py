"""Earth--Moon translational dynamics on shared epoch arrays."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from lunarops.classes.ephemerides import require_tdb_epoch
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.time import Epoch

from .bodies import BodyRole, DynamicalBody, ExternalStateProvider
from .context import AccelerationBatch, AccelerationBreakdown, DynamicsContext, HistoryProvider, PreparedEpoch
from .force_models import ForceGroup


@dataclass(frozen=True, slots=True)
class FrameProviders:
    inertial_to_earth_body_fixed: Callable[[Epoch], np.ndarray] | None = None
    inertial_to_moon_body_fixed: Callable[[Epoch], np.ndarray] | None = None

    def evaluate(self, epoch: Epoch) -> dict[str, np.ndarray]:
        earth = np.eye(3) if self.inertial_to_earth_body_fixed is None else self.inertial_to_earth_body_fixed(epoch)
        moon = np.eye(3) if self.inertial_to_moon_body_fixed is None else self.inertial_to_moon_body_fixed(epoch)
        return {"EARTH": np.asarray(earth, dtype=float), "MOON": np.asarray(moon, dtype=float)}


class EarthMoonDynamics:
    """Evaluate one or more Earth/Moon state channels at a shared epoch."""

    integrated_body_names = ("EARTH", "MOON")

    def __init__(
        self,
        earth: DynamicalBody,
        moon: DynamicalBody,
        external_bodies: Sequence[DynamicalBody] = (),
        provider: ExternalStateProvider | None = None,
        force_group: ForceGroup | None = None,
        frames: FrameProviders | None = None,
    ) -> None:
        if earth.body_id != "EARTH" or moon.body_id != "MOON":
            raise ValueError("The first integrated bodies must be EARTH and MOON")
        if earth.role != BodyRole.INTEGRATED or moon.role != BodyRole.INTEGRATED:
            raise ValueError("Earth and Moon must have integrated role")
        self.earth = earth
        self.moon = moon
        self.external_bodies = tuple(external_bodies)
        self.provider = provider
        self.force_group = ForceGroup() if force_group is None else force_group
        self.frames = FrameProviders() if frames is None else frames
        bodies = (earth, moon, *self.external_bodies)
        self.body_names = tuple(body.body_id for body in bodies)
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError("Body identifiers must be unique")
        if self.external_bodies and provider is None:
            raise ValueError("External bodies require a state provider")
        self.gravitational_parameters_m3_s2 = np.asarray([body.mu_m3_s2 for body in bodies], dtype=float)
        self.gravitational_parameters_m3_s2.setflags(write=False)
        self.body_indices = {name: index for index, name in enumerate(self.body_names)}
        self.force_group.bind(self.body_names)
        self._context = DynamicsContext(self.body_names, self.gravitational_parameters_m3_s2)
        total = self.earth_mu_m3_s2 + self.moon_mu_m3_s2
        self._earth_fraction = self.earth_mu_m3_s2 / total
        self._moon_fraction = self.moon_mu_m3_s2 / total

    @property
    def earth_mu_m3_s2(self) -> float:
        return self.earth.mu_m3_s2

    @property
    def moon_mu_m3_s2(self) -> float:
        return self.moon.mu_m3_s2

    def prepare_epoch(self, epoch_tdb: Epoch) -> PreparedEpoch:
        return self.prepare_epochs((epoch_tdb,), max_workers=1)[0]

    def prepare_epochs(self, epochs_tdb: Sequence[Epoch], *, max_workers: int = 1) -> tuple[PreparedEpoch, ...]:
        """Prepare a bounded epoch batch; CALCEPH remains serial and numeric force setup may run in parallel."""
        epochs = tuple(epochs_tdb)
        if not epochs:
            return ()
        if not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        epoch_data = []
        for epoch_tdb in epochs:
            require_tdb_epoch(epoch_tdb)
            positions = np.zeros((len(self.body_names), 3), dtype=float)
            velocities = np.zeros_like(positions)
            if self.external_bodies:
                vectors = self._external_state_vectors(epoch_tdb)
                positions[2:] = vectors[:, :3]
                velocities[2:] = vectors[:, 3:]
            epoch_data.append((epoch_tdb, positions, velocities, self.frames.evaluate(epoch_tdb)))

        def prepare_forces(item):
            return self.force_group.prepare_epoch(item[1], integrated_count=len(self.integrated_body_names))

        if max_workers == 1 or len(epoch_data) == 1:
            force_data = tuple(prepare_forces(item) for item in epoch_data)
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(epoch_data))) as executor:
                force_data = tuple(executor.map(prepare_forces, epoch_data))
        return tuple(
            PreparedEpoch(epoch, self.body_names, positions, velocities, frames, forces)
            for (epoch, positions, velocities, frames), forces in zip(epoch_data, force_data, strict=True)
        )

    def _external_state_vectors(self, epoch_tdb: Epoch) -> np.ndarray:
        if self.provider is None:
            return np.empty((0, 6), dtype=float)
        names = tuple(body.body_id for body in self.external_bodies)
        batch = getattr(self.provider, "body_state_vectors_bcrs", None)
        if callable(batch):
            result = np.asarray(batch(names, epoch_tdb), dtype=float)
        else:
            states = self.provider.body_states_bcrs(names, epoch_tdb)
            result = np.asarray(
                [np.concatenate((states[name].position_m, states[name].velocity_mps)) for name in names],
                dtype=float,
            )
        if result.shape != (len(names), 6) or not np.all(np.isfinite(result)):
            raise ValueError("External state provider returned an invalid state matrix")
        return result

    def history_state_vectors(self, body_names: Sequence[str], epoch_tdb: Epoch, vector) -> np.ndarray:
        """Convert one integration vector to requested integrated-body states."""
        names = tuple(body_name(name) for name in body_names)
        if any(name not in self.integrated_body_names for name in names):
            raise KeyError(next(name for name in names if name not in self.integrated_body_names))
        y = np.asarray(vector, dtype=float)
        if y.shape != (12,) or not np.all(np.isfinite(y)):
            raise ValueError("Earth/Moon integration state must be a finite 12-vector")
        earth_position = y[:3] - self._moon_fraction * y[6:9]
        earth_velocity = y[3:6] - self._moon_fraction * y[9:12]
        moon_position = y[:3] + self._earth_fraction * y[6:9]
        moon_velocity = y[3:6] + self._earth_fraction * y[9:12]
        states = {
            "EARTH": np.concatenate((earth_position, earth_velocity)),
            "MOON": np.concatenate((moon_position, moon_velocity)),
        }
        return np.asarray([states[name] for name in names], dtype=float)

    def _bind_context(
        self,
        vector,
        prepared: PreparedEpoch,
        history: HistoryProvider | None,
    ) -> DynamicsContext:
        y = np.asarray(vector, dtype=float)
        if y.shape != (12,) or not np.all(np.isfinite(y)):
            raise ValueError("Earth/Moon integration state must be a finite 12-vector")
        context = self._context
        context.bind_prepared(prepared)
        context.bind_history(history)
        context.positions_m[0] = y[:3] - self._moon_fraction * y[6:9]
        context.velocities_mps[0] = y[3:6] - self._moon_fraction * y[9:12]
        context.positions_m[1] = y[:3] + self._earth_fraction * y[6:9]
        context.velocities_mps[1] = y[3:6] + self._earth_fraction * y[9:12]
        return context

    def _project(self, body_accelerations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        earth = body_accelerations[0]
        moon = body_accelerations[1]
        emb = self._earth_fraction * earth + self._moon_fraction * moon
        return emb, moon - earth

    def accelerations(
        self,
        epoch_tdb: Epoch,
        vector,
        prepared: PreparedEpoch | None = None,
        *,
        history: HistoryProvider | None = None,
    ) -> AccelerationBreakdown:
        """Return a diagnostic force decomposition for one state."""
        prepared = self.prepare_epoch(epoch_tdb) if prepared is None else prepared
        context = self._bind_context(vector, prepared, history)
        body, terms = self.force_group.accelerations(context, collect_terms=True)
        assert terms is not None
        emb, relative = self._project(body)
        return AccelerationBreakdown(self.body_names, body, emb, relative, terms)

    def accelerations_batch(
        self,
        epoch_tdb: Epoch,
        vectors,
        prepared: PreparedEpoch | None = None,
        *,
        histories: Sequence[HistoryProvider | None] | None = None,
    ) -> AccelerationBatch:
        """Evaluate channels while sharing ephemerides, frames, and epoch preparation."""
        states = np.asarray(vectors, dtype=float)
        if states.ndim != 2 or states.shape[1] != 12 or not np.all(np.isfinite(states)):
            raise ValueError("vectors must be a finite (channels,12) array")
        prepared = self.prepare_epoch(epoch_tdb) if prepared is None else prepared
        channel_histories = tuple(histories) if histories is not None else (None,) * len(states)
        if len(channel_histories) != len(states):
            raise ValueError("histories must contain one provider per state channel")
        body = np.empty((len(states), len(self.body_names), 3), dtype=float)
        emb = np.empty((len(states), 3), dtype=float)
        relative = np.empty_like(emb)
        for channel, (state, history) in enumerate(zip(states, channel_histories, strict=True)):
            context = self._bind_context(state, prepared, history)
            body[channel], _ = self.force_group.accelerations(context)
            emb[channel], relative[channel] = self._project(body[channel])
        return AccelerationBatch(self.body_names, body, emb, relative)

    def derivatives(
        self,
        epoch_tdb: Epoch,
        vector,
        prepared: PreparedEpoch | None = None,
        *,
        history: HistoryProvider | None = None,
    ) -> np.ndarray:
        y = np.asarray(vector, dtype=float)
        prepared = self.prepare_epoch(epoch_tdb) if prepared is None else prepared
        context = self._bind_context(y, prepared, history)
        body, _ = self.force_group.accelerations(context)
        emb, relative = self._project(body)
        return np.concatenate((y[3:6], emb, y[9:12], relative))

    def derivatives_batch(
        self,
        epoch_tdb: Epoch,
        vectors,
        prepared: PreparedEpoch | None = None,
        *,
        histories: Sequence[HistoryProvider | None] | None = None,
    ) -> np.ndarray:
        states = np.asarray(vectors, dtype=float)
        batch = self.accelerations_batch(epoch_tdb, states, prepared, histories=histories)
        result = np.empty_like(states)
        result[:, :3] = states[:, 3:6]
        result[:, 3:6] = batch.emb_mps2
        result[:, 6:9] = states[:, 9:12]
        result[:, 9:12] = batch.relative_mps2
        return result

    def detect_events(self, epoch_tdb: Epoch, vector, prepared: PreparedEpoch | None = None):
        return ()

    def state_jacobian(self, epoch_tdb: Epoch, vector, *, step_position_m=1.0, step_velocity_mps=1e-3):
        y = np.asarray(vector, dtype=float)
        if y.shape != (12,):
            raise ValueError("vector must have shape (12,)")
        prepared = self.prepare_epoch(epoch_tdb)
        steps = np.array(
            [step_position_m] * 3 + [step_velocity_mps] * 3 + [step_position_m] * 3 + [step_velocity_mps] * 3
        )
        jacobian = np.empty((12, 12), dtype=float)
        for index, step in enumerate(steps):
            plus = y.copy()
            minus = y.copy()
            plus[index] += step
            minus[index] -= step
            jacobian[:, index] = (
                self.derivatives(epoch_tdb, plus, prepared) - self.derivatives(epoch_tdb, minus, prepared)
            ) / (2 * step)
        return jacobian

    def variational_rhs(self, epoch_tdb: Epoch, vector, transition_matrix):
        phi = np.asarray(transition_matrix, dtype=float)
        if phi.shape != (12, 12):
            raise ValueError("transition_matrix must have shape (12,12)")
        return self.derivatives(epoch_tdb, vector), self.state_jacobian(epoch_tdb, vector) @ phi
