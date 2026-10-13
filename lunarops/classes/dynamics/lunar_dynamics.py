"""Moon-relative translation and optional mantle/core rotational dynamics."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from lunarops.classes.ephemerides import (
    BcrsAccelerationProvider,
    Ephemeris,
    body_accelerations,
    body_state_matrices,
    body_state_matrix,
)
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.time import Epoch, require_tdb_epoch

from .bodies import GravitatingBody
from .context import AccelerationComponents, DynamicsEpochData, DynamicsEvaluationContext, StateHistoryProvider
from .force_models import LunarForceGroup
from .orientation import OrientationProvider
from .rotation import LunarRotationDynamics, quaternion_matrix
from .solar_environment import KuiperBeltRing, SolarGmVariation


class LunarDynamics:
    """Assemble and evaluate the Moon's translational dynamics."""

    def __init__(
        self,
        earth_body: GravitatingBody,
        moon_body: GravitatingBody,
        perturbing_bodies: Sequence[GravitatingBody] = (),
        ephemeris: Ephemeris | None = None,
        force_group: LunarForceGroup | None = None,
        *,
        earth_fixed2inertial_matrix_provider: OrientationProvider | None = None,
        moon_fixed2inertial_matrix_provider: OrientationProvider | None = None,
        rotation: LunarRotationDynamics | None = None,
        kuiper_belt_ring: KuiperBeltRing | None = None,
        solar_gm_variation: SolarGmVariation | None = None,
    ) -> None:
        if earth_body.body_id != "EARTH" or moon_body.body_id != "MOON":
            raise ValueError("The first bodies must be EARTH and MOON")
        if ephemeris is None:
            raise ValueError("Lunar dynamics require an Earth state provider")
        if not isinstance(ephemeris, BcrsAccelerationProvider):
            raise TypeError("Lunar dynamics require a BcrsAccelerationProvider.")
        self.earth_body = earth_body
        self.moon_body = moon_body
        self.perturbing_bodies = tuple(perturbing_bodies)
        self.ephemeris = ephemeris
        self.force_group = LunarForceGroup() if force_group is None else force_group
        self._earth_fixed2inertial_matrix_provider = earth_fixed2inertial_matrix_provider
        self._moon_fixed2inertial_matrix_provider = moon_fixed2inertial_matrix_provider
        self.rotation = rotation
        self.kuiper_belt_ring = kuiper_belt_ring
        self.solar_gm_variation = solar_gm_variation
        if rotation is not None and moon_fixed2inertial_matrix_provider is not None:
            raise ValueError("Integrated Moon rotation and a prescribed Moon orientation are mutually exclusive")
        bodies = (earth_body, moon_body, *self.perturbing_bodies,
                  *(() if kuiper_belt_ring is None else kuiper_belt_ring.bodies))
        self.body_names = tuple(body.body_id for body in bodies)
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError("Body identifiers must be unique")
        self.gravitational_parameters_m3_s2 = np.asarray(
            [body.gravitational_parameter_m3_s2 for body in bodies], dtype=float
        )
        self.gravitational_parameters_m3_s2.setflags(write=False)
        self.force_group.configure_bodies(self.body_names)
        self._inputs = DynamicsEvaluationContext(self.body_names, self.gravitational_parameters_m3_s2)
        self.history_body_names = ("EARTH", *(body.body_id for body in self.perturbing_bodies))
        self._history_body_indices = np.asarray((0, *range(2, 2 + len(self.perturbing_bodies))), dtype=int)

    def build_epoch_data(self, epoch_tdb: Epoch) -> DynamicsEpochData:
        require_tdb_epoch(epoch_tdb)
        names = ("EARTH", *tuple(body.body_id for body in self.perturbing_bodies))
        state_vectors = body_state_matrix(self.ephemeris, names, epoch_tdb)
        earth_acceleration = np.asarray(self.ephemeris.body_acceleration_bcrs("EARTH", epoch_tdb), dtype=float)
        return self._make_epoch_data(epoch_tdb, state_vectors, earth_acceleration)

    def build_epoch_data_batch(self, epochs_tdb: Sequence[Epoch]) -> tuple[DynamicsEpochData, ...]:
        epochs = tuple(require_tdb_epoch(epoch, name="epoch_tdb") for epoch in epochs_tdb)
        if not epochs:
            return ()
        names = ("EARTH", *tuple(body.body_id for body in self.perturbing_bodies))
        state_vectors = body_state_matrices(self.ephemeris, names, epochs)
        earth_accelerations = body_accelerations(self.ephemeris, "EARTH", epochs)
        moon_matrices = self._matrix_batch(self._moon_fixed2inertial_matrix_provider, epochs)
        earth_matrices = self._matrix_batch(self._earth_fixed2inertial_matrix_provider, epochs)
        return tuple(
            self._make_epoch_data(epoch, states, acceleration, moon_matrix=moon_matrix, earth_matrix=earth_matrix)
            for epoch, states, acceleration, moon_matrix, earth_matrix in zip(
                epochs, state_vectors, earth_accelerations, moon_matrices, earth_matrices, strict=True
            )
        )

    def _matrix_batch(self, provider, epochs: tuple[Epoch, ...]) -> tuple[np.ndarray, ...]:
        if provider is None:
            identity = np.eye(3)
            return tuple(identity for _ in epochs)
        matrices = np.asarray(provider.matrices(epochs), dtype=float)
        if matrices.shape != (len(epochs), 3, 3) or not np.all(np.isfinite(matrices)):
            raise ValueError("Orientation provider returned an invalid epoch matrix batch")
        return tuple(matrices)

    def _make_epoch_data(
        self,
        epoch_tdb: Epoch,
        state_vectors: np.ndarray,
        earth_acceleration: np.ndarray,
        *,
        moon_matrix: np.ndarray | None = None,
        earth_matrix: np.ndarray | None = None,
    ) -> DynamicsEpochData:
        names = ("EARTH", *tuple(body.body_id for body in self.perturbing_bodies))
        if state_vectors.shape != (len(names), 6) or not np.all(np.isfinite(state_vectors)):
            raise ValueError("Ephemeris returned an invalid body state matrix")
        if earth_acceleration.shape != (3,) or not np.all(np.isfinite(earth_acceleration)):
            raise ValueError("Ephemeris returned an invalid Earth acceleration")
        positions = np.zeros((len(self.body_names), 3), dtype=float)
        velocities = np.zeros_like(positions)
        positions[0] = state_vectors[0, :3]
        velocities[0] = state_vectors[0, 3:]
        end = 2 + len(self.perturbing_bodies)
        if self.perturbing_bodies:
            positions[2:end] = state_vectors[1:, :3]
            velocities[2:end] = state_vectors[1:, 3:]
        gm = self.gravitational_parameters_m3_s2
        if self.solar_gm_variation is not None:
            gm = gm.copy()
            gm[self.body_names.index("SUN")] *= self.solar_gm_variation.factor(epoch_tdb)
        if self.kuiper_belt_ring is not None:
            sun = self.body_names.index("SUN")
            positions[end:], velocities[end:] = self.kuiper_belt_ring.states(
                epoch_tdb, positions[sun], velocities[sun]
            )
        earth_fixed2inertial_matrix = (
            earth_matrix
            if earth_matrix is not None
            else (
                np.eye(3)
                if self._earth_fixed2inertial_matrix_provider is None
                else self._earth_fixed2inertial_matrix_provider.matrix(epoch_tdb)
            )
        )
        if moon_matrix is None:
            provider = self._moon_fixed2inertial_matrix_provider
            moon_matrix = np.eye(3) if provider is None else provider.matrix(epoch_tdb)
        return DynamicsEpochData(
            epoch_tdb=epoch_tdb,
            body_names=self.body_names,
            positions_m=positions,
            velocities_mps=velocities,
            earth_fixed2inertial_matrix=earth_fixed2inertial_matrix,
            moon_fixed2inertial_matrix=moon_matrix,
            ephemeris_earth_acceleration_mps2=earth_acceleration,
            point_mass_gravity_cache=self.force_group.build_point_mass_gravity_cache(
                positions,
                integrated_body_count=2,
                gravitational_parameters_m3_s2=gm,
            ),
            gravitational_parameters_m3_s2=gm,
        )

    def reconstruct_moon_state(self, epoch_tdb: Epoch, relative_state) -> np.ndarray:
        """Reconstruct the Moon's BCRS state from its Earth-relative state."""
        y = np.asarray(relative_state, dtype=float)
        if y.shape != (6,) or not np.all(np.isfinite(y)):
            raise ValueError("Moon relative integration state must be a finite 6-vector")
        earth_state = self.ephemeris.body_state_bcrs("EARTH", epoch_tdb)
        earth = np.concatenate((earth_state.position_m, earth_state.velocity_mps))
        return earth + y

    def history_state_matrix(self, epoch_data: DynamicsEpochData) -> np.ndarray:
        """Return the prescribed-body states in the order used by delayed history."""
        positions = epoch_data.positions_m[self._history_body_indices]
        velocities = epoch_data.velocities_mps[self._history_body_indices]
        return np.concatenate((positions, velocities), axis=1)

    def build_history_provider(
        self,
        relative_state_at: Callable[[Epoch], np.ndarray],
        external_provider: StateHistoryProvider | None,
        *,
        history_state_at: Callable[[Sequence[str], Epoch], np.ndarray] | None = None,
    ) -> StateHistoryProvider:
        def history(body_names: Sequence[str], epoch_tdb: Epoch) -> np.ndarray:
            names = tuple(body_name(name) for name in body_names)
            result = np.empty((len(names), 6), dtype=float)
            moon_slots = [index for index, name in enumerate(names) if name == "MOON"]
            external_slots = [index for index, name in enumerate(names) if name != "MOON"]
            if len(moon_slots) > 1:
                raise ValueError("History queries may contain MOON at most once")
            cached_names = tuple(
                dict.fromkeys((("EARTH",) if moon_slots else ()) + tuple(names[index] for index in external_slots))
            )
            cached_states = (
                None if history_state_at is None or not cached_names else history_state_at(cached_names, epoch_tdb)
            )
            if cached_states is not None:
                cached_states = np.asarray(cached_states, dtype=float)
                if cached_states.shape != (len(cached_names), 6) or not np.all(np.isfinite(cached_states)):
                    raise ValueError("Delayed history returned an invalid prescribed-body state matrix")
                cached_indices = {name: index for index, name in enumerate(cached_names)}
            else:
                cached_indices = {}
            if moon_slots:
                if cached_states is None:
                    result[moon_slots[0]] = self.reconstruct_moon_state(epoch_tdb, relative_state_at(epoch_tdb)[:6])
                else:
                    earth = cached_states[cached_indices["EARTH"]]
                    relative = np.asarray(relative_state_at(epoch_tdb), dtype=float)[:6]
                    result[moon_slots[0]] = earth + relative
            if external_slots:
                requested = tuple(names[index] for index in external_slots)
                if cached_states is not None:
                    try:
                        result[external_slots] = cached_states[[cached_indices[name] for name in requested]]
                    except KeyError as error:
                        raise KeyError(error.args[0]) from error
                else:
                    if external_provider is None:
                        raise KeyError(names[external_slots[0]])
                    result[external_slots] = external_provider(requested, epoch_tdb)
            return result

        return history

    def accelerations(
        self,
        epoch_tdb: Epoch,
        vector,
        epoch_data: DynamicsEpochData | None = None,
        *,
        history=None,
        evaluation_cache: dict[object, object] | None = None,
    ):
        epoch_data = self.build_epoch_data(epoch_tdb) if epoch_data is None else epoch_data
        inputs = self._load_evaluation(epoch_data, vector, history, evaluation_cache)
        body, terms = self.force_group.compute_accelerations(inputs, collect_terms=True)
        assert terms is not None
        return AccelerationComponents(self.body_names, body, terms)

    def derivatives(
        self,
        epoch_tdb: Epoch,
        vector,
        epoch_data: DynamicsEpochData | None = None,
        *,
        history=None,
        evaluation_cache: dict[object, object] | None = None,
    ) -> np.ndarray:
        y = np.asarray(vector, dtype=float)
        epoch_data = self.build_epoch_data(epoch_tdb) if epoch_data is None else epoch_data
        inputs = self._load_evaluation(epoch_data, y, history, evaluation_cache)
        body, _ = self.force_group.compute_accelerations(inputs)
        parts = [y[3:6], body[1] - epoch_data.ephemeris_earth_acceleration_mps2]
        if self.rotation is not None:
            parts.append(self.rotation.evaluate(inputs, y[6:]).derivative)
        return np.concatenate(parts)

    def _load_evaluation(self, epoch_data, vector, history, evaluation_cache, *, rigid=False):
        matrix = None if self.rotation is None else quaternion_matrix(vector[6:10])
        inputs = self._inputs.load_evaluation(
            epoch_data,
            vector,
            history,
            evaluation_cache,
            moon_matrix=matrix,
        )
        if self.rotation is not None and self.rotation.deformation is not None:
            inputs.lunar_inertia = self.rotation.deformation.evaluate(inputs, rigid=rigid)
        return inputs

    def rotation_evaluation(self, epoch, vector, epoch_data=None, *, history=None):
        data = self.build_epoch_data(epoch) if epoch_data is None else epoch_data
        inputs = self._load_evaluation(data, vector, history, None)
        return self.rotation.evaluate(inputs, vector[6:])

    def rigid_seed_derivatives(self, epoch, vector, *, history=None):
        """Explicit constant-inertia backward seed; NOT a backward DDE solve."""
        data = self.build_epoch_data(epoch)
        inputs = self._load_evaluation(data, vector, history, None, rigid=True)
        body, _ = self.force_group.compute_accelerations(inputs)
        return np.concatenate((
            vector[3:6], body[1] - data.ephemeris_earth_acceleration_mps2,
            self.rotation.evaluate(inputs, vector[6:]).derivative,
        ))
