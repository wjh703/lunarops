"""Moon-relative translation and optional mantle/core rotational dynamics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np

from lunarops.classes.ephemerides import Ephemeris
from lunarops.classes.time import Epoch

from .bodies import GravitatingBody
from .context import AccelerationComponents, DynamicsEpochData, DynamicsEvaluationContext, StateHistoryProvider
from .earth_figure_orientation import OrientationProvider
from .force_models import LunarForceGroup
from .rotation import LunarRotationDynamics, quaternion_matrix
from .solar_environment import KuiperBeltRing, SolarGmVariation

_IDENTITY3 = np.eye(3)
_IDENTITY3.setflags(write=False)


class LunarDynamics:
    """Assemble and evaluate the Moon's translational dynamics."""

    def __init__(
        self,
        earth_body: GravitatingBody,
        moon_body: GravitatingBody,
        perturbing_bodies: Sequence[GravitatingBody],
        ephemeris: Ephemeris,
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
        self.earth_body = earth_body
        self.moon_body = moon_body
        self.perturbing_bodies = tuple(perturbing_bodies)
        self.ephemeris = ephemeris
        self.force_group = LunarForceGroup() if force_group is None else force_group
        self._earth_fixed2inertial_matrix_provider = earth_fixed2inertial_matrix_provider
        self._moon_fixed2inertial_matrix_provider = moon_fixed2inertial_matrix_provider
        active = set(self.force_group.active_force_names)
        needs_earth = "figure_earth" in active or (
            rotation is not None and any(model.name == "figure:EARTH_J2" for model in rotation.torque_group.models)
        )
        needs_moon = "figure_moon" in active
        for model in self.force_group.active_models:
            if hasattr(model, "gravity_fields"):
                needs_earth |= "EARTH" in model.gravity_fields
                needs_moon |= "MOON" in model.gravity_fields
        if needs_earth and earth_fixed2inertial_matrix_provider is None:
            raise ValueError("Active Earth figure requires an Earth orientation provider")
        if needs_moon and rotation is None and moon_fixed2inertial_matrix_provider is None:
            raise ValueError("Active Moon figure requires prescribed or integrated Moon orientation")
        self.rotation = rotation
        self.kuiper_belt_ring = kuiper_belt_ring
        self.solar_gm_variation = solar_gm_variation
        if rotation is not None and moon_fixed2inertial_matrix_provider is not None:
            raise ValueError("Integrated Moon rotation and a prescribed Moon orientation are mutually exclusive")
        bodies = (
            earth_body,
            moon_body,
            *self.perturbing_bodies,
            *(() if kuiper_belt_ring is None else kuiper_belt_ring.bodies),
        )
        self.body_names = tuple(body.body_id for body in bodies)
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError("Body identifiers must be unique")
        self.gravitational_parameters_m3_s2 = np.asarray(
            [body.gravitational_parameter_m3_s2 for body in bodies], dtype=float
        )
        self.gravitational_parameters_m3_s2.setflags(write=False)
        self._inputs = DynamicsEvaluationContext(self.body_names, self.gravitational_parameters_m3_s2)
        self.history_body_names = ("EARTH", *(body.body_id for body in self.perturbing_bodies))
        self._history_body_indices = np.asarray((0, *range(2, 2 + len(self.perturbing_bodies))), dtype=int)

    def build_epoch_data(self, epoch_tdb: Epoch) -> DynamicsEpochData:
        return self.build_epoch_data_batch((epoch_tdb,))[0]

    def build_epoch_data_batch(self, epochs_tdb: Sequence[Epoch]) -> tuple[DynamicsEpochData, ...]:
        epochs = tuple(epochs_tdb)
        state_vectors = self.ephemeris.body_state_matrices_bcrs(self.history_body_names, epochs)
        earth_accelerations = self.ephemeris.body_accelerations_bcrs("EARTH", epochs)
        moon_matrices = self._matrix_batch(self._moon_fixed2inertial_matrix_provider, epochs)
        earth_matrices = self._matrix_batch(self._earth_fixed2inertial_matrix_provider, epochs)
        positions = np.zeros((len(epochs), len(self.body_names), 3), dtype=float)
        velocities = np.zeros_like(positions)
        gm = np.empty((len(epochs), len(self.body_names)), dtype=float)
        for index, (epoch, states) in enumerate(zip(epochs, state_vectors, strict=True)):
            self._fill_epoch_bodies(epoch, states, positions[index], velocities[index], gm[index])
        if self.kuiper_belt_ring is not None:
            end = 2 + len(self.perturbing_bodies)
            sun = self.body_names.index("SUN")
            positions[:, end:], velocities[:, end:] = self.kuiper_belt_ring.states_many(epochs, positions[:, sun], velocities[:, sun])
        caches = self.force_group.build_point_mass_gravity_cache_batch(
            positions, integrated_body_count=2, gravitational_parameters_m3_s2=gm
        )
        return tuple(
            DynamicsEpochData(
                epoch_tdb=epoch,
                body_names=self.body_names,
                positions_m=positions[index],
                velocities_mps=velocities[index],
                earth_fixed2inertial_matrix=earth_matrices[index],
                moon_fixed2inertial_matrix=moon_matrices[index],
                ephemeris_earth_acceleration_mps2=earth_accelerations[index],
                gravitational_parameters_m3_s2=gm[index],
                point_mass_gravity_cache=None if caches is None else caches[index],
            )
            for index, epoch in enumerate(epochs)
        )

    @staticmethod
    def _matrix_batch(provider, epochs: tuple[Epoch, ...]) -> tuple[np.ndarray, ...]:
        if provider is None:
            return tuple(_IDENTITY3 for _ in epochs)
        matrices = np.asarray(provider.matrices(epochs), dtype=float)
        matrices.setflags(write=False)
        return tuple(matrices)

    def _fill_epoch_bodies(self, epoch_tdb, state_vectors, positions, velocities, gm) -> None:
        positions[0] = state_vectors[0, :3]
        velocities[0] = state_vectors[0, 3:]
        end = 2 + len(self.perturbing_bodies)
        positions[2:end] = state_vectors[1:, :3]
        velocities[2:end] = state_vectors[1:, 3:]
        gm[:] = self.gravitational_parameters_m3_s2
        if self.solar_gm_variation is not None:
            gm[self.body_names.index("SUN")] *= self.solar_gm_variation.factor(epoch_tdb)

    def history_state_matrix(self, epoch_data: DynamicsEpochData) -> np.ndarray:
        """Return the prescribed-body states in the order used by delayed history."""
        positions = epoch_data.positions_m[self._history_body_indices]
        velocities = epoch_data.velocities_mps[self._history_body_indices]
        return np.concatenate((positions, velocities), axis=1)

    @staticmethod
    def build_history_provider(
        relative_state_at: Callable[[Epoch], np.ndarray],
        history_state_at: Callable[[Sequence[str], Epoch], np.ndarray],
    ) -> StateHistoryProvider:
        """Combine the integrated Moon with interpolated prescribed-body states."""

        def history(body_names: Sequence[str], epoch_tdb: Epoch) -> np.ndarray:
            names = tuple(body_names)
            if names.count("MOON") > 1:
                raise ValueError("History queries may contain MOON at most once")
            externals = tuple(name for name in names if name != "MOON")
            cached_names = tuple(dict.fromkeys(("EARTH", *externals) if "MOON" in names else externals))
            cached_states = history_state_at(cached_names, epoch_tdb)
            rows = {name: cached_states[index] for index, name in enumerate(cached_names)}
            result = np.empty((len(names), 6), dtype=float)
            for index, name in enumerate(names):
                if name == "MOON":
                    result[index] = rows["EARTH"] + np.asarray(relative_state_at(epoch_tdb), dtype=float)[:6]
                else:
                    result[index] = rows[name]
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
        vectors = self.force_group.acceleration_terms(inputs)
        terms = {}
        for name, force_vector in vectors.items():
            values = np.zeros_like(inputs.positions_m)
            values[inputs.moon_index] = force_vector
            terms[name] = values
        total = np.zeros_like(inputs.positions_m)
        for term in terms.values():
            total += term
        return AccelerationComponents(self.body_names, total, terms)

    def derivatives(
        self,
        epoch_tdb: Epoch,
        vector,
        epoch_data: DynamicsEpochData | None = None,
        *,
        history=None,
        evaluation_cache: dict[object, object] | None = None,
        rigid: bool = False,
    ) -> np.ndarray:
        """Return the state derivative; ``rigid`` uses the constant-inertia backward seed model."""
        y = np.asarray(vector, dtype=float)
        epoch_data = self.build_epoch_data(epoch_tdb) if epoch_data is None else epoch_data
        inputs = self._load_evaluation(epoch_data, y, history, evaluation_cache, rigid=rigid)
        body = self.force_group.compute_accelerations(inputs)
        parts = [y[3:6], body - epoch_data.ephemeris_earth_acceleration_mps2]
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
        return cast(LunarRotationDynamics, self.rotation).evaluate(inputs, vector[6:])
