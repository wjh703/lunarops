"""Moon-relative translation and optional mantle/core rotational dynamics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np

from lunarops.classes.ephemerides import Ephemeris
from lunarops.classes.time import Epoch

from .bodies import GravitatingBody
from .context import AccelerationComponents, DynamicsEvaluationContext, StateHistoryProvider, StepData
from .epoch_data import DynamicsEpochData, EphemerisLunarAxes, EpochInputs, OrientationProvider
from .force_models import EarthFigureForce, EarthTideForce, LunarForceGroup, MoonFigureForce
from .fused_rhs import compile_fused_rhs
from .rotation import LunarRotationDynamics, quaternion_matrix
from .solar_environment import KuiperBeltRing, SolarGmVariation
from .torques import EarthFigureTorque


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
        moon_fixed2inertial_matrix_provider: OrientationProvider | EphemerisLunarAxes | None = None,
        rotation: LunarRotationDynamics | None = None,
        kuiper_belt_ring: KuiperBeltRing | None = None,
        solar_gm_variation: SolarGmVariation | None = None,
        fused_rhs: bool = True,
    ) -> None:
        if earth_body.body_id != "EARTH" or moon_body.body_id != "MOON":
            raise ValueError("The first bodies must be EARTH and MOON")
        self.earth_body = earth_body
        self.moon_body = moon_body
        self.perturbing_bodies = tuple(perturbing_bodies)
        self.ephemeris = ephemeris
        self.force_group = LunarForceGroup() if force_group is None else force_group
        active = self.force_group.active_models
        needs_earth = any(isinstance(model, EarthFigureForce) for model in active) or (
            rotation is not None and any(isinstance(model, EarthFigureTorque) for model in rotation.torque_group.models)
        )
        needs_moon = any(isinstance(model, MoonFigureForce) for model in active)
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
        self.prescribed_body_names = ("EARTH", *(body.body_id for body in self.perturbing_bodies))
        # Prescribed bodies the Earth tide reads at its fixed delays; they depend on time
        # only, so they are fetched together with the rest of the epoch data.
        self._earth_axes = earth_fixed2inertial_matrix_provider
        self._moon_axes = moon_fixed2inertial_matrix_provider
        # The native fused right-hand side replaces the Python force/torque objects when every
        # active model is a known type; ``fused_rhs=False`` keeps the reference path.
        self._fused = compile_fused_rhs(self) if fused_rhs else None

    @property
    def epoch_inputs(self) -> EpochInputs:
        """Picklable recipe of the time-only inputs, reflecting the current ring and GM drift."""
        tide = next((m.tide_model for m in self.force_group.active_models if isinstance(m, EarthTideForce)), None)
        return EpochInputs(
            body_names=self.body_names,
            gravitational_parameters_m3_s2=self.gravitational_parameters_m3_s2,
            prescribed_body_names=self.prescribed_body_names,
            solar_gm_variation=self.solar_gm_variation,
            kuiper_belt_ring=self.kuiper_belt_ring,
            earth_axes=self._earth_axes,
            moon_axes=self._moon_axes,
            point_mass=self.force_group.has_point_mass,
            delayed_external_names=() if tide is None else tide.external_names,
            delayed_offsets_s=() if tide is None else tide.delay_seconds,
        )

    def build_epoch_data(self, epoch_tdb: Epoch) -> DynamicsEpochData:
        return self.build_epoch_data_batch((epoch_tdb,))[0]

    def build_epoch_data_batch(self, epochs_tdb: Sequence[Epoch]) -> tuple[DynamicsEpochData, ...]:
        epochs = tuple(epochs_tdb)
        return EpochInputs.records(epochs, self.epoch_inputs.arrays(self.ephemeris, epochs))

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
    ):
        epoch_data = self.build_epoch_data(epoch_tdb) if epoch_data is None else epoch_data
        inputs = self._load_evaluation(epoch_data, vector, history)
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

    def prepare_step(self, epoch_data: DynamicsEpochData, history, *, rigid: bool = False) -> StepData:
        """Evaluate everything that depends on accepted history but not on the candidate state."""
        inertia = None
        if self.rotation is not None and self.rotation.deformation is not None:
            inertia = self.rotation.deformation.evaluate(epoch_data.epoch_tdb, history, rigid=rigid)
        return StepData(self.force_group.prepare_step(epoch_data, history), inertia)

    def derivatives(
        self,
        epoch_tdb: Epoch,
        vector,
        epoch_data: DynamicsEpochData | None = None,
        *,
        history=None,
        rigid: bool = False,
        step: StepData | None = None,
    ) -> np.ndarray:
        """Return the state derivative; ``rigid`` uses the constant-inertia backward seed model.

        ``step`` carries the candidate-independent data of :meth:`prepare_step`.
        """
        y = np.asarray(vector, dtype=float)
        epoch_data = self.build_epoch_data(epoch_tdb) if epoch_data is None else epoch_data
        if self._fused is not None:
            if step is None:
                step = self.prepare_step(epoch_data, history, rigid=rigid)
            return self._fused.derivative(epoch_data, step, y)
        inputs = self._load_evaluation(epoch_data, y, history, rigid=rigid, step=step)
        body = self.force_group.compute_accelerations(inputs)
        parts = [y[3:6], body - epoch_data.ephemeris_earth_acceleration_mps2]
        if self.rotation is not None:
            parts.append(self.rotation.evaluate(inputs, y[6:]).derivative)
        return np.concatenate(parts)

    def _load_evaluation(self, epoch_data, vector, history, *, rigid=False, step: StepData | None = None):
        matrix = None if self.rotation is None else quaternion_matrix(vector[6:10])
        inputs = self._inputs.load_evaluation(
            epoch_data,
            vector,
            history,
            moon_matrix=matrix,
            step_payloads=None if step is None else step.force_payloads,
        )
        deformation = None if self.rotation is None else self.rotation.deformation
        if deformation is not None:
            inertia = (
                deformation.evaluate(epoch_data.epoch_tdb, history, rigid=rigid) if step is None else step.lunar_inertia
            )
            inputs.lunar_inertia = inertia
            if inertia is not None and inertia.degree2 is not None:
                inputs.coefficient_overrides[id(deformation.field)] = deformation.field.coefficients_with_degree2(
                    inertia.degree2
                )
        return inputs

    def rotation_evaluation(self, epoch, vector, epoch_data=None, *, history=None):
        data = self.build_epoch_data(epoch) if epoch_data is None else epoch_data
        inputs = self._load_evaluation(data, vector, history)
        return cast(LunarRotationDynamics, self.rotation).evaluate(inputs, vector[6:])
