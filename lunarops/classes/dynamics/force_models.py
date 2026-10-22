"""Composable array-oriented acceleration models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from .context import DynamicsEvaluationContext
from .deformation import LunarDegree2GravityModel
from .earth_figure_orientation import earth_j2_increment
from .forces import (
    EarthTideModel,
    SolarSourceParameters,
    evaluate_eih_correction,
    evaluate_newtonian_point_mass_system,
)
from .gravity import GravityField


class ForceModel(Protocol):
    @property
    def name(self) -> str: ...

    def acceleration(self, inputs: DynamicsEvaluationContext) -> np.ndarray: ...


class LunarForceGroup:
    """Force composition compiled once for the integrated Moon."""

    def __init__(
        self,
        force_models: Sequence[ForceModel] = (),
        enabled_force_names: Sequence[str] | None = None,
    ):
        self.force_models = tuple(force_models)
        model_names = tuple(model.name for model in self.force_models)
        terms = frozenset(model_names if enabled_force_names is None else enabled_force_names)
        self._active_models = tuple(model for model in self.force_models if model.name in terms)
        self._step_models = tuple(model for model in self._active_models if hasattr(model, "prepare_step"))
        self._point_mass_model = next((model for model in self._active_models if model.name == "point_mass"), None)

    @property
    def active_models(self) -> tuple[ForceModel, ...]:
        return self._active_models

    @property
    def has_point_mass(self) -> bool:
        return self._point_mass_model is not None

    @property
    def active_force_names(self) -> tuple[str, ...]:
        return tuple(model.name for model in self._active_models)

    def _contribution(self, model, inputs: DynamicsEvaluationContext) -> np.ndarray:
        if model is not self._point_mass_model:
            return model.acceleration(inputs)
        inputs.newtonian_evaluation = NewtonianPointMassForce.evaluate(cast(NewtonianPointMassForce, model), inputs)
        return inputs.newtonian_evaluation.accelerations_mps2[inputs.moon_index]

    def compute_accelerations(self, inputs: DynamicsEvaluationContext) -> np.ndarray:
        total = np.zeros(3)
        for model in self._active_models:
            total += self._contribution(model, inputs)
        return total

    def acceleration_terms(self, inputs: DynamicsEvaluationContext) -> dict[str, np.ndarray]:
        return {model.name: self._contribution(model, inputs) for model in self._active_models}

    def prepare_step(self, epoch_data, history) -> dict[str, object]:
        """Candidate-independent data of the models that need accepted history, once per ABM step."""
        payloads = {}
        for model in self._step_models:
            payload = model.prepare_step(epoch_data, history)  # type: ignore[attr-defined]
            if payload is not None:
                payloads[model.name] = payload
        return payloads


@dataclass(frozen=True, slots=True)
class NewtonianPointMassForce:
    name: str = "point_mass"

    def evaluate(self, inputs):
        return evaluate_newtonian_point_mass_system(
            inputs.positions_m, inputs.gravitational_parameters_m3_s2, inputs.epoch_data.point_mass_gravity_cache
        )

    def acceleration(self, inputs):
        return self.evaluate(inputs).accelerations_mps2[inputs.moon_index]


@dataclass(frozen=True, slots=True)
class EihPointMassForce:
    name: str = "eih_1pn"

    def acceleration(self, inputs):
        return evaluate_eih_correction(
            inputs.positions_m,
            inputs.velocities_mps,
            inputs.gravitational_parameters_m3_s2,
            inputs.newtonian_evaluation,
            inputs.moon_targets,
        )[inputs.moon_index]


@dataclass(frozen=True, slots=True)
class SolarJ2Force:
    gravity_field: GravityField
    inertial2solar_fixed_matrix: np.ndarray
    name: str = "solar_j2"

    def __post_init__(self) -> None:
        matrix = np.array(self.inertial2solar_fixed_matrix, dtype=float, copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "inertial2solar_fixed_matrix", matrix)

    def acceleration(self, inputs):
        sun = inputs.body_indices["SUN"]
        targets = inputs.moon_targets
        relative = (inputs.positions_m[targets] - inputs.positions_m[sun]) @ self.inertial2solar_fixed_matrix.T
        scale = inputs.gravitational_parameters_m3_s2[sun] / self.gravity_field.gm_m3_s2
        return (scale * (self.gravity_field.nonspherical_accelerations(relative) @ self.inertial2solar_fixed_matrix))[0]


@dataclass(frozen=True, slots=True)
class LenseThirringForce:
    solar_parameters: SolarSourceParameters
    name: str = "lense_thirring"

    def acceleration(self, inputs):
        sun = inputs.body_indices["SUN"]
        moon = inputs.moon_index
        out = self.solar_parameters.lense_thirring_acceleration(
            inputs.positions_m[moon] - inputs.positions_m[sun],
            inputs.velocities_mps[moon] - inputs.velocities_mps[sun],
        )
        out *= inputs.gravitational_parameters_m3_s2[sun] / self.solar_parameters.gravitational_parameter_m3_s2
        return out


@dataclass(frozen=True, slots=True)
class SolarRadiationPressureForce:
    """Solar radiation pressure on the Moon; ``epsilon`` is its area-to-mass coupling."""

    solar_parameters: SolarSourceParameters
    epsilon: float
    name: str = "solar_radiation_pressure"

    def acceleration(self, inputs):
        sun = inputs.body_indices["SUN"]
        moon = inputs.moon_index
        out = self.solar_parameters.radiation_pressure_acceleration(
            inputs.positions_m[moon] - inputs.positions_m[sun],
            self.epsilon,
        )
        out *= inputs.gravitational_parameters_m3_s2[sun] / self.solar_parameters.gravitational_parameter_m3_s2
        return out


@dataclass(frozen=True, slots=True)
class EarthTideForce:
    tide_model: EarthTideModel
    name: str = "tide"

    def prepare_step(self, epoch_data, history):
        matrix = np.ascontiguousarray(epoch_data.earth_fixed2inertial_matrix.T)
        return (
            matrix,
            *self.tide_model.delayed_vectors(
                epoch_data.epoch_tdb, matrix, history, epoch_data.delayed_external_states
            ),
        )

    def acceleration(self, inputs):
        payload = None if inputs.step_payloads is None else inputs.step_payloads.get(self.name)
        if payload is None:
            payload = self.prepare_step(inputs.epoch_data, inputs.history)
        matrix, vectors, raiser_gm = payload
        delta = self.tide_model.acceleration_from_vectors(
            inputs.positions_m[inputs.moon_index] - inputs.positions_m[inputs.earth_index], matrix, vectors, raiser_gm
        )
        # JPL's tidal kernel returns Moon-minus-Earth acceleration and leaves
        # the EMB unchanged. The prescribed Earth already includes its share.
        mu_earth = self.tide_model.earth_gravitational_parameter_m3_s2
        mu_moon = self.tide_model.moon_gravitational_parameter_m3_s2
        return (mu_earth / (mu_earth + mu_moon)) * delta


@dataclass(frozen=True, slots=True)
class EarthFigureForce:
    """Earth's non-spherical field acting on the Moon, with the JPL J2(t) polynomial.

    ``j2_rate_per_year`` and ``j2_quadratic_per_year2`` are polynomial coefficients
    (JPL ``J2EDOT``/``J2ET2``), so the quadratic term carries no 1/2 factor.
    """

    field: GravityField
    j2_rate_per_year: float = 0.0
    j2_quadratic_per_year2: float = 0.0
    reference_jd_tdb: float = 2_451_545.0
    name: str = "figure_earth"

    def c20(self, epoch) -> float:
        delta = earth_j2_increment(epoch, self.reference_jd_tdb, self.j2_rate_per_year, self.j2_quadratic_per_year2)
        return float(self.field.coefficients.coeffs[0, 2, 0]) - delta / np.sqrt(5.0)

    def acceleration(self, inputs):
        inputs.coefficient_overrides[id(self.field)] = self.field.coefficients_with_c20(
            self.c20(inputs.epoch_data.epoch_tdb)
        )
        _, direct = inputs.figure_accelerations(inputs.earth_index, inputs.moon_targets, self.field)
        out = np.zeros(3)
        out += (direct @ inputs.epoch_data.earth_fixed2inertial_matrix.T)[0]
        return out


@dataclass(frozen=True, slots=True)
class MoonFigureForce:
    """The Moon's reaction to its own non-spherical field acting on ``partners``.

    With ``lunar_degree2_gravity_model`` the degree 2 follows the prescribed delayed
    deformation; with integrated rotation it follows the dynamics' delayed inertia.
    """

    field: GravityField
    partners: tuple[str, ...]
    lunar_degree2_gravity_model: LunarDegree2GravityModel | None = None
    name: str = "figure_moon"

    def __post_init__(self) -> None:
        object.__setattr__(self, "partners", tuple(self.partners))

    def _degree2(self, epoch_data, history) -> np.ndarray:
        model = self.lunar_degree2_gravity_model
        assert model is not None

        def relative_position(epoch):
            states = history(("EARTH", "MOON"), epoch)
            return states[0, :3] - states[1, :3]

        return model.degree2_coefficients(epoch_data.epoch_tdb, relative_position)

    def prepare_step(self, epoch_data, history):
        """Delayed degree 2 once per step; every corrector evaluation reuses it."""
        if self.lunar_degree2_gravity_model is None:
            return None
        return self._degree2(epoch_data, history)

    def acceleration(self, inputs):
        if self.lunar_degree2_gravity_model is not None:
            degree2 = None if inputs.step_payloads is None else inputs.step_payloads.get(self.name)
            if degree2 is None:
                degree2 = self._degree2(inputs.epoch_data, inputs.history)
            inputs.coefficient_overrides[id(self.field)] = self.field.coefficients_with_degree2(degree2)
        moon = inputs.moon_index
        mu = inputs.gravitational_parameters_m3_s2
        targets = np.asarray([inputs.body_indices[name] for name in self.partners], dtype=int)
        _, direct = inputs.figure_accelerations(moon, targets, self.field)
        direct_inertial = direct @ inputs.moon_fixed2inertial_matrix.T
        out = np.zeros(3)
        out -= np.sum((mu[targets] / mu[moon])[:, None] * direct_inertial, axis=0)
        return out
