"""Composable array-oriented acceleration models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np

from .context import DynamicsEvaluationContext
from .earth_figure_orientation import earth_j2_increment
from .forces import (
    EarthTideModel,
    PointMassGravityCache,
    SolarSourceParameters,
    build_point_mass_gravity_cache_batch,
    evaluate_eih_correction,
    evaluate_newtonian_point_mass_system,
)
from .gravity import GravityField, make_j2_coefficients
from .inertia import LunarDegree2GravityModel


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
        if len(model_names) != len(set(model_names)):
            raise ValueError("Force model names must be unique within a LunarForceGroup.")
        if "eih_1pn" in model_names and (
            "point_mass" not in model_names or model_names.index("eih_1pn") < model_names.index("point_mass")
        ):
            raise ValueError("LunarForceGroup must place point_mass before eih_1pn.")
        terms = frozenset(model_names if enabled_force_names is None else enabled_force_names)
        unknown = set(terms) - set(model_names)
        if unknown:
            raise ValueError(f"enabled_force_names contains unknown force terms: {sorted(unknown)}")
        if "eih_1pn" in terms and "point_mass" not in terms:
            raise ValueError("enabled_force_names enables eih_1pn without point_mass")
        self._active_models = tuple(model for model in self.force_models if model.name in terms)
        self._point_mass_model = next((model for model in self._active_models if model.name == "point_mass"), None)

    @property
    def active_models(self) -> tuple[ForceModel, ...]:
        return self._active_models

    @property
    def active_force_names(self) -> tuple[str, ...]:
        return tuple(model.name for model in self._active_models)

    def _contribution(self, model, inputs: DynamicsEvaluationContext) -> np.ndarray:
        if model is not self._point_mass_model:
            return model.acceleration(inputs)
        inputs.newtonian_evaluation = NewtonianPointMassForce.evaluate(model, inputs)
        contribution = np.zeros_like(inputs.positions_m)
        contribution[inputs.moon_index] = inputs.newtonian_evaluation.accelerations_mps2[inputs.moon_index]
        return contribution

    def compute_accelerations(self, inputs: DynamicsEvaluationContext) -> np.ndarray:
        total = np.zeros_like(inputs.positions_m)
        for model in self._active_models:
            total += self._contribution(model, inputs)
        return total

    def acceleration_terms(self, inputs: DynamicsEvaluationContext) -> dict[str, np.ndarray]:
        return {model.name: self._contribution(model, inputs) for model in self._active_models}

    def build_point_mass_gravity_cache_batch(
        self,
        body_positions_m: np.ndarray,
        integrated_body_count: int,
        gravitational_parameters_m3_s2: np.ndarray,
    ) -> tuple[PointMassGravityCache, ...] | None:
        """Build point-mass caches for a batch of epochs."""
        if self._point_mass_model is None:
            return None
        return build_point_mass_gravity_cache_batch(
            body_positions_m, gravitational_parameters_m3_s2, integrated_body_count=integrated_body_count
        )


@dataclass(frozen=True, slots=True)
class NewtonianPointMassForce:
    name: str = "point_mass"

    def evaluate(self, inputs):
        return evaluate_newtonian_point_mass_system(
            inputs.positions_m, inputs.gravitational_parameters_m3_s2, inputs.epoch_data.point_mass_gravity_cache
        )

    def acceleration(self, inputs):
        return self.evaluate(inputs).accelerations_mps2


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
        )


@dataclass(frozen=True, slots=True)
class SolarJ2Force:
    gravity_field: GravityField
    inertial2solar_fixed_matrix: np.ndarray
    name: str = "solar_j2"

    def __post_init__(self) -> None:
        if not isinstance(self.gravity_field, GravityField):
            raise TypeError("gravity_field must be a GravityField")
        matrix = np.array(self.inertial2solar_fixed_matrix, dtype=float, copy=True)
        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix @ matrix.T, np.eye(3), rtol=0, atol=1e-14)
            or not np.isclose(np.linalg.det(matrix), 1.0, rtol=0, atol=1e-14)
        ):
            raise ValueError("inertial2solar_fixed_matrix must be a proper orthogonal matrix")
        matrix.setflags(write=False)
        object.__setattr__(self, "inertial2solar_fixed_matrix", matrix)

    def acceleration(self, inputs):
        sun = inputs.body_indices["SUN"]
        targets = inputs.moon_targets
        out = np.zeros_like(inputs.positions_m)
        relative = (inputs.positions_m[targets] - inputs.positions_m[sun]) @ self.inertial2solar_fixed_matrix.T
        scale = inputs.gravitational_parameters_m3_s2[sun] / self.gravity_field.gm_m3_s2
        out[targets] = scale * (self.gravity_field.nonspherical_accelerations(relative) @ self.inertial2solar_fixed_matrix)
        return out


@dataclass(frozen=True, slots=True)
class LenseThirringForce:
    solar_parameters: SolarSourceParameters
    name: str = "lense_thirring"

    def acceleration(self, inputs):
        sun = inputs.body_indices["SUN"]
        moon = inputs.moon_index
        out = np.zeros_like(inputs.positions_m)
        out[moon] = self.solar_parameters.lense_thirring_acceleration(
            inputs.positions_m[moon] - inputs.positions_m[sun],
            inputs.velocities_mps[moon] - inputs.velocities_mps[sun],
        )
        out[moon] *= inputs.gravitational_parameters_m3_s2[sun] / self.solar_parameters.gravitational_parameter_m3_s2
        return out


@dataclass(frozen=True, slots=True)
class SolarRadiationPressureForce:
    solar_parameters: SolarSourceParameters
    epsilon_by_body: Mapping[str, float]
    name: str = "solar_radiation_pressure"

    def __post_init__(self) -> None:
        values = {key: float(value) for key, value in self.epsilon_by_body.items()}
        if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("epsilon_by_body values must be finite and nonnegative")
        object.__setattr__(
            self,
            "epsilon_by_body",
            MappingProxyType(values),
        )

    def acceleration(self, inputs):
        sun = inputs.body_indices["SUN"]
        moon = inputs.moon_index
        out = np.zeros_like(inputs.positions_m)
        out[moon] = self.solar_parameters.radiation_pressure_acceleration(
            inputs.positions_m[moon] - inputs.positions_m[sun],
            self.epsilon_by_body.get("MOON", 0.0),
        )
        out[moon] *= inputs.gravitational_parameters_m3_s2[sun] / self.solar_parameters.gravitational_parameter_m3_s2
        return out


@dataclass(frozen=True, slots=True)
class EarthTideForce:
    tide_model: EarthTideModel
    name: str = "tide"

    def acceleration(self, inputs):
        earth = inputs.body_indices["EARTH"]
        moon = inputs.body_indices["MOON"]
        delta = self.tide_model.relative_acceleration(
            inputs.epoch_data.epoch_tdb,
            inputs.positions_m[earth],
            inputs.positions_m[moon],
            inputs.epoch_data.earth_fixed2inertial_matrix.T,
            inputs.history,
        )
        out = np.zeros_like(inputs.positions_m)
        # JPL's tidal kernel returns Moon-minus-Earth acceleration and leaves
        # the EMB unchanged. The prescribed Earth already includes its share.
        mu_earth = self.tide_model.earth_gravitational_parameter_m3_s2
        mu_moon = self.tide_model.moon_gravitational_parameter_m3_s2
        out[moon] = (mu_earth / (mu_earth + mu_moon)) * delta
        return out


@dataclass(frozen=True, slots=True)
class FigureForce:
    gravity_fields: Mapping[str, GravityField]
    figure_partners: Mapping[str, Sequence[str]]
    name: str = "figure"
    lunar_degree2_gravity_model: LunarDegree2GravityModel | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gravity_fields", MappingProxyType(dict(self.gravity_fields)))
        if set(self.gravity_fields) - {"EARTH", "MOON"}:
            raise ValueError("FigureForce supports Earth and Moon gravity fields; use SolarJ2Force for the Sun")
        partners = {source: tuple(values) for source, values in self.figure_partners.items()}
        if set(partners) != set(self.gravity_fields):
            raise ValueError("Figure interaction partners must be declared for every and only every figure source.")
        for source, values in partners.items():
            if not values or len(values) != len(set(values)) or source in values:
                raise ValueError(f"Figure source {source!r} requires unique, non-self interaction partners.")
        object.__setattr__(self, "figure_partners", MappingProxyType(partners))

    def _acceleration_with_fields(self, inputs, fields):
        out = np.zeros_like(inputs.positions_m)
        mu = inputs.gravitational_parameters_m3_s2
        for source_name, field_model in fields.items():
            source = inputs.body_indices[source_name]
            targets = np.asarray([inputs.body_indices[name] for name in self.figure_partners[source_name]], dtype=int)
            # Both stored matrices map body-fixed axes to inertial axes;
            # gravity evaluation uses their transpose.
            fixed2inertial_matrix = (
                inputs.epoch_data.earth_fixed2inertial_matrix
                if source_name == "EARTH"
                else inputs.moon_fixed2inertial_matrix
            )
            inertial2fixed_matrix = fixed2inertial_matrix.T
            _relative_body_fixed, direct_fixed = inputs.figure_accelerations(source, targets, field_model)
            direct_inertial = direct_fixed @ inertial2fixed_matrix
            out[targets] += direct_inertial
            out[source] -= np.sum((mu[targets] / mu[source])[:, None] * direct_inertial, axis=0)
        result = np.zeros_like(out)
        result[inputs.moon_index] = out[inputs.moon_index]
        return result

    def acceleration(self, inputs):
        if self.lunar_degree2_gravity_model is not None:
            def relative_position(epoch):
                states = inputs.history(("EARTH", "MOON"), epoch)
                return states[0, :3] - states[1, :3]
            self.lunar_degree2_gravity_model.update_gravity_field(
                inputs.epoch_data.epoch_tdb, self.gravity_fields["MOON"],
                earth_minus_moon_position_provider=relative_position,
            )
        return self._acceleration_with_fields(inputs, self.gravity_fields)


@dataclass(frozen=True, slots=True)
class TimeVaryingEarthFigureForce(FigureForce):
    """Earth figure field with a J2000-referenced J2 time polynomial.

    ``j2_rate_per_year`` and ``j2_quadratic_per_year2`` are polynomial
    coefficients, so the quadratic term is applied directly without a 1/2
    factor.  This represents the JPL ``J2EDOT``/``J2ET2`` convention used by
    DE440 while retaining the linear DE430 case.
    """

    j2_rate_per_year: float = 0.0
    j2_quadratic_per_year2: float = 0.0
    reference_jd_tdb: float = 2_451_545.0
    name: str = "figure_earth"
    _unit_j2_field: GravityField = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if set(self.gravity_fields) != {"EARTH"}:
            raise ValueError("TimeVaryingEarthFigureForce requires exactly one EARTH field")
        if not all(
            np.isfinite(value) for value in (self.j2_rate_per_year, self.j2_quadratic_per_year2, self.reference_jd_tdb)
        ):
            raise ValueError("Earth J2 polynomial coefficients and reference epoch must be finite")
        earth = self.gravity_fields["EARTH"]
        unit = GravityField(
            make_j2_coefficients(
                gm_m3_s2=earth.gm_m3_s2,
                radius_m=earth.radius_m,
                j2=1.0,
                name="unit Earth J2 increment",
            )
        )
        object.__setattr__(self, "_unit_j2_field", unit)

    def acceleration(self, inputs):
        static = super().acceleration(inputs)
        delta_j2 = earth_j2_increment(inputs.epoch_data.epoch_tdb, self.reference_jd_tdb, self.j2_rate_per_year, self.j2_quadratic_per_year2)
        if delta_j2 == 0.0:
            return static
        unit = self._acceleration_with_fields(
            inputs,
            {"EARTH": self._unit_j2_field},
        )
        return static + delta_j2 * unit


