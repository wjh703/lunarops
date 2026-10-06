"""Composable array-oriented acceleration models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np

from lunarops.classes.ephemerides.body_ids import body_name

from .context import ForceEvaluationContext
from .forces import (
    EarthTideModel,
    PointMassGravityCache,
    PointMassGravityEvaluation,
    SolarSourceParameters,
    build_point_mass_gravity_cache,
    evaluate_eih_correction,
    evaluate_newtonian_point_mass_system,
)
from .gravity import GravityField, ensure_gravity_field, make_j2_coefficients
from .inertia import LunarDegree2GravityModel


class ForceModel(Protocol):
    @property
    def name(self) -> str: ...

    def acceleration(
        self,
        inputs: ForceEvaluationContext,
        *,
        target_body_mask: np.ndarray,
        newtonian_evaluation: PointMassGravityEvaluation | None = None,
    ) -> np.ndarray: ...


def _apply_target_mask(body_accelerations_mps2: np.ndarray, target_body_mask: np.ndarray) -> np.ndarray:
    result = np.asarray(body_accelerations_mps2, dtype=float)
    if np.all(target_body_mask):
        return result
    result = result.copy()
    result[~target_body_mask] = 0.0
    return result


class LunarForceGroup:
    """Fixed force composition with masks compiled for one body ordering."""

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
        values = () if enabled_force_names is None else enabled_force_names
        if isinstance(values, (str, bytes)):
            raise TypeError("enabled_force_names must be a sequence of force names")
        terms = frozenset(str(term).strip() for term in values) if enabled_force_names is not None else frozenset(model_names)
        unknown = set(terms) - set(model_names)
        if unknown:
            raise ValueError(f"enabled_force_names contains unknown force terms: {sorted(unknown)}")
        if "eih_1pn" in terms and "point_mass" not in terms:
            raise ValueError("enabled_force_names enables eih_1pn without point_mass")
        self.enabled_force_names = terms
        point_mass_model = next((model for model in self.force_models if model.name == "point_mass"), None)
        if point_mass_model is not None and not isinstance(point_mass_model, NewtonianPointMassForce):
            raise TypeError("The point_mass model must be NewtonianPointMassForce")
        self._point_mass_model = point_mass_model
        self._body_names: tuple[str, ...] | None = None
        self._target_masks: dict[str, np.ndarray] = {}

    def configure_bodies(self, body_names: Sequence[str]) -> None:
        names = tuple(body_name(name) for name in body_names)
        if "MOON" not in names:
            raise ValueError("LunarForceGroup requires MOON in the dynamics system")
        self._body_names = names
        self._target_masks = {
            model.name: np.asarray(
                [name == "MOON" and model.name in self.enabled_force_names for name in names],
                dtype=bool,
            )
            for model in self.force_models
        }
        for value in self._target_masks.values():
            value.setflags(write=False)

    def compute_accelerations(self, inputs: ForceEvaluationContext, *, collect_terms: bool = False):
        if self._body_names != inputs.body_names:
            raise ValueError("LunarForceGroup body order does not match ForceEvaluationContext")
        total = np.zeros_like(inputs.positions_m)
        terms: dict[str, np.ndarray] | None = {} if collect_terms else None
        newtonian_evaluation: PointMassGravityEvaluation | None = None
        for model in self.force_models:
            mask = self._target_masks[model.name]
            if not np.any(mask):
                continue
            if model is self._point_mass_model:
                assert isinstance(model, NewtonianPointMassForce)
                newtonian_evaluation = model.evaluate(inputs)
                contribution = _apply_target_mask(newtonian_evaluation.accelerations_mps2, mask)
            else:
                contribution = model.acceleration(
                    inputs,
                    target_body_mask=mask,
                    newtonian_evaluation=newtonian_evaluation,
                )
            total += contribution
            if terms is not None:
                terms[model.name] = np.array(contribution, copy=True)
        return total, terms

    def build_point_mass_gravity_cache(
        self,
        body_positions_m: np.ndarray,
        integrated_body_count: int,
    ) -> PointMassGravityCache | None:
        """Build the point-mass gravity cache for one epoch."""
        if self._point_mass_model is None or "point_mass" not in self.enabled_force_names:
            return None
        return self._point_mass_model.build_cache(body_positions_m, integrated_body_count)


@dataclass(frozen=True, slots=True)
class NewtonianPointMassForce:
    body_parameters: tuple[tuple[str, float], ...]
    name: str = "point_mass"
    _body_names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _mu: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = tuple(body_name(name) for name, _ in self.body_parameters)
        mu = np.asarray([float(value) for _, value in self.body_parameters], dtype=float)
        if not names or len(names) != len(set(names)):
            raise ValueError("Point-mass bodies must have unique non-empty names")
        if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
            raise ValueError("Point-mass gravitational parameters must be positive and finite")
        mu.setflags(write=False)
        object.__setattr__(self, "body_parameters", tuple(zip(names, mu.tolist())))
        object.__setattr__(self, "_body_names", names)
        object.__setattr__(self, "_mu", mu)

    def build_cache(self, body_positions_m, integrated_body_count):
        return build_point_mass_gravity_cache(
            body_positions_m,
            self._mu,
            integrated_body_count=integrated_body_count,
        )

    def evaluate(self, inputs):
        if inputs.body_names != self._body_names:
            raise ValueError("NewtonianPointMassForce body order does not match the dynamics system")
        return evaluate_newtonian_point_mass_system(inputs.positions_m, self._mu, inputs.point_mass_gravity_cache)

    def acceleration(self, inputs, *, target_body_mask, newtonian_evaluation=None):
        evaluation = self.evaluate(inputs) if newtonian_evaluation is None else newtonian_evaluation
        return _apply_target_mask(evaluation.accelerations_mps2, target_body_mask)


@dataclass(frozen=True, slots=True)
class EihPointMassForce:
    body_parameters: tuple[tuple[str, float], ...]
    name: str = "eih_1pn"
    _body_names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _mu: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = tuple(body_name(name) for name, _ in self.body_parameters)
        mu = np.asarray([float(value) for _, value in self.body_parameters], dtype=float)
        if not names or len(names) != len(set(names)):
            raise ValueError("EIH bodies must have unique non-empty names")
        if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
            raise ValueError("EIH gravitational parameters must be positive and finite")
        mu.setflags(write=False)
        object.__setattr__(self, "body_parameters", tuple(zip(names, mu.tolist())))
        object.__setattr__(self, "_body_names", names)
        object.__setattr__(self, "_mu", mu)

    def acceleration(self, inputs, *, target_body_mask, newtonian_evaluation=None):
        if inputs.body_names != self._body_names:
            raise ValueError("EihPointMassForce body order does not match the dynamics system")
        if newtonian_evaluation is None:
            raise ValueError("EIH requires the Newtonian point-mass acceleration from the same RHS")
        return evaluate_eih_correction(
            inputs.positions_m,
            inputs.velocities_mps,
            self._mu,
            newtonian_evaluation=newtonian_evaluation,
            target_body_indices=np.flatnonzero(target_body_mask),
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

    def acceleration(self, inputs, *, target_body_mask, newtonian_evaluation=None):
        sun = inputs.body_indices["SUN"]
        targets = np.flatnonzero(target_body_mask)
        targets = targets[targets != sun]
        out = np.zeros_like(inputs.positions_m)
        if len(targets):
            relative = (inputs.positions_m[targets] - inputs.positions_m[sun]) @ self.inertial2solar_fixed_matrix.T
            out[targets] = self.gravity_field.nonspherical_accelerations(relative) @ self.inertial2solar_fixed_matrix
        return out


@dataclass(frozen=True, slots=True)
class LenseThirringForce:
    solar_parameters: SolarSourceParameters
    name: str = "lense_thirring"

    def acceleration(self, inputs, *, target_body_mask, newtonian_evaluation=None):
        sun = inputs.body_indices["SUN"]
        out = np.zeros_like(inputs.positions_m)
        for index in np.flatnonzero(target_body_mask):
            if index != sun:
                out[index] = self.solar_parameters.lense_thirring_acceleration(
                    inputs.positions_m[index] - inputs.positions_m[sun],
                    inputs.velocities_mps[index] - inputs.velocities_mps[sun],
                )
        return out


@dataclass(frozen=True, slots=True)
class SolarRadiationPressureForce:
    solar_parameters: SolarSourceParameters
    epsilon_by_body: Mapping[str, float]
    name: str = "solar_radiation_pressure"

    def __post_init__(self) -> None:
        values = {body_name(key): float(value) for key, value in self.epsilon_by_body.items()}
        if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("epsilon_by_body values must be finite and nonnegative")
        object.__setattr__(
            self,
            "epsilon_by_body",
            MappingProxyType(values),
        )

    def acceleration(self, inputs, *, target_body_mask, newtonian_evaluation=None):
        sun = inputs.body_indices["SUN"]
        out = np.zeros_like(inputs.positions_m)
        for index in np.flatnonzero(target_body_mask):
            if index != sun:
                out[index] = self.solar_parameters.radiation_pressure_acceleration(
                    inputs.positions_m[index] - inputs.positions_m[sun],
                    self.epsilon_by_body.get(inputs.body_names[index], 0.0),
                )
        return out


@dataclass(frozen=True, slots=True)
class EarthTideForce:
    tide_model: EarthTideModel
    name: str = "tide"

    def acceleration(self, inputs, *, target_body_mask, newtonian_evaluation=None):
        if inputs.epoch_tdb is None:
            raise RuntimeError("ForceEvaluationContext has no loaded epoch")
        earth = inputs.body_indices["EARTH"]
        moon = inputs.body_indices["MOON"]
        delta = self.tide_model.relative_acceleration(
            inputs.epoch_tdb,
            inputs.positions_m[earth],
            inputs.positions_m[moon],
            inputs.earth_fixed2inertial_matrix.T,
            history=inputs.history,
        )
        out = np.zeros_like(inputs.positions_m)
        out[moon] = delta
        return _apply_target_mask(out, target_body_mask)


@dataclass(frozen=True, slots=True)
class FigureForce:
    gravity_fields: Mapping[str, GravityField]
    figure_partners: Mapping[str, Sequence[str]]
    name: str = "figure"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gravity_fields",
            MappingProxyType({body_name(key): ensure_gravity_field(value) for key, value in self.gravity_fields.items()}),
        )
        if set(self.gravity_fields) - {"EARTH", "MOON"}:
            raise ValueError("FigureForce supports Earth and Moon gravity fields; use SolarJ2Force for the Sun")
        partners = {
            body_name(source): tuple(body_name(partner) for partner in values)
            for source, values in self.figure_partners.items()
        }
        if set(partners) != set(self.gravity_fields):
            raise ValueError("Figure interaction partners must be declared for every and only every figure source.")
        for source, values in partners.items():
            if not values or len(values) != len(set(values)) or source in values:
                raise ValueError(f"Figure source {source!r} requires unique, non-self interaction partners.")
        object.__setattr__(self, "figure_partners", MappingProxyType(partners))

    def _acceleration_with_fields(self, inputs, target_body_mask, fields):
        out = np.zeros_like(inputs.positions_m)
        mu = inputs.gravitational_parameters_m3_s2
        for source_name, field_model in fields.items():
            if source_name not in inputs.body_indices:
                continue
            source = inputs.body_indices[source_name]
            missing = set(self.figure_partners[source_name]) - set(inputs.body_indices)
            if missing:
                raise ValueError(f"Figure source {source_name!r} is missing interaction partners: {sorted(missing)}")
            targets = np.asarray(
                [inputs.body_indices[name] for name in self.figure_partners[source_name]], dtype=int
            )
            # Both stored matrices map body-fixed axes to inertial axes;
            # gravity evaluation uses their transpose.
            fixed2inertial_matrix = (
                inputs.earth_fixed2inertial_matrix
                if source_name == "EARTH"
                else inputs.moon_fixed2inertial_matrix
            )
            inertial2fixed_matrix = fixed2inertial_matrix.T
            relative_body_fixed = (inputs.positions_m[targets] - inputs.positions_m[source]) @ inertial2fixed_matrix.T
            direct_inertial = field_model.nonspherical_accelerations(relative_body_fixed) @ inertial2fixed_matrix
            out[targets] += direct_inertial
            out[source] -= np.sum((mu[targets] / mu[source])[:, None] * direct_inertial, axis=0)
        return _apply_target_mask(out, target_body_mask)

    def acceleration(self, inputs, *, target_body_mask, newtonian_evaluation=None):
        return self._acceleration_with_fields(inputs, target_body_mask, self.gravity_fields)


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
            np.isfinite(value)
            for value in (self.j2_rate_per_year, self.j2_quadratic_per_year2, self.reference_jd_tdb)
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

    def acceleration(self, inputs, *, target_body_mask, newtonian_evaluation=None):
        static = super().acceleration(
            inputs,
            target_body_mask=target_body_mask,
            newtonian_evaluation=newtonian_evaluation,
        )
        if inputs.epoch_tdb is None:
            raise RuntimeError("ForceEvaluationContext has no loaded epoch")
        elapsed_years = (
            float(inputs.epoch_tdb.jd1 + inputs.epoch_tdb.jd2) - self.reference_jd_tdb
        ) / 365.25
        delta_j2 = (
            self.j2_rate_per_year * elapsed_years
            + self.j2_quadratic_per_year2 * elapsed_years * elapsed_years
        )
        if delta_j2 == 0.0:
            return static
        unit = self._acceleration_with_fields(
            inputs,
            target_body_mask,
            {"EARTH": self._unit_j2_field},
        )
        return static + delta_j2 * unit


@dataclass(frozen=True, slots=True)
class LunarDegree2GravityCorrectionForce(FigureForce):
    """Acceleration increment from the Moon's time-varying degree-2 field.

    The static lunar field is evaluated separately as ``figure_moon``.  This
    model returns the dynamic field minus that static field so their sum is
    equivalent to a dynamic lunar field while evaluating only the degree-2
    increment in the production path.
    """

    lunar_degree2_gravity_model: LunarDegree2GravityModel | None = None
    name: str = "lunar_degree2_gravity"

    def acceleration(self, inputs, *, target_body_mask, newtonian_evaluation=None):
        if self.lunar_degree2_gravity_model is None or "MOON" not in self.gravity_fields:
            return np.zeros_like(inputs.positions_m)
        if inputs.epoch_tdb is None:
            raise RuntimeError("ForceEvaluationContext has no loaded epoch")
        earth_minus_moon_position_provider = None
        if inputs.history is not None:
            def earth_minus_moon_position_provider(epoch):
                states = np.asarray(inputs.history(("EARTH", "MOON"), epoch), dtype=float)
                if states.shape != (2, 6) or not np.all(np.isfinite(states)):
                    raise ValueError("Lunar history must return a finite (2,6) state matrix")
                return states[0, :3] - states[1, :3]

        cache = inputs.evaluation_cache
        cache_key = (
            "lunar-degree2-correction",
            id(self.lunar_degree2_gravity_model),
            id(self.gravity_fields["MOON"].coefficients),
            float(inputs.epoch_tdb.jd1),
            float(inputs.epoch_tdb.jd2),
            inputs.epoch_tdb.scale,
        )
        increment_field = None if cache is None else cache.get(cache_key)
        if increment_field is None:
            increment_field = self.lunar_degree2_gravity_model.degree2_gravity_correction_field(
                inputs.epoch_tdb,
                self.gravity_fields["MOON"].coefficients,
                earth_minus_moon_position_provider=earth_minus_moon_position_provider,
            )
            if cache is not None:
                cache[cache_key] = increment_field
        return self._acceleration_with_fields(
            inputs,
            target_body_mask,
            {"MOON": increment_field},
        )
