"""Composable array-oriented acceleration models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from lunarops.classes.ephemerides.body_ids import body_name

from .context import DynamicsContext
from .forces import (
    EarthTideModel,
    NewtonianPointMassEvaluation,
    SolarForceParameters,
    eih_acceleration,
    evaluate_point_mass_system,
    prepare_point_mass_system,
)
from .gravity import GravityField, as_gravity_field
from .inertia import LunarInertiaModel

_IDENTITY3 = np.eye(3)


class ForceModel(Protocol):
    @property
    def name(self) -> str: ...

    def acceleration(
        self,
        context: DynamicsContext,
        *,
        target_mask: np.ndarray,
        point_mass_evaluation: NewtonianPointMassEvaluation | None = None,
    ) -> np.ndarray: ...


def _apply_mask(acceleration: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    result = np.asarray(acceleration, dtype=float)
    if np.all(target_mask):
        return result
    result = result.copy()
    result[~target_mask] = 0.0
    return result


class ForceGroup:
    """Fixed force composition with masks compiled for one body ordering."""

    def __init__(self, models: Sequence[ForceModel] = (), body_terms: Mapping[str, Sequence[str]] | None = None):
        self.models = tuple(models)
        model_names = tuple(model.name for model in self.models)
        if len(model_names) != len(set(model_names)):
            raise ValueError("Force model names must be unique within a ForceGroup.")
        if "eih_1pn" in model_names and (
            "point_mass" not in model_names or model_names.index("eih_1pn") < model_names.index("point_mass")
        ):
            raise ValueError("ForceGroup must place point_mass before eih_1pn.")
        terms: dict[str, frozenset[str]] = {}
        for body, values in (body_terms or {}).items():
            if isinstance(values, (str, bytes)):
                raise TypeError(f"body_terms[{body!r}] must be a sequence of force names")
            terms[body_name(body)] = frozenset(str(term).strip() for term in values)
        unknown = {term for values in terms.values() for term in values if term not in set(model_names)}
        if unknown:
            raise ValueError(f"body_terms contains unknown force terms: {sorted(unknown)}")
        for body, enabled_terms in terms.items():
            if "eih_1pn" in enabled_terms and "point_mass" not in enabled_terms:
                raise ValueError(f"Body {body!r} enables eih_1pn without point_mass.")
        self.body_terms = terms
        self._body_names: tuple[str, ...] | None = None
        self._target_masks: dict[str, np.ndarray] = {}
        self._all_targets = np.empty(0, dtype=bool)

    def bind(self, body_names: Sequence[str]) -> None:
        names = tuple(body_name(name) for name in body_names)
        if self.body_terms:
            unknown_bodies = set(self.body_terms) - set(names)
            if unknown_bodies:
                raise ValueError(f"body_terms contains bodies outside the dynamics system: {sorted(unknown_bodies)}")
        self._body_names = names
        self._target_masks = {
            model.name: np.asarray(
                [not self.body_terms or model.name in self.body_terms.get(name, frozenset()) for name in names],
                dtype=bool,
            )
            for model in self.models
        }
        for value in self._target_masks.values():
            value.setflags(write=False)
        for model in self.models:
            validate = getattr(model, "validate_target_mask", None)
            if callable(validate):
                validate(names, self._target_masks[model.name])
        self._all_targets = np.ones(len(names), dtype=bool)
        self._all_targets.setflags(write=False)

    def accelerations(self, context: DynamicsContext, *, collect_terms: bool = False):
        if self._body_names != context.body_names:
            raise ValueError("ForceGroup is not bound to this DynamicsContext body order")
        total = np.zeros_like(context.positions_m)
        terms: dict[str, np.ndarray] | None = {} if collect_terms else None
        point_mass_evaluation: NewtonianPointMassEvaluation | None = None
        for model in self.models:
            mask = self._target_masks[model.name]
            if model.name == "point_mass":
                if not isinstance(model, PointMassForce):
                    raise TypeError("The point_mass model must be PointMassForce")
                point_mass_evaluation = model.evaluate(context)
                contribution = _apply_mask(point_mass_evaluation.accelerations_mps2, mask)
            else:
                if not np.any(mask):
                    continue
                contribution = model.acceleration(
                    context,
                    target_mask=mask,
                    point_mass_evaluation=point_mass_evaluation,
                )
            total += contribution
            if terms is not None:
                terms[model.name] = np.array(contribution, copy=True)
        if any(model.name == "eih_1pn" for model in self.models) and point_mass_evaluation is None:
            raise ValueError("EIH requires point_mass in the same ForceGroup")
        return total, terms

    def prepare_epoch(self, positions_m: np.ndarray, integrated_count: int) -> dict[str, object]:
        """Prepare force data that depends only on prescribed states at one epoch."""
        result: dict[str, object] = {}
        for model in self.models:
            prepare = getattr(model, "prepare_epoch", None)
            if callable(prepare):
                result[model.name] = prepare(positions_m, integrated_count)
        return result


@dataclass(frozen=True, slots=True)
class PointMassForce:
    bodies: tuple[tuple[str, float], ...]
    name: str = "point_mass"
    _body_names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _mu: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = tuple(body_name(name) for name, _ in self.bodies)
        mu = np.asarray([float(value) for _, value in self.bodies], dtype=float)
        if not names or len(names) != len(set(names)):
            raise ValueError("Point-mass bodies must have unique non-empty names")
        if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
            raise ValueError("Point-mass gravitational parameters must be positive and finite")
        mu.setflags(write=False)
        object.__setattr__(self, "bodies", tuple(zip(names, mu.tolist())))
        object.__setattr__(self, "_body_names", names)
        object.__setattr__(self, "_mu", mu)

    def prepare_epoch(self, positions_m, integrated_count):
        return prepare_point_mass_system(positions_m, self._mu, integrated_count=integrated_count)

    def evaluate(self, context):
        if context.body_names != self._body_names:
            raise ValueError("PointMassForce body order does not match the dynamics system")
        return evaluate_point_mass_system(context.positions_m, self._mu, context.force_data.get(self.name))

    def acceleration(self, context, *, target_mask, point_mass_evaluation=None):
        evaluation = self.evaluate(context) if point_mass_evaluation is None else point_mass_evaluation
        return _apply_mask(evaluation.accelerations_mps2, target_mask)


@dataclass(frozen=True, slots=True)
class EihForce:
    bodies: tuple[tuple[str, float], ...]
    name: str = "eih_1pn"
    _body_names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _mu: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = tuple(body_name(name) for name, _ in self.bodies)
        mu = np.asarray([float(value) for _, value in self.bodies], dtype=float)
        if not names or len(names) != len(set(names)):
            raise ValueError("EIH bodies must have unique non-empty names")
        if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
            raise ValueError("EIH gravitational parameters must be positive and finite")
        mu.setflags(write=False)
        object.__setattr__(self, "bodies", tuple(zip(names, mu.tolist())))
        object.__setattr__(self, "_body_names", names)
        object.__setattr__(self, "_mu", mu)

    def acceleration(self, context, *, target_mask, point_mass_evaluation=None):
        if context.body_names != self._body_names:
            raise ValueError("EihForce body order does not match the dynamics system")
        if point_mass_evaluation is None:
            raise ValueError("EIH requires the Newtonian point-mass acceleration from the same RHS")
        return eih_acceleration(
            context.positions_m,
            context.velocities_mps,
            self._mu,
            point_mass_evaluation,
            np.flatnonzero(target_mask),
        )


@dataclass(frozen=True, slots=True)
class SolarJ2Force:
    field: GravityField
    inertial_to_solar_body_fixed: np.ndarray
    name: str = "solar_j2"

    def __post_init__(self) -> None:
        if not isinstance(self.field, GravityField):
            raise TypeError("field must be a GravityField")
        frame = np.array(self.inertial_to_solar_body_fixed, dtype=float, copy=True)
        if (
            frame.shape != (3, 3)
            or not np.all(np.isfinite(frame))
            or not np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-14)
            or not np.isclose(np.linalg.det(frame), 1.0, rtol=0, atol=1e-14)
        ):
            raise ValueError("inertial_to_solar_body_fixed must be a proper orthogonal matrix")
        frame.setflags(write=False)
        object.__setattr__(self, "inertial_to_solar_body_fixed", frame)

    def acceleration(self, context, *, target_mask, point_mass_evaluation=None):
        sun = context.body_indices["SUN"]
        targets = np.flatnonzero(target_mask)
        targets = targets[targets != sun]
        out = np.zeros_like(context.positions_m)
        if len(targets):
            relative = (context.positions_m[targets] - context.positions_m[sun]) @ self.inertial_to_solar_body_fixed.T
            out[targets] = self.field.figure_accelerations(relative) @ self.inertial_to_solar_body_fixed
        return out


@dataclass(frozen=True, slots=True)
class LenseThirringForce:
    parameters: SolarForceParameters
    name: str = "lense_thirring"

    def acceleration(self, context, *, target_mask, point_mass_evaluation=None):
        sun = context.body_indices["SUN"]
        out = np.zeros_like(context.positions_m)
        for index in np.flatnonzero(target_mask):
            if index != sun:
                out[index] = self.parameters.lense_thirring(
                    context.positions_m[index] - context.positions_m[sun],
                    context.velocities_mps[index] - context.velocities_mps[sun],
                )
        return out


@dataclass(frozen=True, slots=True)
class SolarRadiationPressureForce:
    parameters: SolarForceParameters
    epsilon_by_body: Mapping[str, float]
    name: str = "solar_radiation_pressure"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epsilon_by_body",
            {body_name(key): float(value) for key, value in self.epsilon_by_body.items()},
        )

    def acceleration(self, context, *, target_mask, point_mass_evaluation=None):
        sun = context.body_indices["SUN"]
        out = np.zeros_like(context.positions_m)
        for index in np.flatnonzero(target_mask):
            if index != sun:
                out[index] = self.parameters.radiation_pressure(
                    context.positions_m[index] - context.positions_m[sun],
                    self.epsilon_by_body.get(context.body_names[index], 0.0),
                )
        return out


@dataclass(frozen=True, slots=True)
class EarthTideForce:
    model: EarthTideModel
    name: str = "tide"

    @staticmethod
    def validate_target_mask(body_names, target_mask) -> None:
        indices = {name: index for index, name in enumerate(body_names)}
        if "EARTH" not in indices or "MOON" not in indices:
            raise ValueError("Earth tide requires EARTH and MOON in the dynamics system.")
        if bool(target_mask[indices["EARTH"]]) != bool(target_mask[indices["MOON"]]):
            raise ValueError("Earth tide must be enabled for EARTH and MOON together.")

    def acceleration(self, context, *, target_mask, point_mass_evaluation=None):
        if context.epoch_tdb is None:
            raise RuntimeError("DynamicsContext is not bound to an epoch")
        earth = context.body_indices["EARTH"]
        moon = context.body_indices["MOON"]
        delta = self.model.relative_acceleration(
            context.epoch_tdb,
            context.positions_m[earth],
            context.positions_m[moon],
            context.frames.get("EARTH"),
            history=context.history,
        )
        split = EarthTideModel.split_relative_acceleration(delta, self.model.earth_gm, self.model.moon_gm)
        out = np.zeros_like(context.positions_m)
        out[earth], out[moon] = split
        return _apply_mask(out, target_mask)


@dataclass(frozen=True, slots=True)
class FigureForce:
    fields: Mapping[str, GravityField]
    interaction_partners: Mapping[str, Sequence[str]]
    name: str = "figure"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fields",
            {body_name(key): as_gravity_field(value) for key, value in self.fields.items()},
        )
        partners = {
            body_name(source): tuple(body_name(partner) for partner in values)
            for source, values in self.interaction_partners.items()
        }
        if set(partners) != set(self.fields):
            raise ValueError("Figure interaction partners must be declared for every and only every figure source.")
        for source, values in partners.items():
            if not values or len(values) != len(set(values)) or source in values:
                raise ValueError(f"Figure source {source!r} requires unique, non-self interaction partners.")
        object.__setattr__(self, "interaction_partners", partners)

    def _acceleration_with_fields(self, context, target_mask, fields):
        out = np.zeros_like(context.positions_m)
        mu = context.gravitational_parameters_m3_s2
        for source_name, field_model in fields.items():
            if source_name not in context.body_indices:
                continue
            source = context.body_indices[source_name]
            missing = set(self.interaction_partners[source_name]) - set(context.body_indices)
            if missing:
                raise ValueError(f"Figure source {source_name!r} is missing interaction partners: {sorted(missing)}")
            targets = np.asarray(
                [context.body_indices[name] for name in self.interaction_partners[source_name]], dtype=int
            )
            frame = context.frames.get(source_name, _IDENTITY3)
            relative_body_fixed = (context.positions_m[targets] - context.positions_m[source]) @ frame.T
            direct_inertial = field_model.figure_accelerations(relative_body_fixed) @ frame
            out[targets] += direct_inertial
            out[source] -= np.sum((mu[targets] / mu[source])[:, None] * direct_inertial, axis=0)
        return _apply_mask(out, target_mask)

    def acceleration(self, context, *, target_mask, point_mass_evaluation=None):
        return self._acceleration_with_fields(context, target_mask, self.fields)


@dataclass(frozen=True, slots=True)
class LunarInertiaFigureForce(FigureForce):
    inertia: LunarInertiaModel | None = None
    name: str = "figure_moon"

    def acceleration(self, context, *, target_mask, point_mass_evaluation=None):
        if self.inertia is None or "MOON" not in self.fields:
            return super().acceleration(
                context,
                target_mask=target_mask,
                point_mass_evaluation=point_mass_evaluation,
            )
        if context.epoch_tdb is None:
            raise RuntimeError("DynamicsContext is not bound to an epoch")
        field, _ = self.inertia.field(
            context.epoch_tdb,
            self.fields["MOON"].coefficients,
            history=context.history,
        )
        fields = dict(self.fields)
        fields["MOON"] = as_gravity_field(field)
        return self._acceleration_with_fields(context, target_mask, fields)
