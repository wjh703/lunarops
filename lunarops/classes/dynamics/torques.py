"""Moon-only external torques in PA, normalized by whole-Moon M R²."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .context import DynamicsEvaluationContext
from .gravity import GravityField


class LunarTorqueModel(Protocol):
    name: str

    def torque(self, inputs: DynamicsEvaluationContext) -> np.ndarray: ...


class LunarTorqueGroup:
    def __init__(self, models=()):
        self.models = tuple(models)
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("Lunar torque names must be unique")

    def evaluate(self, inputs):
        terms = {model.name: model.torque(inputs) for model in self.models}
        return sum(terms.values(), start=np.zeros(3)), terms


@dataclass(frozen=True, slots=True)
class PointMassGravityTorque:
    """Quadrupole-only baseline; inertia is the full Moon's tensor."""

    source: str
    total_inertia: np.ndarray

    @property
    def name(self):
        return f"quadrupole:{self.source}"

    def torque(self, inputs):
        source = inputs.body_indices[self.source]
        r = inputs.moon_fixed2inertial_matrix.T @ (inputs.positions_m[source] - inputs.positions_m[1])
        distance = np.linalg.norm(r)
        u = r / distance
        K = self.total_inertia if inputs.lunar_inertia is None else inputs.lunar_inertia.total
        return 3 * inputs.gravitational_parameters_m3_s2[source] / distance**3 * np.cross(u, K @ u)


@dataclass(frozen=True, slots=True)
class LunarFigureTorque:
    """DE430 Eq. 43 / DE440 Eq. 56, all degrees in the shared field."""

    source: str
    field: GravityField

    @property
    def name(self):
        return f"figure:{self.source}"

    def torque(self, inputs):
        source = inputs.body_indices[self.source]
        # Same target/field evaluation as FigureForce. This is the direct
        # acceleration of the point mass, not the Moon reaction acceleration.
        relative, direct = inputs.figure_accelerations(1, (source,), self.field)
        reaction = -inputs.gravitational_parameters_m3_s2[source] / inputs.gravitational_parameters_m3_s2[1] * direct[0]
        return np.cross(relative[0], reaction) / self.field.radius_m**2


@dataclass(frozen=True, slots=True)
class EarthFigureTorque:
    """The three dominant figure-figure terms (DE440 Eq. 57)."""

    total_inertia: np.ndarray
    earth_radius_m: float
    earth_j2: float
    j2_rate_per_year: float = 0.0
    j2_quadratic_per_year2: float = 0.0
    reference_jd_tdb: float = 2451545.0
    name: str = "figure:EARTH_J2"

    def torque(self, inputs):
        R = inputs.moon_fixed2inertial_matrix
        r = R.T @ (inputs.positions_m[0] - inputs.positions_m[1])
        distance = np.linalg.norm(r)
        u = r / distance
        p = R.T @ inputs.epoch_data.earth_fixed2inertial_matrix[:, 2]
        s = u @ p
        K = self.total_inertia if inputs.lunar_inertia is None else inputs.lunar_inertia.total
        years = ((inputs.epoch_data.epoch_tdb.jd1 - self.reference_jd_tdb) + inputs.epoch_data.epoch_tdb.jd2) / 365.25
        j2 = self.earth_j2 + self.j2_rate_per_year * years + self.j2_quadratic_per_year2 * years**2
        terms = (
            (1 - 7 * s * s) * np.cross(u, K @ u)
            + 2 * s * (np.cross(u, K @ p) + np.cross(p, K @ u))
            - 0.4 * np.cross(p, K @ p)
        )
        return 7.5 * inputs.gravitational_parameters_m3_s2[0] * self.earth_radius_m**2 * j2 / distance**5 * terms
