"""Moon-only external torques in PA, normalized by whole-Moon M R²."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._vector import cross3
from .earth_figure_orientation import earth_j2_increment
from .gravity import GravityField


class LunarTorqueGroup:
    def __init__(self, models=()):
        self.models = tuple(models)

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
        r = inputs.moon_fixed2inertial_matrix.T @ (inputs.positions_m[source] - inputs.positions_m[inputs.moon_index])
        distance = np.linalg.norm(r)
        u = r / distance
        K = self.total_inertia if inputs.lunar_inertia is None else inputs.lunar_inertia.total
        return 3 * inputs.gravitational_parameters_m3_s2[source] / distance**3 * cross3(u, K @ u)


@dataclass(frozen=True, slots=True)
class LunarFigureTorque:
    """DE430 Eq. 43 / DE440 Eq. 56, all degrees of the shared field (current degree 2)."""

    source: str
    field: GravityField

    @property
    def name(self):
        return f"figure:{self.source}"

    def torque(self, inputs):
        source = inputs.body_indices[self.source]
        # Same target/field evaluation as MoonFigureForce. This is the direct
        # acceleration of the point mass, not the Moon reaction acceleration.
        relative, direct = inputs.figure_accelerations(inputs.moon_index, (source,), self.field)
        reaction = -inputs.gravitational_parameters_m3_s2[source] / inputs.gravitational_parameters_m3_s2[inputs.moon_index] * direct[0]
        return cross3(relative[0], reaction) / self.field.radius_m**2


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
        r = R.T @ (inputs.positions_m[inputs.earth_index] - inputs.positions_m[inputs.moon_index])
        distance = np.linalg.norm(r)
        u = r / distance
        p = R.T @ inputs.epoch_data.earth_fixed2inertial_matrix[:, 2]
        s = u @ p
        K = self.total_inertia if inputs.lunar_inertia is None else inputs.lunar_inertia.total
        j2 = self.earth_j2 + earth_j2_increment(inputs.epoch_data.epoch_tdb, self.reference_jd_tdb, self.j2_rate_per_year, self.j2_quadratic_per_year2)
        terms = (
            (1 - 7 * s * s) * cross3(u, K @ u)
            + 2 * s * (cross3(u, K @ p) + cross3(p, K @ u))
            - 0.4 * cross3(p, K @ p)
        )
        return 7.5 * inputs.gravitational_parameters_m3_s2[inputs.earth_index] * self.earth_radius_m**2 * j2 / distance**5 * terms
