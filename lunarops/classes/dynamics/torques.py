"""Moon-only external torques in PA, normalized by whole-Moon M R²."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gravity import GravityField


class LunarTorqueGroup:
    def __init__(self, models=()):
        self.models = tuple(models)


@dataclass(frozen=True, slots=True)
class PointMassGravityTorque:
    """Quadrupole-only baseline; inertia is the full Moon's tensor."""

    source: str
    total_inertia: np.ndarray

    @property
    def name(self):
        return f"quadrupole:{self.source}"


@dataclass(frozen=True, slots=True)
class LunarFigureTorque:
    """DE430 Eq. 43 / DE440 Eq. 56, all degrees of the shared field (current degree 2)."""

    source: str
    field: GravityField

    @property
    def name(self):
        return f"figure:{self.source}"


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
