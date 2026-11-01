"""Composable array-oriented acceleration models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .deformation import LunarDegree2GravityModel
from .forces import EarthTideModel, SolarSourceParameters
from .gravity import GravityField


class ForceModel(Protocol):
    @property
    def name(self) -> str: ...


class LunarForceGroup:

    def __init__(
        self,
        force_models: Sequence[ForceModel] = (),
        enabled_force_names: Sequence[str] | None = None,
    ):
        self.force_models = tuple(force_models)
        model_names = tuple(model.name for model in self.force_models)
        terms = frozenset(model_names if enabled_force_names is None else enabled_force_names)
        self._active_models = tuple(model for model in self.force_models if model.name in terms)
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


@dataclass(frozen=True, slots=True)
class NewtonianPointMassForce:
    name: str = "point_mass"


@dataclass(frozen=True, slots=True)
class EihPointMassForce:
    name: str = "eih_1pn"


@dataclass(frozen=True, slots=True)
class SolarJ2Force:
    gravity_field: GravityField
    inertial2solar_fixed_matrix: np.ndarray
    name: str = "solar_j2"


@dataclass(frozen=True, slots=True)
class LenseThirringForce:
    solar_parameters: SolarSourceParameters
    name: str = "lense_thirring"


@dataclass(frozen=True, slots=True)
class SolarRadiationPressureForce:
    """Solar radiation pressure on the Moon; ``epsilon`` is its area-to-mass coupling."""

    solar_parameters: SolarSourceParameters
    epsilon: float
    name: str = "solar_radiation_pressure"


@dataclass(frozen=True, slots=True)
class EarthTideForce:
    tide_model: EarthTideModel
    name: str = "tide"


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
