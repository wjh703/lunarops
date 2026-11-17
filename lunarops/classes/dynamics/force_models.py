"""Force-model records acting on the Moon and the Earth.

Each record declares one acceleration term and its parameters; the native
right-hand side compiled by :func:`~lunarops.classes.dynamics.fused_rhs.compile_fused_rhs`
evaluates every term on both the Moon and the Earth, placed about the
prescribed Earth-Moon barycentre. All accelerations are BCRS accelerations
with ICRF axes, in m s⁻². The integrated quantity is the Moon-minus-Earth
acceleration, the difference of the two.

:func:`enabled_force_terms` reads the active terms of a ``LunarOrbitPropagation``
config, the ``bodyForceTerms`` list, which is the only switch of a term.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lunarops.base.constants import J2000_JD

from .deformation import LunarDegree2GravityModel
from .forces import EarthTideModel, SolarSourceParameters
from .gravity import GravityField

#: The Moon force terms in summation order, each with the config key of the parameters it requires (``None``
#: for a term without parameters). ``bodyForceTerms`` lists the active terms.
FORCE_TERMS: tuple[tuple[str, str | None], ...] = (
    ("point_mass", None),
    ("eih_1pn", None),
    ("solar_j2", "solarJ2"),
    ("figure_earth", "gravityFields.earth"),
    ("figure_moon", "gravityFields.moon"),
    ("tide", "earthTide"),
    ("lense_thirring", None),
    ("solar_radiation_pressure", "solarRelativistic"),
)


def enabled_force_terms(config: Mapping) -> tuple[str, ...]:
    """Return the active Moon force terms of a ``LunarOrbitPropagation`` config, its ``bodyForceTerms``.

    ``bodyForceTerms`` is the only switch of a force term: a term is active exactly when it is listed. A listed
    term must have its parameters configured (see ``FORCE_TERMS``); parameters of a term that is not listed
    are accepted and unused.

    Parameters
    ----------
    config : Mapping
        ``LunarOrbitPropagation`` config.

    Returns
    -------
    tuple of str
        The listed terms, in the order of ``bodyForceTerms``.

    Raises
    ------
    ValueError
        If a term is unknown or listed twice, ``eih_1pn`` is listed without ``point_mass``, or the parameters
        of a listed term are not configured.
    """
    terms = tuple(config["bodyForceTerms"])
    requirements = dict(FORCE_TERMS)
    if unknown := [term for term in terms if term not in requirements]:
        raise ValueError(f"bodyForceTerms lists unknown force terms {unknown}; known: {', '.join(requirements)}")
    if len(set(terms)) != len(terms):
        raise ValueError("bodyForceTerms lists a force term twice")
    if "eih_1pn" in terms and "point_mass" not in terms:
        raise ValueError("bodyForceTerms lists eih_1pn, which requires point_mass")
    for term in terms:
        key = requirements[term]
        if key is not None and _config_value(config, key) is None:
            raise ValueError(f"bodyForceTerms lists {term}, which requires {key}")
    return terms


def _config_value(config: Mapping, dotted_key: str):
    """Return the value at a dotted key of ``config``; ``None`` when a key on the way is absent or null."""
    value = config
    for key in dotted_key.split("."):
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


class ForceModel(Protocol):
    """Protocol of a force-model record."""

    @property
    def name(self) -> str:
        """str: Unique term name used in ``bodyForceTerms`` and diagnostics."""
        ...


class LunarForceGroup:
    """Available force models and the subset that is active.

    Parameters
    ----------
    force_models : sequence of ForceModel, optional
        All configured force models.
    enabled_force_names : sequence of str or None, optional
        Names of the active models; all models are active when ``None``.

    Raises
    ------
    ValueError
        If more than one Earth tide force is active.
    """

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
        tides = [model for model in self._active_models if isinstance(model, EarthTideForce)]
        if len(tides) > 1:
            raise ValueError("At most one Earth tide force may be active")
        self._earth_tide = tides[0] if tides else None

    @property
    def active_models(self) -> tuple[ForceModel, ...]:
        """Active force models, in configuration order."""
        return self._active_models

    @property
    def has_point_mass(self) -> bool:
        """bool: Whether the Newtonian point-mass term is active."""
        return self._point_mass_model is not None

    @property
    def earth_tide(self) -> EarthTideForce | None:
        """EarthTideForce or None: The active Earth tide force; ``None`` when the Earth tide is off."""
        return self._earth_tide


@dataclass(frozen=True, slots=True)
class NewtonianPointMassForce:
    """Newtonian point-mass attraction of every body on the Moon and the Earth.

    All bodies (external bodies and ring nodes, and each other) attract the
    Moon and the Earth. The same Newtonian sums supply the auxiliary
    accelerations and potentials of the EIH term.

    Attributes
    ----------
    name : str, default "point_mass"
        Term name.
    """

    name: str = "point_mass"


@dataclass(frozen=True, slots=True)
class EihPointMassForce:
    """First post-Newtonian (EIH) point-mass correction on the Moon and the Earth.

    The correction uses the PPN parameters β = γ = 1. The accelerations
    inside the 1PN term are the Newtonian point-mass accelerations; figure
    and tide accelerations do not enter. Requires :class:`NewtonianPointMassForce`.

    Attributes
    ----------
    name : str, default "eih_1pn"
        Term name.

    References
    ----------
    Park et al. (2021), Eq. 27.
    """

    name: str = "eih_1pn"


@dataclass(frozen=True, slots=True)
class SolarJ2Force:
    """Acceleration of the Moon and the Earth by the solar oblateness J2.

    Attributes
    ----------
    gravity_field : GravityField
        Degree-2 solar field built from J2.
    inertial2solar_fixed_matrix : numpy.ndarray
        Rotation from ICRF axes to solar axes with z along the solar spin
        pole, shape ``(3, 3)``.
    name : str, default "solar_j2"
        Term name.
    """

    gravity_field: GravityField
    inertial2solar_fixed_matrix: np.ndarray
    name: str = "solar_j2"


@dataclass(frozen=True, slots=True)
class LenseThirringForce:
    """Lense-Thirring (frame-dragging) acceleration from the solar spin, on the Moon and the Earth.

    Attributes
    ----------
    solar_parameters : SolarSourceParameters
        Solar GM, spin angular momentum and PPN γ.
    name : str, default "lense_thirring"
        Term name.
    """

    solar_parameters: SolarSourceParameters
    name: str = "lense_thirring"


@dataclass(frozen=True, slots=True)
class SolarRadiationPressureForce:
    """Solar radiation pressure on the Moon and the Earth.

    Each acceleration points away from the Sun, ``ε GM_sun r / |r|³`` with
    ``r`` the Sun-to-body vector (Park et al. 2021, Eqs. 40-41).

    Attributes
    ----------
    solar_parameters : SolarSourceParameters
        Solar parameters providing GM.
    epsilon : float
        Dimensionless ratio of radiation-pressure to solar-gravity
        acceleration on the Moon.
    earth_epsilon : float, default 0.0
        The same ratio for the Earth.
    name : str, default "solar_radiation_pressure"
        Term name.
    """

    solar_parameters: SolarSourceParameters
    epsilon: float
    earth_epsilon: float = 0.0
    name: str = "solar_radiation_pressure"


@dataclass(frozen=True, slots=True)
class EarthTideForce:
    """Acceleration of the Moon by the tides it and the Sun raise on the Earth.

    The Moon takes ``GM_E / (GM_E + GM_M)`` of the relative tidal acceleration
    and the Earth the reaction, so the term leaves the barycentre unaffected.

    Attributes
    ----------
    tide_model : EarthTideModel
        Earth-tide model.
    name : str, default "tide"
        Term name.
    """

    tide_model: EarthTideModel
    name: str = "tide"


@dataclass(frozen=True, slots=True)
class EarthFigureForce:
    """Interaction of the Earth's non-spherical gravity field with point masses.

    The field attracts each partner point mass. The Moon, when it is a
    partner, takes that acceleration directly; the Earth takes the indirect
    acceleration ``-GM_partner / GM_E`` times it for every partner (Park et
    al. 2021, Eq. 29). The Earth J2 varies with time as
    ``J2(t) = J2 + rate·t + quadratic·t²`` with ``t`` in Julian years of TDB
    since ``reference_jd_tdb``; C̄20 of the field is updated accordingly at
    every epoch. The field is evaluated in the Earth gravity-field axes of the
    configured pole model.

    Attributes
    ----------
    field : GravityField
        Earth gravity field.
    partners : tuple of str, default ("MOON",)
        Canonical identifiers of the point masses interacting with the
        field. DE440 uses the Moon, the Sun, Mercury, Venus, Mars, Jupiter
        and Saturn.
    j2_rate_per_year : float, default 0.0
        Linear J2 coefficient per Julian year (JPL ``J2EDOT``).
    j2_quadratic_per_year2 : float, default 0.0
        Quadratic J2 coefficient per Julian year squared (JPL ``J2ET2``); a
        polynomial coefficient without a factor 1/2.
    reference_jd_tdb : float, default 2451545.0
        Reference epoch of the J2 polynomial as a TDB Julian date.
    name : str, default "figure_earth"
        Term name.
    """

    field: GravityField
    partners: tuple[str, ...] = ("MOON",)
    j2_rate_per_year: float = 0.0
    j2_quadratic_per_year2: float = 0.0
    reference_jd_tdb: float = J2000_JD
    name: str = "figure_earth"


@dataclass(frozen=True, slots=True)
class MoonFigureForce:
    """Reaction on the Moon of its own non-spherical field acting on partners.

    The Moon field attracts each partner point mass; the Moon receives the
    reaction ``-GM_partner / GM_moon`` times that acceleration, and the
    Earth, when it is a partner, the acceleration itself. With
    ``lunar_degree2_gravity_model`` the degree 2 follows the prescribed
    delayed deformation; with integrated rotation and deformation it follows
    the delayed inertia of the rotational dynamics.

    Attributes
    ----------
    field : GravityField
        Moon gravity field in PA axes.
    partners : tuple of str
        Canonical identifiers of the bodies interacting with the field.
    lunar_degree2_gravity_model : LunarDegree2GravityModel or None, default None
        Prescribed delayed degree-2 model.
    name : str, default "figure_moon"
        Term name.
    """

    field: GravityField
    partners: tuple[str, ...]
    lunar_degree2_gravity_model: LunarDegree2GravityModel | None = None
    name: str = "figure_moon"
