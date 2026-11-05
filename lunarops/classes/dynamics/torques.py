"""External torque models acting on the lunar mantle.

The records here only declare which torques act; the native right-hand side
in :mod:`lunarops._dynamics_core` evaluates them. Torques are expressed in
Moon principal-axis (PA) components and normalised by the whole-Moon
``M R²``, so they have units of s⁻².
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gravity import GravityField


class LunarTorqueGroup:
    """Ordered collection of the torque models acting on the Moon.

    Parameters
    ----------
    models : iterable of PointMassGravityTorque, LunarFigureTorque or EarthFigureTorque, optional
        Torque models. The native kernel sums them in this order.

    Attributes
    ----------
    models : tuple
        The torque models as an immutable tuple.
    """

    def __init__(self, models=()):
        self.models = tuple(models)


@dataclass(frozen=True, slots=True)
class PointMassGravityTorque:
    """Quadrupole torque of an external point mass on the whole-Moon inertia.

    The torque is ``3 GM / r⁵ (r × I r)`` with ``r`` the Moon-to-source vector
    in PA components and ``I`` the current whole-Moon inertia tensor, so only
    the degree-2 part of the lunar figure contributes.

    Attributes
    ----------
    source : str
        Canonical identifier of the torque-raising body.
    total_inertia : numpy.ndarray
        Undeformed whole-Moon inertia tensor in PA axes, shape ``(3, 3)``,
        normalised by ``M R²``. It must equal the rotation model's tensor.
    """

    source: str
    total_inertia: np.ndarray

    @property
    def name(self):
        """str: Torque name, ``"quadrupole:<source>"``."""
        return f"quadrupole:{self.source}"


@dataclass(frozen=True, slots=True)
class LunarFigureTorque:
    """Torque of an external point mass on the full lunar gravity field.

    Every degree of ``field`` contributes. When the rotation has a
    deformation model, degree 2 is replaced per step by the delayed inertia.

    Attributes
    ----------
    source : str
        Canonical identifier of the torque-raising body.
    field : GravityField
        Moon gravity field in PA axes, shared with
        :class:`~lunarops.classes.dynamics.force_models.MoonFigureForce`.

    Notes
    -----
    The torque is the negative of the moment of the figure force on the
    source body.

    References
    ----------
    Folkner et al. (2014), Eq. 43; Park et al. (2021), Eq. 56.
    """

    source: str
    field: GravityField

    @property
    def name(self):
        """str: Torque name, ``"figure:<source>"``."""
        return f"figure:{self.source}"


@dataclass(frozen=True, slots=True)
class EarthFigureTorque:
    """Torque from the interaction of the Earth's J2 with the lunar inertia.

    The model keeps the three dominant figure-figure terms. Earth J2 varies
    with time as ``J2(t) = earth_j2 + rate·t + quadratic·t²`` with ``t`` in
    Julian years of TDB since ``reference_jd_tdb``.

    Attributes
    ----------
    total_inertia : numpy.ndarray
        Undeformed whole-Moon inertia tensor in PA axes, shape ``(3, 3)``,
        normalised by ``M R²``. It must equal the rotation model's tensor.
    earth_radius_m : float
        Reference radius of the Earth field in m.
    earth_j2 : float
        Unnormalised Earth J2 at ``reference_jd_tdb`` (dimensionless,
        ``J2 = -√5 C̄20``).
    j2_rate_per_year : float, default 0.0
        Linear J2 coefficient per Julian year.
    j2_quadratic_per_year2 : float, default 0.0
        Quadratic J2 coefficient per Julian year squared; it is a polynomial
        coefficient and carries no factor 1/2.
    reference_jd_tdb : float, default 2451545.0
        Reference epoch of the J2 polynomial as a TDB Julian date.
    name : str, default "figure:EARTH_J2"
        Torque name.

    References
    ----------
    Park et al. (2021), Eq. 57.
    """

    total_inertia: np.ndarray
    earth_radius_m: float
    earth_j2: float
    j2_rate_per_year: float = 0.0
    j2_quadratic_per_year2: float = 0.0
    reference_jd_tdb: float = 2451545.0
    name: str = "figure:EARTH_J2"
