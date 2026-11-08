"""Gravitating-body records used by the lunar dynamics model.

A :class:`GravitatingBody` pairs a canonical body identifier with its
gravitational parameter. Every force, torque and epoch-data component refers
to bodies through these records.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.classes.ephemerides.body_ids import body_name


@dataclass(frozen=True, slots=True)
class GravitatingBody:
    """A body that acts as a Newtonian point-mass source.

    Attributes
    ----------
    body_id : str
        Canonical body identifier: an exact ``BodyId`` name such as
        ``"EARTH"`` or ``"JUPITER BARYCENTER"``, an explicit
        ``"NAIF:<integer>"`` identifier, or a generated ``"KBO_RING_<nn>"``
        ring node. A ``NAIF:`` identifier of a known major body is stored
        under its ``BodyId`` name.
    gravitational_parameter_m3_s2 : float
        Gravitational parameter GM in m³ s⁻² (TDB-compatible value).

    Raises
    ------
    TypeError
        If ``body_id`` is neither a string nor a ``BodyId``.
    ValueError
        If ``body_id`` is empty, padded with whitespace or unknown.
    """

    body_id: str
    gravitational_parameter_m3_s2: float

    def __post_init__(self):
        """Replace ``body_id`` by its canonical spelling."""
        key = body_name(self.body_id)
        object.__setattr__(self, "body_id", key)


EMB = "EARTH MOON BARYCENTER"


def earth_moon_about_barycenter(barycenter, relative, earth_gm_m3_s2: float, moon_gm_m3_s2: float):
    """Place the Earth and the Moon about the Earth-Moon barycentre.

    ``r_E = r_B - GM_M / (GM_E + GM_M) y`` and ``r_M = r_B + GM_E / (GM_E + GM_M) y``
    with ``y`` the Moon-minus-Earth vector; velocities follow the same rule.

    Parameters
    ----------
    barycenter : array_like
        BCRS barycentre states, shape ``(..., k)``.
    relative : array_like
        Moon-minus-Earth states of the same shape.
    earth_gm_m3_s2, moon_gm_m3_s2 : float
        Earth and Moon GM in m³ s⁻²; only their ratio is used.

    Returns
    -------
    earth, moon : numpy.ndarray
        BCRS Earth and Moon states, each of the input shape.
    """
    barycenter = np.asarray(barycenter, dtype=float)
    relative = np.asarray(relative, dtype=float)
    total = earth_gm_m3_s2 + moon_gm_m3_s2
    return barycenter - (moon_gm_m3_s2 / total) * relative, barycenter + (earth_gm_m3_s2 / total) * relative
