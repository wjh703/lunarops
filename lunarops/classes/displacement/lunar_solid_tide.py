"""LCRS solid-tide displacement of lunar retroreflectors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from lunarops.classes.displacement.constants import LUNAR_H2, LUNAR_L2, MOON_REFERENCE_RADIUS_M
from lunarops.classes.frames.coordinate_scaling import GM_EARTH, GM_MOON, GM_SUN
from lunarops.classes.frames.relativistic import RelativisticFrameTransform
from lunarops.config.schema import ConfigSchema, number

from .base import ReflectorDisplacementInput


class LunarSolidTide:
    """Degree-2 lunar solid tide displacement of a lunar retroreflector.

    The displacement raised by the Earth and the Sun is

        dr = sum_B (GM_B / GM_M) (R^4 / r_B^3) [h2 (3 cos^2(psi_B) - 1) / 2 u
                                                 + 3 l2 cos(psi_B) (b_B - cos(psi_B) u)],

    where ``u`` is the unit vector to the reflector, ``b_B`` the unit vector to body B at distance r_B
    from the Moon's centre, cos(psi_B) = b_B . u, and R the lunar reference radius. Body positions are the
    ephemeris positions at the TDB epoch transformed to the LCRS.

    Parameters
    ----------
    transform : RelativisticFrameTransform
        BCRS<->LCRS transformation and body states.
    h2 : float, optional
        Lunar radial displacement Love number (dimensionless).
    l2 : float, optional
        Lunar horizontal displacement Love number (dimensionless).
    moon_radius_m : float, optional
        Lunar reference radius R, in metres.

    References
    ----------
    .. [1] Pavlov, D. A., Williams, J. G., & Suvorkin, V. V. (2016). Determining parameters of Moon's
       orbital and rotational motion from LLR observations using GRAIL and IERS-recommended models.
       Celestial Mechanics and Dynamical Astronomy, 126, 61-88, Eq. (24).
    """

    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(
        fields=(
            number("h2", default=LUNAR_H2, description="Lunar radial displacement Love number h2."),
            number("l2", default=LUNAR_L2, description="Lunar horizontal displacement Love number l2."),
            number(
                "moonRadiusM",
                default=MOON_REFERENCE_RADIUS_M,
                gt=0.0,
                description="Lunar reference radius of the tidal potential, in metres.",
            ),
        ),
        type_name="lunarSolidTide",
    )

    @classmethod
    def from_config(cls, cfg, ctx, *, frames):
        """Create the model from its configuration and the run's relativistic transform."""
        return cls(
            transform=frames.relativistic_transform,
            h2=float(cfg["h2"]),
            l2=float(cfg["l2"]),
            moon_radius_m=float(cfg["moonRadiusM"]),
        )

    def __init__(
        self,
        transform: RelativisticFrameTransform,
        h2: float = LUNAR_H2,
        l2: float = LUNAR_L2,
        moon_radius_m: float = MOON_REFERENCE_RADIUS_M,
    ) -> None:
        self.transform = transform
        self.h2 = h2
        self.l2 = l2
        self.moon_radius_m = moon_radius_m

    def _tidal_geometry(self, data: ReflectorDisplacementInput):
        """Return the reflector direction and, per tide-raising body, ``(scale, cosine, body direction)``.

        ``scale`` is ``(GM_B / GM_M) R^4 / r_B^3`` in metres and ``cosine`` is ``b_B . u``.
        """
        epoch = data.epoch_tdb
        reflector = data.reference_position_lcrs_m
        reflector_norm = float(np.linalg.norm(reflector))
        if reflector_norm <= 0.0:
            raise ValueError("reference_position_lcrs_m must have a positive norm.")
        reflector_direction = reflector / reflector_norm

        transform = self.transform
        bodies = []
        for name, body_gm_m3_s2 in (("EARTH", GM_EARTH), ("SUN", GM_SUN)):
            body_lcrs_m = transform.bcrs2lcrs(transform.body_state(name, epoch).position_m, epoch)
            distance_m = float(np.linalg.norm(body_lcrs_m))
            if distance_m <= 0.0:
                raise RuntimeError("Ephemeris returned a zero Moon-to-body vector.")
            body_direction = body_lcrs_m / distance_m
            cosine = float(np.dot(body_direction, reflector_direction))
            scale = body_gm_m3_s2 * self.moon_radius_m**4 / (GM_MOON * distance_m**3)
            bodies.append((scale, cosine, body_direction))
        return reflector_direction, bodies

    def displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        """Return the solid tide displacement of a reflector.

        Parameters
        ----------
        data : ReflectorDisplacementInput
            Reflector LCRS position and TDB epoch.

        Returns
        -------
        ndarray, shape (3,)
            Displacement on LCRS (ICRF-aligned) axes, in metres.

        Raises
        ------
        ValueError
            If the reflector position has zero norm.
        """
        reflector_direction, bodies = self._tidal_geometry(data)
        total = np.zeros(3, dtype=float)
        for scale, cosine, body_direction in bodies:
            radial = 0.5 * self.h2 * (3.0 * cosine * cosine - 1.0) * reflector_direction
            tangential = 3.0 * self.l2 * cosine * (body_direction - cosine * reflector_direction)
            total += scale * (radial + tangential)
        return total

    def displacement_per_h2_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        """Return the partial derivative of the displacement with respect to ``h2``.

        The displacement is linear in ``h2`` (Pavlov et al. 2016, Eq. 24), so this is the radial term of unit
        Love number, ``sum_B (GM_B / GM_M) (R^4 / r_B^3) (3 cos^2(psi_B) - 1) / 2 u``.

        Parameters
        ----------
        data : ReflectorDisplacementInput
            Reflector LCRS position and TDB epoch.

        Returns
        -------
        ndarray, shape (3,)
            ``d(displacement)/d(h2)`` on LCRS (ICRF-aligned) axes, in metres.

        Raises
        ------
        ValueError
            If the reflector position has zero norm.
        """
        reflector_direction, bodies = self._tidal_geometry(data)
        factor = sum(scale * 0.5 * (3.0 * cosine * cosine - 1.0) for scale, cosine, _ in bodies)
        return factor * reflector_direction
