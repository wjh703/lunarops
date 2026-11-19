"""LCRS solid-tide displacement of lunar retroreflectors."""

from __future__ import annotations

from typing import ClassVar

import erfa
import numpy as np

from lunarops.base.constants import DAYS_PER_JULIAN_CENTURY, J2000_JD
from lunarops.classes.displacement.constants import (
    LUNAR_H2,
    LUNAR_L2,
    LUNAR_TIDAL_DISSIPATION_COEFFICIENTS_M,
    MOON_REFERENCE_RADIUS_M,
)
from lunarops.classes.frames.coordinate_scaling import GM_EARTH, GM_MOON, GM_SUN
from lunarops.classes.frames.relativistic import RelativisticFrameTransform
from lunarops.config.schema import ConfigSchema, number

from .base import ReflectorDisplacementInput


class LunarSolidTide:
    """Solid tide displacement of a lunar retroreflector: elastic degree 2 and 3 plus the dissipative lag.

    The elastic displacement raised by the Earth and the Sun is

        dr = sum_B (GM_B / GM_M) (R^4 / r_B^3) [h2 (3 cos^2(psi_B) - 1) / 2 u
                                                 + 3 l2 cos(psi_B) (b_B - cos(psi_B) u)]
           + sum_B (GM_B / GM_M) (R^5 / r_B^4) [h3 (5/2 cos^3(psi_B) - 3/2 cos(psi_B)) u
                                                 + l3 (15/2 cos^2(psi_B) - 3/2) (b_B - cos(psi_B) u)],

    where ``u`` is the unit vector to the reflector, ``b_B`` the unit vector to body B at distance r_B
    from the Moon's centre, cos(psi_B) = b_B . u, and R the lunar reference radius. Body positions are the
    ephemeris positions at the TDB epoch transformed to the LCRS. The degree-3 form is that of the IERS
    Conventions (2010), Eq. 7.6, with the Moon as the deformed body.

    Tidal dissipation delays the response. Its largest effect on the displacement is at the anomalistic
    (argument l, 27.555 d) and nodical (F, 27.212 d) months; following the JPL LLR model it is added as

        dr_Q = sum_{a in (l, F)} [(h2/Q_h) (U_c cos a + U_s sin a) e_up
                                  + (l2/Q_l) ((E_c cos a + E_s sin a) e_east + (N_c cos a + N_s sin a) e_north)],

    with the per-reflector coefficients of Williams & Boggs (2020), Table A.1, the local up, east and north
    unit vectors of the reflector in the PA frame, and the IERS 2003 Delaunay arguments at the TDB epoch.

    Parameters
    ----------
    transform : RelativisticFrameTransform
        BCRS<->LCRS transformation and body states.
    h2, l2 : float, optional
        Lunar degree-2 radial and horizontal displacement Love numbers (dimensionless).
    moon_radius_m : float, optional
        Lunar reference radius R, in metres.
    h3, l3 : float, optional
        Lunar degree-3 radial and horizontal displacement Love numbers; 0 omits the degree-3 tide.
    h2_over_q, l2_over_q : float, optional
        Dissipation factors h2/Q_h and l2/Q_l; both 0 omit the dissipative displacement.

    References
    ----------
    .. [1] Pavlov, D. A., Williams, J. G., & Suvorkin, V. V. (2016). Determining parameters of Moon's
       orbital and rotational motion from LLR observations using GRAIL and IERS-recommended models.
       Celestial Mechanics and Dynamical Astronomy, 126, 61-88, Eq. (24).
    .. [2] Williams, J. G., & Boggs, D. H. (2020). The JPL lunar laser range model 2020. JPL IOM 335N-20-01,
       Sec. 3.12 and Appendix A1.
    .. [3] Petit, G., & Luzum, B. (Eds.) (2010). IERS Conventions (2010), IERS Technical Note 36, Eq. 7.6.
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
            number(
                "h3",
                default=0.0,
                description="Lunar degree-3 radial Love number h3 (JPL DE440 model: 0.0234). 0 omits it.",
            ),
            number(
                "l3",
                default=0.0,
                description="Lunar degree-3 horizontal Love number l3 (JPL DE440 model: 0.0030). 0 omits it.",
            ),
            number(
                "h2OverQ",
                default=0.0,
                ge=0.0,
                description=(
                    "Radial dissipation factor h2/Q_h of the monthly tides (JPL DE440 model: 1.0e-3). "
                    "0 omits the dissipative displacement."
                ),
            ),
            number(
                "l2OverQ",
                default=0.0,
                ge=0.0,
                description=(
                    "Horizontal dissipation factor l2/Q_l of the monthly tides (JPL DE440 model: 3.0e-5). "
                    "0 omits the dissipative displacement."
                ),
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
            h3=float(cfg["h3"]),
            l3=float(cfg["l3"]),
            h2_over_q=float(cfg["h2OverQ"]),
            l2_over_q=float(cfg["l2OverQ"]),
        )

    def __init__(
        self,
        transform: RelativisticFrameTransform,
        h2: float = LUNAR_H2,
        l2: float = LUNAR_L2,
        moon_radius_m: float = MOON_REFERENCE_RADIUS_M,
        h3: float = 0.0,
        l3: float = 0.0,
        h2_over_q: float = 0.0,
        l2_over_q: float = 0.0,
    ) -> None:
        self.transform = transform
        self.h2 = h2
        self.l2 = l2
        self.moon_radius_m = moon_radius_m
        self.h3 = h3
        self.l3 = l3
        self.h2_over_q = h2_over_q
        self.l2_over_q = l2_over_q

    def _tidal_geometry(self, data: ReflectorDisplacementInput):
        """Return the reflector direction and, per tide-raising body, ``(scale, cosine, body direction, ratio)``.

        ``scale`` is ``(GM_B / GM_M) R^4 / r_B^3`` in metres, ``cosine`` is ``b_B . u`` and ``ratio`` is
        ``R / r_B``, the factor from the degree-2 to the degree-3 scale.
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
            bodies.append((scale, cosine, body_direction, self.moon_radius_m / distance_m))
        return reflector_direction, bodies

    def elastic_displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        """Return the elastic degree-2 and degree-3 displacement on LCRS axes, shape (3,), in metres."""
        reflector_direction, bodies = self._tidal_geometry(data)
        total = np.zeros(3, dtype=float)
        for scale, cosine, body_direction, ratio in bodies:
            horizontal = body_direction - cosine * reflector_direction
            total += scale * (
                0.5 * self.h2 * (3.0 * cosine * cosine - 1.0) * reflector_direction
                + 3.0 * self.l2 * cosine * horizontal
            )
            if self.h3 or self.l3:
                total += (scale * ratio) * (
                    self.h3 * (2.5 * cosine**3 - 1.5 * cosine) * reflector_direction
                    + self.l3 * (7.5 * cosine * cosine - 1.5) * horizontal
                )
        return total

    def dissipation_displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        """Return the dissipative monthly displacement on LCRS axes, shape (3,), in metres.

        Raises
        ------
        ValueError
            If dissipation is modelled and the input lacks the reflector identifier, a reflector of
            Table A.1 or the PA-to-LCRS rotation.
        """
        if not (self.h2_over_q or self.l2_over_q):
            return np.zeros(3, dtype=float)
        try:
            coefficients = LUNAR_TIDAL_DISSIPATION_COEFFICIENTS_M[str(data.reflector_id).upper()]
        except KeyError:
            raise ValueError(
                f"lunarSolidTide dissipation has no coefficients for reflector {data.reflector_id!r}; "
                f"known: {sorted(LUNAR_TIDAL_DISSIPATION_COEFFICIENTS_M)}."
            ) from None
        if data.pa_to_lcrs_matrix is None:
            raise ValueError("lunarSolidTide dissipation needs the PA-to-LCRS rotation of the reflector input.")
        pa_to_lcrs = np.asarray(data.pa_to_lcrs_matrix, dtype=float)
        up = data.reference_position_lcrs_m / np.linalg.norm(data.reference_position_lcrs_m)
        east = np.cross(pa_to_lcrs[:, 2], up)
        east /= np.linalg.norm(east)
        north = np.cross(up, east)

        t = (data.epoch_tdb.jd1 - J2000_JD + data.epoch_tdb.jd2) / DAYS_PER_JULIAN_CENTURY
        arguments = {"l": erfa.fal03(t), "F": erfa.faf03(t)}
        up_m = east_m = north_m = 0.0
        for name, (up_c, up_s, east_c, east_s, north_c, north_s) in coefficients.items():
            cos_a, sin_a = np.cos(arguments[name]), np.sin(arguments[name])
            up_m += up_c * cos_a + up_s * sin_a
            east_m += east_c * cos_a + east_s * sin_a
            north_m += north_c * cos_a + north_s * sin_a
        return self.h2_over_q * up_m * up + self.l2_over_q * (east_m * east + north_m * north)

    def displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        """Return the solid tide displacement of a reflector.

        Parameters
        ----------
        data : ReflectorDisplacementInput
            Reflector LCRS position and TDB epoch; with dissipation also the reflector identifier and the
            PA-to-LCRS rotation.

        Returns
        -------
        ndarray, shape (3,)
            Displacement on LCRS (ICRF-aligned) axes, in metres.

        Raises
        ------
        ValueError
            If the reflector position has zero norm, or dissipation lacks its inputs.
        """
        return self.elastic_displacement_lcrs_m(data) + self.dissipation_displacement_lcrs_m(data)

    def displacement_per_h2_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        """Return the partial derivative of the displacement with respect to ``h2``.

        The displacement is linear in ``h2`` (Pavlov et al. 2016, Eq. 24), so this is the radial term of unit
        Love number, ``sum_B (GM_B / GM_M) (R^4 / r_B^3) (3 cos^2(psi_B) - 1) / 2 u``. The dissipative term has
        its own factor ``h2/Q_h`` and does not enter.

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
        factor = sum(scale * 0.5 * (3.0 * cosine * cosine - 1.0) for scale, cosine, _, _ in bodies)
        return factor * reflector_direction
