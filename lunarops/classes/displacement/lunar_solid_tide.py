"""LCRS solid-tide displacement of lunar retroreflectors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from lunarops.classes.displacement.constants import LUNAR_H2, LUNAR_L2, MOON_REFERENCE_RADIUS_M
from lunarops.classes.ephemerides import Ephemeris
from lunarops.classes.frames.coordinate_scaling import GM_EARTH, GM_MOON, GM_SUN, LunarRelativisticScale
from lunarops.classes.frames.relativistic import RelativisticFrameTransform
from lunarops.config.schema import ConfigSchema, number

from .base import ReflectorDisplacementInput


class LunarSolidTide:
    """Degree-2 lunar solid tide following Pavlov et al. (2016), Eq. (24)."""

    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(
        fields=(
            number("h2", default=LUNAR_H2),
            number("l2", default=LUNAR_L2),
            number("moonRadiusM", default=MOON_REFERENCE_RADIUS_M, gt=0.0),
        ),
        type_name="lunarSolidTide",
    )

    @classmethod
    def from_config(cls, cfg, ctx, *, ephemeris, frames):
        return cls(
            ephemeris=ephemeris,
            lunar_scale=frames.lunar_scale,
            h2=float(cfg["h2"]),
            l2=float(cfg["l2"]),
            moon_radius_m=float(cfg["moonRadiusM"]),
        )

    def __init__(
        self,
        ephemeris: Ephemeris,
        lunar_scale: LunarRelativisticScale,
        h2: float = LUNAR_H2,
        l2: float = LUNAR_L2,
        moon_radius_m: float = MOON_REFERENCE_RADIUS_M,
    ) -> None:
        self.ephemeris = ephemeris
        self.lunar_scale = lunar_scale
        self.h2 = h2
        self.l2 = l2
        self.moon_radius_m = moon_radius_m

    def displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        epoch = data.epoch_tdb
        reflector = data.reference_position_lcrs_m
        reflector_norm = float(np.linalg.norm(reflector))
        if reflector_norm <= 0.0:
            raise ValueError("reference_position_lcrs_m must have a positive norm.")
        reflector_direction = reflector / reflector_norm

        transform = RelativisticFrameTransform(self.ephemeris, self.lunar_scale)
        earth_lcrs = transform.bcrs2lcrs(
            self.ephemeris.body_position_bcrs("EARTH", epoch),
            epoch,
        )
        sun_lcrs = transform.bcrs2lcrs(
            self.ephemeris.body_position_bcrs("SUN", epoch),
            epoch,
        )

        def body_term(body_lcrs_m: np.ndarray, body_gm_m3_s2: float) -> np.ndarray:
            distance_m = float(np.linalg.norm(body_lcrs_m))
            if distance_m <= 0.0:
                raise RuntimeError("Ephemeris returned a zero Moon-to-body vector.")
            body_direction = body_lcrs_m / distance_m
            cosine = float(np.dot(body_direction, reflector_direction))
            radial = 0.5 * self.h2 * (3.0 * cosine * cosine - 1.0) * reflector_direction
            tangential = 3.0 * self.l2 * cosine * (body_direction - cosine * reflector_direction)
            scale = body_gm_m3_s2 * self.moon_radius_m**4 / (GM_MOON * distance_m**3)
            return scale * (radial + tangential)

        return body_term(earth_lcrs, GM_EARTH) + body_term(
            sun_lcrs,
            GM_SUN,
        )
