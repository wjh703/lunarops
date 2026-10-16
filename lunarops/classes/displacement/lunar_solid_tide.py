"""LCRS solid-tide displacement of lunar retroreflectors."""

from __future__ import annotations

import numpy as np

from lunarops.classes.displacement.constants import (
    LUNAR_H2,
    LUNAR_L2,
    MOON_REFERENCE_RADIUS_M,
)
from lunarops.classes.ephemerides import Ephemeris
from lunarops.classes.frames.relativistic import RelativisticFrameTransform
from lunarops.classes.relativistic.constants import GM_EARTH, GM_MOON, GM_SUN
from lunarops.classes.relativistic.lunar_scale import LunarRelativisticScale
from lunarops.classes.time import require_tdb_epoch

from .base import ReflectorDisplacementInput


class LunarSolidTide:
    """Degree-2 lunar solid tide following Pavlov et al. (2016), Eq. (24)."""

    def __init__(
        self,
        ephemeris: Ephemeris,
        lunar_scale: LunarRelativisticScale,
        h2: float = LUNAR_H2,
        l2: float = LUNAR_L2,
        moon_radius_m: float = MOON_REFERENCE_RADIUS_M,
    ) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        if not isinstance(lunar_scale, LunarRelativisticScale):
            raise TypeError("lunar_scale must be a LunarRelativisticScale.")
        scalar_values = {
            "h2": h2,
            "l2": l2,
            "moon_radius_m": moon_radius_m,
        }
        normalized: dict[str, float] = {}
        for name, value in scalar_values.items():
            try:
                normalized[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a real scalar.") from exc
            if not np.isfinite(normalized[name]):
                raise ValueError(f"{name} must be finite.")
        if normalized["moon_radius_m"] <= 0.0:
            raise ValueError("moon_radius_m must be positive.")
        self.ephemeris = ephemeris
        self.lunar_scale = lunar_scale
        self.h2 = normalized["h2"]
        self.l2 = normalized["l2"]
        self.moon_radius_m = normalized["moon_radius_m"]

    def displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        epoch = require_tdb_epoch(data.epoch_tdb, name="data.epoch_tdb")
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
