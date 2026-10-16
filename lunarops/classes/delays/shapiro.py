from __future__ import annotations

import numpy as np

from lunarops.base.constants import C2
from lunarops.classes.delays.base import GravitationalDelay
from lunarops.classes.ephemerides import Ephemeris
from lunarops.classes.relativistic.constants import GM_BY_BODY
from lunarops.classes.time import Epoch

# IERS Conventions 2010 S11.2 (Eq. 11.17) one-way path delay used for LLR.
# Pavlov, Williams & Suvorkin (2016) S4 explicitly include Sun, Earth, Moon,
# Jupiter and Saturn as point masses contributing to the gravitational delay.
_DEFAULT_LLR_SHAPIRO_BODIES = (
    "SUN",
    "EARTH",
    "MOON",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
)


class Iers2010ShapiroDelay(GravitationalDelay):
    """IERS 2010 Eq. (11.17) one-way gravitational path delay."""

    def __init__(self, ephemeris: Ephemeris) -> None:
        self.ephemeris = ephemeris

    def path_delay_m(
        self,
        transmitter_bcrs_m: np.ndarray,
        receiver_bcrs_m: np.ndarray,
        epoch_tdb: Epoch,
    ) -> float:
        x1 = transmitter_bcrs_m
        x2 = receiver_bcrs_m
        rho = float(np.linalg.norm(x2 - x1))

        total = 0.0
        for body in _DEFAULT_LLR_SHAPIRO_BODIES:
            xb = self.ephemeris.body_position_bcrs(body, epoch_tdb)
            r1 = float(np.linalg.norm(x1 - xb))
            r2 = float(np.linalg.norm(x2 - xb))
            denom = r1 + r2 - rho
            numer = r1 + r2 + rho
            if denom <= 0.0 or numer <= 0.0:
                continue
            total += 2.0 * GM_BY_BODY[body] / C2 * np.log(numer / denom)
        return float(total)
