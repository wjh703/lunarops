"""Relativistic and gravitational constants used by frame/time models."""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Final, Mapping

# IAU/JPL relativistic scale constants used by coordinate transforms.
# The GCRS<->BCRS scale uses the explicit L_B-L_G form.
L_B = 1.550519768e-8
L_G = 6.969290134e-10
L_B_MINUS_L_G = L_B - L_G

# Lunar surface coordinate-time scale constant L_L from Turyshev et al.
# Table 2 gives L_L = 0.003139054 x 1e-8. It is derived from lunar gravity
# and spin, analogous to L_G for Earth; it is not DE440 kernel metadata.
L_L_LUNAR_SURFACE = 0.003139054e-8
L_B_MINUS_L_L_LUNAR_SURFACE = L_B - L_L_LUNAR_SURFACE


class LunarRelativisticScaleConvention(StrEnum):
    """Coordinate-scale convention used by lunar ephemeris coordinates."""

    TDB_COMPATIBLE_LUNAR_SURFACE = "tdbCompatibleLunarSurface"
    ALREADY_SCALED = "alreadyScaled"


def normalize_lunar_relativistic_scale_convention(
    value: LunarRelativisticScaleConvention | str,
) -> LunarRelativisticScaleConvention:
    if isinstance(value, LunarRelativisticScaleConvention):
        return value
    if not isinstance(value, str):
        raise TypeError(
            "lunar relativistic scale convention must be a string or "
            "LunarRelativisticScaleConvention."
        )
    text = value.strip().casefold()
    for convention in LunarRelativisticScaleConvention:
        if convention.value.casefold() == text:
            return convention
    allowed = ", ".join(convention.value for convention in LunarRelativisticScaleConvention)
    raise ValueError(
        f"Unsupported lunar relativistic scale convention {value!r}; expected one of: {allowed}."
    )


def l_b_minus_l_l_for_convention(convention: LunarRelativisticScaleConvention) -> float:
    if convention is LunarRelativisticScaleConvention.TDB_COMPATIBLE_LUNAR_SURFACE:
        return float(L_B_MINUS_L_L_LUNAR_SURFACE)
    return 0.0


GM_SUN = 1.32712440041e20
GM_EARTH = 3.986004355e14
GM_MOON = 4.9028002e12
GM_MERCURY = 2.2032e13
GM_VENUS = 3.24859e14
GM_MARS = 4.282837e13
GM_JUPITER = 1.26686534e17
GM_SATURN = 3.7931187e16
GM_URANUS = 5.793939e15
GM_NEPTUNE = 6.836529e15

# Keys are canonical upper-case body names used by the LLR processing code.
# For LLR the planets are best looked up via their barycenters because the
# barycenter is what is reliably available in standard DE ephemerides.
GM_BY_BODY: Final[Mapping[str, float]] = MappingProxyType({
    "SUN": GM_SUN,
    "EARTH": GM_EARTH,
    "MOON": GM_MOON,
    "MERCURY BARYCENTER": GM_MERCURY,
    "VENUS BARYCENTER": GM_VENUS,
    "MARS BARYCENTER": GM_MARS,
    "JUPITER BARYCENTER": GM_JUPITER,
    "SATURN BARYCENTER": GM_SATURN,
    "URANUS BARYCENTER": GM_URANUS,
    "NEPTUNE BARYCENTER": GM_NEPTUNE,
})

# External-potential body lists used in BCRS<->GCRS / BCRS<->LCRS
# coordinate-time conversions (IERS Eq. 11.18 / 11.19; paper Eq. 21 / 23).
EARTH_EXTERNAL_POTENTIAL_BODIES = (
    "SUN",
    "MOON",
    "MERCURY BARYCENTER",
    "VENUS BARYCENTER",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
)
MOON_EXTERNAL_POTENTIAL_BODIES = (
    "SUN",
    "EARTH",
    "MERCURY BARYCENTER",
    "VENUS BARYCENTER",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
)

__all__ = [
    "EARTH_EXTERNAL_POTENTIAL_BODIES",
    "GM_BY_BODY",
    "GM_EARTH",
    "GM_JUPITER",
    "GM_MARS",
    "GM_MERCURY",
    "GM_MOON",
    "GM_NEPTUNE",
    "GM_SATURN",
    "GM_SUN",
    "GM_URANUS",
    "GM_VENUS",
    "L_B",
    "L_B_MINUS_L_G",
    "L_B_MINUS_L_L_LUNAR_SURFACE",
    "L_G",
    "L_L_LUNAR_SURFACE",
    "LunarRelativisticScaleConvention",
    "MOON_EXTERNAL_POTENTIAL_BODIES",
    "l_b_minus_l_l_for_convention",
    "normalize_lunar_relativistic_scale_convention",
]
