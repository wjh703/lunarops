"""Canonical NAIF body identifiers used by ephemeris kernels."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from functools import lru_cache
from types import MappingProxyType
from typing import Final


class BodyId(StrEnum):
    SOLAR_SYSTEM_BARYCENTER = "SOLAR SYSTEM BARYCENTER"
    MERCURY_BARYCENTER = "MERCURY BARYCENTER"
    VENUS_BARYCENTER = "VENUS BARYCENTER"
    EARTH_MOON_BARYCENTER = "EARTH MOON BARYCENTER"
    MARS_BARYCENTER = "MARS BARYCENTER"
    JUPITER_BARYCENTER = "JUPITER BARYCENTER"
    SATURN_BARYCENTER = "SATURN BARYCENTER"
    URANUS_BARYCENTER = "URANUS BARYCENTER"
    NEPTUNE_BARYCENTER = "NEPTUNE BARYCENTER"
    PLUTO_BARYCENTER = "PLUTO BARYCENTER"
    SUN = "SUN"
    MERCURY = "MERCURY"
    VENUS = "VENUS"
    MOON = "MOON"
    EARTH = "EARTH"
    MARS = "MARS"
    JUPITER = "JUPITER"
    SATURN = "SATURN"
    URANUS = "URANUS"
    NEPTUNE = "NEPTUNE"
    PLUTO = "PLUTO"


NAIF_ID_BY_BODY: Final[Mapping[BodyId, int]] = MappingProxyType(
    {
        BodyId.SOLAR_SYSTEM_BARYCENTER: 0,
        BodyId.MERCURY_BARYCENTER: 1,
        BodyId.VENUS_BARYCENTER: 2,
        BodyId.EARTH_MOON_BARYCENTER: 3,
        BodyId.MARS_BARYCENTER: 4,
        BodyId.JUPITER_BARYCENTER: 5,
        BodyId.SATURN_BARYCENTER: 6,
        BodyId.URANUS_BARYCENTER: 7,
        BodyId.NEPTUNE_BARYCENTER: 8,
        BodyId.PLUTO_BARYCENTER: 9,
        BodyId.SUN: 10,
        BodyId.MERCURY: 199,
        BodyId.VENUS: 299,
        BodyId.MOON: 301,
        BodyId.EARTH: 399,
        BodyId.MARS: 499,
        BodyId.JUPITER: 599,
        BodyId.SATURN: 699,
        BodyId.URANUS: 799,
        BodyId.NEPTUNE: 899,
        BodyId.PLUTO: 999,
    }
)
BODY_BY_NAIF_ID: Final[Mapping[int, BodyId]] = MappingProxyType(
    {naif_id: body for body, naif_id in NAIF_ID_BY_BODY.items()}
)

_ALIASES: Final[Mapping[str, BodyId]] = MappingProxyType(
    {
        "SSB": BodyId.SOLAR_SYSTEM_BARYCENTER,
        "SOLAR-SYSTEM BARYCENTER": BodyId.SOLAR_SYSTEM_BARYCENTER,
        "EMB": BodyId.EARTH_MOON_BARYCENTER,
        "EARTH BARYCENTER": BodyId.EARTH_MOON_BARYCENTER,
        "EARTH-MOON BARYCENTER": BodyId.EARTH_MOON_BARYCENTER,
    }
)


@lru_cache(maxsize=4096)
def _canonical_body_id_string(value: str) -> BodyId | str:
    text = " ".join(value.strip().upper().split())
    if not text:
        raise ValueError("body identifier must not be empty")
    alias = _ALIASES.get(text)
    if alias is not None:
        return alias
    if text.startswith("NAIF:"):
        try:
            naif_id = int(text.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid NAIF body identifier: {value!r}") from exc
        known = BODY_BY_NAIF_ID.get(naif_id)
        return known if known is not None else f"NAIF:{naif_id}"
    try:
        return BodyId(text)
    except ValueError:
        return text


def canonical_body_id(value: BodyId | str) -> BodyId | str:
    """Return an unambiguous canonical kernel identifier.

    Known major bodies become :class:`BodyId`. Unknown kernel names remain
    normalized strings, and explicit ``NAIF:<integer>`` identifiers are
    retained for minor bodies.
    """
    if isinstance(value, BodyId):
        return value
    if not isinstance(value, str):
        raise TypeError("body identifier must be a BodyId or string")
    return _canonical_body_id_string(value)


def body_name(value: BodyId | str) -> str:
    return str(canonical_body_id(value))


def naif_id(value: BodyId | str) -> int | None:
    canonical = canonical_body_id(value)
    if isinstance(canonical, BodyId):
        return NAIF_ID_BY_BODY[canonical]
    if canonical.startswith("NAIF:"):
        return int(canonical.split(":", 1)[1])
    return None


__all__ = [
    "BODY_BY_NAIF_ID",
    "NAIF_ID_BY_BODY",
    "BodyId",
    "body_name",
    "canonical_body_id",
    "naif_id",
]
