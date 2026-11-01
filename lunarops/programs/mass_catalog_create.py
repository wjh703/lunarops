"""Create a normalized gravitational-parameter catalog from inline records."""

from __future__ import annotations

import numpy as np

from lunarops.config.context import RunContext
from lunarops.config.schema import ConfigSchema, integer, mapping, number, sequence, string
from lunarops.fileio.mass_catalog import BodyMass, MassCatalog, write_mass_catalog
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program

AU_M = 149_597_870_700.0
DAY_S = 86_400.0
GM_UNIT_SCALE = {"m3/s2": 1.0, "au3/day2": AU_M**3 / DAY_S**2}

BODY_MASS_SCHEMA = ConfigSchema(
    (
        string("bodyId", required=True, non_empty=True, allow_none=False),
        string("name", default="", allow_none=False),
        integer("ephemerisTarget", required=True, allow_none=False),
        number("gm", default=None, minimum=0, minimum_exclusive=True, allow_none=True),
        string("gmUnit", default="m3/s2", choices=tuple(GM_UNIT_SCALE), allow_none=False),
        sequence("groups", required=True, item_kind="string", min_items=1, non_empty=True, allow_none=False),
        string("source", default="", allow_none=False),
    )
)

EARTH_MOON_CANONICAL_SCHEMA = ConfigSchema(
    (
        number("gmbAu3Day2", required=True, minimum=0, minimum_exclusive=True, allow_none=False),
        number("emrat", required=True, minimum=0, minimum_exclusive=True, allow_none=False),
    )
)


def earth_moon_gm_from_gmb_emrat(gmb_au3_day2: float, emrat: float) -> tuple[float, float]:
    """Derive canonical Earth and Moon GM values in SI units."""
    system_gm = float(gmb_au3_day2) * GM_UNIT_SCALE["au3/day2"]
    ratio = float(emrat)
    if not np.isfinite(system_gm) or system_gm <= 0 or not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("GMB and EMRAT must be positive and finite.")
    moon_gm = system_gm / (1.0 + ratio)
    return system_gm - moon_gm, moon_gm


@program(
    ProgramSpec(
        name="MassCatalogCreate",
        summary="Normalize configured body GMs and create a native dynamics mass catalog.",
        outputs=(ArtifactSlot("outputFileMassCatalog", "MassCatalogFile"),),
        fields=(
            string("catalogName", required=True, non_empty=True, allow_none=False),
            string("source", default="", allow_none=False),
            integer("expectedBodyCount", default=None, minimum=1, allow_none=True),
            mapping("expectedGroupCounts", default={}, allow_none=False),
            mapping(
                "earthMoonCanonical",
                default=None,
                nested=EARTH_MOON_CANONICAL_SCHEMA,
                allow_none=True,
            ),
            sequence(
                "bodyMasses",
                required=True,
                item_kind="mapping",
                item_nested=BODY_MASS_SCHEMA,
                min_items=1,
                allow_none=False,
            ),
        ),
    )
)
def mass_catalog_create(config: dict, context: RunContext):
    canonical = config["earthMoonCanonical"]
    canonical_gm = None
    if canonical is not None:
        earth_gm, moon_gm = earth_moon_gm_from_gmb_emrat(canonical["gmbAu3Day2"], canonical["emrat"])
        canonical_gm = {"EARTH": earth_gm, "MOON": moon_gm}
    bodies = []
    for record in config["bodyMasses"]:
        body_id = record["bodyId"].strip().upper()
        if canonical_gm is not None and body_id in canonical_gm:
            if record["gm"] is not None:
                raise ValueError(f"Body {body_id} GM must come only from earthMoonCanonical.")
            gm_m3_s2 = canonical_gm[body_id]
        else:
            if record["gm"] is None:
                raise ValueError(f"Body {body_id} requires gm when it is not canonically derived.")
            gm_m3_s2 = float(record["gm"]) * GM_UNIT_SCALE[record["gmUnit"]]
        bodies.append(
            BodyMass(
                body_id=record["bodyId"],
                gm_m3_s2=gm_m3_s2,
                groups=tuple(record["groups"]),
                ephemeris_target=record["ephemerisTarget"],
                name=record["name"],
                source=record["source"],
            )
        )
    catalog = MassCatalog(bodies, name=config["catalogName"], source=config["source"])
    expected_body_count = config["expectedBodyCount"]
    if expected_body_count is not None and len(catalog.bodies) != expected_body_count:
        raise ValueError(f"Mass catalog has {len(catalog.bodies)} bodies; expected {expected_body_count}")
    for group, expected_count in config["expectedGroupCounts"].items():
        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0:
            raise TypeError(f"expectedGroupCounts[{group!r}] must be a nonnegative integer")
        actual_count = sum(group in body.groups for body in catalog.bodies)
        if actual_count != expected_count:
            raise ValueError(f"Mass catalog group {group!r} has {actual_count} bodies; expected {expected_count}")
    output = write_mass_catalog(catalog, context.resolve_path(config["outputFileMassCatalog"]))
    print(f"[MassCatalogCreate] {len(catalog.bodies)} body mass(es) -> {output}")
    return output


__all__ = ["earth_moon_gm_from_gmb_emrat", "mass_catalog_create"]
