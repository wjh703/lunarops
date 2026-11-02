from __future__ import annotations

import numpy as np

from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord
from lunarops.config.context import RunContext
from lunarops.config.coordinates import REFLECTOR_COORDINATE_SCHEMA, STATION_COORDINATE_SCHEMA, validate_coordinate_keys
from lunarops.config.schema import ConfigSchema, integer, mapping, number, sequence, string
from lunarops.fileio.catalog_sources import load_reflector_coordinate_source, load_station_coordinate_source
from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog
from lunarops.fileio.mass_catalog import BodyMass, MassCatalog, write_mass_catalog

from .registry import ArtifactSlot, program, program_spec


def exclusive_source(config, path_name, file_key, inline_key):
    if (config.get(file_key) is not None) == (config.get(inline_key) is not None):
        raise ValueError(f"{path_name} requires exactly one of '{file_key}' or '{inline_key}'.")
    return config


def station_catalog_from_coordinates(entries, *, path: str = "stationCoordinates"):
    catalog = {}
    for entry in entries:
        key = str(entry["key"]).strip()
        catalog[key] = StationRecord(
            name=key,
            itrf_xyz_m=entry["xyzM"],
            itrf_velocity_m_per_year=entry["velocityMPerYear"],
            position_epoch_utc=entry["positionEpochUtc"],
        )
    return catalog


def reflector_catalog_from_coordinates(entries, *, path: str = "reflectorCoordinates"):
    catalog = {}
    for entry in entries:
        key = str(entry["key"]).strip()
        catalog[key] = ReflectorRecord(name=key, moon_fixed_xyz_m=entry["xyzM"])
    return catalog


def _validate_station(config: dict, path_name: str):
    exclusive_source(config, path_name, "inputFileStationCoordinates", "stationCoordinates")
    if config.get("stationCoordinates") is not None:
        validate_coordinate_keys(config["stationCoordinates"], path_name)
    return config


@program(
    program_spec(
        name="StationCatalogCreate",
        summary="Create a native ITRF station catalog from XYZ or WGS84 geodetic coordinates.",
        inputs=(
            ArtifactSlot(
                "inputFileStationCoordinates",
                "ExternalStationCoordinatesFile", external=True,
                required=False,
            ),
        ),
        outputs=(ArtifactSlot("outputFileStationCatalog", "stationCatalog"),),
        fields=(
            sequence(
                "stationCoordinates",
                item_kind="mapping",
                item_nested=STATION_COORDINATE_SCHEMA,
                min_items=1,
                allow_none=True,
            ),
        ),
        validator=_validate_station,
    )
)
def station_catalog_create(config: dict, context: RunContext):
    entries = (
        load_station_coordinate_source(context.resolve_path(config["inputFileStationCoordinates"]))
        if config.get("inputFileStationCoordinates") is not None
        else config["stationCoordinates"]
    )
    catalog = station_catalog_from_coordinates(entries)
    output = write_station_catalog(catalog, context.resolve_path(config["outputFileStationCatalog"]))
    print(f"[StationCatalogCreate] {len(catalog)} station(s) -> {output}")
    return output


def _validate_reflector(config: dict, path_name: str):
    exclusive_source(config, path_name, "inputFileReflectorCoordinates", "reflectorCoordinates")
    if config.get("reflectorCoordinates") is not None:
        validate_coordinate_keys(config["reflectorCoordinates"], path_name)
    return config


@program(
    program_spec(
        name="ReflectorCatalogCreate",
        summary="Create a native Moon-PA reflector catalog from a source file or inline XYZ.",
        inputs=(
            ArtifactSlot(
                "inputFileReflectorCoordinates",
                "ExternalReflectorCoordinatesFile", external=True,
                required=False,
            ),
        ),
        outputs=(ArtifactSlot("outputFileReflectorCatalog", "reflectorCatalog"),),
        fields=(
            sequence(
                "reflectorCoordinates",
                item_kind="mapping",
                item_nested=REFLECTOR_COORDINATE_SCHEMA,
                min_items=0,
                allow_none=True,
            ),
        ),
        validator=_validate_reflector,
    )
)
def reflector_catalog_create(config: dict, context: RunContext):
    entries = (
        load_reflector_coordinate_source(context.resolve_path(config["inputFileReflectorCoordinates"]))
        if config.get("inputFileReflectorCoordinates") is not None
        else config["reflectorCoordinates"]
    )
    catalog = reflector_catalog_from_coordinates(entries)
    output = write_reflector_catalog(catalog, context.resolve_path(config["outputFileReflectorCatalog"]))
    print(f"[ReflectorCatalogCreate] {len(catalog)} reflector(s) -> {output}")
    return output


AU_M = 149_597_870_700.0
DAY_S = 86_400.0
GM_UNIT_SCALE = {"m3/s2": 1.0, "au3/day2": AU_M**3 / DAY_S**2}

BODY_MASS_SCHEMA = ConfigSchema(
    (
        string("bodyId", required=True, non_empty=True),
        string("name", default="", non_empty=False),
        integer("ephemerisTarget", required=True),
        number("gm", default=None, gt=0, allow_none=True),
        string("gmUnit", default="m3/s2", choices=tuple(GM_UNIT_SCALE)),
        sequence("groups", required=True, item_kind="string", min_items=1, non_empty=True),
        string("source", default="", non_empty=False),
    )
)

EARTH_MOON_CANONICAL_SCHEMA = ConfigSchema(
    (
        number("gmbAu3Day2", required=True, gt=0),
        number("emrat", required=True, gt=0),
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
    program_spec(
        name="MassCatalogCreate",
        summary="Normalize configured body GMs and create a native dynamics mass catalog.",
        outputs=(ArtifactSlot("outputFileMassCatalog", "massCatalog"),),
        fields=(
            string("catalogName", required=True, non_empty=True),
            string("source", default="", non_empty=False),
            integer("expectedBodyCount", default=None, ge=1, allow_none=True),
            mapping("expectedGroupCounts", default={}),
            mapping("earthMoonCanonical", default=None, nested=EARTH_MOON_CANONICAL_SCHEMA, allow_none=True),
            sequence("bodyMasses", required=True, item_kind="mapping", item_nested=BODY_MASS_SCHEMA, min_items=1),
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
