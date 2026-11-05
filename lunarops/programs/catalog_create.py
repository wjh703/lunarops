"""Programs that create the station, reflector and mass catalogs.

``StationCatalogCreate`` writes ITRF station positions and velocities, ``ReflectorCatalogCreate`` writes
reflector positions in the Moon principal-axis (PA) frame, and ``MassCatalogCreate`` writes the
gravitational parameters (GM) of the bodies used by the lunar dynamics, in m^3/s^2.
"""

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
    """Check that a config gives exactly one of a source file and inline entries.

    Parameters
    ----------
    config : dict
        Program config.
    path_name : str
        Config path used in error messages.
    file_key : str
        Key of the source file.
    inline_key : str
        Key of the inline entries.

    Returns
    -------
    dict
        The config, unchanged.

    Raises
    ------
    ValueError
        If both or neither key is set.
    """
    if (config.get(file_key) is not None) == (config.get(inline_key) is not None):
        raise ValueError(f"{path_name} requires exactly one of '{file_key}' or '{inline_key}'.")
    return config


def station_catalog_from_coordinates(entries, *, path: str = "stationCoordinates"):
    """Build station records from resolved station coordinate entries.

    Parameters
    ----------
    entries : list of dict
        Entries with ``key``, ``xyzM`` (ITRF position, m), ``velocityMPerYear`` (ITRF velocity, m/yr) and
        ``positionEpochUtc`` (UTC reference epoch of the position).
    path : str, optional
        Config path of the entries (unused).

    Returns
    -------
    dict of str to StationRecord
        Records keyed by station ID.
    """
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
    """Build reflector records from resolved reflector coordinate entries.

    Parameters
    ----------
    entries : list of dict
        Entries with ``key`` and ``xyzM`` (position in the Moon PA frame of the ephemeris, m).
    path : str, optional
        Config path of the entries (unused).

    Returns
    -------
    dict of str to ReflectorRecord
        Records keyed by reflector ID.
    """
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
                "ExternalStationCoordinatesFile",
                external=True,
                required=False,
                description=(
                    "YAML coordinate source with a `stationCoordinates` list. Give either this or the inline "
                    "`stationCoordinates`."
                ),
            ),
        ),
        outputs=(
            ArtifactSlot(
                "outputFileStationCatalog",
                "stationCatalog",
                description="Output native ITRF station catalog.",
            ),
        ),
        fields=(
            sequence(
                "stationCoordinates",
                item_kind="mapping",
                item_nested=STATION_COORDINATE_SCHEMA,
                min_items=1,
                allow_none=True,
                description="Inline station coordinates.",
            ),
        ),
        validator=_validate_station,
    )
)
def station_catalog_create(config: dict, context: RunContext):
    """Write a station catalog from a coordinate source file or inline coordinates.

    Reads the station entries from ``inputFileStationCoordinates`` or ``stationCoordinates`` (geodetic WGS84
    coordinates are already converted to ITRF XYZ by the schema) and writes the ``stationCatalog`` artifact
    ``outputFileStationCatalog``. See the program reference (``docs/reference/programs.md``) for every
    configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    Path
        Path of the written catalog.
    """
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
                "ExternalReflectorCoordinatesFile",
                external=True,
                required=False,
                description=(
                    "YAML coordinate source with a `reflectorCoordinates` list. Give either this or the inline "
                    "`reflectorCoordinates`."
                ),
            ),
        ),
        outputs=(
            ArtifactSlot(
                "outputFileReflectorCatalog",
                "reflectorCatalog",
                description="Output native Moon-PA reflector catalog.",
            ),
        ),
        fields=(
            sequence(
                "reflectorCoordinates",
                item_kind="mapping",
                item_nested=REFLECTOR_COORDINATE_SCHEMA,
                min_items=0,
                allow_none=True,
                description="Inline reflector coordinates.",
            ),
        ),
        validator=_validate_reflector,
    )
)
def reflector_catalog_create(config: dict, context: RunContext):
    """Write a reflector catalog from a coordinate source file or inline coordinates.

    Reads the reflector entries (Moon PA frame positions, m) from ``inputFileReflectorCoordinates`` or
    ``reflectorCoordinates`` and writes the ``reflectorCatalog`` artifact ``outputFileReflectorCatalog``.
    See the program reference (``docs/reference/programs.md``) for every configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    Path
        Path of the written catalog.
    """
    entries = (
        load_reflector_coordinate_source(context.resolve_path(config["inputFileReflectorCoordinates"]))
        if config.get("inputFileReflectorCoordinates") is not None
        else config["reflectorCoordinates"]
    )
    catalog = reflector_catalog_from_coordinates(entries)
    output = write_reflector_catalog(catalog, context.resolve_path(config["outputFileReflectorCatalog"]))
    print(f"[ReflectorCatalogCreate] {len(catalog)} reflector(s) -> {output}")
    return output


#: Astronomical unit, in m (IAU 2012 Resolution B2).
AU_M = 149_597_870_700.0
#: Length of one day, in s.
DAY_S = 86_400.0
#: Factor converting a GM in the given unit to m^3/s^2.
GM_UNIT_SCALE ={"m3/s2": 1.0, "au3/day2": AU_M**3 / DAY_S**2}

BODY_MASS_SCHEMA = ConfigSchema(
    (
        string(
            "bodyId",
            required=True,
            non_empty=True,
            description="Canonical body ID with NAIF semantics, such as `SUN`, `MARS BARYCENTER` or `NAIF:2000001`.",
        ),
        string("name", default="", non_empty=False, description="Human-readable body name."),
        integer(
            "ephemerisTarget",
            required=True,
            description="NAIF integer ID used to query the body state from the ephemeris.",
        ),
        number(
            "gm",
            default=None,
            gt=0,
            allow_none=True,
            description=(
                "Gravitational parameter in `gmUnit`. Omitted for EARTH and MOON when `earthMoonCanonical` is given."
            ),
        ),
        string(
            "gmUnit",
            default="m3/s2",
            choices=tuple(GM_UNIT_SCALE),
            description="Unit of `gm`: `m3/s2` or `au3/day2`.",
        ),
        sequence(
            "groups",
            required=True,
            item_kind="string",
            min_items=1,
            non_empty=True,
            description=(
                "Group names used to select the body as a perturber, for example `de440_major` or `sb441_asteroid`."
            ),
        ),
        string("source", default="", non_empty=False, description="Provenance of the value, stored in the catalog."),
    )
)

EARTH_MOON_CANONICAL_SCHEMA = ConfigSchema(
    (
        number("gmbAu3Day2", required=True, gt=0, description="GM of the Earth-Moon system, in au^3/day^2."),
        number("emrat", required=True, gt=0, description="Earth-to-Moon mass ratio."),
    )
)


def earth_moon_gm_from_gmb_emrat(gmb_au3_day2: float, emrat: float) -> tuple[float, float]:
    """Split the Earth-Moon system GM into Earth and Moon GM using the Earth/Moon mass ratio.

    GM_Moon = GMB / (1 + EMRAT) and GM_Earth = GMB - GM_Moon.

    Parameters
    ----------
    gmb_au3_day2 : float
        GM of the Earth-Moon system, in au^3/day^2.
    emrat : float
        Earth-to-Moon mass ratio.

    Returns
    -------
    tuple of (float, float)
        Earth GM and Moon GM, in m^3/s^2.

    Raises
    ------
    ValueError
        If either input is not positive and finite.
    """
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
        outputs=(ArtifactSlot("outputFileMassCatalog", "massCatalog", description="Output native mass catalog."),),
        fields=(
            string("catalogName", required=True, non_empty=True, description="Name stored in the catalog header."),
            string("source", default="", non_empty=False, description="Provenance of the catalog as a whole."),
            integer(
                "expectedBodyCount",
                default=None,
                ge=1,
                allow_none=True,
                description="If set, the run fails unless the catalog contains exactly this many bodies.",
            ),
            mapping(
                "expectedGroupCounts",
                default={},
                description="Expected number of bodies per group; the run fails on any mismatch.",
            ),
            mapping(
                "earthMoonCanonical",
                default=None,
                nested=EARTH_MOON_CANONICAL_SCHEMA,
                allow_none=True,
                description=(
                    "Derive the Earth and Moon GM from the Earth-Moon system GM and the mass ratio, so all models "
                    "share one consistent pair."
                ),
            ),
            sequence(
                "bodyMasses",
                required=True,
                item_kind="mapping",
                item_nested=BODY_MASS_SCHEMA,
                min_items=1,
                description="Bodies of the catalog.",
            ),
        ),
    )
)
def mass_catalog_create(config: dict, context: RunContext):
    """Write a mass catalog from the configured body GMs.

    Each entry of ``bodyMasses`` is converted to m^3/s^2 from its ``gmUnit``. With ``earthMoonCanonical``,
    the EARTH and MOON GMs are derived from the Earth-Moon system GM and the mass ratio and must not be
    given in ``bodyMasses``. The catalog is checked against ``expectedBodyCount`` and
    ``expectedGroupCounts`` and written as the ``massCatalog`` artifact ``outputFileMassCatalog``. See the
    program reference (``docs/reference/programs.md``) for every configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    Path
        Path of the written catalog.

    Raises
    ------
    ValueError
        If a GM is missing or given twice, or the body or group counts do not match.
    TypeError
        If an expected group count is not a non-negative integer.
    """
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
