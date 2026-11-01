from __future__ import annotations

from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord
from lunarops.config.context import RunContext
from lunarops.config.coordinates import REFLECTOR_COORDINATE_SCHEMA, STATION_COORDINATE_SCHEMA, validate_coordinate_keys
from lunarops.config.schema import sequence
from lunarops.fileio.catalog_sources import load_reflector_coordinate_source, load_station_coordinate_source
from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog

from .registry import ArtifactSlot, ProgramSpec, program


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
    ProgramSpec(
        name="StationCatalogCreate",
        summary="Create a native ITRF station catalog from XYZ or WGS84 geodetic coordinates.",
        inputs=(
            ArtifactSlot(
                "inputFileStationCoordinates",
                "ExternalStationCoordinatesFile",
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
    ProgramSpec(
        name="ReflectorCatalogCreate",
        summary="Create a native Moon-PA reflector catalog from a source file or inline XYZ.",
        inputs=(
            ArtifactSlot(
                "inputFileReflectorCoordinates",
                "ExternalReflectorCoordinatesFile",
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
