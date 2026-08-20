"""Create a native station catalog from a source file or inline coordinates."""

from __future__ import annotations

from lunarops.config.context import RunContext
from lunarops.config.schema import sequence
from lunarops.fileio.catalog_sources import (
    load_station_coordinate_source,
    station_catalog_from_coordinates,
)
from lunarops.fileio.catalogs import write_station_catalog
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program
from lunarops.programs.specs import STATION_COORDINATE_SCHEMA


def _validate_config(config: dict, path_name: str):
    has_file = config.get("inputFileStationCoordinates") is not None
    has_inline = config.get("stationCoordinates") is not None
    if has_file == has_inline:
        raise ValueError(
            f"{path_name} requires exactly one of 'inputFileStationCoordinates' "
            "or 'stationCoordinates'."
        )
    return config


@program(
    ProgramSpec(
        name="StationCatalogCreate",
        summary="Create a native ITRF station catalog from a source file or coordinates and rates.",
        inputs=(
            ArtifactSlot(
                "inputFileStationCoordinates",
                "ExternalStationCoordinatesFile",
                required=False,
            ),
        ),
        outputs=(ArtifactSlot("outputFileStationCatalog", "StationCatalogFile"),),
        fields=(
            sequence(
                "stationCoordinates",
                item_kind="mapping",
                item_nested=STATION_COORDINATE_SCHEMA,
                min_items=1,
                allow_none=True,
            ),
        ),
        validator=_validate_config,
    )
)
def station_catalog_create(config: dict, context: RunContext):
    if config.get("inputFileStationCoordinates") is not None:
        catalog = load_station_coordinate_source(
            context.resolve_path(config["inputFileStationCoordinates"])
        )
    else:
        catalog = station_catalog_from_coordinates(config["stationCoordinates"])
    output = write_station_catalog(
        catalog,
        context.resolve_path(config["outputFileStationCatalog"]),
    )
    print(f"[StationCatalogCreate] {len(catalog)} station(s) -> {output}")
    return output


__all__ = ["station_catalog_create"]
