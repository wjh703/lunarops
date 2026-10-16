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

from ._catalog_create import create_catalog, exclusive_source


def _validate_config(config: dict, path_name: str):
    return exclusive_source(config, path_name, "inputFileStationCoordinates", "stationCoordinates")


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
    return create_catalog(
        config,
        context,
        file_key="inputFileStationCoordinates",
        inline_key="stationCoordinates",
        output_key="outputFileStationCatalog",
        load=load_station_coordinate_source,
        inline=station_catalog_from_coordinates,
        write=write_station_catalog,
        program="StationCatalogCreate",
        noun="station",
    )


__all__ = ["station_catalog_create"]
