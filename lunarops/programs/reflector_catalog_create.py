"""Create a native reflector catalog from a source file or inline XYZ."""

from __future__ import annotations

from lunarops.config.context import RunContext
from lunarops.config.schema import sequence
from lunarops.fileio.catalog_sources import (
    load_reflector_coordinate_source,
    reflector_catalog_from_coordinates,
)
from lunarops.fileio.catalogs import write_reflector_catalog
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program
from lunarops.programs.specs import REFLECTOR_COORDINATE_SCHEMA

from ._catalog_create import create_catalog, exclusive_source


def _validate_config(config: dict, path_name: str):
    return exclusive_source(config, path_name, "inputFileReflectorCoordinates", "reflectorCoordinates")


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
        outputs=(ArtifactSlot("outputFileReflectorCatalog", "ReflectorCatalogFile"),),
        fields=(
            sequence(
                "reflectorCoordinates",
                item_kind="mapping",
                item_nested=REFLECTOR_COORDINATE_SCHEMA,
                min_items=0,
                allow_none=True,
            ),
        ),
        validator=_validate_config,
    )
)
def reflector_catalog_create(config: dict, context: RunContext):
    return create_catalog(
        config,
        context,
        file_key="inputFileReflectorCoordinates",
        inline_key="reflectorCoordinates",
        output_key="outputFileReflectorCatalog",
        load=load_reflector_coordinate_source,
        inline=reflector_catalog_from_coordinates,
        write=write_reflector_catalog,
        program="ReflectorCatalogCreate",
        noun="reflector",
    )


__all__ = ["reflector_catalog_create"]
