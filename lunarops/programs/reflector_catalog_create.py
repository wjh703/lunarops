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


def _validate_config(config: dict, path_name: str):
    has_file = config.get("inputFileReflectorCoordinates") is not None
    has_inline = config.get("reflectorCoordinates") is not None
    if has_file == has_inline:
        raise ValueError(
            f"{path_name} requires exactly one of 'inputFileReflectorCoordinates' "
            "or 'reflectorCoordinates'."
        )
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
    if config.get("inputFileReflectorCoordinates") is not None:
        catalog = load_reflector_coordinate_source(
            context.resolve_path(config["inputFileReflectorCoordinates"])
        )
    else:
        catalog = reflector_catalog_from_coordinates(config["reflectorCoordinates"])
    output = write_reflector_catalog(
        catalog,
        context.resolve_path(config["outputFileReflectorCatalog"]),
    )
    print(f"[ReflectorCatalogCreate] {len(catalog)} reflector(s) -> {output}")
    return output


__all__ = ["reflector_catalog_create"]
