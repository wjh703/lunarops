"""Create a native reflector catalog from inline Moon-PA coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lunarops.classes.observation.catalogs import ReflectorRecord
from lunarops.config.context import RunContext
from lunarops.config.schema import sequence
from lunarops.fileio.catalogs import write_reflector_catalog
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program
from lunarops.programs.specs import REFLECTOR_COORDINATE_SCHEMA


def _validate_config(config: dict, path: str):
    entries = config.get("reflectorCoordinates")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence) or not entries:
        raise ValueError(f"{path}.reflectorCoordinates must be a non-empty sequence.")
    seen = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"{path}.reflectorCoordinates[{index}] must be a mapping.")
        key = str(entry["key"]).strip()
        folded = key.casefold()
        if folded in seen:
            raise ValueError(f"{path}.reflectorCoordinates contains duplicate key {key!r}.")
        seen.add(folded)
    return config


@program(
    ProgramSpec(
        name="ReflectorCatalogCreate",
        summary="Create a native Moon-PA reflector catalog from inline coordinates.",
        outputs=(ArtifactSlot("outputFileReflectorCatalog", "ReflectorCatalogFile"),),
        fields=(
            sequence(
                "reflectorCoordinates",
                required=True,
                item_kind="mapping",
                item_nested=REFLECTOR_COORDINATE_SCHEMA,
                min_items=1,
                allow_none=False,
            ),
        ),
        validator=_validate_config,
    )
)
def reflector_catalog_create(config: dict, context: RunContext):
    catalog = {}
    for entry in config["reflectorCoordinates"]:
        key = str(entry["key"]).strip()
        catalog[key] = ReflectorRecord(name=key, moon_fixed_xyz_m=entry["xyzM"])
    output = write_reflector_catalog(
        catalog,
        context.resolve_path(config["outputFileReflectorCatalog"]),
    )
    print(f"[ReflectorCatalogCreate] {len(catalog)} reflector(s) -> {output}")
    return output


__all__ = ["reflector_catalog_create"]
