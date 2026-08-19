"""Create a native reflector PA-coordinate catalog."""

from __future__ import annotations

import csv
from pathlib import Path

from lunarops.classes.observation.catalogs import ReflectorRecord
from lunarops.config.context import RunContext
from lunarops.config.schema import string
from lunarops.fileio.catalogs import load_reflector_catalog, write_reflector_catalog
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program


def _validate_config(config: dict, path: str):
    source = config["source"]
    input_file = config.get("inputFileCoordinates")
    if source == "csv" and input_file is None:
        raise ValueError(f"{path}.inputFileCoordinates is required when source is csv.")
    if source == "builtin" and input_file is not None:
        raise ValueError(f"{path}.inputFileCoordinates is not allowed when source is builtin.")
    return config


def _aliases(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(item.strip() for item in value.split("|") if item.strip())


def _read_csv(path: Path) -> dict[str, ReflectorRecord]:
    builtin = load_reflector_catalog("builtin")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Reflector coordinate CSV has no header: {path}")
        fields = set(reader.fieldnames)
        key_field = "key" if "key" in fields else "reflector_key" if "reflector_key" in fields else None
        coordinate_fields = next(
            (
                candidate
                for candidate in (
                    ("x_m", "y_m", "z_m"),
                    ("final_x_m", "final_y_m", "final_z_m"),
                    ("moon_fixed_x_m", "moon_fixed_y_m", "moon_fixed_z_m"),
                )
                if set(candidate) <= fields
            ),
            None,
        )
        if key_field is None or coordinate_fields is None:
            raise ValueError(
                "Reflector CSV requires key or reflector_key and one supported XYZ column set."
            )
        result: dict[str, ReflectorRecord] = {}
        for row_number, row in enumerate(reader, start=2):
            key = str(row.get(key_field, "")).strip()
            if not key:
                raise ValueError(f"Reflector CSV row {row_number} has an empty key.")
            if key in result:
                raise ValueError(f"Reflector CSV contains duplicate key {key!r}.")
            try:
                coordinates = [float(row[field]) for field in coordinate_fields]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Reflector CSV row {row_number} has invalid coordinates.") from exc
            fallback = builtin.get(key)
            name = str(row.get("name", "")).strip() or (fallback.name if fallback is not None else key)
            aliases = _aliases(row.get("aliases"))
            if not aliases and fallback is not None:
                aliases = tuple(fallback.aliases)
            result[key] = ReflectorRecord(name=name, moon_fixed_xyz_m=coordinates, aliases=aliases)
    if not result:
        raise ValueError(f"Reflector coordinate CSV contains no records: {path}")
    return result


@program(
    ProgramSpec(
        name="ReflectorCatalogCreate",
        summary="Create a native Moon-PA reflector catalog from builtin coordinates or CSV.",
        inputs=(
            ArtifactSlot(
                "inputFileCoordinates",
                "ExternalReflectorCoordinatesFile",
                required=False,
            ),
        ),
        outputs=(ArtifactSlot("outputFileReflectorCatalog", "ReflectorCatalogFile"),),
        fields=(
            string(
                "source",
                default="builtin",
                choices=("builtin", "csv"),
                non_empty=True,
                allow_none=False,
            ),
        ),
        validator=_validate_config,
    )
)
def reflector_catalog_create(config: dict, context: RunContext):
    if config["source"] == "builtin":
        catalog = load_reflector_catalog("builtin")
    else:
        catalog = _read_csv(context.resolve_path(config["inputFileCoordinates"]))
    output = write_reflector_catalog(
        catalog,
        context.resolve_path(config["outputFileReflectorCatalog"]),
    )
    print(f"[ReflectorCatalogCreate] {len(catalog)} reflector(s) -> {output}")
    return output


__all__ = ["reflector_catalog_create"]
