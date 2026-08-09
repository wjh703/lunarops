"""Canonical catalog creation and solution application."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from lunarops.config.context import RunContext
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program


@program(
    ProgramSpec(
        name="CatalogCreate",
        summary="Publish the builtin station and reflector catalogs as typed text files.",
        outputs=(
            ArtifactSlot("outputFileStationCatalog", "StationCatalogFile"),
            ArtifactSlot("outputFileReflectorCatalog", "ReflectorCatalogFile"),
        ),
    )
)
def catalog_create(config: dict, context: RunContext):
    from lunarops.classes.observation.builtin_catalogs import REFLECTORS, STATIONS
    from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog

    station_path = context.resolve_path(config["outputFileStationCatalog"])
    reflector_path = context.resolve_path(config["outputFileReflectorCatalog"])
    write_station_catalog(STATIONS, station_path)
    write_reflector_catalog(REFLECTORS, reflector_path)
    print(
        f"[CatalogCreate] {len(STATIONS)} station(s) -> {station_path}; "
        f"{len(REFLECTORS)} reflector(s) -> {reflector_path}"
    )
    return station_path, reflector_path


@program(
    ProgramSpec(
        name="LlrApplySolution",
        summary="Apply reflector-position and range-bias corrections from a parameter vector.",
        inputs=(
            ArtifactSlot("inputFileSolution", "ParameterVectorFile"),
            ArtifactSlot("inputFileReflectorCatalog", "ReflectorCatalogFile"),
        ),
        outputs=(
            ArtifactSlot("outputFileReflectorCatalog", "ReflectorCatalogFile"),
            ArtifactSlot("outputFileModelState", "ModelStateFile"),
        ),
    )
)
def llr_apply_solution(config: dict, context: RunContext):
    from lunarops.fileio.catalogs import read_reflector_catalog, write_reflector_catalog
    from lunarops.fileio.parameters import read_parameter_vector
    from lunarops.fileio.structured_text import write_structured_text

    solution = read_parameter_vector(context.resolve_path(config["inputFileSolution"]))
    if solution.kind not in {"estimate", "correction"}:
        raise ValueError(f"LlrApplySolution requires vectorKind estimate or correction, found {solution.kind!r}.")
    input_catalog_path = context.resolve_path(config["inputFileReflectorCatalog"])
    output_catalog_path = context.resolve_path(config["outputFileReflectorCatalog"])
    if input_catalog_path.resolve() == output_catalog_path.resolve():
        raise ValueError("LlrApplySolution input and output catalogs must differ.")
    catalog = read_reflector_catalog(input_catalog_path)
    position_values: dict[str, np.ndarray] = {}
    position_axes: dict[str, set[int]] = {}
    range_biases: dict[str, float] = {}
    for name, unit, value in zip(solution.parameter_names, solution.units, solution.values):
        if name.parameter_type.startswith("position."):
            if unit != "m":
                raise ValueError(f"Position parameter {name} must use metres, found {unit!r}.")
            if name.object_name not in catalog:
                raise KeyError(f"Solution references unknown reflector {name.object_name!r}.")
            axis = {"position.x": 0, "position.y": 1, "position.z": 2}.get(name.parameter_type)
            if axis is None:
                raise ValueError(f"Unsupported reflector-position parameter {name}.")
            if solution.kind == "estimate":
                position_values.setdefault(
                    name.object_name,
                    np.asarray(catalog[name.object_name].moon_fixed_xyz_m, dtype=float).copy(),
                )[axis] = float(value)
            elif solution.kind == "correction":
                position_values.setdefault(name.object_name, np.zeros(3))[axis] += float(value)
            position_axes.setdefault(name.object_name, set()).add(axis)
        elif name.parameter_type.casefold() == "rangebias":
            if unit != "m":
                raise ValueError(f"Range-bias parameter {name} must use metres, found {unit!r}.")
            range_biases[str(name)] = range_biases.get(str(name), 0.0) + float(value)
        else:
            raise ValueError(f"LlrApplySolution does not support parameter type {name.parameter_type!r}.")

    for key, values in position_values.items():
        if solution.kind == "estimate" and position_axes[key] != {0, 1, 2}:
            raise ValueError(f"Absolute reflector estimate for {key!r} must contain x, y, and z.")
        position = (
            values if solution.kind == "estimate" else np.asarray(catalog[key].moon_fixed_xyz_m, dtype=float) + values
        )
        catalog[key] = replace(
            catalog[key],
            moon_fixed_xyz_m=position,
        )
    catalog_path = output_catalog_path
    state_path = context.resolve_path(config["outputFileModelState"])
    write_reflector_catalog(catalog, catalog_path)
    write_structured_text(
        state_path,
        "llrModelState",
        {
            "solutionKind": solution.kind,
            "reflectorPositionValuesM": {key: values.tolist() for key, values in sorted(position_values.items())},
            "rangeBiasValuesM": dict(sorted(range_biases.items())),
        },
    )
    print(
        f"[LlrApplySolution] {len(position_values)} reflector(s), {len(range_biases)} range bias(es) -> {catalog_path}"
    )
    return catalog


__all__ = ["catalog_create", "llr_apply_solution"]
