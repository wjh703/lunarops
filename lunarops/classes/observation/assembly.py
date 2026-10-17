"""Observation configuration, catalogs and runtime assembly."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from lunarops.classes.builtin_registry import ensure_registered
from lunarops.config.schema import ConfigSchema, class_config, class_list
if TYPE_CHECKING:
    from lunarops.classes.ephemerides import Ephemeris
    from lunarops.classes.frames import EarthOrientationProvider, ReferenceFrameSystem
    from lunarops.classes.observation import LightTimeSolver
    from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord
    from lunarops.config.context import RunContext

OBSERVATION_MODEL_CATEGORIES = (
    "ephemerides",
    "earthRotation",
    "troposphere",
    "relativity",
    "stationDisplacement",
    "reflectorDisplacement",
    "rangeBias",
)


def resolve_model_configs(context, program_config: Mapping[str, object]) -> dict[str, object]:
    """Merge program overrides with run defaults."""
    merged: dict[str, object] = {}
    required = set(OBSERVATION_MODEL_CATEGORIES) - {"rangeBias"}
    program = program_config.get("program", program_config.get("name", "LLR observation assembly"))
    for category in OBSERVATION_MODEL_CATEGORIES:
        value = context.class_config(category, dict(program_config))
        if value is None:
            if category in required:
                raise ValueError(
                    f"{program}: missing model {category}; declare program.{category} or observationModel.{category}/shared.{category}; use explicit type: none only for optional corrections"
                )
            continue
        merged[category] = value
    return merged


def observation_model_schema() -> ConfigSchema:
    return ConfigSchema(
        fields=tuple(
            class_list(category, category, min_items=1)
            if category == "stationDisplacement"
            else class_config(category, category)
            for category in OBSERVATION_MODEL_CATEGORIES
        )
    )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ObservationAssembly:
    """Resolved model configuration and catalogs shared by serial and MPI."""

    model_configs: dict
    station_catalog: Mapping[str, StationRecord]
    reflector_catalog: Mapping[str, ReflectorRecord]


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ObservationRuntime:
    """Physical observation services shared by residual and prediction runs."""

    assembly: ObservationAssembly
    frames: ReferenceFrameSystem
    light_time_solver: LightTimeSolver


@dataclass(frozen=True, slots=True)
class _ObservationDependencies:
    run_context: RunContext
    ephemeris: Ephemeris
    earth_orientation_provider: EarthOrientationProvider
    frames: ReferenceFrameSystem
    cache_namespace: str

    def resolve_path(self, value: object) -> Path:
        return self.run_context.resolve_path(value)

    def create_class(self, category: str, config, *, cache: bool = True):
        return self.run_context.create_class(
            category,
            config,
            cache=cache,
            factory_context=self,
            cache_namespace=self.cache_namespace,
        )


def _compose_station_displacements(factory_context, configs):
    from lunarops.classes.displacement import CompositeStationDisplacement

    return CompositeStationDisplacement(
        tuple(factory_context.create_class("stationDisplacement", component, cache=True) for component in configs)
    )


def resolve_observation_assembly(
    context,
    program_config: dict,
    *,
    station_catalog=None,
    reflector_catalog=None,
) -> ObservationAssembly:
    """Resolve configs, paths, and catalogs once for every execution backend."""
    ensure_registered()
    from lunarops.fileio.catalogs import read_reflector_catalog, read_station_catalog

    merged = resolve_model_configs(context, program_config)

    stations = (
        station_catalog
        if station_catalog is not None
        else read_station_catalog(context.resolve_path(program_config["inputFileStationCatalog"]))
    )
    reflectors = (
        reflector_catalog
        if reflector_catalog is not None
        else read_reflector_catalog(context.resolve_path(program_config["inputFileReflectorCatalog"]))
    )
    return ObservationAssembly(merged, stations, reflectors)


def build_observation_processor(
    context,
    program_config: dict,
    *,
    station_catalog=None,
    reflector_catalog=None,
):
    """Assemble :class:`LlrObservationProcessor` from config.

    Expected class configs (program entry overrides ``globals:``)::

        ephemerides:           {type: calceph, directory: ..., lunarRelativisticScaleConvention: alreadyScaled,
                                longitudeLibrationCorrection: none}
        earthRotation:         {type: file, file: native EarthOrientationParameter artifact}
        troposphere:           mendesPavlis
        relativity:            iersShapiro
        stationDisplacement:   [{type: iers2010SolidEarthTide}, ...]
        reflectorDisplacement: lunarSolidTide | none
        rangeBias:             none | inpop21a | {type: table, file: ...} | {type: table, biases: [...]}

    Observation uncertainty is read directly from each normal-point record.
    """
    ensure_registered()
    from lunarops.classes.observation import (
        LlrObservationModel,
        LlrObservationProcessor,
        ObservationCatalogState,
        ObservationResolver,
    )

    runtime = build_observation_runtime(
        context,
        program_config,
        station_catalog=station_catalog,
        reflector_catalog=reflector_catalog,
    )
    model_configs = runtime.assembly.model_configs

    solver = runtime.light_time_solver
    model_state = ObservationCatalogState(
        runtime.assembly.station_catalog,
        runtime.assembly.reflector_catalog,
    )
    resolver = ObservationResolver(model_state)
    if "rangeBias" not in model_configs:
        raise ValueError(
            "LLR measurement assembly: missing rangeBias; declare program.rangeBias or observationModel.rangeBias, including explicit type: none"
        )
    range_bias_cfg = model_configs["rangeBias"]
    range_bias = context.create_class(
        "rangeBias",
        range_bias_cfg,
        cache=True,
    )
    observation_model = LlrObservationModel(
        frame_system=runtime.frames,
        light_time_solver=solver,
        range_bias_model=range_bias,
    )
    processor = LlrObservationProcessor(
        resolver,
        observation_model,
    )
    return processor


def build_observation_runtime(
    context,
    program_config: dict,
    *,
    station_catalog=None,
    reflector_catalog=None,
) -> ObservationRuntime:
    """Build physical observation services without requiring normal points."""
    ensure_registered()
    from lunarops.classes.frames import ReferenceFrameSystem
    from lunarops.classes.observation import LightTimeSolver

    assembly = resolve_observation_assembly(
        context,
        program_config,
        station_catalog=station_catalog,
        reflector_catalog=reflector_catalog,
    )
    model_configs = assembly.model_configs

    ephemeris = context.create_class("ephemerides", model_configs["ephemerides"], cache=True)
    earth_orientation_provider = context.create_class("earthRotation", model_configs["earthRotation"], cache=True)
    frames = ReferenceFrameSystem(
        ephemeris,
        earth_orientation_provider,
        ephemeris.lunar_orientation,
        ephemeris.relativistic_scale,
    )
    factory_context = _ObservationDependencies(
        run_context=context,
        ephemeris=ephemeris,
        earth_orientation_provider=earth_orientation_provider,
        frames=frames,
        cache_namespace=f"observation:{context.next_observation_spec_id()}",
    )
    solver = LightTimeSolver(
        frame_system=frames,
        gravitational_delay_model=factory_context.create_class("relativity", model_configs["relativity"], cache=False),
        troposphere_delay_model=factory_context.create_class("troposphere", model_configs["troposphere"], cache=True),
        station_displacement_model=_compose_station_displacements(
            factory_context, model_configs["stationDisplacement"]
        ),
        reflector_displacement_model=factory_context.create_class(
            "reflectorDisplacement",
            model_configs["reflectorDisplacement"],
            cache=True,
        ),
    )
    return ObservationRuntime(assembly, frames, solver)


