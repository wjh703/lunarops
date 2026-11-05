"""Observation configuration, catalogs and runtime assembly.

The functions here resolve the observation-model declarations of a program (ephemerides, Earth rotation,
troposphere, relativity, station and reflector displacements, range bias), read the station and reflector
catalogs, and build the frame system, light-time solver and ``LlrObservationProcessor`` from them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lunarops.config.schema import ConfigSchema, class_config, class_list

if TYPE_CHECKING:
    from lunarops.classes.frames import ReferenceFrameSystem
    from lunarops.classes.observation import LightTimeSolver
    from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord

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
    """Resolve the declaration of every observation-model category for one program.

    Each category in ``OBSERVATION_MODEL_CATEGORIES`` is looked up with ``context.class_config``: first in the
    program configuration, then in ``observationModel``, then in ``shared``.

    Parameters
    ----------
    context : RunContext
        Run context holding the ``observationModel`` and ``shared`` declarations.
    program_config : Mapping[str, object]
        Program configuration; its ``program`` or ``name`` entry labels error messages.

    Returns
    -------
    dict[str, object]
        Declaration per category. ``rangeBias`` is omitted when it resolves to ``None``.

    Raises
    ------
    ValueError
        If a category other than ``rangeBias`` resolves to ``None``.
    """
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
    """Return the schema of the ``observationModel`` section.

    ``stationDisplacement`` is a list of at least one declaration; every other category is a single declaration.
    """
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
    """Resolved observation-model declarations and the station and reflector catalogs.

    Attributes
    ----------
    model_configs : dict
        Declaration per observation-model category, as returned by ``resolve_model_configs``.
    station_catalog : Mapping[str, StationRecord]
        Station records by catalog key.
    reflector_catalog : Mapping[str, ReflectorRecord]
        Reflector records by catalog key.
    """

    model_configs: dict
    station_catalog: Mapping[str, StationRecord]
    reflector_catalog: Mapping[str, ReflectorRecord]


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ObservationRuntime:
    """Instantiated observation models built from an ``ObservationAssembly``.

    Attributes
    ----------
    assembly : ObservationAssembly
        Declarations and catalogs the runtime was built from.
    frames : ReferenceFrameSystem
        Frame and time-scale transformations built from the ephemeris and Earth-rotation models.
    light_time_solver : LightTimeSolver
        Two-way light-time solver with the configured delay and displacement models.
    """

    assembly: ObservationAssembly
    frames: ReferenceFrameSystem
    light_time_solver: LightTimeSolver


def _compose_station_displacements(context, configs, **dependencies):
    """Create the declared station displacements and sum them in one composite model.

    Declarations of type ``none`` are skipped. ``iers2010SolidEarthTide`` receives ``frames``; the pole-tide
    types receive ``earth_orientation_provider``.
    """
    from lunarops.classes.displacement import CompositeStationDisplacement
    from lunarops.config.registry import create

    components = []
    for config in configs:
        typ = config["type"]
        if typ == "none":
            continue
        needed = (
            {"frames": dependencies["frames"]}
            if typ == "iers2010SolidEarthTide"
            else {"earth_orientation_provider": dependencies["earth_orientation_provider"]}
            if typ in {"iers2010PoleTide", "iers2010OceanPoleTide"}
            else {}
        )
        components.append(create("stationDisplacement", config, context, **needed))
    return CompositeStationDisplacement(tuple(components))


def resolve_observation_assembly(
    context,
    program_config: dict,
    *,
    station_catalog=None,
    reflector_catalog=None,
) -> ObservationAssembly:
    """Resolve the observation-model declarations and load the station and reflector catalogs.

    Parameters
    ----------
    context : RunContext
        Run context used to resolve declarations and relative paths.
    program_config : dict
        Program configuration. ``inputFileStationCatalog`` and ``inputFileReflectorCatalog`` are read when the
        corresponding catalog argument is ``None``.
    station_catalog : Mapping[str, StationRecord], optional
        Station catalog to use instead of reading ``inputFileStationCatalog``.
    reflector_catalog : Mapping[str, ReflectorRecord], optional
        Reflector catalog to use instead of reading ``inputFileReflectorCatalog``.

    Returns
    -------
    ObservationAssembly
        Resolved declarations and catalogs.
    """
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
    """Build the LLR observation processor for one program.

    The processor uses the runtime from ``build_observation_runtime``, a resolver over a mutable copy of the
    catalogs, and the configured ``rangeBias`` model. The a-priori observation uncertainty is taken from each
    normal-point record.

    Parameters
    ----------
    context : RunContext
        Run context used to resolve declarations, paths and cached model instances.
    program_config : dict
        Program configuration.
    station_catalog : Mapping[str, StationRecord], optional
        Station catalog to use instead of reading ``inputFileStationCatalog``.
    reflector_catalog : Mapping[str, ReflectorRecord], optional
        Reflector catalog to use instead of reading ``inputFileReflectorCatalog``.

    Returns
    -------
    LlrObservationProcessor
        Processor that evaluates normal-point datasets.

    Raises
    ------
    ValueError
        If no ``rangeBias`` declaration is resolved (``type: none`` must be given explicitly).
    """
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
    """Build the frame system and light-time solver from the observation-model declarations.

    Relativity, troposphere and reflector-displacement declarations of type ``none`` contribute no model.

    Parameters
    ----------
    context : RunContext
        Run context used to resolve declarations, paths and cached model instances.
    program_config : dict
        Program configuration.
    station_catalog : Mapping[str, StationRecord], optional
        Station catalog to use instead of reading ``inputFileStationCatalog``.
    reflector_catalog : Mapping[str, ReflectorRecord], optional
        Reflector catalog to use instead of reading ``inputFileReflectorCatalog``.

    Returns
    -------
    ObservationRuntime
        Assembly, frame system and light-time solver.
    """
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
    from lunarops.config.registry import create

    relativity = model_configs["relativity"]
    reflector = model_configs["reflectorDisplacement"]
    solver = LightTimeSolver(
        frame_system=frames,
        gravitational_delay_models=() if relativity["type"] == "none" else (create(
            "relativity",
            relativity,
            context,
            frames=frames,
        ),),
        troposphere_delay_models=() if model_configs["troposphere"]["type"] == "none" else (
            context.create_class("troposphere", model_configs["troposphere"]),
        ),
        station_displacement_model=_compose_station_displacements(
            context,
            model_configs["stationDisplacement"],
            frames=frames,
            earth_orientation_provider=earth_orientation_provider,
        ),
        reflector_displacement_models=() if reflector["type"] == "none" else (create(
            "reflectorDisplacement",
            reflector,
            context,
            frames=frames,
        ),),
    )
    return ObservationRuntime(assembly, frames, solver)
