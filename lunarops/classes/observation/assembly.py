"""Observation configuration, catalogs and runtime assembly.

The functions here resolve the observation-model declarations of a program (ephemerides, Earth rotation,
troposphere, relativity, station and reflector displacements, range bias), read the station and reflector
catalogs, and build the frame system, light-time solver and ``LlrObservationProcessor`` from them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    "reflectorDelay",
    "rangeBias",
)
# Categories that ``resolve_model_configs`` may leave undeclared: only the observation processor needs a range bias.
_PROCESSOR_ONLY_CATEGORIES = frozenset({"rangeBias"})
# Categories that default to ``type: none`` when neither the program nor ``models`` declares them.
_DEFAULT_MODEL_CONFIGS = {"reflectorDelay": {"type": "none"}}
# Categories whose instance is shared through the run cache: they open files or tables and need no runtime object.
_RUN_CACHED_CATEGORIES = frozenset({"ephemerides", "earthRotation", "troposphere", "reflectorDelay", "rangeBias"})
# Runtime objects passed to each station-displacement type besides its configuration; other types get none.
_STATION_DISPLACEMENT_DEPENDENCIES = {
    "iers2010SolidEarthTide": ("frames",),
    "iers2010PoleTide": ("earth_orientation_provider",),
    "iers2010OceanPoleTide": ("earth_orientation_provider",),
}


def _program_label(program_config: Mapping[str, object]) -> object:
    return program_config.get("program", program_config.get("name", "LLR observation assembly"))


def _require_model_config(model_configs: Mapping[str, object], category: str, program: object) -> object:
    """Return the declaration of ``category``; raise ``ValueError`` naming the config keys when it is missing."""
    if category not in model_configs:
        raise ValueError(
            f"{program}: missing model {category}; declare program.{category} or models.{category} with an "
            "explicit type (type: none disables an optional correction)"
        )
    return model_configs[category]


def resolve_model_configs(context, program_config: Mapping[str, object]) -> dict[str, object]:
    """Resolve the declaration of every observation-model category for one program.

    Each category in ``OBSERVATION_MODEL_CATEGORIES`` is looked up with ``context.class_config``: first in the
    program configuration, then in ``models``.

    Parameters
    ----------
    context : RunContext
        Run context holding the ``models`` declarations.
    program_config : Mapping[str, object]
        Program configuration; its ``program`` or ``name`` entry labels error messages.

    Returns
    -------
    dict[str, object]
        Declaration per declared category. ``rangeBias`` is omitted when it resolves to ``None``; the observation
        processor requires it.

    Raises
    ------
    ValueError
        If a category other than ``rangeBias`` resolves to ``None``.
    """
    merged: dict[str, object] = {}
    for category in OBSERVATION_MODEL_CATEGORIES:
        if (
            category in _DEFAULT_MODEL_CONFIGS
            and category not in program_config
            and category not in context.model_configs
        ):
            merged[category] = dict(_DEFAULT_MODEL_CONFIGS[category])
            continue
        value = context.class_config(category, dict(program_config))
        if value is not None:
            merged[category] = value
    program = _program_label(program_config)
    for category in OBSERVATION_MODEL_CATEGORIES:
        if category not in _PROCESSOR_ONLY_CATEGORIES:
            _require_model_config(merged, category, program)
    return merged


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


def _create_model(context, category: str, config, **dependencies):
    """Create the model declared by ``config`` in ``category``.

    Categories in ``_RUN_CACHED_CATEGORIES`` take no dependencies and are created through the run cache, so
    identical declarations share one instance. The others are created for one runtime with ``dependencies``
    (``frames``, ``earth_orientation_provider``) passed to the registered factory.
    """
    from lunarops.config.registry import create

    if category in _RUN_CACHED_CATEGORIES:
        if dependencies:
            raise TypeError(f"Run-cached {category} models take no runtime dependencies, got {sorted(dependencies)}.")
        return context.create_class(category, config, cache=True)
    return create(category, config, context, **dependencies)


def _optional_model(context, category: str, config, **dependencies):
    """Return the model of an optional correction, or ``None`` when it is declared with ``type: none``."""
    if config["type"] == "none":
        return None
    return _create_model(context, category, config, **dependencies)


def _optional_models(context, category: str, config, **dependencies) -> tuple:
    """Return the model of an optional correction as a 1-tuple, or ``()`` for ``type: none``."""
    model = _optional_model(context, category, config, **dependencies)
    return () if model is None else (model,)


def _compose_station_displacements(context, configs, **dependencies):
    """Create the declared station displacements and sum them in one composite model.

    Declarations of type ``none`` are skipped. Each type receives the ``dependencies`` named for it in
    ``_STATION_DISPLACEMENT_DEPENDENCIES``.
    """
    from lunarops.classes.displacement import CompositeStationDisplacement

    components = []
    for config in configs:
        needed = {name: dependencies[name] for name in _STATION_DISPLACEMENT_DEPENDENCIES.get(config["type"], ())}
        model = _optional_model(context, "stationDisplacement", config, **needed)
        if model is not None:
            components.append(model)
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
        If no ``rangeBias`` declaration is resolved (``type: none`` must be given explicitly), or a required
        observation model is missing.
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
    range_bias_config = _require_model_config(
        runtime.assembly.model_configs, "rangeBias", _program_label(program_config)
    )
    model_state = ObservationCatalogState(
        runtime.assembly.station_catalog,
        runtime.assembly.reflector_catalog,
    )
    observation_model = LlrObservationModel(
        frame_system=runtime.frames,
        light_time_solver=runtime.light_time_solver,
        range_bias_model=_create_model(context, "rangeBias", range_bias_config),
    )
    return LlrObservationProcessor(ObservationResolver(model_state), observation_model)


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

    ephemeris = _create_model(context, "ephemerides", model_configs["ephemerides"])
    earth_orientation_provider = _create_model(context, "earthRotation", model_configs["earthRotation"])
    frames = ReferenceFrameSystem(
        ephemeris,
        earth_orientation_provider,
        ephemeris.lunar_orientation,
        ephemeris.relativistic_scale,
    )
    solver = LightTimeSolver(
        frame_system=frames,
        gravitational_delay_models=_optional_models(context, "relativity", model_configs["relativity"], frames=frames),
        troposphere_delay_models=_optional_models(context, "troposphere", model_configs["troposphere"]),
        station_displacement_model=_compose_station_displacements(
            context,
            model_configs["stationDisplacement"],
            frames=frames,
            earth_orientation_provider=earth_orientation_provider,
        ),
        reflector_displacement_models=_optional_models(
            context, "reflectorDisplacement", model_configs["reflectorDisplacement"], frames=frames
        ),
        reflector_delay_models=_optional_models(context, "reflectorDelay", model_configs["reflectorDelay"]),
    )
    return ObservationRuntime(assembly, frames, solver)
