"""Register model implementations and assemble the LLR observation workflow.

This is the single place where config ``type`` names map to physics classes.
The factories expose only the parameters supported by the project
configuration; fixed physical constants stay in the model modules.

Registered categories and types
-------------------------------
ephemerides            : calceph
earthRotation          : iersC04
troposphere            : none | mendesPavlis
relativity             : none | iersShapiro
stationDisplacement    : none | sum | iers2010SolidEarthTide | iers2010PoleTide | iers2010OceanPoleTide | iers2010OceanTidalLoading
reflectorDisplacement  : none | lunarSolidTide
rangeBias             : none | inpop21a | table
parametrization        : reflectorPosition | stationRangeBias   (registered in their modules)

``RunContext.create_class(..., cache=True)`` is intentionally used here for
immutable/heavy backends such as CALCEPH and Earth-orientation sources.  Mutable model state
(catalog coordinates, station-bias values, future EOP/orbit corrections) stays
inside the returned ``LlrObservationProcessor`` instance.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Protocol

from lunarops.config.registry import (
    register_factory,
    registration_transaction,
)
from lunarops.config.schema import ConfigSchema, class_list, field, number, path, sequence, string

if TYPE_CHECKING:
    from lunarops.classes.ephemerides import Ephemeris
    from lunarops.classes.frames import EarthOrientationProvider, ReferenceFrameSystem
    from lunarops.config.context import RunContext
    from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord


_MODEL_CATEGORIES = (
    "ephemerides",
    "earthRotation",
    "troposphere",
    "relativity",
    "stationDisplacement",
    "reflectorDisplacement",
    "rangeBias",
)
_UNSET_CONFIG = object()
_PARAMETRIZATION_MODULES = (
    "lunarops.classes.parametrization.reflector_position",
    "lunarops.classes.parametrization.station_range_bias",
)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ObservationAssembly:
    """Resolved model configuration and catalogs shared by serial and MPI."""

    model_configs: dict
    station_catalog: Mapping[str, StationRecord]
    reflector_catalog: Mapping[str, ReflectorRecord]


class _PathResolver(Protocol):
    def resolve_path(self, value: object) -> Path: ...


@dataclass(frozen=True, slots=True)
class _ObservationDependencies:
    run_context: RunContext
    ephemeris: Ephemeris
    earth_orientation_provider: EarthOrientationProvider
    frames: ReferenceFrameSystem
    cache_namespace: str

    @property
    def mpi_resources(self):
        return self.run_context.mpi_resources

    def resolve_path(self, value: object) -> Path:
        return self.run_context.resolve_path(value)

    def create_class(self, category: str, config=_UNSET_CONFIG, *, cache: bool = True):
        if config is _UNSET_CONFIG:
            return self.run_context.create_class(
                category,
                cache=cache,
                factory_context=self,
                cache_namespace=self.cache_namespace,
            )
        return self.run_context.create_class(
            category,
            config,
            cache=cache,
            factory_context=self,
            cache_namespace=self.cache_namespace,
        )


def _resolve_required_path(ctx: _PathResolver, value: object, *, name: str) -> Path:
    if value in (None, ""):
        raise ValueError(f"{name} must be a non-empty path.")
    return ctx.resolve_path(value)


def _required_observation_dependency(ctx: object, attribute: str):
    value = getattr(ctx, attribute, None)
    if value is None:
        raise RuntimeError(
            "This model requires an assembled observation context with "
            f"{attribute!r}. Use build_observation_processor() instead of "
            "RunContext.create_class() for models that depend on observation services."
        )
    return value


def _register_all() -> None:
    # Imports are local so that merely importing the registry does not load
    # CALCEPH or optional physical-model backends.
    from lunarops.classes.delays.base import ZeroGravitationalDelay, ZeroTroposphereDelay
    from lunarops.classes.delays.shapiro import Iers2010ShapiroDelay
    from lunarops.classes.delays.troposphere import Iers2010MendesPavlisTroposphere
    from lunarops.classes.displacement import (
        CompositeStationDisplacement,
        Iers2010OceanPoleTide,
        Iers2010OceanTidalLoading,
        Iers2010SolidEarthPoleTide,
        Iers2010SolidEarthTide,
        LunarSolidTide,
        OceanPoleTideGrid,
        OceanTidalLoadingCatalog,
        ZeroReflectorDisplacement,
        ZeroStationDisplacement,
    )
    from lunarops.classes.ephemerides import load_calceph_ephemeris
    from lunarops.classes.frames import TabulatedEarthOrientation, load_iers_eop
    from lunarops.classes.range_bias.models import (
        TableRangeBiasModel,
        ZeroRangeBiasModel,
    )
    from lunarops.classes.range_bias.table import (
        AdditiveRangeBiasTable,
        builtin_additive_range_bias_table,
        load_additive_range_bias_table,
    )

    def _class_schema(type_name: str, *fields_, validator=None) -> ConfigSchema:
        return ConfigSchema(tuple(fields_), type_name=type_name, validator=validator)

    def _validate_range_bias_table(config: dict, path_name: str):
        selected = {key for key in ("file", "biases") if config.get(key) is not None}
        if len(selected) != 1:
            raise ValueError(f"{path_name} requires exactly one of 'file' or 'biases'.")
        if selected == {"file"}:
            incompatible = sorted(set(config) & {"source", "biases"})
            if incompatible:
                raise ValueError(f"{path_name} file form cannot include {incompatible}.")
        elif "file" in config:
            raise ValueError(f"{path_name} inline form cannot include 'file'.")
        return config

    range_bias_row_schema = ConfigSchema(
        fields=(
            string("station", required=True, non_empty=True, allow_none=False),
            field("start", "time", required=True, allow_none=False),
            field("end", "time", required=True, allow_none=False),
            number("correctionTwoWayCm", required=True, allow_none=False),
            string("source", non_empty=True),
        )
    )

    def _calceph(cfg: dict, ctx):
        return load_calceph_ephemeris(
            _resolve_required_path(ctx, cfg["file"], name="ephemerides/calceph file"),
            lunar_relativistic_scale_convention=cfg["lunarRelativisticScaleConvention"],
            longitude_libration_correction_type=cfg.get(
                "longitudeLibrationCorrection",
                "none",
            ),
        )

    def _iers_c04(cfg: dict, ctx):
        payload = ctx.mpi_resources.get("earthRotation")
        if payload is not None:
            return TabulatedEarthOrientation.from_mpi_payload(payload)
        return load_iers_eop(
            _resolve_required_path(ctx, cfg["file"], name="earthRotation/iersC04 file"),
            duplicate_mjd_policy=cfg.get("duplicateMjdPolicy", "error"),
        )

    register_factory(
        "ephemerides",
        "calceph",
        _calceph,
        schema=_class_schema(
            "calceph",
            path("file", required=True, non_empty=True, allow_none=False),
            string(
                "lunarRelativisticScaleConvention",
                required=True,
                non_empty=True,
                choices=("tdbCompatibleLunarSurface", "alreadyScaled"),
                allow_none=False,
            ),
            string(
                "longitudeLibrationCorrection",
                default="none",
                non_empty=True,
                choices=("none", "inpop21a"),
                allow_none=False,
            ),
        ),
        global_scope=True,
    )
    register_factory(
        "earthRotation",
        "iersC04",
        _iers_c04,
        schema=_class_schema(
            "iersC04",
            path("file", required=True, non_empty=True, allow_none=False),
            string(
                "duplicateMjdPolicy",
                default="error",
                non_empty=True,
                choices=("error", "first", "last", "mean"),
                allow_none=False,
            ),
        ),
        global_scope=True,
    )

    def _zero_troposphere(cfg: dict, ctx):
        return ZeroTroposphereDelay()

    def _mendes_pavlis(cfg: dict, ctx):
        return Iers2010MendesPavlisTroposphere()

    register_factory(
        "troposphere",
        "none",
        _zero_troposphere,
        schema=_class_schema("none"),
        global_scope=True,
    )
    register_factory(
        "troposphere",
        "mendesPavlis",
        _mendes_pavlis,
        schema=_class_schema("mendesPavlis"),
        global_scope=True,
    )

    def _zero_relativity(cfg: dict, ctx):
        return ZeroGravitationalDelay()

    register_factory(
        "relativity",
        "none",
        _zero_relativity,
        schema=_class_schema("none"),
        global_scope=True,
    )

    def _iers_shapiro(cfg: dict, ctx):
        return Iers2010ShapiroDelay(ephemeris=_required_ephemeris(ctx))

    register_factory(
        "relativity",
        "iersShapiro",
        _iers_shapiro,
        schema=_class_schema("iersShapiro"),
        global_scope=True,
    )

    def _required_earth_orientation(ctx):
        return _required_observation_dependency(ctx, "earth_orientation_provider")

    def _required_ephemeris(ctx):
        return _required_observation_dependency(ctx, "ephemeris")

    def _required_frames(ctx):
        return _required_observation_dependency(ctx, "frames")

    def _station_sum(cfg: dict, ctx) -> CompositeStationDisplacement:
        components = tuple(
            ctx.create_class("stationDisplacement", component, cache=True) for component in cfg["components"]
        )
        return CompositeStationDisplacement(components)

    def _station_ocean_pole_tide(cfg: dict, ctx) -> Iers2010OceanPoleTide:
        coefficient_file = _resolve_required_path(
            ctx,
            cfg["coefficientFile"],
            name="stationDisplacement/iers2010OceanPoleTide coefficientFile",
        )
        return Iers2010OceanPoleTide(
            grid=OceanPoleTideGrid(coefficient_file),
            earth_orientation_provider=_required_earth_orientation(ctx),
        )

    def _station_ocean_tidal_loading(cfg: dict, ctx) -> Iers2010OceanTidalLoading:
        coefficient_file = _resolve_required_path(
            ctx,
            cfg["coefficientFile"],
            name="stationDisplacement/iers2010OceanTidalLoading coefficientFile",
        )
        catalog = OceanTidalLoadingCatalog(coefficient_file)
        expected_model = cfg.get("model")
        actual_model = catalog.info.tidal_model
        if expected_model is not None and (
            actual_model is None or str(expected_model).casefold() != actual_model.casefold()
        ):
            raise ValueError(
                "stationDisplacement/iers2010OceanTidalLoading model mismatch: "
                f"config requests {expected_model!r}, BLQ file declares {actual_model!r}."
            )
        return Iers2010OceanTidalLoading(catalog=catalog)

    def _zero_station_displacement(cfg: dict, ctx):
        return ZeroStationDisplacement()

    def _solid_earth_tide(cfg: dict, ctx):
        return Iers2010SolidEarthTide(frame_system=_required_frames(ctx))

    def _solid_earth_pole_tide(cfg: dict, ctx):
        return Iers2010SolidEarthPoleTide(earth_orientation_provider=_required_earth_orientation(ctx))

    register_factory(
        "stationDisplacement",
        "none",
        _zero_station_displacement,
        schema=_class_schema("none"),
        global_scope=True,
    )
    register_factory(
        "stationDisplacement",
        "sum",
        _station_sum,
        schema=_class_schema(
            "sum",
            class_list("components", "stationDisplacement", required=True, min_items=1),
        ),
        global_scope=True,
    )
    register_factory(
        "stationDisplacement",
        "iers2010SolidEarthTide",
        _solid_earth_tide,
        schema=_class_schema("iers2010SolidEarthTide"),
        global_scope=True,
    )
    register_factory(
        "stationDisplacement",
        "iers2010PoleTide",
        _solid_earth_pole_tide,
        schema=_class_schema("iers2010PoleTide"),
        global_scope=True,
    )
    register_factory(
        "stationDisplacement",
        "iers2010OceanPoleTide",
        _station_ocean_pole_tide,
        schema=_class_schema(
            "iers2010OceanPoleTide",
            path("coefficientFile", required=True, non_empty=True, allow_none=False),
        ),
        global_scope=True,
    )
    register_factory(
        "stationDisplacement",
        "iers2010OceanTidalLoading",
        _station_ocean_tidal_loading,
        schema=_class_schema(
            "iers2010OceanTidalLoading",
            path("coefficientFile", required=True, non_empty=True, allow_none=False),
            string("model", non_empty=True),
        ),
        global_scope=True,
    )

    def _zero_reflector_displacement(cfg: dict, ctx):
        return ZeroReflectorDisplacement()

    def _lunar_solid_tide(cfg: dict, ctx):
        return LunarSolidTide(
            ephemeris=_required_ephemeris(ctx),
            h2=float(cfg["h2"]),
            l2=float(cfg["l2"]),
            moon_radius_m=float(cfg["moonRadiusM"]),
        )

    register_factory(
        "reflectorDisplacement",
        "none",
        _zero_reflector_displacement,
        schema=_class_schema("none"),
        global_scope=True,
    )
    register_factory(
        "reflectorDisplacement",
        "lunarSolidTide",
        _lunar_solid_tide,
        schema=_class_schema(
            "lunarSolidTide",
            number("h2", default=0.0423, allow_none=False),
            number("l2", default=0.0107, allow_none=False),
            number("moonRadiusM", default=1_737_400.0, minimum=0.0, minimum_exclusive=True, allow_none=False),
        ),
        global_scope=True,
    )

    def _range_bias_table(cfg: dict, ctx) -> TableRangeBiasModel:
        if "file" in cfg:
            table = load_additive_range_bias_table(
                _resolve_required_path(ctx, cfg["file"], name="rangeBias/table file")
            )
        else:
            table = AdditiveRangeBiasTable.from_mapping(cfg)
        return TableRangeBiasModel(table)

    def _zero_range_bias(cfg: dict, ctx):
        return ZeroRangeBiasModel()

    def _builtin_range_bias(cfg: dict, ctx):
        return TableRangeBiasModel(builtin_additive_range_bias_table("inpop21a"))

    register_factory(
        "rangeBias",
        "none",
        _zero_range_bias,
        schema=_class_schema("none"),
        global_scope=True,
    )
    register_factory(
        "rangeBias",
        "inpop21a",
        _builtin_range_bias,
        schema=_class_schema("inpop21a"),
        global_scope=True,
    )
    register_factory(
        "rangeBias",
        "table",
        _range_bias_table,
        schema=_class_schema(
            "table",
            path("file", non_empty=True),
            string("source", non_empty=True),
            sequence("biases", item_kind="mapping", item_nested=range_bias_row_schema, min_items=1),
            validator=_validate_range_bias_table,
        ),
        global_scope=True,
    )

_REGISTERED = False
_REGISTRATION_LOCK = RLock()


def ensure_registered() -> None:
    global _REGISTERED
    with _REGISTRATION_LOCK:
        if _REGISTERED:
            return
        missing = object()
        previous_modules = {name: sys.modules.get(name, missing) for name in _PARAMETRIZATION_MODULES}
        try:
            # Parametrization modules use registry decorators at import time;
            # keep those declarations in the same transaction as the built-in
            # observation models so a failed batch leaves no half-registry.
            with registration_transaction():
                for module_name in _PARAMETRIZATION_MODULES:
                    importlib.import_module(module_name)
                _register_all()
        except Exception:
            for module_name, previous in previous_modules.items():
                if previous is missing:
                    sys.modules.pop(module_name, None)
            raise
        _REGISTERED = True


def resolve_observation_assembly(
    context,
    program_config: dict,
    *,
    station_catalog=None,
    reflector_catalog=None,
) -> ObservationAssembly:
    """Resolve configs, paths, and catalogs once for every execution backend."""
    ensure_registered()
    from lunarops.fileio.catalogs import load_reflector_catalog, load_station_catalog
    from lunarops.config.registry import validate_class_config

    merged = {
        category: validate_class_config(category, value, path=f"observation.{category}")
        for category in _MODEL_CATEGORIES
        if (value := context.class_config(category, program_config)) is not None
    }

    def catalog_source(name: str):
        value = program_config.get(name, context.global_class_configs.get(name))
        if isinstance(value, str) and value not in ("builtin", ""):
            return context.resolve_path(value)
        return value

    stations = load_station_catalog(catalog_source("stationCatalog")) if station_catalog is None else station_catalog
    reflectors = (
        load_reflector_catalog(catalog_source("reflectorCatalog")) if reflector_catalog is None else reflector_catalog
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

        ephemerides:           {type: calceph, file: ..., lunarRelativisticScaleConvention: alreadyScaled,
                                longitudeLibrationCorrection: none}
        earthRotation:         {type: iersC04, file: ..., duplicateMjdPolicy: error|first|last|mean}
        troposphere:           mendesPavlis
        relativity:            iersShapiro
        stationDisplacement:   {type: sum, components: [...]} | none
        reflectorDisplacement: lunarSolidTide | none
        rangeBias:             none | inpop21a | {type: table, file: ...} | {type: table, biases: [...]}

    Observation uncertainty is read directly from each normal-point record.
    """
    ensure_registered()
    from lunarops.classes.frames import EarthOrientationProvider, ReferenceFrameSystem
    from lunarops.classes.observation import (
        LightTimeSolver,
        LlrObservationModel,
        LlrObservationProcessor,
        ObservationCatalogState,
        ObservationResolver,
    )

    assembly = resolve_observation_assembly(
        context,
        program_config,
        station_catalog=station_catalog,
        reflector_catalog=reflector_catalog,
    )
    model_configs = assembly.model_configs

    def cfg(category: str):
        try:
            return model_configs[category]
        except KeyError as exc:
            raise KeyError(
                f"Observation processing requires explicit {category!r} in the program or globals config."
            ) from exc

    eph_cfg = cfg("ephemerides")
    eop_cfg = cfg("earthRotation")

    from lunarops.classes.ephemerides import Ephemeris

    ephemeris = context.create_class("ephemerides", eph_cfg, cache=True)
    if not isinstance(ephemeris, Ephemeris):
        raise TypeError(
            "ephemerides factory must return an Ephemeris implementation, "
            f"got {type(ephemeris).__name__}."
        )
    earth_orientation_provider = context.create_class("earthRotation", eop_cfg, cache=True)
    if not isinstance(earth_orientation_provider, EarthOrientationProvider):
        raise TypeError(
            "earthRotation factory must return an EarthOrientationProvider implementation, "
            f"got {type(earth_orientation_provider).__name__}."
        )
    frames = ReferenceFrameSystem(
        ephemeris=ephemeris,
        earth_orientation_provider=earth_orientation_provider,
    )
    factory_context = _ObservationDependencies(
        run_context=context,
        ephemeris=ephemeris,
        earth_orientation_provider=earth_orientation_provider,
        frames=frames,
        cache_namespace=f"observation:{context.next_observation_spec_id()}",
    )
    station_displacement = factory_context.create_class(
        "stationDisplacement",
        cfg("stationDisplacement"),
        cache=True,
    )
    reflector_displacement = factory_context.create_class(
        "reflectorDisplacement",
        cfg("reflectorDisplacement"),
        cache=True,
    )
    solver = LightTimeSolver(
        frame_system=frames,
        gravitational_delay_model=factory_context.create_class("relativity", cfg("relativity"), cache=False),
        troposphere_delay_model=factory_context.create_class("troposphere", cfg("troposphere"), cache=True),
        station_displacement_model=station_displacement,
        reflector_displacement_model=reflector_displacement,
    )
    model_state = ObservationCatalogState(
        assembly.station_catalog,
        assembly.reflector_catalog,
    )
    resolver = ObservationResolver(model_state)
    range_bias_cfg = cfg("rangeBias")
    range_bias = factory_context.create_class(
        "rangeBias",
        range_bias_cfg,
        cache=True,
    )
    observation_model = LlrObservationModel(
        frame_system=frames,
        light_time_solver=solver,
        range_bias_model=range_bias,
    )
    processor = LlrObservationProcessor(
        resolver,
        observation_model,
    )
    return processor


__all__ = [
    "ObservationAssembly",
    "build_observation_processor",
    "ensure_registered",
    "resolve_observation_assembly",
]
