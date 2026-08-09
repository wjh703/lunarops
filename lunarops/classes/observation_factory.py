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

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Protocol

from lunarops.config.registry import (
    _registration_transaction,
    normalize_class_config,
    register_factory,
)

if TYPE_CHECKING:
    from lunarops.classes.ephemerides import Ephemeris
    from lunarops.classes.frames import EarthOrientationProvider, ReferenceFrameSystem
    from lunarops.config.context import RunContext
    from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord


_REMOVED_UNCERTAINTY_CONFIG_KEYS = frozenset({"uncertainty", "uncertaintyModel"})
_MODEL_CATEGORIES = (
    "ephemerides",
    "earthRotation",
    "troposphere",
    "relativity",
    "stationDisplacement",
    "reflectorDisplacement",
    "rangeBias",
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

    def create_class(self, category: str, config=None, *, cache: bool = True):
        return self.run_context.create_class(
            category,
            config,
            cache=cache,
            factory_context=self,
            cache_namespace=self.cache_namespace,
        )


def validate_observation_config(
    program_config: dict,
    global_config: dict | None = None,
) -> None:
    """Reject uncertainty selectors now owned by normal-point records."""
    for scope, config in (
        ("program", program_config),
        ("globals", global_config or {}),
    ):
        removed = sorted(_REMOVED_UNCERTAINTY_CONFIG_KEYS.intersection(config))
        if removed:
            raise ValueError(
                f"{scope} contains removed uncertainty configuration key(s) "
                f"{removed}; every normal-point record must provide "
                "uncertainty_two_way_s."
            )


def _resolve_optional_path(ctx: _PathResolver, value: object):
    if value in (None, ""):
        return None
    return ctx.resolve_path(value)


def _resolve_required_path(ctx: _PathResolver, value: object, *, name: str) -> Path:
    path = _resolve_optional_path(ctx, value)
    if path is None:
        raise ValueError(f"{name} must be a non-empty path.")
    return path


def _reject_unknown_keys(config: Mapping[str, object], allowed: set[str], *, name: str) -> None:
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"{name} has unknown key(s) {sorted(unknown)}.")


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

    def _calceph(cfg: dict, ctx):
        _reject_unknown_keys(
            cfg,
            {
                "type",
                "file",
                "longitudeLibrationCorrection",
                "lunarRelativisticScaleConvention",
            },
            name="ephemerides/calceph",
        )
        if "file" not in cfg:
            raise ValueError("ephemerides/calceph requires 'file'.")
        if "lunarRelativisticScaleConvention" not in cfg:
            raise ValueError(
                "ephemerides/calceph requires explicit "
                "'lunarRelativisticScaleConvention' "
                "(tdbCompatibleLunarSurface or alreadyScaled)."
            )
        return load_calceph_ephemeris(
            _resolve_required_path(ctx, cfg["file"], name="ephemerides/calceph file"),
            lunar_relativistic_scale_convention=cfg["lunarRelativisticScaleConvention"],
            longitude_libration_correction_type=cfg.get(
                "longitudeLibrationCorrection",
                "none",
            ),
        )

    def _iers_c04(cfg: dict, ctx):
        _reject_unknown_keys(
            cfg,
            {"type", "file", "duplicateMjdPolicy"},
            name="earthRotation/iersC04",
        )
        payload = ctx.mpi_resources.get("earthRotation")
        if payload is not None:
            return TabulatedEarthOrientation.from_mpi_payload(payload)
        if "file" not in cfg:
            raise ValueError("earthRotation/iersC04 requires 'file'.")
        return load_iers_eop(
            _resolve_required_path(ctx, cfg["file"], name="earthRotation/iersC04 file"),
            duplicate_mjd_policy=cfg.get("duplicateMjdPolicy", "error"),
        )

    register_factory("ephemerides", "calceph", _calceph)
    register_factory("earthRotation", "iersc04", _iers_c04)

    def _zero_troposphere(cfg: dict, ctx):
        _reject_unknown_keys(cfg, {"type"}, name="troposphere/none")
        return ZeroTroposphereDelay()

    def _mendes_pavlis(cfg: dict, ctx):
        _reject_unknown_keys(cfg, {"type"}, name="troposphere/mendesPavlis")
        return Iers2010MendesPavlisTroposphere()

    register_factory("troposphere", "none", _zero_troposphere)
    register_factory("troposphere", "mendespavlis", _mendes_pavlis)

    def _zero_relativity(cfg: dict, ctx):
        _reject_unknown_keys(cfg, {"type"}, name="relativity/none")
        return ZeroGravitationalDelay()

    register_factory("relativity", "none", _zero_relativity)

    def _iers_shapiro(cfg: dict, ctx):
        _reject_unknown_keys(cfg, {"type"}, name="relativity/iersShapiro")
        return Iers2010ShapiroDelay(ephemeris=_required_ephemeris(ctx))

    register_factory(
        "relativity",
        "iersshapiro",
        _iers_shapiro,
    )

    def _required_earth_orientation(ctx):
        return _required_observation_dependency(ctx, "earth_orientation_provider")

    def _required_ephemeris(ctx):
        return _required_observation_dependency(ctx, "ephemeris")

    def _required_frames(ctx):
        return _required_observation_dependency(ctx, "frames")

    def _station_sum(cfg: dict, ctx) -> CompositeStationDisplacement:
        _reject_unknown_keys(cfg, {"type", "components"}, name="stationDisplacement/sum")
        if "components" not in cfg:
            raise ValueError("stationDisplacement/sum requires at least one component in 'components'.")
        components_cfg = cfg.get("components")
        if not isinstance(components_cfg, list):
            raise TypeError("stationDisplacement/sum components list must be a list.")
        if not components_cfg:
            raise ValueError("stationDisplacement/sum requires at least one component.")
        components = tuple(
            ctx.create_class("stationDisplacement", component, cache=True) for component in components_cfg
        )
        return CompositeStationDisplacement(components)

    def _station_ocean_pole_tide(cfg: dict, ctx) -> Iers2010OceanPoleTide:
        _reject_unknown_keys(
            cfg,
            {"type", "coefficientFile"},
            name="stationDisplacement/iers2010OceanPoleTide",
        )
        coefficient_file = _resolve_optional_path(ctx, cfg.get("coefficientFile"))
        if coefficient_file is None:
            raise ValueError("stationDisplacement/iers2010OceanPoleTide requires 'coefficientFile'.")
        return Iers2010OceanPoleTide(
            grid=OceanPoleTideGrid(coefficient_file),
            earth_orientation_provider=_required_earth_orientation(ctx),
        )

    def _station_ocean_tidal_loading(cfg: dict, ctx) -> Iers2010OceanTidalLoading:
        _reject_unknown_keys(
            cfg,
            {"type", "coefficientFile", "model"},
            name="stationDisplacement/iers2010OceanTidalLoading",
        )
        coefficient_file = _resolve_optional_path(ctx, cfg.get("coefficientFile"))
        if coefficient_file is None:
            raise ValueError("stationDisplacement/iers2010OceanTidalLoading requires 'coefficientFile'.")
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
        _reject_unknown_keys(cfg, {"type"}, name="stationDisplacement/none")
        return ZeroStationDisplacement()

    def _solid_earth_tide(cfg: dict, ctx):
        _reject_unknown_keys(cfg, {"type"}, name="stationDisplacement/iers2010SolidEarthTide")
        return Iers2010SolidEarthTide(frame_system=_required_frames(ctx))

    def _solid_earth_pole_tide(cfg: dict, ctx):
        _reject_unknown_keys(cfg, {"type"}, name="stationDisplacement/iers2010PoleTide")
        return Iers2010SolidEarthPoleTide(earth_orientation_provider=_required_earth_orientation(ctx))

    register_factory("stationDisplacement", "none", _zero_station_displacement)
    register_factory("stationDisplacement", "sum", _station_sum)
    register_factory("stationDisplacement", "iers2010solidearthtide", _solid_earth_tide)
    register_factory("stationDisplacement", "iers2010poletide", _solid_earth_pole_tide)
    register_factory(
        "stationDisplacement",
        "iers2010oceanpoletide",
        _station_ocean_pole_tide,
    )
    register_factory(
        "stationDisplacement",
        "iers2010oceantidalloading",
        _station_ocean_tidal_loading,
    )

    def _zero_reflector_displacement(cfg: dict, ctx):
        _reject_unknown_keys(cfg, {"type"}, name="reflectorDisplacement/none")
        return ZeroReflectorDisplacement()

    def _lunar_solid_tide(cfg: dict, ctx):
        _reject_unknown_keys(
            cfg,
            {"type", "h2", "l2", "moonRadiusM"},
            name="reflectorDisplacement/lunarSolidTide",
        )
        return LunarSolidTide(
            ephemeris=_required_ephemeris(ctx),
            h2=float(cfg.get("h2", 0.0423)),
            l2=float(cfg.get("l2", 0.0107)),
            moon_radius_m=float(cfg.get("moonRadiusM", 1_737_400.0)),
        )

    register_factory("reflectorDisplacement", "none", _zero_reflector_displacement)
    register_factory(
        "reflectorDisplacement",
        "lunarsolidtide",
        _lunar_solid_tide,
    )

    def _range_bias_table(cfg: dict, ctx) -> TableRangeBiasModel:
        has_file = "file" in cfg
        has_biases = "biases" in cfg
        if has_file == has_biases:
            raise ValueError("rangeBias/table requires exactly one of 'file' or 'biases'.")
        if has_file:
            unknown = set(cfg) - {"type", "file"}
            if unknown:
                raise ValueError(f"rangeBias/table file config has unknown key(s) {sorted(unknown)}.")
            table = load_additive_range_bias_table(
                _resolve_required_path(ctx, cfg["file"], name="rangeBias/table file")
            )
        else:
            unknown = set(cfg) - {"type", "source", "biases"}
            if unknown:
                raise ValueError(f"rangeBias/table inline config has unknown key(s) {sorted(unknown)}.")
            table = AdditiveRangeBiasTable.from_mapping(cfg)
        return TableRangeBiasModel(table)

    def _zero_range_bias(cfg: dict, ctx):
        _reject_unknown_keys(cfg, {"type"}, name="rangeBias/none")
        return ZeroRangeBiasModel()

    def _builtin_range_bias(cfg: dict, ctx):
        _reject_unknown_keys(cfg, {"type"}, name="rangeBias/inpop21a")
        return TableRangeBiasModel(builtin_additive_range_bias_table("inpop21a"))

    register_factory("rangeBias", "none", _zero_range_bias)
    register_factory(
        "rangeBias",
        "inpop21a",
        _builtin_range_bias,
    )
    register_factory("rangeBias", "table", _range_bias_table)

    # Parametrizations register themselves on import.
    import lunarops.classes.parametrization.reflector_position
    import lunarops.classes.parametrization.station_range_bias  # noqa: F401


_REGISTERED = False
_REGISTRATION_LOCK = RLock()


def ensure_registered() -> None:
    global _REGISTERED
    with _REGISTRATION_LOCK:
        if _REGISTERED:
            return
        with _registration_transaction():
            _register_all()
        _REGISTERED = True


def resolve_observation_assembly(
    context,
    program_config: dict,
    *,
    station_catalog=None,
    reflector_catalog=None,
) -> ObservationAssembly:
    """Resolve configs, paths, and catalogs once for every execution backend."""
    from lunarops.fileio.catalogs import load_reflector_catalog, load_station_catalog

    validate_observation_config(program_config, context.global_class_configs)
    merged = {
        category: value
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
        return normalize_class_config(context.class_config(category, model_configs))

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
        normalize_class_config(context.class_config("stationDisplacement", model_configs)),
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
    range_bias_cfg = context.class_config("rangeBias", model_configs)
    if range_bias_cfg is None:
        raise KeyError("Observation processing requires explicit 'rangeBias' in the program or globals config.")
    range_bias = factory_context.create_class(
        "rangeBias",
        normalize_class_config(range_bias_cfg),
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
    "validate_observation_config",
]
