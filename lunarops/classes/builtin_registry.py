"""Built-in configurable model declarations, loaded on demand."""
from __future__ import annotations

import importlib
from functools import partial, lru_cache
from lunarops.config.registry import register_factory
from lunarops.config.schema import ConfigSchema, boolean, number, path, sequence, string

_PARAMETRIZATION_MODULES = (
    "lunarops.classes.parametrization.reflector_position",
    "lunarops.classes.parametrization.station_range_bias",
)


@lru_cache(maxsize=16)
def _coefficient_file(factory, path):
    return factory(path)


def _register_all() -> None:
    # Imports are local so that merely importing the registry does not load
    # CALCEPH or optional physical-model backends.
    from lunarops.classes.delays.base import ZeroGravitationalDelay, ZeroTroposphereDelay
    from lunarops.classes.delays.shapiro import Iers2010ShapiroDelay
    from lunarops.classes.delays.troposphere import Iers2010MendesPavlisTroposphere
    from lunarops.classes.displacement import (
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
    from lunarops.classes.displacement.constants import LUNAR_H2, LUNAR_L2, MOON_REFERENCE_RADIUS_M
    from lunarops.classes.ephemerides import CalcephEphemeris
    from lunarops.classes.frames import TabulatedEarthOrientation
    from lunarops.classes.range_bias.models import (
        TableRangeBiasModel,
        ZeroRangeBiasModel,
    )
    from lunarops.classes.range_bias.table import (
        RANGE_BIAS_ROW_SCHEMA,
        AdditiveRangeBiasTable,
        builtin_additive_range_bias_table,
        load_additive_range_bias_table,
    )
    from lunarops.fileio.earth_orientation import load_earth_orientation_parameter

    register_shared = partial(register_factory, scope="shared")

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

    def _calceph(cfg: dict, ctx):
        return CalcephEphemeris(
            ctx.resolve_path(cfg["directory"]),
            lunar_relativistic_scale_convention=cfg["lunarRelativisticScaleConvention"],
            longitude_libration_correction=cfg.get(
                "longitudeLibrationCorrection",
                "none",
            ),
        )

    def _earth_orientation_file(cfg: dict, ctx):
        payload = ctx.mpi_resources.get("earthRotation")
        if payload is not None:
            return TabulatedEarthOrientation.from_mpi_payload(payload)
        return load_earth_orientation_parameter(
            ctx.resolve_path(cfg["file"]),
            allow_predicted=cfg["allowPredicted"],
        )

    register_shared(
        "ephemerides",
        "calceph",
        _calceph,
        schema=_class_schema(
            "calceph",
            path("directory", required=True, non_empty=True, allow_none=False),
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
    )
    register_shared(
        "earthRotation",
        "file",
        _earth_orientation_file,
        schema=_class_schema(
            "file",
            path("file", required=True, non_empty=True, allow_none=False),
            boolean("allowPredicted", default=True, allow_none=False),
        ),
    )

    def _zero_troposphere(cfg: dict, ctx):
        return ZeroTroposphereDelay()

    def _mendes_pavlis(cfg: dict, ctx):
        return Iers2010MendesPavlisTroposphere()

    register_shared(
        "troposphere",
        "none",
        _zero_troposphere,
        schema=_class_schema("none"),
    )
    register_shared(
        "troposphere",
        "mendesPavlis",
        _mendes_pavlis,
        schema=_class_schema("mendesPavlis"),
    )

    def _zero_relativity(cfg: dict, ctx):
        return ZeroGravitationalDelay()

    register_shared(
        "relativity",
        "none",
        _zero_relativity,
        schema=_class_schema("none"),
    )

    def _iers_shapiro(cfg: dict, ctx):
        return Iers2010ShapiroDelay(ephemeris=ctx.ephemeris)

    register_shared(
        "relativity",
        "iersShapiro",
        _iers_shapiro,
        schema=_class_schema("iersShapiro"),
    )

    def _station_ocean_pole_tide(cfg: dict, ctx) -> Iers2010OceanPoleTide:
        return Iers2010OceanPoleTide(
            grid=_coefficient_file(OceanPoleTideGrid, ctx.resolve_path(cfg["coefficientFile"])),
            earth_orientation_provider=ctx.earth_orientation_provider,
        )

    def _station_ocean_tidal_loading(cfg: dict, ctx) -> Iers2010OceanTidalLoading:
        catalog = _coefficient_file(OceanTidalLoadingCatalog, ctx.resolve_path(cfg["coefficientFile"]))
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
        return Iers2010SolidEarthTide(frame_system=ctx.frames)

    def _solid_earth_pole_tide(cfg: dict, ctx):
        return Iers2010SolidEarthPoleTide(earth_orientation_provider=ctx.earth_orientation_provider)

    register_shared(
        "stationDisplacement",
        "none",
        _zero_station_displacement,
        schema=_class_schema("none"),
    )
    register_shared(
        "stationDisplacement",
        "iers2010SolidEarthTide",
        _solid_earth_tide,
        schema=_class_schema("iers2010SolidEarthTide"),
    )
    register_shared(
        "stationDisplacement",
        "iers2010PoleTide",
        _solid_earth_pole_tide,
        schema=_class_schema("iers2010PoleTide"),
    )
    register_shared(
        "stationDisplacement",
        "iers2010OceanPoleTide",
        _station_ocean_pole_tide,
        schema=_class_schema(
            "iers2010OceanPoleTide",
            path("coefficientFile", required=True, non_empty=True, allow_none=False),
        ),
    )
    register_shared(
        "stationDisplacement",
        "iers2010OceanTidalLoading",
        _station_ocean_tidal_loading,
        schema=_class_schema(
            "iers2010OceanTidalLoading",
            path("coefficientFile", required=True, non_empty=True, allow_none=False),
            string("model", non_empty=True),
        ),
    )

    def _zero_reflector_displacement(cfg: dict, ctx):
        return ZeroReflectorDisplacement()

    def _lunar_solid_tide(cfg: dict, ctx):
        return LunarSolidTide(
            ephemeris=ctx.ephemeris,
            lunar_scale=ctx.frames.lunar_scale,
            h2=float(cfg["h2"]),
            l2=float(cfg["l2"]),
            moon_radius_m=float(cfg["moonRadiusM"]),
        )

    register_shared(
        "reflectorDisplacement",
        "none",
        _zero_reflector_displacement,
        schema=_class_schema("none"),
    )
    register_shared(
        "reflectorDisplacement",
        "lunarSolidTide",
        _lunar_solid_tide,
        schema=_class_schema(
            "lunarSolidTide",
            number("h2", default=LUNAR_H2, allow_none=False),
            number("l2", default=LUNAR_L2, allow_none=False),
            number("moonRadiusM", default=MOON_REFERENCE_RADIUS_M, minimum=0.0, minimum_exclusive=True, allow_none=False),
        ),
    )

    def _range_bias_table(cfg: dict, ctx) -> TableRangeBiasModel:
        if "file" in cfg:
            table = load_additive_range_bias_table(
                ctx.resolve_path(cfg["file"])
            )
        else:
            table = AdditiveRangeBiasTable.from_mapping(cfg)
        return TableRangeBiasModel(table)

    def _zero_range_bias(cfg: dict, ctx):
        return ZeroRangeBiasModel()

    def _builtin_range_bias(cfg: dict, ctx):
        return TableRangeBiasModel(builtin_additive_range_bias_table("inpop21a"))

    register_shared(
        "rangeBias",
        "none",
        _zero_range_bias,
        schema=_class_schema("none"),
    )
    register_shared(
        "rangeBias",
        "inpop21a",
        _builtin_range_bias,
        schema=_class_schema("inpop21a"),
    )
    register_shared(
        "rangeBias",
        "table",
        _range_bias_table,
        schema=_class_schema(
            "table",
            path("file", non_empty=True),
            string("source", non_empty=True),
            sequence("biases", item_kind="mapping", item_nested=RANGE_BIAS_ROW_SCHEMA, min_items=1),
            validator=_validate_range_bias_table,
        ),
    )


_REGISTERED = False


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    for module_name in _PARAMETRIZATION_MODULES:
        importlib.import_module(module_name)
    _register_all()
    _REGISTERED = True


