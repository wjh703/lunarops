"""Lookup table of the built-in model classes selectable from configuration.

``BUILTINS`` maps a ``(category, type)`` pair, as spelled in the configuration (for example
``("stationDisplacement", "iers2010SolidEarthTide")``), to the import path ``"module:ClassName"`` of the
class that implements it. The configuration registry imports the class lazily on first use.
"""

BUILTINS = {
    ("ephemerides", "chebyshev"): "lunarops.classes.ephemerides.chebyshev:ChebyshevEphemeris",
    ("earthRotation", "file"): "lunarops.classes.frames.earth_orientation:TabulatedEarthOrientation",
    ("troposphere", "mendesPavlis"): "lunarops.classes.delays.troposphere:Iers2010MendesPavlisTroposphere",
    ("relativity", "iersShapiro"): "lunarops.classes.delays.shapiro:Iers2010ShapiroDelay",
    (
        "stationDisplacement",
        "iers2010SolidEarthTide",
    ): "lunarops.classes.displacement.solid_earth_tide:Iers2010SolidEarthTide",
    ("stationDisplacement", "iers2010PoleTide"): "lunarops.classes.displacement.pole_tide:Iers2010SolidEarthPoleTide",
    (
        "stationDisplacement",
        "iers2010OceanPoleTide",
    ): "lunarops.classes.displacement.ocean_pole_tide:Iers2010OceanPoleTide",
    (
        "stationDisplacement",
        "iers2010OceanTidalLoading",
    ): "lunarops.classes.displacement.ocean_tidal_loading:Iers2010OceanTidalLoading",
    ("reflectorDisplacement", "lunarSolidTide"): "lunarops.classes.displacement.lunar_solid_tide:LunarSolidTide",
    ("rangeBias", "none"): "lunarops.classes.range_bias.table:RangeBiasTable",
    ("rangeBias", "inpop21a"): "lunarops.classes.range_bias.table:RangeBiasTable",
    ("rangeBias", "table"): "lunarops.classes.range_bias.table:RangeBiasTable",
    (
        "parametrization",
        "reflectorPosition",
    ): "lunarops.classes.parametrization.reflector_position:ReflectorPositionParametrization",
    (
        "parametrization",
        "stationPosition",
    ): "lunarops.classes.parametrization.station_position:StationPositionParametrization",
    (
        "parametrization",
        "stationRangeBias",
    ): "lunarops.classes.parametrization.station_range_bias:StationRangeBiasParametrization",
    (
        "parametrization",
        "empiricalLibration",
    ): "lunarops.classes.parametrization.empirical_libration:EmpiricalLibrationParametrization",
}
