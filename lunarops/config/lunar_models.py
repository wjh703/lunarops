"""Pure configuration contracts shared by lunar models and artifact validation."""

from lunarops.classes.dynamics.deformation import LunarDegree2GravityParameters
from lunarops.classes.dynamics.forces import EarthTideParameters

from .schema import ConfigSchema, boolean, class_config, integer, mapping, number, path, sequence, string


def _validate_degree2(config, path):
    if not config["gamma"] < config["beta"]:
        raise ValueError(f"{path}.gamma must be less than {path}.beta.")
    return config


_DEGREE2_DEFAULTS = LunarDegree2GravityParameters()
LUNAR_DEGREE2_SCHEMA = ConfigSchema(
    (
        number(
            "j2Undistorted",
            default=_DEGREE2_DEFAULTS.j2_undistorted,
            gt=0,
        ),
        number(
            "beta",
            default=_DEGREE2_DEFAULTS.beta,
            gt=0,
            lt=1,
        ),
        number(
            "gamma",
            default=_DEGREE2_DEFAULTS.gamma,
            gt=0,
            lt=1,
        ),
        number(
            "loveK2",
            default=_DEGREE2_DEFAULTS.love_k2,
            ge=0,
        ),
        number(
            "tidalDelayDays",
            default=_DEGREE2_DEFAULTS.tidal_delay_days,
            ge=0,
        ),
        number(
            "radiusM",
            default=_DEGREE2_DEFAULTS.radius_m,
            gt=0,
        ),
        number(
            "meanMotionRadS",
            default=_DEGREE2_DEFAULTS.mean_motion_rad_s,
            gt=0,
        ),
        boolean(
            "includeTidalDeformation",
            default=_DEGREE2_DEFAULTS.include_tidal_deformation,
        ),
        boolean(
            "includeRotationalDeformation",
            default=_DEGREE2_DEFAULTS.include_rotational_deformation,
        ),
        number(
            "c21Static",
            default=_DEGREE2_DEFAULTS.c21_static,
        ),
        number(
            "s21Static",
            default=_DEGREE2_DEFAULTS.s21_static,
        ),
        number(
            "s22Static",
            default=_DEGREE2_DEFAULTS.s22_static,
        ),
    ),
    description="Delayed degree-2 gravity using prescribed lunar attitude; separate from integrated-rotation deformation.",
    validator=_validate_degree2,
)


LUNAR_CORE_SCHEMA = ConfigSchema(
    (
        number(
            "polarMomentFraction",
            required=True,
            gt=0,
            lt=1,
        ),
        number(
            "flattening",
            required=True,
            ge=0,
            lt=1,
        ),
        number(
            "viscosityPerS",
            required=True,
            ge=0,
        ),
    )
)


_GRAVITY_FIELD_SCHEMA = ConfigSchema(
    (
        path(
            "path",
            required=True,
            non_empty=True,
        ),
        integer(
            "maxDegree",
            ge=0,
        ),
        number(
            "radiusM",
            gt=0,
        ),
        string(
            "name",
            non_empty=True,
        ),
        boolean(
            "allowMissingCoefficients",
            default=False,
        ),
    )
)

_GRAVITY_FIELDS_SCHEMA = ConfigSchema(
    (
        mapping(
            "earth",
            nested=_GRAVITY_FIELD_SCHEMA,
        ),
        mapping(
            "moon",
            nested=_GRAVITY_FIELD_SCHEMA,
        ),
    )
)

_DEFORMATION_SCHEMA = ConfigSchema(
    (
        number(
            "loveK2",
            required=True,
            ge=0,
        ),
        number(
            "delayDays",
            required=True,
            gt=0,
        ),
        number(
            "meanMotionRadS",
            required=True,
            gt=0,
        ),
        number(
            "seedDurationSeconds",
            required=True,
            gt=0,
        ),
        string(
            "seedModel",
            required=True,
            choices=("rigidBackward",),
        ),
        boolean(
            "includeTidal",
            default=True,
        ),
        boolean(
            "includeRotational",
            default=True,
        ),
        number(
            "c21Static",
            default=0.0,
        ),
        number(
            "s21Static",
            default=0.0,
        ),
        number(
            "s22Static",
            default=0.0,
        ),
    )
)

_ROTATION_SCHEMA = ConfigSchema(
    (
        sequence(
            "inertiaMoments",
            required=True,
            item_kind="number",
            min_items=3,
            max_items=3,
        ),
        number(
            "radiusM",
            required=True,
            gt=0,
        ),
        sequence(
            "initialQuaternion",
            required=True,
            item_kind="number",
            min_items=4,
            max_items=4,
        ),
        sequence(
            "initialAngularVelocityRadS",
            required=True,
            item_kind="number",
            min_items=3,
            max_items=3,
        ),
        sequence(
            "initialCoreAngularVelocityRadS",
            default=None,
            item_kind="number",
            min_items=3,
            max_items=3,
            allow_none=True,
        ),
        sequence(
            "sources",
            required=True,
            item_kind="string",
        ),
        string(
            "gravityTorque",
            required=True,
            choices=("quadrupole", "figure"),
        ),
        boolean(
            "earthFigureTorque",
            default=False,
        ),
        boolean(
            "geodeticPrecession",
            default=False,
        ),
        mapping("core", nested=LUNAR_CORE_SCHEMA, default=None, allow_none=True),
        mapping("deformation", nested=_DEFORMATION_SCHEMA, default=None, allow_none=True),
    )
)

_TIDE_DEFAULTS = EarthTideParameters()
_EARTH_TIDE_SCHEMA = ConfigSchema(
    (
        sequence(
            "tideRaisers",
            default=["MOON", "SUN"],
            item_kind="string",
        ),
        number(
            "k20",
            default=_TIDE_DEFAULTS.k20,
        ),
        number(
            "k21",
            default=_TIDE_DEFAULTS.k21,
        ),
        number(
            "k22",
            default=_TIDE_DEFAULTS.k22,
        ),
        sequence(
            "tauOrbDays",
            default=_TIDE_DEFAULTS.tau_orb_days,
            item_kind="number",
            min_items=3,
            max_items=3,
        ),
        sequence(
            "tauRotDays",
            default=_TIDE_DEFAULTS.tau_rot_days,
            item_kind="number",
            min_items=3,
            max_items=3,
        ),
        number(
            "earthRotationRateRadS",
            default=_TIDE_DEFAULTS.earth_rotation_rate_rad_s,
        ),
        number(
            "earthRadiusM",
            default=6_378_136.6,
            gt=0,
        ),
    )
)
_EARTH_J2_SCHEMA = ConfigSchema(
    (
        number(
            "linearPerYear",
            default=0.0,
        ),
        number(
            "quadraticPerYear2",
            default=0.0,
        ),
        number(
            "referenceJdTdb",
            default=2_451_545.0,
        ),
    )
)
_SOLAR_RELATIVISTIC_SCHEMA = ConfigSchema(
    (
        boolean(
            "includeLenseThirring",
            default=True,
        ),
        boolean(
            "includeRadiationPressure",
            default=False,
        ),
        number(
            "moonEpsilon",
            default=0.0,
        ),
    )
)
_RESOLVED_FORCE_SCHEMA = ConfigSchema(
    (
        sequence(
            "active",
            item_kind="string",
        ),
        string(
            "source",
        ),
    )
)
_GRAVITY_SOURCE_SCHEMA = ConfigSchema(
    (
        *(
            string(
                name,
                required=True,
            )
            for name in ("path", "norm", "tideSystem", "gmSource", "radiusSource")
        ),
        integer(
            "declaredDegree",
            required=True,
            ge=0,
        ),
        integer(
            "loadedDegree",
            required=True,
            ge=0,
        ),
        boolean(
            "allowMissingCoefficients",
            required=True,
        ),
        sequence(
            "missingCoefficients",
            required=True,
        ),
        number(
            "fileGmM3S2",
            required=True,
            gt=0,
        ),
        number(
            "fileRadiusM",
            required=True,
            gt=0,
        ),
    )
)

LUNAR_ORBIT_FIELDS = (
    class_config(
        "ephemerides",
        "ephemerides",
        required=True,
    ),
    number(
        "initialTdbJd1",
        required=True,
    ),
    number(
        "initialTdbJd2",
        default=0.0,
    ),
    number(
        "durationSeconds",
        required=True,
    ),
    number(
        "outputStepSeconds",
        default=3600.0,
        gt=0,
    ),
    number(
        "accelerationDiagnosticsStepSeconds",
        default=None,
        gt=0,
        allow_none=True,
    ),
    boolean(
        "includeEih",
        default=False,
    ),
    mapping("lunarDegree2Gravity", nested=LUNAR_DEGREE2_SCHEMA, default=None, allow_none=True),
    mapping("lunarRotation", nested=_ROTATION_SCHEMA, default=None, allow_none=True),
    mapping("gravityFields", default=None, nested=_GRAVITY_FIELDS_SCHEMA, allow_none=True),
    sequence("moonFigurePartners", default=None, item_kind="string", allow_none=True),
    string(
        "earthDynamicsFrame",
        default="de440",
        choices=("de440", "de430"),
    ),
    sequence(
        "externalBodyGroups",
        default=[],
        item_kind="string",
    ),
    string(
        "earthPoleModel",
        default="ephemeris",
        choices=("ephemeris", "de430", "de440", "iau2006_2000a"),
    ),
    sequence(
        "externalBodyIds",
        default=[],
        item_kind="string",
    ),
    mapping("earthTide", default=None, nested=_EARTH_TIDE_SCHEMA, allow_none=True),
    sequence(
        "bodyForceTerms",
        item_kind="string",
        min_items=1,
    ),
    number("solarJ2"),
    mapping(
        "solarGmVariation",
        default=None,
        nested=ConfigSchema(
            (
                number(
                    "fractionalRatePerYear",
                    required=True,
                ),
                number(
                    "referenceJdTdb",
                    required=True,
                ),
            )
        ),
        allow_none=True,
    ),
    mapping(
        "kuiperBeltRing",
        default=None,
        nested=ConfigSchema(
            (
                number(
                    "pointGmM3S2",
                    required=True,
                    gt=0,
                ),
                number(
                    "radiusAu",
                    default=44.0,
                    gt=0,
                ),
                integer(
                    "pointCount",
                    default=36,
                    ge=3,
                ),
                number(
                    "obliquityArcsec",
                    default=84381.448,
                ),
                number(
                    "referenceJdTdb",
                    required=True,
                ),
                number(
                    "phaseDegrees",
                    default=0.0,
                ),
            )
        ),
        allow_none=True,
    ),
    mapping(
        "earthJ2TimeVariation",
        default={},
        nested=_EARTH_J2_SCHEMA,
    ),
    mapping("solarRelativistic", default=None, nested=_SOLAR_RELATIVISTIC_SCHEMA, allow_none=True),
    number(
        "solarRadiusM",
        default=696000000.0,
        gt=0,
    ),
    number(
        "stepSeconds",
        default=5400.0,
        gt=0,
    ),
    integer(
        "integratorOrder",
        default=13,
        ge=2,
        le=13,
    ),
    integer(
        "correctorIterations",
        default=2,
        ge=1,
    ),
    integer(
        "historyInterpolationOrder",
        default=13,
        ge=1,
        le=13,
    ),
    integer(
        "trajectoryInterpolationOrder",
        default=13,
        ge=1,
        le=13,
    ),
    number(
        "startupStepSeconds",
        default=675.0,
        gt=0,
    ),
)

LUNAR_ORBIT_SNAPSHOT_SCHEMA = ConfigSchema(
    LUNAR_ORBIT_FIELDS
    + (
        path(
            "inputFileMassCatalog",
        ),
        path(
            "outputFileOrbit",
        ),
        path(
            "outputFileAccelerationDiagnostics",
        ),
        path(
            "outputFileMetadata",
        ),
        path("outputFileRotation"),
        mapping("resolvedForceInventory", nested=_RESOLVED_FORCE_SCHEMA),
        mapping("resolvedGravityFields"),
    )
)


def validate_config_snapshot(value, schema, *, path):
    from collections.abc import Mapping

    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object.")
    schema._check_structure(dict(value), path)
    fields = {field.name: field for field in schema.fields}
    for name, item in value.items():
        if name == "type" and schema.type_name is not None:
            continue
        field = fields[name]
        field_path = f"{path}.{name}"
        if name == "resolvedGravityFields":
            field.validate(item, field_path)
            for body, source in item.items():
                if body not in {"EARTH", "MOON"}:
                    raise ValueError(f"{field_path} has unknown body ID {body!r}.")
                validate_config_snapshot(source, _GRAVITY_SOURCE_SCHEMA, path=f"{field_path}[{body}]")
                for pair in source["missingCoefficients"]:
                    if (
                        not isinstance(pair, (list, tuple))
                        or len(pair) != 2
                        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in pair)
                    ):
                        raise ValueError(f"{field_path}[{body}].missingCoefficients must contain degree/order pairs.")
        elif field.nested is not None and item is not None:
            validate_config_snapshot(item, field.nested, path=field_path)
        elif field.kind == "class" and item is not None:
            from .registry import _declaration

            declaration = item
            registered = _declaration(field.class_category, declaration["type"])
            validate_config_snapshot(item, registered.schema, path=field_path)
        else:
            field.validate(item, field_path)
    if schema is LUNAR_DEGREE2_SCHEMA:
        _validate_degree2(
            {"beta": value.get("beta", _DEGREE2_DEFAULTS.beta), "gamma": value.get("gamma", _DEGREE2_DEFAULTS.gamma)},
            path,
        )
    return dict(value)
