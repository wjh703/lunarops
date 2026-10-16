"""Pure configuration contracts shared by lunar models and artifact validation."""

from lunarops.classes.dynamics.inertia import LunarDegree2GravityParameters

from .schema import ConfigSchema, UiHints, boolean, class_config, integer, mapping, number, path, sequence, string


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
            minimum=0,
            minimum_exclusive=True,
            allow_none=False,
            ui=UiHints(unit="1"),
        ),
        number(
            "beta",
            default=_DEGREE2_DEFAULTS.beta,
            minimum=0,
            minimum_exclusive=True,
            maximum=1,
            maximum_exclusive=True,
            allow_none=False,
            ui=UiHints(unit="1"),
        ),
        number(
            "gamma",
            default=_DEGREE2_DEFAULTS.gamma,
            minimum=0,
            minimum_exclusive=True,
            maximum=1,
            maximum_exclusive=True,
            allow_none=False,
            ui=UiHints(unit="1"),
        ),
        number("loveK2", default=_DEGREE2_DEFAULTS.love_k2, minimum=0, allow_none=False, ui=UiHints(unit="1")),
        number(
            "tidalDelayDays",
            default=_DEGREE2_DEFAULTS.tidal_delay_days,
            minimum=0,
            allow_none=False,
            ui=UiHints(unit="day"),
        ),
        number(
            "radiusM",
            default=_DEGREE2_DEFAULTS.radius_m,
            minimum=0,
            minimum_exclusive=True,
            allow_none=False,
            ui=UiHints(unit="m"),
        ),
        number(
            "meanMotionRadS",
            default=_DEGREE2_DEFAULTS.mean_motion_rad_s,
            minimum=0,
            minimum_exclusive=True,
            allow_none=False,
            ui=UiHints(unit="rad/s"),
        ),
        boolean("includeTidalDeformation", default=_DEGREE2_DEFAULTS.include_tidal_deformation, allow_none=False),
        boolean(
            "includeRotationalDeformation", default=_DEGREE2_DEFAULTS.include_rotational_deformation, allow_none=False
        ),
        number("c21Static", default=_DEGREE2_DEFAULTS.c21_static, allow_none=False, ui=UiHints(unit="1")),
        number("s21Static", default=_DEGREE2_DEFAULTS.s21_static, allow_none=False, ui=UiHints(unit="1")),
        number("s22Static", default=_DEGREE2_DEFAULTS.s22_static, allow_none=False, ui=UiHints(unit="1")),
    ),
    description="Delayed degree-2 gravity using prescribed lunar attitude; separate from integrated-rotation deformation.",
    validator=_validate_degree2,
)


LUNAR_CORE_SCHEMA = ConfigSchema(
    (
        number(
            "polarMomentFraction",
            required=True,
            minimum=0,
            maximum=1,
            minimum_exclusive=True,
            maximum_exclusive=True,
            allow_none=False,
        ),
        number("flattening", required=True, minimum=0, maximum=1, maximum_exclusive=True, allow_none=False),
        number("viscosityPerS", required=True, minimum=0, allow_none=False, ui=UiHints(unit="s^-1")),
        string("pressureSpin", required=True, choices=("core", "mantle"), allow_none=False),
    )
)


_GRAVITY_FIELD_SCHEMA = ConfigSchema(
    (
        path("path", required=True, non_empty=True, allow_none=False),
        string("format", default="icgem", choices=("icgem",), allow_none=False),
        integer("maxDegree", minimum=0, allow_none=False),
        number("radiusM", minimum=0, minimum_exclusive=True, allow_none=False),
        string("name", non_empty=True, allow_none=False),
        boolean("allowMissingCoefficients", default=False, allow_none=False),
    )
)

_GRAVITY_FIELDS_SCHEMA = ConfigSchema(
    (
        mapping("earth", nested=_GRAVITY_FIELD_SCHEMA, allow_none=False),
        mapping("moon", nested=_GRAVITY_FIELD_SCHEMA, allow_none=False),
    )
)

_DEFORMATION_SCHEMA = ConfigSchema(
    (
        number("loveK2", required=True, minimum=0, allow_none=False),
        number("delayDays", required=True, minimum=0, minimum_exclusive=True, allow_none=False),
        number("meanMotionRadS", required=True, minimum=0, minimum_exclusive=True, allow_none=False),
        number("seedDurationSeconds", required=True, minimum=0, minimum_exclusive=True, allow_none=False),
        string("seedModel", required=True, choices=("rigidBackward",), allow_none=False),
        boolean("includeTidal", default=True, allow_none=False),
        boolean("includeRotational", default=True, allow_none=False),
        number("c21Static", default=0.0, allow_none=False),
        number("s21Static", default=0.0, allow_none=False),
        number("s22Static", default=0.0, allow_none=False),
    )
)

_ROTATION_SCHEMA = ConfigSchema(
    (
        sequence("inertiaMoments", required=True, item_kind="number", min_items=3, max_items=3, allow_none=False),
        number("radiusM", required=True, minimum=0, minimum_exclusive=True, allow_none=False),
        sequence("initialQuaternion", required=True, item_kind="number", min_items=4, max_items=4, allow_none=False),
        sequence(
            "initialAngularVelocityRadS", required=True, item_kind="number", min_items=3, max_items=3, allow_none=False
        ),
        sequence("initialCoreAngularVelocityRadS", default=None, item_kind="number", min_items=3, max_items=3),
        sequence("sources", required=True, item_kind="string", allow_none=False),
        string("gravityTorque", required=True, choices=("quadrupole", "figure"), allow_none=False),
        boolean("earthFigureTorque", default=False, allow_none=False),
        boolean("geodeticPrecession", default=False, allow_none=False),
        mapping("core", nested=LUNAR_CORE_SCHEMA, default=None),
        mapping("deformation", nested=_DEFORMATION_SCHEMA, default=None),
    )
)

_EARTH_TIDE_SCHEMA = ConfigSchema(
    (
        sequence("tideRaisers", item_kind="string", allow_none=False),
        number("k20", allow_none=False),
        number("k21", allow_none=False),
        number("k22", allow_none=False),
        sequence("tauOrbDays", item_kind="number", min_items=3, max_items=3, allow_none=False),
        sequence("tauRotDays", item_kind="number", min_items=3, max_items=3, allow_none=False),
        number("earthRotationRateRadS", allow_none=False),
        number("earthRadiusM", minimum=0, minimum_exclusive=True, allow_none=False),
    )
)
_EARTH_J2_SCHEMA = ConfigSchema(
    (
        number("linearPerYear", allow_none=False),
        number("quadraticPerYear2", allow_none=False),
        number("referenceJdTdb", allow_none=False),
    )
)
_SOLAR_RELATIVISTIC_SCHEMA = ConfigSchema(
    (
        boolean("includeLenseThirring", allow_none=False),
        boolean("includeRadiationPressure", allow_none=False),
        number("earthEpsilon", allow_none=False),
        number("moonEpsilon", allow_none=False),
    )
)
_RESOLVED_FORCE_SCHEMA = ConfigSchema(
    (sequence("active", item_kind="string", allow_none=False), string("source", allow_none=False))
)
_GRAVITY_SOURCE_SCHEMA = ConfigSchema(
    (
        *(
            string(name, required=True, allow_none=False)
            for name in ("path", "norm", "tideSystem", "gmSource", "radiusSource")
        ),
        integer("declaredDegree", required=True, minimum=0, allow_none=False),
        integer("loadedDegree", required=True, minimum=0, allow_none=False),
        boolean("allowMissingCoefficients", required=True, allow_none=False),
        sequence("missingCoefficients", required=True, allow_none=False),
        number("fileGmM3S2", required=True, minimum=0, minimum_exclusive=True, allow_none=False),
        number("fileRadiusM", required=True, minimum=0, minimum_exclusive=True, allow_none=False),
    )
)

LUNAR_ORBIT_FIELDS = (
    class_config("ephemerides", "ephemerides", required=True, allow_none=False),
    number("initialTdbJd1", required=True, allow_none=False),
    number("initialTdbJd2", default=0.0, allow_none=False),
    number("durationSeconds", required=True, allow_none=False),
    number("outputStepSeconds", default=3600.0, minimum=0, minimum_exclusive=True, allow_none=False),
    number(
        "accelerationDiagnosticsStepSeconds",
        default=None,
        minimum=0,
        minimum_exclusive=True,
        allow_none=True,
    ),
    boolean("includeEih", default=False, allow_none=False),
    mapping("lunarDegree2Gravity", nested=LUNAR_DEGREE2_SCHEMA, default=None, allow_none=True),
    mapping("lunarRotation", nested=_ROTATION_SCHEMA, default=None),
    mapping("gravityFields", default=None, nested=_GRAVITY_FIELDS_SCHEMA, allow_none=True),
    mapping(
        "figurePartners",
        default=None,
        allow_none=True,
        nested=ConfigSchema(
            (
                sequence("earth", item_kind="string", allow_none=False),
                sequence("moon", item_kind="string", allow_none=False),
            )
        ),
    ),
    string(
        "earthDynamicsFrame",
        default="de440",
        choices=("de440", "de430"),
        allow_none=False,
    ),
    sequence("externalBodyGroups", default=[], item_kind="string", allow_none=False),
    string(
        "earthPoleModel",
        default="ephemeris",
        choices=("ephemeris", "de430", "de440", "iau2006_2000a"),
        allow_none=False,
    ),
    sequence("externalBodyIds", default=[], item_kind="string", allow_none=False),
    mapping("earthTide", default=None, nested=_EARTH_TIDE_SCHEMA, allow_none=True),
    sequence("bodyForceTerms", item_kind="string", min_items=1, allow_none=False),
    number("solarJ2"),
    mapping(
        "solarGmVariation",
        default=None,
        nested=ConfigSchema(
            (
                number("fractionalRatePerYear", required=True, allow_none=False),
                number("referenceJdTdb", required=True, allow_none=False),
            )
        ),
    ),
    mapping(
        "kuiperBeltRing",
        default=None,
        nested=ConfigSchema(
            (
                number("pointGmM3S2", required=True, minimum=0, minimum_exclusive=True, allow_none=False),
                number("radiusAu", default=44.0, minimum=0, minimum_exclusive=True, allow_none=False),
                integer("pointCount", default=36, minimum=3, allow_none=False),
                number("obliquityArcsec", default=84381.448, allow_none=False),
                number("referenceJdTdb", required=True, allow_none=False),
                number("phaseDegrees", default=0.0, allow_none=False),
            )
        ),
    ),
    mapping("earthJ2TimeVariation", default=None, nested=_EARTH_J2_SCHEMA, allow_none=True),
    mapping("solarRelativistic", default=None, nested=_SOLAR_RELATIVISTIC_SCHEMA, allow_none=True),
    number("solarRadiusM", default=696000000.0, minimum=0, minimum_exclusive=True, allow_none=False),
    number("stepSeconds", default=5400.0, minimum=0, minimum_exclusive=True, allow_none=False),
    integer("integratorOrder", default=13, minimum=2, maximum=13, allow_none=False),
    integer("correctorIterations", default=2, minimum=1, allow_none=False),
    integer("historyInterpolationOrder", default=13, minimum=1, maximum=13, allow_none=False),
    integer("trajectoryInterpolationOrder", default=13, minimum=1, maximum=13, allow_none=False),
    number("startupStepSeconds", default=675.0, minimum=0, minimum_exclusive=True, allow_none=False),
)

LUNAR_ORBIT_SNAPSHOT_SCHEMA = ConfigSchema(
    LUNAR_ORBIT_FIELDS
    + (
        path("inputFileMassCatalog", allow_none=False),
        path("outputFileOrbit", allow_none=False),
        path("outputFileAccelerationDiagnostics", allow_none=False),
        path("outputFileMetadata", allow_none=False),
        path("outputFileRotation"),
        mapping("resolvedForceInventory", nested=_RESOLVED_FORCE_SCHEMA),
        mapping("resolvedGravityFields", freeform_identifiers=True),
    )
)


def validate_config_snapshot(value, schema, *, path):
    """Check only present canonical fields without filling snapshot defaults."""
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
            from lunarops.classes.builtin_registry import ensure_registered

            from .registry import get_class_declaration, normalize_class_config

            ensure_registered()
            declaration = normalize_class_config(item)
            registered = get_class_declaration(field.class_category, declaration["type"])
            validate_config_snapshot(item, registered.schema, path=field_path)
        else:
            field.validate(item, field_path)
    if schema is LUNAR_DEGREE2_SCHEMA:
        _validate_degree2(
            {"beta": value.get("beta", _DEGREE2_DEFAULTS.beta), "gamma": value.get("gamma", _DEGREE2_DEFAULTS.gamma)},
            path,
        )
    return dict(value)
