"""Config schemas of the lunar dynamics models.

``LUNAR_ORBIT_FIELDS`` are the fields of the ``LunarOrbitPropagation`` program: initial TDB epoch and span,
force terms, gravity fields, Earth tides, integrated lunar rotation and integrator settings.
``LUNAR_DEGREE2_SCHEMA`` describes the time-variable lunar degree-2 gravity field under the ephemeris
attitude, and ``LUNAR_CORE_SCHEMA`` the fluid core of the integrated lunar rotation.
"""

from lunarops.classes.dynamics.deformation import LunarDegree2GravityParameters
from lunarops.classes.dynamics.forces import EarthTideParameters
from lunarops.config.schema import ConfigSchema, boolean, class_config, integer, mapping, number, path, sequence, string


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
            description="Undistorted lunar J2 (unnormalized), before tidal and rotational deformation.",
        ),
        number(
            "beta",
            default=_DEGREE2_DEFAULTS.beta,
            gt=0,
            lt=1,
            description="Lunar moment-of-inertia ratio beta = (C - A) / B.",
        ),
        number(
            "gamma",
            default=_DEGREE2_DEFAULTS.gamma,
            gt=0,
            lt=1,
            description="Lunar moment-of-inertia ratio gamma = (B - A) / C; must be less than `beta`.",
        ),
        number(
            "loveK2",
            default=_DEGREE2_DEFAULTS.love_k2,
            ge=0,
            description="Lunar degree-2 potential Love number k2.",
        ),
        number(
            "tidalDelayDays",
            default=_DEGREE2_DEFAULTS.tidal_delay_days,
            ge=0,
            description="Time delay of the lunar tidal and rotational response, in days.",
        ),
        number(
            "radiusM",
            default=_DEGREE2_DEFAULTS.radius_m,
            gt=0,
            description="Reference radius of the lunar gravity field, in metres.",
        ),
        number(
            "meanMotionRadS",
            default=_DEGREE2_DEFAULTS.mean_motion_rad_s,
            gt=0,
            description="Lunar orbital mean motion used in the rotational deformation, in rad/s.",
        ),
        boolean(
            "includeTidalDeformation",
            default=_DEGREE2_DEFAULTS.include_tidal_deformation,
            description="Include the deformation raised by the Earth and Sun tides.",
        ),
        boolean(
            "includeRotationalDeformation",
            default=_DEGREE2_DEFAULTS.include_rotational_deformation,
            description="Include the deformation caused by the lunar spin.",
        ),
        number(
            "c21Static",
            default=_DEGREE2_DEFAULTS.c21_static,
            description="Static unnormalized C21 added to the deformed degree-2 field.",
        ),
        number(
            "s21Static",
            default=_DEGREE2_DEFAULTS.s21_static,
            description="Static unnormalized S21 added to the deformed degree-2 field.",
        ),
        number(
            "s22Static",
            default=_DEGREE2_DEFAULTS.s22_static,
            description="Static unnormalized S22 added to the deformed degree-2 field.",
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
            description="Polar moment of inertia of the fluid core as a fraction of the total polar moment C.",
        ),
        number(
            "flattening",
            required=True,
            ge=0,
            lt=1,
            description="Geometric flattening of the core-mantle boundary.",
        ),
        number(
            "viscosityPerS",
            required=True,
            ge=0,
            description="Viscous core-mantle coupling coefficient k_v / C_c, in 1/s. Zero disables viscous coupling.",
        ),
    )
)


_GRAVITY_FIELD_SCHEMA = ConfigSchema(
    (
        path("path", required=True, non_empty=True, description="Spherical-harmonic gravity field file."),
        integer("maxDegree", ge=0, description="Highest degree loaded from the file. Defaults to the full field."),
        number(
            "radiusM",
            gt=0,
            description="Reference radius of the field, in metres. Defaults to the radius in the file header.",
        ),
        string("name", non_empty=True, description="Label of the field in metadata. Defaults to the body name."),
        boolean(
            "allowMissingCoefficients",
            default=False,
            description=(
                "Accept a file that omits some (degree, order) pairs up to `maxDegree`; the missing coefficients are "
                "zero and are listed in the metadata."
            ),
        ),
    )
)

_GRAVITY_FIELDS_SCHEMA = ConfigSchema(
    (
        mapping(
            "earth",
            nested=_GRAVITY_FIELD_SCHEMA,
            description="Static Earth gravity field, evaluated in the Earth-fixed axes selected by `earthPoleModel`.",
        ),
        mapping(
            "moon",
            nested=_GRAVITY_FIELD_SCHEMA,
            description="Static Moon gravity field, evaluated in the lunar principal-axis frame.",
        ),
    )
)

_DEFORMATION_SCHEMA = ConfigSchema(
    (
        number("loveK2", required=True, ge=0, description="Lunar degree-2 potential Love number k2."),
        number(
            "delayDays",
            required=True,
            gt=0,
            description=(
                "Time delay of the lunar deformation response, in days. Must not be shorter than `stepSeconds`."
            ),
        ),
        number(
            "meanMotionRadS",
            required=True,
            gt=0,
            description="Lunar orbital mean motion used in the rotational deformation, in rad/s.",
        ),
        number(
            "seedDurationSeconds",
            required=True,
            gt=0,
            description=(
                "Length of the seed history integrated backwards before the start epoch to supply delayed states, in "
                "seconds. Must cover `delayDays`."
            ),
        ),
        string(
            "seedModel",
            required=True,
            choices=("rigidBackward",),
            description=(
                "How the seed history is produced: `rigidBackward` integrates the rigid Moon backwards from the "
                "initial state."
            ),
        ),
        boolean(
            "includeTidal",
            default=True,
            description="Include the deformation raised by the external bodies' tides.",
        ),
        boolean("includeRotational", default=True, description="Include the deformation caused by the lunar spin."),
        number("c21Static", default=0.0, description="Static unnormalized C21 added to the deformed degree-2 field."),
        number("s21Static", default=0.0, description="Static unnormalized S21 added to the deformed degree-2 field."),
        number("s22Static", default=0.0, description="Static unnormalized S22 added to the deformed degree-2 field."),
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
            description=(
                "Principal moments of inertia (A, B, C) of the undistorted Moon, normalized by M R^2 with R = "
                "`radiusM`."
            ),
        ),
        number(
            "radiusM",
            required=True,
            gt=0,
            description=(
                "Reference radius used to normalize the moments of inertia, in metres. Must equal the Moon field "
                "radius."
            ),
        ),
        sequence(
            "initialEulerAnglesRad",
            required=True,
            item_kind="number",
            min_items=3,
            max_items=3,
            description=(
                "Initial mantle attitude as 3-1-3 Euler angles [phi, theta, psi] in radians, rotating PA axes to "
                "inertial axes as Rz(phi) Rx(theta) Rz(psi). sin(theta) must be nonzero."
            ),
        ),
        sequence(
            "initialAngularVelocityRadS",
            required=True,
            item_kind="number",
            min_items=3,
            max_items=3,
            description="Initial mantle angular velocity in PA-frame components, in rad/s.",
        ),
        sequence(
            "initialCoreAngularVelocityRadS",
            default=None,
            item_kind="number",
            min_items=3,
            max_items=3,
            allow_none=True,
            description=(
                "Initial fluid-core angular velocity in mantle PA-frame components, in rad/s. Required exactly when "
                "`core` is set."
            ),
        ),
        sequence(
            "sources",
            required=True,
            item_kind="string",
            description=(
                "Bodies whose gravity exerts a torque on the lunar figure, by mass-catalog body ID. Each must be in "
                "the selected bodies."
            ),
        ),
        string(
            "gravityTorque",
            required=True,
            choices=("quadrupole", "figure"),
            description=(
                "Torque model for the `sources`: `quadrupole` uses the inertia tensor, `figure` the full Moon gravity"
                " field."
            ),
        ),
        boolean(
            "earthFigureTorque",
            default=False,
            description="Add the torque from the Earth J2 acting on the lunar figure. Requires an Earth gravity field.",
        ),
        boolean(
            "geodeticPrecession",
            default=False,
            description="Include the relativistic geodetic precession in the transport of angular momentum.",
        ),
        mapping(
            "core",
            nested=LUNAR_CORE_SCHEMA,
            default=None,
            allow_none=True,
            description="Fluid-core model. Null integrates a rigid mantle without a core.",
        ),
        mapping(
            "deformation",
            nested=_DEFORMATION_SCHEMA,
            default=None,
            allow_none=True,
            description="Delayed tidal and rotational deformation of the integrated Moon. Null keeps the figure rigid.",
        ),
    )
)

_TIDE_DEFAULTS = EarthTideParameters()
_EARTH_TIDE_SCHEMA = ConfigSchema(
    (
        sequence(
            "tideRaisers",
            default=["MOON", "SUN"],
            item_kind="string",
            description="Bodies that raise tides on the Earth, by mass-catalog body ID.",
        ),
        number("k20", default=_TIDE_DEFAULTS.k20, description="Earth degree-2, order-0 potential Love number."),
        number("k21", default=_TIDE_DEFAULTS.k21, description="Earth degree-2, order-1 potential Love number."),
        number("k22", default=_TIDE_DEFAULTS.k22, description="Earth degree-2, order-2 potential Love number."),
        sequence(
            "tauOrbDays",
            default=_TIDE_DEFAULTS.tau_orb_days,
            item_kind="number",
            min_items=3,
            max_items=3,
            description="Orbital time delays of the order-0, order-1 and order-2 tides, in days.",
        ),
        sequence(
            "tauRotDays",
            default=_TIDE_DEFAULTS.tau_rot_days,
            item_kind="number",
            min_items=3,
            max_items=3,
            description="Rotational time delays of the order-0, order-1 and order-2 tides, in days.",
        ),
        number(
            "earthRotationRateRadS",
            default=_TIDE_DEFAULTS.earth_rotation_rate_rad_s,
            description="Earth rotation rate used in the tidal delay, in rad/s.",
        ),
        number(
            "earthRadiusM",
            default=6_378_136.6,
            gt=0,
            description="Earth reference radius of the tidal potential, in metres.",
        ),
    )
)
_EARTH_J2_SCHEMA = ConfigSchema(
    (
        number("linearPerYear", default=0.0, description="Linear rate of the Earth J2, per Julian year."),
        number(
            "quadraticPerYear2",
            default=0.0,
            description=(
                "Quadratic coefficient of the Earth J2 polynomial, per Julian year squared (multiplies T^2 directly)."
            ),
        ),
        number(
            "referenceJdTdb",
            default=2_451_545.0,
            description="Reference epoch of the Earth J2 polynomial, as a TDB Julian date.",
        ),
    )
)
_SOLAR_RELATIVISTIC_SCHEMA = ConfigSchema(
    (
        boolean(
            "includeLenseThirring",
            default=True,
            description="Include the Lense-Thirring acceleration from the solar spin.",
        ),
        boolean(
            "includeRadiationPressure",
            default=False,
            description="Include solar radiation pressure on the Moon and the Earth.",
        ),
        number("moonEpsilon", default=0.0, description="Radiation-pressure coefficient of the Moon (dimensionless)."),
        number("earthEpsilon", default=0.0, description="Radiation-pressure coefficient of the Earth (dimensionless)."),
    )
)
LUNAR_ORBIT_FIELDS = (
    class_config(
        "ephemerides",
        "ephemerides",
        required=True,
        description=(
            "Ephemeris that supplies the prescribed Earth-Moon barycentre and external-body states, the initial"
            " Earth-Moon state and, for a rigid prescribed Moon, the lunar attitude."
        ),
    ),
    number("initialTdbJd1", required=True, description="First part of the TDB Julian date of the initial epoch."),
    number("initialTdbJd2", default=0.0, description="Second part of the TDB Julian date of the initial epoch."),
    number(
        "durationSeconds",
        required=True,
        description=(
            "Propagation span, in seconds; negative values propagate backwards. Must be a whole multiple of "
            "`stepSeconds`."
        ),
    ),
    number(
        "backwardDurationSeconds",
        default=0.0,
        ge=0,
        description=(
            "Length, in seconds, of an additional arc propagated backward from the initial epoch, for a two-sided "
            "propagation with the forward arc of `durationSeconds`, which must then be positive. A whole multiple of "
            "`stepSeconds`."
        ),
    ),
    number(
        "outputStepSeconds",
        default=3600.0,
        gt=0,
        description="Sampling interval of the orbit and rotation output tables, in seconds.",
    ),
    number(
        "accelerationDiagnosticsStepSeconds",
        default=None,
        gt=0,
        allow_none=True,
        description=(
            "Sampling interval of the acceleration diagnostics, in seconds; a whole multiple of `stepSeconds`. Null "
            "writes no diagnostic rows."
        ),
    ),
    boolean(
        "includeEih",
        default=False,
        description="Add the Einstein-Infeld-Hoffmann 1PN point-mass acceleration (`eih_1pn`).",
    ),
    mapping(
        "lunarDegree2Gravity",
        nested=LUNAR_DEGREE2_SCHEMA,
        default=None,
        allow_none=True,
        description=(
            "Time-variable lunar degree-2 field under the ephemeris attitude. Requires a Moon gravity field; not "
            "combinable with `lunarRotation`."
        ),
    ),
    mapping(
        "lunarRotation",
        nested=_ROTATION_SCHEMA,
        default=None,
        allow_none=True,
        description=(
            "Integrate the mantle (and optional core) rotation together with the orbit. Requires `outputFileRotation`."
        ),
    ),
    mapping(
        "gravityFields",
        default=None,
        nested=_GRAVITY_FIELDS_SCHEMA,
        allow_none=True,
        description="Static spherical-harmonic fields of the Earth and the Moon. At least one must be given.",
    ),
    sequence(
        "moonFigurePartners",
        default=None,
        item_kind="string",
        allow_none=True,
        description=(
            "Point-mass bodies that interact with the Moon gravity field, by mass-catalog body ID. Null uses EARTH "
            "and the default planets."
        ),
    ),
    sequence(
        "earthFigurePartners",
        default=None,
        item_kind="string",
        allow_none=True,
        description=(
            "Point-mass bodies that interact with the Earth gravity field, by mass-catalog body ID. The Moon takes "
            "the field acceleration directly and the Earth the reaction of every partner. Null mirrors "
            "`moonFigurePartners` with MOON in place of EARTH."
        ),
    ),
    string(
        "earthDynamicsFrame",
        default="de440",
        choices=("de440", "de430"),
        description=(
            "Earth-orientation model of the gravity axes when `earthPoleModel` is `ephemeris`: `de440` or `de430`."
        ),
    ),
    sequence(
        "externalBodyGroups",
        default=[],
        item_kind="string",
        description="Mass-catalog groups whose bodies act as prescribed perturbers.",
    ),
    string(
        "earthPoleModel",
        default="ephemeris",
        choices=("ephemeris", "de430", "de440", "iau2006_2000a"),
        description=(
            "Earth-orientation model of the Earth gravity field axes: `ephemeris` (use `earthDynamicsFrame`), "
            "`de430`, `de440` or `iau2006_2000a`."
        ),
    ),
    sequence(
        "externalBodyIds",
        default=[],
        item_kind="string",
        description="Additional mass-catalog body IDs that act as prescribed perturbers.",
    ),
    mapping(
        "earthTide",
        default=None,
        nested=_EARTH_TIDE_SCHEMA,
        allow_none=True,
        description="Delayed Earth tides raised by `tideRaisers`, acting on the Moon. Null disables the tide.",
    ),
    sequence(
        "bodyForceTerms",
        item_kind="string",
        min_items=1,
        description=(
            "Force terms applied to the Moon, from `point_mass`, `eih_1pn`, `solar_j2`, `figure_earth`, "
            "`figure_moon`, `tide`, `lense_thirring` and `solar_radiation_pressure`. Defaults to every configured "
            "term; `eih_1pn` requires `point_mass`."
        ),
    ),
    number("solarJ2", description="Solar J2 (unnormalized), applied about the IAU solar pole. Omit to disable."),
    mapping(
        "solarGmVariation",
        default=None,
        nested=ConfigSchema(
            (
                number(
                    "fractionalRatePerYear",
                    required=True,
                    description="Fractional rate of the solar GM, (dGM/dt)/GM, per Julian year.",
                ),
                number(
                    "referenceJdTdb",
                    required=True,
                    description="Epoch at which the solar GM equals its catalog value, as a TDB Julian date.",
                ),
            )
        ),
        allow_none=True,
        description="Linear drift of the solar GM. Null keeps it constant.",
    ),
    mapping(
        "kuiperBeltRing",
        default=None,
        nested=ConfigSchema(
            (
                number("pointGmM3S2", required=True, gt=0, description="GM of each ring point, in m^3/s^2."),
                number("radiusAu", default=44.0, gt=0, description="Ring radius, in astronomical units."),
                integer("pointCount", default=36, ge=3, description="Number of ring points."),
                number(
                    "obliquityArcsec",
                    default=84381.448,
                    description=(
                        "Inclination of the ring plane to the ICRF equator (the ecliptic obliquity), in arcseconds."
                    ),
                ),
                number(
                    "referenceJdTdb",
                    required=True,
                    description="Epoch at which the ring phase is `phaseDegrees`, as a TDB Julian date.",
                ),
                number(
                    "phaseDegrees",
                    default=0.0,
                    description="Longitude of the first ring point at `referenceJdTdb`, in degrees.",
                ),
            )
        ),
        allow_none=True,
        description=(
            "Ring of equal point masses in circular heliocentric orbits that represents the Kuiper belt. Null "
            "disables it."
        ),
    ),
    mapping(
        "earthJ2TimeVariation",
        default={},
        nested=_EARTH_J2_SCHEMA,
        description=(
            "Polynomial time variation of the Earth J2: J2(t) = J2 + linearPerYear*T + quadraticPerYear2*T^2, with T "
            "in Julian years from `referenceJdTdb`."
        ),
    ),
    mapping(
        "solarRelativistic",
        default=None,
        nested=_SOLAR_RELATIVISTIC_SCHEMA,
        allow_none=True,
        description="Solar Lense-Thirring and radiation-pressure accelerations. Null disables both.",
    ),
    number(
        "solarRadiusM",
        default=696000000.0,
        gt=0,
        description="Solar reference radius for the J2 field and the radiation model, in metres.",
    ),
    number(
        "stepSeconds",
        default=5400.0,
        gt=0,
        description="Fixed step of the Adams-Bashforth-Moulton integrator, in seconds.",
    ),
    integer(
        "integratorOrder",
        default=13,
        ge=2,
        le=13,
        description="Order of the Adams-Bashforth-Moulton integrator (2 to 13).",
    ),
    integer(
        "correctorIterations",
        default=2,
        ge=1,
        description="Number of corrector passes per step; 2 gives a PECEC scheme.",
    ),
    integer(
        "historyInterpolationOrder",
        default=13,
        ge=1,
        le=13,
        description="Order of the local interpolation of the accepted history for delayed-state queries.",
    ),
    integer(
        "trajectoryInterpolationOrder",
        default=13,
        ge=1,
        le=13,
        description="Order of the local interpolation between integration nodes for the output.",
    ),
    number(
        "startupStepSeconds",
        default=675.0,
        gt=0,
        description=(
            "Step of the Dormand-Prince (DOP853) start-up integration that produces the first nodes, in seconds."
        ),
    ),
)
