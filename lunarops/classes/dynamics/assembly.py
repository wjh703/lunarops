"""Assembly of a lunar propagation problem from a program configuration.

:func:`build_lunar_propagation_problem` reads the mass catalog, gravity
fields and ephemeris named in the configuration and builds the force models,
Earth tide, Kuiper-belt ring, lunar rotation and integrator of one
propagation. The configuration is the validated ``LunarOrbitPropagation``
mapping with camelCase keys.

Attributes
----------
DEFAULT_PLANETARY_PARTNERS : tuple of str
    Bodies besides the Earth and the Moon that interact with the Moon figure
    and the Earth figure when ``moonFigurePartners`` or
    ``earthFigurePartners`` is not configured (DE440 convention).
SOLAR_FORCE_TERMS : tuple of tuple of str
    Diagnostic name and force-term name of each optional solar term.
DIAGNOSTIC_FORCE_GROUPS : tuple
    Diagnostic group names and the force names summed into each group.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from lunarops import console
from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.classes.dynamics.deformation import (
    LunarDeformation,
    LunarDeformationParameters,
    LunarDegree2GravityModel,
    LunarDegree2GravityParameters,
)
from lunarops.classes.dynamics.earth_figure_orientation import (
    EARTH_FIGURE_ORIENTATIONS,
    inertial2fixed_matrix_from_pole,
)
from lunarops.classes.dynamics.epoch_data import EphemerisLunarAxes, OrientationProvider
from lunarops.classes.dynamics.gravity import GravityField, make_j2_coefficients
from lunarops.classes.dynamics.rotation import LunarCoreParameters, LunarRotationDynamics, LunarRotationState
from lunarops.classes.dynamics.solar_environment import AU_M, KuiperBeltRing, SolarGmVariation
from lunarops.classes.dynamics.torques import (
    EarthFigureTorque,
    LunarFigureTorque,
    LunarTorqueGroup,
    PointMassGravityTorque,
)
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.time import Epoch, TimeScale
from lunarops.fileio.mass_catalog import read_mass_catalog

from .bodies import GravitatingBody
from .context import StateHistoryProvider
from .force_models import (
    FORCE_TERMS,
    EarthFigureForce,
    EarthTideForce,
    EihPointMassForce,
    LenseThirringForce,
    LunarForceGroup,
    MoonFigureForce,
    NewtonianPointMassForce,
    SolarJ2Force,
    SolarRadiationPressureForce,
    enabled_force_terms,
)
from .forces import EarthTideModel, EarthTideParameters, SolarSourceParameters
from .gravity import load_gravity_field
from .history import EphemerisHistory
from .integrators import AdamsBashforthMoultonIntegrator, IntegratorSettings
from .lunar_dynamics import LunarDynamics
from .parameters import (
    INITIAL_VALUE_NAMES,
    LunarDynamicsParameters,
    apply_to_config,
    apply_to_moon_field,
    check_applicable,
    earth_moon_gm,
    initial_orbit,
    nominal_parameters,
    read_lunar_dynamics_parameters,
)

DEFAULT_PLANETARY_PARTNERS = (
    "SUN",
    "MERCURY BARYCENTER",
    "VENUS BARYCENTER",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
)


SOLAR_FORCE_TERMS = (
    ("solar_lense_thirring", "lense_thirring"),
    ("solar_radiation_pressure", "solar_radiation_pressure"),
)


DIAGNOSTIC_FORCE_GROUPS = (
    ("newtonian", ("point_mass",)),
    ("post_newtonian", ("eih_1pn",)),
    ("figure", ("solar_j2", "figure_earth", "figure_moon")),
    ("tide", ("tide",)),
    ("srp", ("solar_radiation_pressure",)),
    ("lt", ("lense_thirring",)),
)


def _load_configured_gravity_fields(config, context, gm_by_body, parameters=None):
    fields = {}
    for body_id in ("EARTH", "MOON"):
        entry = config.get(body_id.lower())
        if entry is None:
            continue
        coefficients = load_gravity_field(
            context.resolve_path(entry["path"]),
            max_degree=entry.get("maxDegree"),
            gm_m3_s2=gm_by_body[body_id],
            reference_radius_m=entry.get("radiusM"),
            name=entry.get("name", body_id),
            allow_missing_coefficients=entry["allowMissingCoefficients"],
        )
        if body_id == "MOON":
            coefficients = apply_to_moon_field(parameters, coefficients)
        fields[body_id] = GravityField(coefficients)
    if not fields:
        raise ValueError("gravityFields must define at least earth or moon")
    return fields


@dataclass(frozen=True, slots=True)
class LunarPropagationProblem:
    """Everything needed to run and record one lunar propagation.

    Attributes
    ----------
    config : Mapping
        Read-only configuration, extended with ``resolvedForceInventory``
        and ``resolvedGravityFields``.
    dynamics : LunarDynamics
        Compiled dynamics.
    integrator : AdamsBashforthMoultonIntegrator
        Configured integrator.
    initial_epoch : Epoch
        Initial epoch in TDB.
    initial_vector : numpy.ndarray
        Initial integration state: ephemeris Moon-minus-Earth position (m)
        and velocity (m s⁻¹) with ICRF axes, followed by the configured
        rotational state.
    duration_s : float
        Signed propagation interval in TDB seconds.
    output_paths : tuple of Path
        Orbit, acceleration-diagnostics and metadata output files.
    external_history : StateHistoryProvider or None
        Ephemeris history used by delayed forces outside the integrated
        history; ``None`` when no delayed force is active.
    solar_force_terms : tuple of str
        Diagnostic names of the active optional solar terms.
    rotation_output : Path or None
        Rotation output file; ``None`` without integrated rotation.
    seed_duration_s : float or None
        Rigid seed-arc length in s for delayed deformation, else ``None``.
    parameters : LunarDynamicsParameters or None
        Current value of every initial value and dynamical parameter of the
        problem (see :mod:`~lunarops.classes.dynamics.parameters`), at the
        initial epoch.
    """

    config: Mapping
    dynamics: LunarDynamics
    integrator: AdamsBashforthMoultonIntegrator
    initial_epoch: Epoch
    initial_vector: np.ndarray
    duration_s: float
    backward_duration_s: float
    output_paths: tuple[Path, Path, Path]
    external_history: StateHistoryProvider | None
    solar_force_terms: tuple[str, ...]
    rotation_output: Path | None
    seed_duration_s: float | None
    parameters: LunarDynamicsParameters | None = None


def _moon_field(fields, user: str, radius_m: float | None = None) -> GravityField:
    """Return the Moon gravity field that ``user`` requires.

    Raises
    ------
    ValueError
        If ``fields`` has no Moon field, or its reference radius differs from ``radius_m`` (when given).
    """
    if fields is None or "MOON" not in fields:
        raise ValueError(f"{user} requires a Moon gravity field")
    moon = fields["MOON"]
    if radius_m is not None and not np.isclose(moon.radius_m, radius_m, rtol=0, atol=1e-9):
        raise ValueError(f"{user} and the Moon gravity field must use the same reference radius")
    return moon


def _figure_partners(body: str, configured, all_body_gm) -> tuple[str, ...]:
    """Return the figure partners of ``body`` by body name; raise ValueError if one is not a selected body."""
    partners = tuple(body_name(value) for value in configured)
    missing = sorted(set(partners) - set(all_body_gm))
    if missing:
        raise ValueError(f"{body} figure partners are missing from the selected bodies: {missing}")
    return partners


def build_rotation(config, fields, moon_gm, earth_gm):
    """Build the integrated lunar rotation from the configuration.

    Parameters
    ----------
    config : Mapping
        Program configuration; ``lunarRotation`` selects the model.
    fields : dict of str to GravityField or None
        Loaded gravity fields keyed by ``"EARTH"`` and ``"MOON"``.
    moon_gm : float
        Moon GM in m³ s⁻².
    earth_gm : float
        Earth GM in m³ s⁻².

    Returns
    -------
    rotation : LunarRotationDynamics or None
        Rotation model; ``None`` when ``lunarRotation`` is not configured.
    initial_vector : numpy.ndarray or None
        Initial rotational state of shape ``(7,)`` or ``(10,)``.

    Raises
    ------
    ValueError
        If the rotation is combined with ``lunarDegree2Gravity``, an output
        file or a required gravity field is missing, the core model and its
        initial angular velocity are not configured together, torque sources
        are duplicated or include the Moon, or the Moon field radius or GM
        is inconsistent with the rotation.
    """
    cfg = config["lunarRotation"]
    if cfg is None:
        return None, None
    if config.get("lunarDegree2Gravity") is not None:
        raise ValueError(
            "Integrated rotation uses lunarRotation.deformation, not prescribed-attitude lunarDegree2Gravity"
        )
    if config.get("outputFileRotation") is None:
        raise ValueError("Integrated rotation requires outputFileRotation")
    core = _core_parameters(cfg["core"])
    initial = LunarRotationState(
        cfg["initialEulerAnglesRad"],
        cfg["initialAngularVelocityRadS"],
        cfg["initialCoreAngularVelocityRadS"],
    )
    if (core is None) != (initial.core_angular_velocity_pa_rad_s is None):
        raise ValueError("Core model and core initial angular velocity must be configured together")
    inertia = np.diag(cfg["inertiaMoments"])
    deformation = _deformation(cfg, fields, inertia, earth_gm, moon_gm)
    rotation = LunarRotationDynamics(
        inertia,
        cfg["radiusM"],
        LunarTorqueGroup(_torques(config, fields, inertia, moon_gm)),
        core=core,
        geodetic_precession=cfg["geodeticPrecession"],
        deformation=deformation,
    )
    return rotation, initial.vector()


def _core_parameters(core_cfg) -> LunarCoreParameters | None:
    """Return the fluid-core parameters of ``lunarRotation.core``; ``None`` without a core."""
    if core_cfg is None:
        return None
    return LunarCoreParameters(
        polar_moment_fraction=core_cfg["polarMomentFraction"],
        flattening=core_cfg["flattening"],
        viscosity_per_s=core_cfg["viscosityPerS"],
    )


def _deformation(cfg, fields, inertia, earth_gm, moon_gm) -> LunarDeformation | None:
    """Return the delayed deformation of ``lunarRotation.deformation``; ``None`` when not configured."""
    p = cfg["deformation"]
    if p is None:
        return None
    moon = _moon_field(fields, "Integrated deformation", cfg["radiusM"])
    return LunarDeformation(
        LunarDeformationParameters(
            p["loveK2"],
            p["delayDays"] * SECONDS_PER_DAY,
            p["meanMotionRadS"],
            p["includeTidal"],
            p["includeRotational"],
            c21_static=p["c21Static"],
            s21_static=p["s21Static"],
            s22_static=p["s22Static"],
        ),
        inertia,
        cfg["radiusM"],
        earth_gm,
        moon_gm,
        moon,
    )


def _torques(config, fields, inertia, moon_gm) -> list[LunarFigureTorque | PointMassGravityTorque | EarthFigureTorque]:
    """Return the torques of ``lunarRotation``: the gravity torque of every source and the Earth figure torque."""
    cfg = config["lunarRotation"]
    torques: list[LunarFigureTorque | PointMassGravityTorque | EarthFigureTorque] = []
    sources = tuple(body_name(name) for name in cfg["sources"])
    if len(sources) != len(set(sources)) or "MOON" in sources:
        raise ValueError("Lunar torque sources must be unique and external to the Moon")
    if cfg["gravityTorque"] == "figure":
        moon = _moon_field(fields, "Figure torques", cfg["radiusM"])
        if not np.isclose(moon.gm_m3_s2, moon_gm, rtol=1e-14):
            raise ValueError("Moon field and rotational dynamics must use the same GM")
        torques.extend(LunarFigureTorque(source, moon) for source in sources)
    else:
        torques.extend(PointMassGravityTorque(source, inertia) for source in sources)
    if cfg["earthFigureTorque"]:
        if fields is None or "EARTH" not in fields:
            raise ValueError("Earth figure torque requires an Earth gravity field")
        earth = fields["EARTH"]
        j2_time = config["earthJ2TimeVariation"]
        torques.append(
            EarthFigureTorque(
                inertia,
                earth.radius_m,
                -np.sqrt(5) * earth.coefficients.coeffs[0, 2, 0],
                j2_time["linearPerYear"],
                j2_time["quadraticPerYear2"],
                j2_time["referenceJdTdb"],
            )
        )
    return torques


def _build_ring(
    config,
    gm_by_body,
    ephemeris,
    epoch,
    earth_gm,
    moon_gm,
    solar_gm_variation,
):
    """Build and prepropagate the Kuiper-belt ring, or return ``None``.

    The prepropagated interval covers the propagation, the rigid seed arc
    and the integrator warm-up on either side of the initial epoch.
    """
    ring_config = config["kuiperBeltRing"]
    if ring_config is None:
        return None
    ring = KuiperBeltRing(
        point_gm_m3_s2=ring_config["pointGmM3S2"],
        solar_gm_m3_s2=gm_by_body["SUN"],
        radius_m=ring_config["radiusAu"] * AU_M,
        point_count=ring_config["pointCount"],
        obliquity_rad=np.deg2rad(ring_config["obliquityArcsec"] / 3600),
        reference_jd_tdb=ring_config["referenceJdTdb"],
        phase_rad=np.deg2rad(ring_config["phaseDegrees"]),
    )
    ring_sources = (("EARTH", earth_gm), ("MOON", moon_gm), *gm_by_body.items())
    console.info(f"precomputing {ring.point_count} KBO ring trajectories")
    rotation_config = config["lunarRotation"]
    seed_duration = (
        0
        if rotation_config is None or rotation_config["deformation"] is None
        else rotation_config["deformation"]["seedDurationSeconds"]
    )
    duration = config["durationSeconds"]
    warmup = config["integratorOrder"] * config["stepSeconds"]
    lower = min(0, duration, -seed_duration, -config["backwardDurationSeconds"], -warmup if duration > 0 else 0)
    upper = max(0, duration, warmup if duration < 0 else 0)
    return ring.prepare(
        ephemeris,
        tuple(name for name, _ in ring_sources),
        np.array([gm for _, gm in ring_sources]),
        epoch.shifted(lower),
        upper - lower,
        solar_gm_variation,
    )


def build_tides(
    config,
    perturbing_bodies,
    earth_gm,
    moon_gm,
    solar_gm_variation,
):
    """Build the Earth-tide model from the configuration.

    Parameters
    ----------
    config : Mapping
        Program configuration; ``earthTide`` selects the model.
    perturbing_bodies : sequence of GravitatingBody
        Selected external bodies, providing the raiser GMs.
    earth_gm : float
        Earth GM in m³ s⁻².
    moon_gm : float
        Moon GM in m³ s⁻².
    solar_gm_variation : SolarGmVariation or None
        Solar GM drift.

    Returns
    -------
    EarthTideModel or None
        Earth-tide model; ``None`` when ``bodyForceTerms`` does not list ``tide``.

    Raises
    ------
    ValueError
        If a tide raiser is not among the selected bodies.
    """
    if "tide" not in enabled_force_terms(config):
        return None
    tide_config = config["earthTide"]
    external_gm = {body.body_id: body.gravitational_parameter_m3_s2 for body in perturbing_bodies}
    raisers = tuple(body_name(name) for name in tide_config["tideRaisers"])
    raiser_gm = {"MOON": moon_gm}
    raiser_gm.update({name: external_gm[name] for name in raisers if name != "MOON" and name in external_gm})
    missing = [name for name in raisers if name not in raiser_gm]
    if missing:
        raise ValueError(f"Earth-tide raisers are missing from the mass-catalog selection: {missing}")
    orb_values = tuple(tide_config["tauOrbDays"])
    rot_values = tuple(tide_config["tauRotDays"])
    params = EarthTideParameters(
        k20=tide_config["k20"],
        k21=tide_config["k21"],
        k22=tide_config["k22"],
        tau_orb_days=(orb_values[0], orb_values[1], orb_values[2]),
        tau_rot_days=(rot_values[0], rot_values[1], rot_values[2]),
        earth_rotation_rate_rad_s=tide_config["earthRotationRateRadS"],
    )
    return EarthTideModel(
        earth_gm,
        moon_gm,
        earth_radius_m=tide_config["earthRadiusM"],
        tide_parameters=params,
        tide_raisers=raisers,
        tide_raiser_gm=raiser_gm,
        solar_gm_variation=solar_gm_variation,
    )


def build_forces(config, fields, all_body_gm, lunar_degree2_gravity, earth_tide):
    """Build the force models of the terms that ``bodyForceTerms`` lists.

    The models are built in the summation order of ``FORCE_TERMS``, whatever the order of the list. The solar
    spin pole is RA 286.13°, Dec 63.87° (ICRF).

    Parameters
    ----------
    config : Mapping
        Program configuration.
    fields : dict of str to GravityField or None
        Loaded gravity fields keyed by ``"EARTH"`` and ``"MOON"``.
    all_body_gm : dict of str to float
        GM in m³ s⁻² of every body, ring points included.
    lunar_degree2_gravity : LunarDegree2GravityModel or None
        Prescribed delayed lunar degree-2 model.
    earth_tide : EarthTideModel or None
        Earth-tide model; required when ``tide`` is listed.

    Returns
    -------
    force_models : list of ForceModel
        Models of the listed terms.
    enabled_force_names : tuple of str
        The listed terms (see :func:`~lunarops.classes.dynamics.force_models.enabled_force_terms`).

    Raises
    ------
    ValueError
        If ``bodyForceTerms`` is inconsistent (see ``enabled_force_terms``), a solar term lacks the Sun, or
        Moon or Earth figure partners are not selected.
    """
    terms = enabled_force_terms(config)
    ra, dec = np.deg2rad([286.13, 63.87])
    solar_pole = (float(np.cos(dec) * np.cos(ra)), float(np.cos(dec) * np.sin(ra)), float(np.sin(dec)))
    builders = {
        "point_mass": NewtonianPointMassForce,
        "eih_1pn": EihPointMassForce,
        "solar_j2": lambda: _solar_j2_force(config, all_body_gm, solar_pole),
        "figure_earth": lambda: _earth_figure_force(config, fields, all_body_gm),
        "figure_moon": lambda: _moon_figure_force(config, fields, all_body_gm, lunar_degree2_gravity),
        "tide": lambda: EarthTideForce(earth_tide),
        "lense_thirring": lambda: LenseThirringForce(_solar_source(config, all_body_gm, solar_pole, "lense_thirring")),
        "solar_radiation_pressure": lambda: SolarRadiationPressureForce(
            _solar_source(config, all_body_gm, solar_pole, "solar_radiation_pressure"),
            config["solarRelativistic"]["moonEpsilon"],
            config["solarRelativistic"]["earthEpsilon"],
        ),
    }
    return [builders[name]() for name, _ in FORCE_TERMS if name in terms], terms


def _solar_j2_force(config, all_body_gm, solar_pole) -> SolarJ2Force:
    """Return the solar J2 force of ``solarJ2``; raise ValueError without the Sun."""
    if "SUN" not in all_body_gm:
        raise ValueError("solar_j2 requires a perturbing SUN body.")
    solar_j2_field = GravityField(
        make_j2_coefficients(
            gm_m3_s2=all_body_gm["SUN"], radius_m=config["solarRadiusM"], j2=config["solarJ2"], name="Solar J2"
        )
    )
    return SolarJ2Force(solar_j2_field, inertial2fixed_matrix_from_pole(solar_pole))


def _earth_figure_force(config, fields, all_body_gm) -> EarthFigureForce:
    """Return the Earth figure force of the Earth gravity field."""
    earth_j2_time = config["earthJ2TimeVariation"]
    configured = config.get("earthFigurePartners")
    if configured is None:
        # Mirror the Moon figure partners with the Moon in place of the Earth.
        moon_partners = config.get("moonFigurePartners")
        configured = (
            ("MOON", *DEFAULT_PLANETARY_PARTNERS)
            if moon_partners is None
            else ["MOON" if body_name(value) == "EARTH" else value for value in moon_partners]
        )
    return EarthFigureForce(
        fields["EARTH"],
        partners=_figure_partners("Earth", configured, all_body_gm),
        j2_rate_per_year=earth_j2_time["linearPerYear"],
        j2_quadratic_per_year2=earth_j2_time["quadraticPerYear2"],
        reference_jd_tdb=earth_j2_time["referenceJdTdb"],
    )


def _moon_figure_force(config, fields, all_body_gm, lunar_degree2_gravity) -> MoonFigureForce:
    """Return the Moon figure force of the Moon gravity field and the prescribed degree 2, if any."""
    configured = config.get("moonFigurePartners")
    raw = ("EARTH", *DEFAULT_PLANETARY_PARTNERS) if configured is None else configured
    return MoonFigureForce(fields["MOON"], _figure_partners("Moon", raw, all_body_gm), lunar_degree2_gravity)


def _solar_source(config, all_body_gm, solar_pole, term) -> SolarSourceParameters:
    """Return the Sun as the source of a solar term; raise ValueError without the Sun."""
    if "SUN" not in all_body_gm:
        raise ValueError(f"{term} requires a perturbing SUN body.")
    return SolarSourceParameters(all_body_gm["SUN"], config["solarRadiusM"], spin_axis=solar_pole)


def _resolve_parameters(config, context, parameters) -> LunarDynamicsParameters | None:
    """Return the parameters of a build: the given ones, else those of ``inputFileLunarDynamicsParameters``.

    Raises
    ------
    ValueError
        If initial values are listed for another epoch than the initial epoch of ``config``.
    """
    if parameters is None:
        path = config.get("inputFileLunarDynamicsParameters")
        if path is None:
            return None
        parameters = read_lunar_dynamics_parameters(context.resolve_path(path))
    if any(name in parameters for name in INITIAL_VALUE_NAMES) and not parameters.is_at(
        config["initialTdbJd1"], config["initialTdbJd2"]
    ):
        raise ValueError(
            "Lunar dynamics parameters hold initial values at JD(TDB) "
            f"{parameters.epoch_tdb_jd1} + {parameters.epoch_tdb_jd2}, but the propagation starts at "
            f"{config['initialTdbJd1']} + {config['initialTdbJd2']}"
        )
    return parameters


def build_lunar_propagation_problem(
    config, context, parameters: LunarDynamicsParameters | None = None, *, kuiper_belt_ring=None
) -> LunarPropagationProblem:
    """Build a complete lunar propagation problem.

    Initial values and dynamical parameters given by ``parameters`` (or by
    the file ``inputFileLunarDynamicsParameters``) replace the values of the
    ephemeris, the configuration, the mass catalog and the Moon gravity field;
    see :mod:`~lunarops.classes.dynamics.parameters`. Quantities that are not
    listed keep those values.

    Parameters
    ----------
    config : Mapping
        Validated ``LunarOrbitPropagation`` configuration.
    context : object
        Program context that resolves input and output paths and creates the
        configured ephemeris.
    parameters : LunarDynamicsParameters or None, optional
        Initial values and dynamical parameters; they take precedence over
        ``inputFileLunarDynamicsParameters``.
    kuiper_belt_ring : KuiperBeltRing or None, optional
        Prepropagated ring of an earlier build of the same configuration, used
        instead of prepropagating it again.

    Returns
    -------
    LunarPropagationProblem
        Dynamics, integrator, initial state and output paths.

    Raises
    ------
    ValueError
        If the mass-catalog selection has no perturbing body, a gravity
        field, force term or rotational source is inconsistent with the
        selected bodies, a sub-model rejects its configuration, or a listed
        parameter belongs to a model that is not configured or holds initial
        values of another epoch.
    """
    parameters = _resolve_parameters(config, context, parameters)
    config = apply_to_config(config, parameters)
    mass_catalog = read_mass_catalog(context.resolve_path(config["inputFileMassCatalog"]))
    earth_gm, moon_gm = earth_moon_gm(
        parameters, mass_catalog.require("EARTH").gm_m3_s2, mass_catalog.require("MOON").gm_m3_s2
    )
    perturbing_bodies = _perturbing_bodies(config, mass_catalog)
    gm_by_body = {body.body_id: body.gravitational_parameter_m3_s2 for body in perturbing_bodies}
    solar_gm_config = config["solarGmVariation"]
    solar_gm_variation = (
        None
        if solar_gm_config is None
        else SolarGmVariation(solar_gm_config["fractionalRatePerYear"], solar_gm_config["referenceJdTdb"])
    )
    ephemeris = context.create_class("ephemerides", config["ephemerides"], cache=True)
    epoch = Epoch(config["initialTdbJd1"], config["initialTdbJd2"], TimeScale.TDB)
    ring = kuiper_belt_ring
    if ring is None:
        ring = _build_ring(
            config,
            gm_by_body,
            ephemeris,
            epoch,
            earth_gm,
            moon_gm,
            solar_gm_variation,
        )
    initial_relative = initial_orbit(parameters, ephemeris.relative_state_matrix("MOON", "EARTH", epoch))
    pole_model = config["earthPoleModel"]
    if pole_model == "ephemeris":
        pole_model = config["earthDynamicsFrame"]
    earth_fixed2inertial_matrix_provider = EARTH_FIGURE_ORIENTATIONS[pole_model]
    fields = (
        _load_configured_gravity_fields(
            config["gravityFields"],
            context,
            {"EARTH": earth_gm, "MOON": moon_gm},
            parameters,
        )
        if config.get("gravityFields")
        else None
    )
    moon_field = fields.get("MOON") if fields else None
    check_applicable(parameters, has_moon_field=moon_field is not None)
    earth_tide = build_tides(
        config,
        perturbing_bodies,
        earth_gm,
        moon_gm,
        solar_gm_variation,
    )

    lunar_degree2_gravity = _lunar_degree2_gravity(config, fields, ephemeris, earth_gm, moon_gm)
    all_body_gm = {"EARTH": earth_gm, "MOON": moon_gm}
    all_body_gm.update(
        (body.body_id, body.gravitational_parameter_m3_s2)
        for body in (*perturbing_bodies, *(() if ring is None else ring.bodies))
    )
    force_models, enabled_force_names = build_forces(config, fields, all_body_gm, lunar_degree2_gravity, earth_tide)
    rotation, rotation_initial = build_rotation(config, fields, moon_gm, earth_gm)
    if rotation is not None:
        _check_rotational_sources(config, rotation, all_body_gm)
    dynamics = LunarDynamics(
        earth_body=GravitatingBody("EARTH", earth_gm),
        moon_body=GravitatingBody("MOON", moon_gm),
        perturbing_bodies=perturbing_bodies,
        ephemeris=ephemeris,
        force_group=LunarForceGroup(force_models, enabled_force_names),
        earth_fixed2inertial_matrix_provider=OrientationProvider(earth_fixed2inertial_matrix_provider),
        moon_fixed2inertial_matrix_provider=EphemerisLunarAxes() if fields and rotation is None else None,
        rotation=rotation,
        kuiper_belt_ring=ring,
        solar_gm_variation=solar_gm_variation,
    )
    config = dict(config)
    config["resolvedForceInventory"] = {
        "active": list(enabled_force_names),
        "source": "explicit",
    }
    config["resolvedGravityFields"] = {
        body: field.coefficients.source_metadata for body, field in (fields or {}).items()
    }
    integrator = AdamsBashforthMoultonIntegrator(_integrator_settings(config))
    initial_vector = initial_relative.copy()
    if rotation_initial is not None:
        initial_vector = np.concatenate((initial_vector, rotation_initial))
    return LunarPropagationProblem(
        MappingProxyType(dict(config)),
        dynamics,
        integrator,
        epoch,
        initial_vector,
        config["durationSeconds"],
        config["backwardDurationSeconds"],
        tuple(
            context.resolve_path(config[key])
            for key in (
                "outputFileOrbit",
                "outputFileAccelerationDiagnostics",
                "outputFileMetadata",
            )
        ),
        EphemerisHistory(ephemeris, earth_gm, moon_gm)
        if earth_tide is not None or lunar_degree2_gravity is not None
        else None,
        tuple(name for name, term in SOLAR_FORCE_TERMS if term in enabled_force_names),
        None if rotation is None else context.resolve_path(config["outputFileRotation"]),
        None
        if rotation is None or rotation.deformation is None
        else config["lunarRotation"]["deformation"]["seedDurationSeconds"],
        nominal_parameters(
            config,
            epoch_tdb_jd1=config["initialTdbJd1"],
            epoch_tdb_jd2=config["initialTdbJd2"],
            initial_orbit_state=initial_relative,
            earth_gm_m3_s2=earth_gm,
            moon_gm_m3_s2=moon_gm,
            moon_coefficients=None if moon_field is None else moon_field.coefficients,
        ),
    )


def _perturbing_bodies(config, mass_catalog) -> tuple[GravitatingBody, ...]:
    """Return the external bodies of the mass-catalog selection; raise ValueError when there is none."""
    selected_external = mass_catalog.select(
        groups=config["externalBodyGroups"],
        body_ids=config["externalBodyIds"],
    )
    selected_external = tuple(body for body in selected_external if body.body_id not in {"EARTH", "MOON"})
    if not selected_external:
        raise ValueError("The mass catalog selection must contain at least one perturbing body")
    return tuple(GravitatingBody(body.body_id, body.gm_m3_s2) for body in selected_external)


def _lunar_degree2_gravity(config, fields, ephemeris, earth_gm, moon_gm) -> LunarDegree2GravityModel | None:
    """Return the prescribed delayed lunar degree 2 of ``lunarDegree2Gravity``; ``None`` when not configured."""
    if config.get("lunarDegree2Gravity") is None:
        return None
    _moon_field(fields, "lunarDegree2Gravity")
    return LunarDegree2GravityModel(
        earth_gm,
        moon_gm,
        ephemeris.lunar_orientation.pa_to_lcrs_matrix,
        angular_velocity_provider=ephemeris.lunar_orientation.angular_velocity_rad_s,
        parameters=_degree2_parameters(config["lunarDegree2Gravity"]),
    )


def _check_rotational_sources(config, rotation, all_body_gm) -> None:
    """Raise ValueError if a torque source, or the Sun of the geodetic precession, is not a selected body."""
    required = {body_name(name) for name in config["lunarRotation"]["sources"]}
    if rotation.geodetic_precession:
        required.add("SUN")
    if missing := required - set(all_body_gm):
        raise ValueError(f"Rotational sources are missing from selected bodies: {sorted(missing)}")


def _integrator_settings(config) -> IntegratorSettings:
    """Return the integrator settings of the configuration."""
    return IntegratorSettings(
        step_s=config["stepSeconds"],
        order=config["integratorOrder"],
        corrector_iterations=config["correctorIterations"],
        history_interpolation_order=config["historyInterpolationOrder"],
        trajectory_interpolation_order=config["trajectoryInterpolationOrder"],
        startup_step_s=config["startupStepSeconds"],
    )


def _degree2_parameters(config):
    return LunarDegree2GravityParameters(
        j2_undistorted=config["j2Undistorted"],
        love_k2=config["loveK2"],
        tidal_delay_days=config["tidalDelayDays"],
        radius_m=config["radiusM"],
        mean_motion_rad_s=config["meanMotionRadS"],
        c21_static=config["c21Static"],
        s21_static=config["s21Static"],
        s22_static=config["s22Static"],
        include_tidal_deformation=config["includeTidalDeformation"],
        include_rotational_deformation=config["includeRotationalDeformation"],
        beta=config["beta"],
        gamma=config["gamma"],
    )
