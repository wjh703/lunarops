"""Assembly of a lunar propagation problem from a program configuration.

:func:`build_lunar_propagation_problem` reads the mass catalog, gravity
fields and ephemeris named in the configuration and builds the force models,
Earth tide, Kuiper-belt ring, lunar rotation and integrator of one
propagation. The configuration is the validated ``LunarOrbitPropagation``
mapping with camelCase keys.

Attributes
----------
DEFAULT_PLANETARY_PARTNERS : tuple of str
    Bodies besides the Earth that interact with the Moon figure when
    ``moonFigurePartners`` is not configured.
SOLAR_FORCE_TERMS : tuple of tuple of str
    Diagnostic name and configuration key of each optional solar term.
DIAGNOSTIC_FORCE_GROUPS : tuple
    Diagnostic group names and the force names summed into each group.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

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
    EarthFigureForce,
    EarthTideForce,
    EihPointMassForce,
    ForceModel,
    LenseThirringForce,
    LunarForceGroup,
    MoonFigureForce,
    NewtonianPointMassForce,
    SolarJ2Force,
    SolarRadiationPressureForce,
)
from .forces import EarthTideModel, EarthTideParameters, SolarSourceParameters
from .gravity import load_gravity_field
from .history import EphemerisHistory
from .integrators import AdamsBashforthMoultonIntegrator, IntegratorSettings
from .lunar_dynamics import LunarDynamics

DEFAULT_PLANETARY_PARTNERS = (
    "SUN", "MERCURY BARYCENTER", "VENUS BARYCENTER", "MARS BARYCENTER", "JUPITER BARYCENTER", "SATURN BARYCENTER",
)


SOLAR_FORCE_TERMS = (
    ("solar_lense_thirring", "includeLenseThirring"), ("solar_radiation_pressure", "includeRadiationPressure"),
)


DIAGNOSTIC_FORCE_GROUPS = (
    ("newtonian", ("point_mass",)),
    ("post_newtonian", ("eih_1pn",)),
    ("figure", ("solar_j2", "figure_earth", "figure_moon")),
    ("tide", ("tide",)),
    ("srp", ("solar_radiation_pressure",)),
    ("lt", ("lense_thirring",)),
)


def _load_configured_gravity_fields(config, context, gm_by_body):
    fields = {}
    for body_id in ("EARTH", "MOON"):
        entry = config.get(body_id.lower())
        if entry is None:
            continue
        fields[body_id] = GravityField(
            load_gravity_field(
                context.resolve_path(entry["path"]),
                max_degree=entry.get("maxDegree"),
                gm_m3_s2=gm_by_body[body_id],
                reference_radius_m=entry.get("radiusM"),
                name=entry.get("name", body_id),
                allow_missing_coefficients=entry["allowMissingCoefficients"],
            )
        )
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
        Active optional solar diagnostic terms.
    rotation_output : Path or None
        Rotation output file; ``None`` without integrated rotation.
    seed_duration_s : float or None
        Rigid seed-arc length in s for delayed deformation, else ``None``.
    """

    config: Mapping
    dynamics: LunarDynamics
    integrator: AdamsBashforthMoultonIntegrator
    initial_epoch: Epoch
    initial_vector: np.ndarray
    duration_s: float
    output_paths: tuple[Path, Path, Path]
    external_history: StateHistoryProvider | None
    solar_force_terms: tuple[str, ...]
    rotation_output: Path | None
    seed_duration_s: float | None


def build_rotation(config, fields, moon_mu, earth_mu):
    """Build the integrated lunar rotation from the configuration.

    Parameters
    ----------
    config : Mapping
        Program configuration; ``lunarRotation`` selects the model.
    fields : dict of str to GravityField or None
        Loaded gravity fields keyed by ``"EARTH"`` and ``"MOON"``.
    moon_mu : float
        Moon GM in m³ s⁻².
    earth_mu : float
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
    core_cfg = cfg["core"]
    core = (
        None
        if core_cfg is None
        else LunarCoreParameters(
            polar_moment_fraction=core_cfg["polarMomentFraction"],
            flattening=core_cfg["flattening"],
            viscosity_per_s=core_cfg["viscosityPerS"],
        )
    )
    initial = LunarRotationState(
        cfg["initialQuaternion"],
        cfg["initialAngularVelocityRadS"],
        cfg["initialCoreAngularVelocityRadS"],
    )
    if (core is None) != (initial.core_angular_velocity_pa_rad_s is None):
        raise ValueError("Core model and core initial angular velocity must be configured together")
    inertia = np.diag(cfg["inertiaMoments"])
    deformation = None
    if cfg["deformation"] is not None:
        p = cfg["deformation"]
        if fields is None or "MOON" not in fields:
            raise ValueError("Integrated deformation requires the shared Moon gravity field")
        if not np.isclose(fields["MOON"].radius_m, cfg["radiusM"], rtol=0, atol=1e-9):
            raise ValueError("Deformation and Moon field must use the same radius")
        deformation = LunarDeformation(
            LunarDeformationParameters(
                p["loveK2"],
                p["delayDays"] * 86400,
                p["meanMotionRadS"],
                p["includeTidal"],
                p["includeRotational"],
                c21_static=p["c21Static"],
                s21_static=p["s21Static"],
                s22_static=p["s22Static"],
            ),
            inertia,
            cfg["radiusM"],
            earth_mu,
            moon_mu,
            fields["MOON"],
        )
    torques: list[LunarFigureTorque | PointMassGravityTorque | EarthFigureTorque] = []
    sources = tuple(body_name(name) for name in cfg["sources"])
    if len(sources) != len(set(sources)) or "MOON" in sources:
        raise ValueError("Lunar torque sources must be unique and external to the Moon")
    if cfg["gravityTorque"] == "figure":
        if fields is None or "MOON" not in fields:
            raise ValueError("Figure torques require a Moon gravity field")
        moon = fields["MOON"]
        if not np.isclose(moon.radius_m, cfg["radiusM"], rtol=0, atol=1e-9):
            raise ValueError("Moon field and rotational inertia must use the same reference radius")
        if not np.isclose(moon.gm_m3_s2, moon_mu, rtol=1e-14):
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
    rotation = LunarRotationDynamics(
        inertia,
        cfg["radiusM"],
        LunarTorqueGroup(torques),
        core=core,
        geodetic_precession=cfg["geodeticPrecession"],
        deformation=deformation,
    )
    return rotation, initial.vector()


def _build_ring(
    config,
    gm_by_body,
    ephemeris,
    epoch,
    earth_gravitational_parameter_m3_s2,
    moon_gravitational_parameter_m3_s2,
    solar_gm_variation,
):
    """Build and prepropagate the Kuiper-belt ring, or return ``None``.

    The prepropagated interval covers the propagation, the rigid seed arc
    and the integrator warm-up on either side of the initial epoch.
    """
    duration = config["durationSeconds"]
    ring_config = config["kuiperBeltRing"]
    ring = (
        None
        if ring_config is None
        else KuiperBeltRing(
            point_gm_m3_s2=ring_config["pointGmM3S2"],
            solar_gm_m3_s2=gm_by_body["SUN"],
            radius_m=ring_config["radiusAu"] * AU_M,
            point_count=ring_config["pointCount"],
            obliquity_rad=np.deg2rad(ring_config["obliquityArcsec"] / 3600),
            reference_jd_tdb=ring_config["referenceJdTdb"],
            phase_rad=np.deg2rad(ring_config["phaseDegrees"]),
        )
    )
    if ring is not None:
        ring_sources = (
            ("EARTH", earth_gravitational_parameter_m3_s2),
            ("MOON", moon_gravitational_parameter_m3_s2),
        ) + tuple(gm_by_body.items())
        print(f"[LunarOrbitPropagation] Precomputing {ring.point_count} independent KBO ring trajectories", flush=True)
        rotation_config = config["lunarRotation"]
        seed_duration = (
            0
            if rotation_config is None or rotation_config["deformation"] is None
            else rotation_config["deformation"]["seedDurationSeconds"]
        )
        warmup = config["integratorOrder"] * config["stepSeconds"]
        lower = min(0, duration, -seed_duration, -warmup if duration > 0 else 0)
        upper = max(0, duration, warmup if duration < 0 else 0)
        ring = ring.prepare(
            ephemeris,
            tuple(name for name, _ in ring_sources),
            np.array([gm for _, gm in ring_sources]),
            epoch.shifted(lower),
            upper - lower,
            solar_gm_variation,
        )
    return ring


def build_tides(
    config,
    perturbing_bodies,
    earth_gravitational_parameter_m3_s2,
    moon_gravitational_parameter_m3_s2,
    solar_gm_variation,
):
    """Build the Earth-tide model from the configuration.

    Parameters
    ----------
    config : Mapping
        Program configuration; ``earthTide`` selects the model.
    perturbing_bodies : sequence of GravitatingBody
        Selected external bodies, providing the raiser GMs.
    earth_gravitational_parameter_m3_s2 : float
        Earth GM in m³ s⁻².
    moon_gravitational_parameter_m3_s2 : float
        Moon GM in m³ s⁻².
    solar_gm_variation : SolarGmVariation or None
        Solar GM drift.

    Returns
    -------
    EarthTideModel or None
        Earth-tide model; ``None`` when ``earthTide`` is not configured.

    Raises
    ------
    ValueError
        If a tide raiser is not among the selected bodies.
    """
    earth_tide = None
    if config.get("earthTide") is not None:
        tide_config = config["earthTide"]
        external_gm = {body.body_id: body.gravitational_parameter_m3_s2 for body in perturbing_bodies}
        raisers = tuple(body_name(name) for name in tide_config["tideRaisers"])
        raiser_gm = {"MOON": moon_gravitational_parameter_m3_s2}
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
        earth_tide = EarthTideModel(
            earth_gravitational_parameter_m3_s2,
            moon_gravitational_parameter_m3_s2,
            earth_radius_m=tide_config["earthRadiusM"],
            tide_parameters=params,
            tide_raisers=raisers,
            tide_raiser_gm=raiser_gm,
            solar_gm_variation=solar_gm_variation,
        )

    return earth_tide


def build_forces(config, fields, all_body_gm, lunar_degree2_gravity, earth_tide):
    """Build the force models and the active force names.

    The Newtonian point-mass force is always built; the other terms follow
    the configuration. ``bodyForceTerms`` selects the active subset, all
    built terms being active by default. The solar spin pole is
    RA 286.13°, Dec 63.87° (ICRF).

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
        Earth-tide model.

    Returns
    -------
    force_models : list of ForceModel
        Built force models.
    enabled_force_names : tuple of str
        Active force names.
    solar_relativistic : Mapping or None
        The ``solarRelativistic`` configuration block.

    Raises
    ------
    ValueError
        If a solar term lacks the Sun, Moon figure partners are not
        selected, ``bodyForceTerms`` names an unavailable term, or
        ``eih_1pn`` is enabled without ``point_mass``.
    """
    force_models: list[ForceModel] = [NewtonianPointMassForce()]
    if config["includeEih"]:
        force_models.append(EihPointMassForce())
    ra, dec = np.deg2rad([286.13, 63.87])
    solar_pole = (float(np.cos(dec) * np.cos(ra)), float(np.cos(dec) * np.sin(ra)), float(np.sin(dec)))
    if config.get("solarJ2") is not None:
        if "SUN" not in all_body_gm:
            raise ValueError("solarJ2 requires a perturbing SUN body.")
        solar_j2_field = GravityField(
            make_j2_coefficients(
                gm_m3_s2=all_body_gm["SUN"], radius_m=config["solarRadiusM"], j2=config["solarJ2"], name="Solar J2"
            )
        )
        force_models.append(SolarJ2Force(solar_j2_field, inertial2fixed_matrix_from_pole(solar_pole)))
    if fields:
        if "EARTH" in fields:
            earth_j2_time = config["earthJ2TimeVariation"]
            force_models.append(
                EarthFigureForce(
                    fields["EARTH"],
                    j2_rate_per_year=earth_j2_time["linearPerYear"],
                    j2_quadratic_per_year2=earth_j2_time["quadraticPerYear2"],
                    reference_jd_tdb=earth_j2_time["referenceJdTdb"],
                )
            )
        if "MOON" in fields:
            configured = config.get("moonFigurePartners")
            raw = ("EARTH", *DEFAULT_PLANETARY_PARTNERS) if configured is None else configured
            partners = tuple(body_name(value) for value in raw)
            missing = sorted(set(partners) - set(all_body_gm))
            if missing:
                raise ValueError(f"Moon figure partners are missing from the selected bodies: {missing}")
            force_models.append(MoonFigureForce(fields["MOON"], partners, lunar_degree2_gravity))
    if earth_tide is not None:
        force_models.append(EarthTideForce(earth_tide))
    solar_relativistic = config.get("solarRelativistic")
    if solar_relativistic is not None:
        if "SUN" not in all_body_gm:
            raise ValueError("solarRelativistic requires a perturbing SUN body.")
        solar_params = SolarSourceParameters(all_body_gm["SUN"], config["solarRadiusM"], spin_axis=solar_pole)
        if solar_relativistic["includeLenseThirring"]:
            force_models.append(LenseThirringForce(solar_params))
        if solar_relativistic["includeRadiationPressure"]:
            force_models.append(SolarRadiationPressureForce(solar_params, solar_relativistic["moonEpsilon"]))
    default_terms = ["point_mass"]
    # Earth and all external bodies are prescribed by the ephemeris.  Only
    # the modeled Moon acceleration is needed by the relative-state RHS.
    for model in force_models[1:]:
        if model.name not in default_terms:
            default_terms.append(model.name)
    enabled_force_names = tuple(config.get("bodyForceTerms", default_terms))
    configured_moon_terms = set(enabled_force_names)
    unknown_force_terms = configured_moon_terms - {model.name for model in force_models}
    if unknown_force_terms:
        raise ValueError(f"bodyForceTerms contains unavailable force terms: {sorted(unknown_force_terms)}")
    if "eih_1pn" in configured_moon_terms and "point_mass" not in configured_moon_terms:
        raise ValueError("bodyForceTerms requires point_mass when eih_1pn is enabled")
    return force_models, enabled_force_names, solar_relativistic


def build_lunar_propagation_problem(config, context) -> LunarPropagationProblem:
    """Build a complete lunar propagation problem.

    Parameters
    ----------
    config : Mapping
        Validated ``LunarOrbitPropagation`` configuration.
    context : object
        Program context that resolves input and output paths and creates the
        configured ephemeris.

    Returns
    -------
    LunarPropagationProblem
        Dynamics, integrator, initial state and output paths.

    Raises
    ------
    ValueError
        If the mass-catalog selection has no perturbing body, a gravity
        field, force term or rotational source is inconsistent with the
        selected bodies, or a sub-model rejects its configuration.
    """
    mass_catalog = read_mass_catalog(context.resolve_path(config["inputFileMassCatalog"]))
    earth_gravitational_parameter_m3_s2 = mass_catalog.require("EARTH").gm_m3_s2
    moon_gravitational_parameter_m3_s2 = mass_catalog.require("MOON").gm_m3_s2
    selected_external = mass_catalog.select(
        groups=config["externalBodyGroups"],
        body_ids=config["externalBodyIds"],
    )
    selected_external = tuple(body for body in selected_external if body.body_id not in {"EARTH", "MOON"})
    if not selected_external:
        raise ValueError("The mass catalog selection must contain at least one perturbing body")
    perturbing_bodies = tuple(GravitatingBody(body.body_id, body.gm_m3_s2) for body in selected_external)
    gm_by_body = {body.body_id: body.gravitational_parameter_m3_s2 for body in perturbing_bodies}
    solar_gm_config = config["solarGmVariation"]
    solar_gm_variation = (
        None
        if solar_gm_config is None
        else SolarGmVariation(solar_gm_config["fractionalRatePerYear"], solar_gm_config["referenceJdTdb"])
    )
    ephemeris = context.create_class("ephemerides", config["ephemerides"], cache=True)
    epoch = Epoch(config["initialTdbJd1"], config["initialTdbJd2"], TimeScale.TDB)
    ring = _build_ring(
        config,
        gm_by_body,
        ephemeris,
        epoch,
        earth_gravitational_parameter_m3_s2,
        moon_gravitational_parameter_m3_s2,
        solar_gm_variation,
    )
    initial_states = ephemeris.body_state_matrix_bcrs(("EARTH", "MOON"), epoch)
    initial_earth = initial_states[0]
    initial_relative = initial_states[1] - initial_earth
    pole_model = config["earthPoleModel"]
    if pole_model == "ephemeris":
        pole_model = config["earthDynamicsFrame"]
    earth_fixed2inertial_matrix_provider = EARTH_FIGURE_ORIENTATIONS[pole_model]
    fields = (
        _load_configured_gravity_fields(
            config["gravityFields"],
            context,
            {"EARTH": earth_gravitational_parameter_m3_s2, "MOON": moon_gravitational_parameter_m3_s2},
        )
        if config.get("gravityFields")
        else None
    )
    earth_tide = build_tides(
        config,
        perturbing_bodies,
        earth_gravitational_parameter_m3_s2,
        moon_gravitational_parameter_m3_s2,
        solar_gm_variation,
    )

    def earth_minus_moon_position_provider(provider_epoch):
        states = ephemeris.body_state_matrix_bcrs(("EARTH", "MOON"), provider_epoch)
        return states[0, :3] - states[1, :3]

    lunar_degree2_gravity = None
    if config.get("lunarDegree2Gravity") is not None:
        if fields is None or "MOON" not in fields:
            raise ValueError("lunarDegree2Gravity requires a configured Moon gravity field.")
        lunar_degree2_gravity = LunarDegree2GravityModel(
            earth_gravitational_parameter_m3_s2,
            moon_gravitational_parameter_m3_s2,
            earth_minus_moon_position_provider,
            ephemeris.lunar_orientation.pa_to_lcrs_matrix,
            angular_velocity_provider=ephemeris.lunar_orientation.angular_velocity_rad_s,
            parameters=_degree2_parameters(config["lunarDegree2Gravity"]),
        )
    all_body_gm = {"EARTH": earth_gravitational_parameter_m3_s2, "MOON": moon_gravitational_parameter_m3_s2}
    all_body_gm.update((body.body_id, body.gravitational_parameter_m3_s2)
                       for body in (*perturbing_bodies, *(() if ring is None else ring.bodies)))
    force_models, enabled_force_names, solar_relativistic = build_forces(
        config, fields, all_body_gm, lunar_degree2_gravity, earth_tide
    )
    rotation, rotation_initial = build_rotation(
        config, fields, moon_gravitational_parameter_m3_s2, earth_gravitational_parameter_m3_s2
    )
    if rotation is not None:
        required = {body_name(name) for name in config["lunarRotation"]["sources"]}
        if rotation.geodetic_precession:
            required.add("SUN")
        if missing := required - set(all_body_gm):
            raise ValueError(f"Rotational sources are missing from selected bodies: {sorted(missing)}")
    dynamics = LunarDynamics(
        earth_body=GravitatingBody("EARTH", earth_gravitational_parameter_m3_s2),
        moon_body=GravitatingBody("MOON", moon_gravitational_parameter_m3_s2),
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
        "source": "explicit" if "bodyForceTerms" in config else "default",
    }
    config["resolvedGravityFields"] = {
        body: field.coefficients.source_metadata for body, field in (fields or {}).items()
    }
    integration_initial = initial_relative
    integrator = AdamsBashforthMoultonIntegrator(
        IntegratorSettings(
            step_s=config["stepSeconds"],
            order=config["integratorOrder"],
            corrector_iterations=config["correctorIterations"],
            history_interpolation_order=config["historyInterpolationOrder"],
            trajectory_interpolation_order=config["trajectoryInterpolationOrder"],
            startup_step_s=config["startupStepSeconds"],
        )
    )
    initial_vector = integration_initial.copy()
    if rotation_initial is not None:
        initial_vector = np.concatenate((initial_vector, rotation_initial))
    return LunarPropagationProblem(
        MappingProxyType(dict(config)),
        dynamics,
        integrator,
        epoch,
        initial_vector,
        config["durationSeconds"],
        tuple(
            context.resolve_path(config[key])
            for key in (
                "outputFileOrbit",
                "outputFileAccelerationDiagnostics",
                "outputFileMetadata",
            )
        ),
        EphemerisHistory(ephemeris)
        if earth_tide is not None or lunar_degree2_gravity is not None
        else None,
        tuple(name for name, key in SOLAR_FORCE_TERMS if solar_relativistic is not None and solar_relativistic[key]),
        None if rotation is None else context.resolve_path(config["outputFileRotation"]),
        None
        if rotation is None or rotation.deformation is None
        else config["lunarRotation"]["deformation"]["seedDurationSeconds"],
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


