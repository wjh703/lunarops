"""Earth/Moon propagation and reference-ephemeris diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    BarycentricEarthMoonState,
    EarthTideForce,
    EarthTideModel,
    EarthTideParameters,
    EihPointMassForce,
    FigureForce,
    ForceModel,
    GravitatingBody,
    IntegratorSettings,
    LenseThirringForce,
    LunarDegree2GravityCorrectionForce,
    LunarDegree2GravityModel,
    LunarDegree2GravityParameters,
    LunarDynamics,
    LunarForceGroup,
    MoonRelativeState,
    NewtonianPointMassForce,
    SolarJ2Force,
    SolarRadiationPressureForce,
    SolarSourceParameters,
    TimeVaryingEarthFigureForce,
    de430_earth_fixed2inertial_matrix,
    de440_earth_fixed2inertial_matrix,
    load_gravity_field,
)
from lunarops.classes.dynamics.deformation import LunarDeformation, LunarDeformationParameters
from lunarops.classes.dynamics.earth_iau2006_orientation import iau2006_earth_fixed2inertial_matrix
from lunarops.classes.dynamics.gravity import GravityField, make_j2_coefficients
from lunarops.classes.dynamics.orientation import OrientationProvider, inertial2fixed_matrix_from_pole
from lunarops.classes.dynamics.rotation import LunarCoreParameters, LunarRotationDynamics, LunarRotationState
from lunarops.classes.dynamics.solar_environment import AU_M, KuiperBeltRing, SolarGmVariation
from lunarops.classes.dynamics.torques import (
    EarthFigureTorque,
    LunarFigureTorque,
    LunarTorqueGroup,
    PointMassGravityTorque,
)
from lunarops.classes.ephemerides import BodyState, body_state_matrix
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.observation_factory import ensure_registered
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.lunar_models import LUNAR_ORBIT_FIELDS
from lunarops.fileio.mass_catalog import read_mass_catalog
from lunarops.fileio.numeric_table import write_numeric_table
from lunarops.fileio.structured_reports import write_lunar_orbit_metadata
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program


def _load_configured_gravity_fields(config, context, gm_by_body):
    """Build the Earth/Moon coefficient map declared by the program config."""
    if not isinstance(config, dict):
        raise TypeError("gravityFields must be a mapping")
    if unknown := set(config) - {"earth", "moon"}:
        raise ValueError(f"gravityFields uses canonical lowercase body mappings only; unknown keys: {sorted(unknown)}")
    fields = {}
    for body_id in ("EARTH", "MOON"):
        entry = config.get(body_id.lower())
        if entry is None:
            continue
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"gravityFields.{body_id.lower()} requires path")
        if "gmM3S2" in entry:
            raise ValueError(f"gravityFields.{body_id.lower()}.gmM3S2 was removed; GM comes from the mass catalog.")
        fields[body_id] = GravityField(
            load_gravity_field(
                context.resolve_path(entry["path"]),
                file_format=entry.get("format", "icgem"),
                max_degree=entry.get("maxDegree"),
                gm_m3_s2=gm_by_body[body_id],
                reference_radius_m=entry.get("radiusM"),
                name=entry.get("name", body_id),
                allow_missing_coefficients=entry.get("allowMissingCoefficients", False),
            )
        )
    if not fields:
        raise ValueError("gravityFields must define at least earth or moon")
    return fields


@dataclass(frozen=True, slots=True)
class LunarPropagationProblem:
    config: Mapping
    dynamics: LunarDynamics
    integrator: AdamsBashforthMoultonIntegrator
    initial_epoch: Epoch
    initial_vector: np.ndarray
    duration_s: float
    output_paths: tuple[Path, Path, Path]
    external_history: Callable | None
    solar_force_terms: tuple[str, ...]
    rotation_output: Path | None
    seed_duration_s: float | None


def _build_rotation(config, fields, moon_mu, earth_mu):
    cfg = config["lunarRotation"]
    if cfg is None:
        return None, None
    if config["lunarDegree2Gravity"] is not None:
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
            pressure_spin=core_cfg["pressureSpin"],
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
    inertia.setflags(write=False)
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
    torques = []
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
        j2_time = config["earthJ2TimeVariation"] or {}
        torques.append(
            EarthFigureTorque(
                inertia,
                earth.radius_m,
                -np.sqrt(5) * earth.coefficients.coeffs[0, 2, 0],
                j2_time.get("linearPerYear", 0.0),
                j2_time.get("quadraticPerYear2", 0.0),
                j2_time.get("referenceJdTdb", 2451545.0),
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


def build_lunar_propagation_problem(config, context) -> LunarPropagationProblem:
    try:
        __import__("scipy")
    except ImportError as exc:
        raise ImportError("LunarOrbitPropagation requires SciPy; install lunarops[dynamics].") from exc
    ensure_registered()
    duration = config["durationSeconds"]
    if duration == 0:
        raise ValueError("durationSeconds must be nonzero.")
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
    ephemeris = context.create_class("ephemerides", config["ephemerides"], cache=True)
    epoch = Epoch(config["initialTdbJd1"], config["initialTdbJd2"], TimeScale.TDB)
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
        ring.prepare(
            ephemeris,
            tuple(name for name, _ in ring_sources),
            np.array([gm for _, gm in ring_sources]),
            epoch.shifted(lower),
            upper - lower,
            solar_gm_variation,
        )
    initial_states = body_state_matrix(ephemeris, ("EARTH", "MOON"), epoch)
    initial = BarycentricEarthMoonState(
        epoch,
        BodyState(initial_states[0, :3], initial_states[0, 3:]),
        BodyState(initial_states[1, :3], initial_states[1, 3:]),
    )
    earth_fixed2inertial_matrix_provider: Callable[[Epoch], np.ndarray]
    pole_model = config["earthPoleModel"]
    if pole_model == "ephemeris":
        pole_model = config["earthDynamicsFrame"]
    earth_fixed2inertial_matrix_provider = {
        "de430": de430_earth_fixed2inertial_matrix,
        "de440": de440_earth_fixed2inertial_matrix,
        "iau2006_2000a": iau2006_earth_fixed2inertial_matrix,
    }[pole_model]
    earth_inertial2fixed_matrix_provider = lambda epoch: earth_fixed2inertial_matrix_provider(epoch).T
    fields = (
        _load_configured_gravity_fields(
            config["gravityFields"],
            context,
            {"EARTH": earth_gravitational_parameter_m3_s2, "MOON": moon_gravitational_parameter_m3_s2},
        )
        if config.get("gravityFields")
        else None
    )
    earth_tide = None
    if config.get("earthTide") is not None:
        tide_config = config.get("earthTide") or {}
        external_gm = {body.body_id: body.gravitational_parameter_m3_s2 for body in perturbing_bodies}
        raisers = tuple(body_name(name) for name in tide_config.get("tideRaisers", ("MOON", "SUN")))
        raiser_gm = {"MOON": moon_gravitational_parameter_m3_s2}
        raiser_gm.update({name: external_gm[name] for name in raisers if name != "MOON" and name in external_gm})
        missing = [name for name in raisers if name not in raiser_gm]
        if missing:
            raise ValueError(f"Earth-tide raisers are missing from the mass-catalog selection: {missing}")
        defaults = EarthTideParameters()
        orb_values = tuple(float(value) for value in tide_config.get("tauOrbDays", defaults.tau_orb_days))
        rot_values = tuple(float(value) for value in tide_config.get("tauRotDays", defaults.tau_rot_days))
        if len(orb_values) != 3 or len(rot_values) != 3:
            raise ValueError("Earth-tide tauOrbDays and tauRotDays must contain three values.")
        params = EarthTideParameters(
            k20=float(tide_config.get("k20", defaults.k20)),
            k21=float(tide_config.get("k21", defaults.k21)),
            k22=float(tide_config.get("k22", defaults.k22)),
            tau_orb_days=(orb_values[0], orb_values[1], orb_values[2]),
            tau_rot_days=(rot_values[0], rot_values[1], rot_values[2]),
            earth_rotation_rate_rad_s=float(
                tide_config.get("earthRotationRateRadS", defaults.earth_rotation_rate_rad_s)
            ),
        )
        earth_tide = EarthTideModel(
            earth_gravitational_parameter_m3_s2,
            moon_gravitational_parameter_m3_s2,
            lambda names, epoch: body_state_matrix(ephemeris, tuple(names), epoch),
            earth_inertial2fixed_matrix_provider,
            earth_radius_m=float(tide_config.get("earthRadiusM", 6_378_136.6)),
            tide_parameters=params,
            tide_raisers=raisers,
            tide_raiser_gm=raiser_gm,
            solar_gm_variation=solar_gm_variation,
        )

    def earth_minus_moon_position_provider(provider_epoch):
        states = body_state_matrix(ephemeris, ("EARTH", "MOON"), provider_epoch)
        if states.shape != (2, 6) or not np.all(np.isfinite(states)):
            raise ValueError("Ephemeris returned an invalid Earth-Moon state")
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
            parameters=_degree2_parameters(config["lunarDegree2Gravity"]),
        )
    all_bodies = (("EARTH", earth_gravitational_parameter_m3_s2), ("MOON", moon_gravitational_parameter_m3_s2)) + tuple(
        (b.body_id, b.gravitational_parameter_m3_s2)
        for b in (*perturbing_bodies, *(() if ring is None else ring.bodies))
    )
    force_models: list[ForceModel] = [NewtonianPointMassForce(all_bodies)]
    if config["includeEih"]:
        force_models.append(EihPointMassForce(all_bodies))
    ra, dec = np.deg2rad([286.13, 63.87])
    solar_pole = (float(np.cos(dec) * np.cos(ra)), float(np.cos(dec) * np.sin(ra)), float(np.sin(dec)))
    if config.get("solarJ2") is not None:
        if "SUN" not in dict(all_bodies):
            raise ValueError("solarJ2 requires a perturbing SUN body.")
        solar_j2_field = GravityField(
            make_j2_coefficients(
                gm_m3_s2=dict(all_bodies)["SUN"], radius_m=config["solarRadiusM"], j2=config["solarJ2"], name="Solar J2"
            )
        )
        force_models.append(SolarJ2Force(solar_j2_field, inertial2fixed_matrix_from_pole(solar_pole)))
    if fields:
        default_planetary_partners = (
            "SUN",
            "MERCURY BARYCENTER",
            "VENUS BARYCENTER",
            "MARS BARYCENTER",
            "JUPITER BARYCENTER",
            "SATURN BARYCENTER",
        )
        figure_partners = config.get("figurePartners") or {}
        if not isinstance(figure_partners, dict):
            raise TypeError("figurePartners must be a mapping")

        def partners_for(body_id: str, counterpart: str) -> tuple[str, ...]:
            configured = figure_partners.get(body_id, figure_partners.get(body_id.lower()))
            raw = (counterpart, *default_planetary_partners) if configured is None else configured
            if not isinstance(raw, (list, tuple)) or not raw:
                raise TypeError(f"figurePartners.{body_id.lower()} must be a non-empty sequence")
            partners = tuple(body_name(value) for value in raw)
            missing = sorted(set(partners) - set(dict(all_bodies)))
            if missing:
                raise ValueError(f"Figure partners for {body_id} are missing from the selected bodies: {missing}")
            return partners

        if "EARTH" in fields:
            earth_partners = {"EARTH": partners_for("EARTH", "MOON")}
            earth_j2_time = config.get("earthJ2TimeVariation")
            if earth_j2_time is not None and not isinstance(earth_j2_time, dict):
                raise TypeError("earthJ2TimeVariation must be a mapping")
            force_models.append(
                TimeVaryingEarthFigureForce(
                    {"EARTH": fields["EARTH"]},
                    earth_partners,
                    j2_rate_per_year=float(earth_j2_time.get("linearPerYear", 0.0)),
                    j2_quadratic_per_year2=float(earth_j2_time.get("quadraticPerYear2", 0.0)),
                    reference_jd_tdb=float(earth_j2_time.get("referenceJdTdb", 2_451_545.0)),
                )
                if earth_j2_time is not None
                else FigureForce(
                    {"EARTH": fields["EARTH"]},
                    earth_partners,
                    name="figure_earth",
                )
            )
        if "MOON" in fields:
            moon_fields = {"MOON": fields["MOON"]}
            moon_partners = {"MOON": partners_for("MOON", "EARTH")}
            force_models.append(
                FigureForce(
                    moon_fields,
                    moon_partners,
                    name="figure_moon",
                )
            )
            if lunar_degree2_gravity is not None:
                force_models.append(
                    LunarDegree2GravityCorrectionForce(
                        moon_fields,
                        moon_partners,
                        lunar_degree2_gravity_model=lunar_degree2_gravity,
                    )
                )
    if earth_tide is not None:
        force_models.append(EarthTideForce(earth_tide))
    solar_relativistic = config.get("solarRelativistic")
    if solar_relativistic is not None:
        if not isinstance(solar_relativistic, dict):
            raise TypeError("solarRelativistic must be a mapping")
        if "SUN" not in dict(all_bodies):
            raise ValueError("solarRelativistic requires a perturbing SUN body.")
        solar_params = SolarSourceParameters(dict(all_bodies)["SUN"], config["solarRadiusM"], spin_axis=solar_pole)
        if bool(solar_relativistic.get("includeLenseThirring", True)):
            force_models.append(LenseThirringForce(solar_params))
        if bool(solar_relativistic.get("includeRadiationPressure", False)):
            force_models.append(
                SolarRadiationPressureForce(
                    solar_params,
                    {
                        "EARTH": float(solar_relativistic.get("earthEpsilon", 0.0)),
                        "MOON": float(solar_relativistic.get("moonEpsilon", 0.0)),
                    },
                )
            )
    default_terms: dict[str, list[str]] = {"MOON": ["point_mass"]}
    if config["includeEih"]:
        default_terms["MOON"].append("eih_1pn")
    # Earth and all external bodies are prescribed by the ephemeris.  Only
    # the modeled Moon acceleration is needed by the relative-state RHS.
    for model in force_models[1:]:
        if model.name not in default_terms["MOON"]:
            default_terms["MOON"].append(model.name)
    enabled_force_names = tuple(config["bodyForceTerms"] if "bodyForceTerms" in config else default_terms["MOON"])
    if not enabled_force_names:
        raise ValueError("bodyForceTerms must not be empty; omit it to select the documented defaults")
    if (
        lunar_degree2_gravity is not None
        and "figure_moon" in enabled_force_names
        and "lunar_degree2_gravity" not in enabled_force_names
    ):
        raise ValueError(
            "figure_moon with lunarDegree2Gravity requires explicit lunar_degree2_gravity in bodyForceTerms"
        )
    configured_moon_terms = set(enabled_force_names)
    known_force_terms = {model.name for model in force_models}
    unknown_force_terms = configured_moon_terms - known_force_terms
    if unknown_force_terms:
        raise ValueError(f"bodyForceTerms contains unavailable force terms: {sorted(unknown_force_terms)}")
    if "eih_1pn" in configured_moon_terms and "point_mass" not in configured_moon_terms:
        raise ValueError("bodyForceTerms requires point_mass when eih_1pn is enabled")
    rotation, rotation_initial = _build_rotation(
        config, fields, moon_gravitational_parameter_m3_s2, earth_gravitational_parameter_m3_s2
    )
    if rotation is not None:
        required = {body_name(name) for name in config["lunarRotation"]["sources"]}
        if rotation.geodetic_precession:
            required.add("SUN")
        if missing := required - set(dict(all_bodies)):
            raise ValueError(f"Rotational sources are missing from selected bodies: {sorted(missing)}")
    dynamics = LunarDynamics(
        earth_body=GravitatingBody("EARTH", earth_gravitational_parameter_m3_s2),
        moon_body=GravitatingBody("MOON", moon_gravitational_parameter_m3_s2),
        perturbing_bodies=perturbing_bodies,
        ephemeris=ephemeris,
        force_group=LunarForceGroup(force_models, enabled_force_names),
        earth_fixed2inertial_matrix_provider=OrientationProvider.from_scalar(earth_fixed2inertial_matrix_provider),
        moon_fixed2inertial_matrix_provider=OrientationProvider(
            ephemeris.lunar_orientation.pa_to_lcrs_matrix,
            ephemeris.lunar_orientation.pa_to_lcrs_matrices,
        )
        if fields and rotation is None
        else None,
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
    integration_initial = MoonRelativeState.from_barycentric_state(initial)
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
    initial_vector = integration_initial.as_relative_vector()
    if rotation_initial is not None:
        initial_vector = np.concatenate((initial_vector, rotation_initial))
    initial_vector.setflags(write=False)
    return LunarPropagationProblem(
        MappingProxyType(dict(config)),
        dynamics,
        integrator,
        epoch,
        initial_vector,
        duration,
        tuple(
            context.resolve_path(config[key])
            for key in (
                "outputFileOrbit",
                "outputFileAccelerationDiagnostics",
                "outputFileMetadata",
            )
        ),
        (lambda names, epoch: body_state_matrix(ephemeris, tuple(names), epoch))
        if earth_tide is not None or lunar_degree2_gravity is not None
        else None,
        tuple(
            name
            for name, enabled in (
                (
                    "solar_lense_thirring",
                    solar_relativistic is not None
                    and bool((solar_relativistic or {}).get("includeLenseThirring", True)),
                ),
                (
                    "solar_radiation_pressure",
                    solar_relativistic is not None
                    and bool((solar_relativistic or {}).get("includeRadiationPressure", False)),
                ),
            )
            if enabled
        ),
        None if rotation is None else context.resolve_path(config["outputFileRotation"]),
        None
        if rotation is None or rotation.deformation is None
        else config["lunarRotation"]["deformation"]["seedDurationSeconds"],
    )




def _degree2_parameters(config):
    """Adapt validated public fields to the unchanged numerical model."""
    return LunarDegree2GravityParameters(
        j2_undistorted=config["j2Undistorted"],
        love_k2=config["loveK2"],
        tidal_delay_days=config["tidalDelayDays"],
        radius_m=config["radiusM"],
        mean_motion_rad_s=config["meanMotionRadS"],
        attitude_derivative_step_s=config["attitudeDerivativeStepS"],
        c21_static=config["c21Static"],
        s21_static=config["s21Static"],
        s22_static=config["s22Static"],
        include_tidal_deformation=config["includeTidalDeformation"],
        include_rotational_deformation=config["includeRotationalDeformation"],
        beta=config["beta"],
        gamma=config["gamma"],
    )




@program(
    ProgramSpec(
        name="LunarOrbitPropagation",
        summary="Propagate the Earth/Moon hybrid subsystem and compare with the input ephemeris.",
        inputs=(ArtifactSlot("inputFileMassCatalog", "MassCatalogFile"),),
        outputs=(
            ArtifactSlot("outputFileOrbit", "LunarOrbitFile"),
            ArtifactSlot("outputFileAccelerationDiagnostics", "LunarAccelerationDiagnosticsFile"),
            ArtifactSlot("outputFileMetadata", "LunarOrbitMetadataFile"),
            ArtifactSlot("outputFileRotation", "LunarRotationFile", required=False),
        ),
        fields=LUNAR_ORBIT_FIELDS,
    )
)
def lunar_orbit_propagation(config, context):
    problem = build_lunar_propagation_problem(config, context)
    config = problem.config
    dynamics, integrator = problem.dynamics, problem.integrator
    ephemeris = dynamics.ephemeris
    epoch, duration = problem.initial_epoch, problem.duration_s
    earth_gravitational_parameter_m3_s2 = dynamics.earth_body.gravitational_parameter_m3_s2
    moon_gravitational_parameter_m3_s2 = dynamics.moon_body.gravitational_parameter_m3_s2
    orbit_output, diagnostic_output, metadata_output = problem.output_paths
    diagnostic_step = config.get("accelerationDiagnosticsStepSeconds")
    diagnostic_stride = None
    if diagnostic_step is not None:
        diagnostic_ratio = float(diagnostic_step) / config["stepSeconds"]
        if not np.isclose(diagnostic_ratio, round(diagnostic_ratio), rtol=0.0, atol=1e-12):
            raise ValueError("accelerationDiagnosticsStepSeconds must be an integer multiple of stepSeconds")
        diagnostic_stride = round(diagnostic_ratio)

    diagnostic_force_groups = (
        ("newtonian", ("point_mass",)),
        ("post_newtonian", ("eih_1pn",)),
        ("figure", ("solar_j2", "figure_earth", "figure_moon")),
        ("tide", ("tide",)),
        ("lunar_degree2_gravity", ("lunar_degree2_gravity",)),
        ("srp", ("solar_radiation_pressure",)),
        ("lt", ("lense_thirring",)),
    )
    diagnostic_rows: list[np.ndarray] = []
    node_index = -1

    def collect_diagnostics(node_epoch, node_state, epoch_data, history) -> None:
        nonlocal node_index
        node_index += 1
        if diagnostic_stride is None or node_index % diagnostic_stride:
            return
        offset = float(node_index * config["stepSeconds"] * np.sign(duration))
        breakdown = dynamics.accelerations(
            node_epoch,
            node_state,
            epoch_data,
            history=history,
        )
        values = [np.array([offset])]
        for _group_name, force_names in diagnostic_force_groups:
            term = sum(
                (
                    breakdown.accelerations_by_force_mps2.get(name, np.zeros((len(dynamics.body_names), 3)))
                    for name in force_names
                ),
                start=np.zeros((len(dynamics.body_names), 3)),
            )
            moon_acceleration = term[1]
            values.append(moon_acceleration)
        diagnostic_rows.append(np.concatenate(values))

    trajectory = integrator.integrate(
        dynamics,
        epoch,
        problem.initial_vector,
        duration,
        external_history_provider=problem.external_history,
        node_diagnostic=collect_diagnostics,
        seed_duration_s=problem.seed_duration_s,
    )
    offsets = np.arange(0.0, abs(duration), config["outputStepSeconds"]) * np.sign(duration)
    offsets = np.append(offsets, duration)
    orbit_columns = (
        "offset_tdb_s",
        *(f"earth_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")),
        *(f"moon_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")),
        *(f"emb_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")),
        *(f"relative_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")),
        *(f"reference_difference_rtn_{axis}_m" for axis in ("r", "t", "n")),
        *(f"reference_difference_velocity_{axis}_mps" for axis in ("x", "y", "z")),
    )
    orbit_values = np.empty((len(offsets), len(orbit_columns)), dtype=float)
    for row_index, offset in enumerate(offsets):
        t = epoch.shifted(float(offset))
        integration_vector = trajectory.vector(t)
        reference_states = body_state_matrix(ephemeris, ("EARTH", "MOON"), t)
        earth_reference = BodyState(reference_states[0, :3], reference_states[0, 3:])
        state = MoonRelativeState.from_relative_vector(t, integration_vector[:6]).to_barycentric_state(earth_reference)
        reference_r = reference_states[1, :3] - reference_states[0, :3]
        reference_v = reference_states[1, 3:] - reference_states[0, 3:]
        radial = reference_r / np.linalg.norm(reference_r)
        normal = np.cross(reference_r, reference_v)
        if np.linalg.norm(normal) == 0:
            raise ValueError("Reference orbit has undefined RTN frame.")
        normal /= np.linalg.norm(normal)
        rtn = np.array([radial, np.cross(normal, radial), normal])
        delta = state.relative_state.position_m - reference_r
        earth_state = np.concatenate((state.earth.position_m, state.earth.velocity_mps))
        moon_state = np.concatenate((state.moon.position_m, state.moon.velocity_mps))
        emb_state = np.concatenate(
            (
                (
                    earth_gravitational_parameter_m3_s2 * state.earth.position_m
                    + moon_gravitational_parameter_m3_s2 * state.moon.position_m
                )
                / (earth_gravitational_parameter_m3_s2 + moon_gravitational_parameter_m3_s2),
                (
                    earth_gravitational_parameter_m3_s2 * state.earth.velocity_mps
                    + moon_gravitational_parameter_m3_s2 * state.moon.velocity_mps
                )
                / (earth_gravitational_parameter_m3_s2 + moon_gravitational_parameter_m3_s2),
            )
        )
        orbit_values[row_index] = np.concatenate(
            (
                np.array([offset]),
                earth_state,
                moon_state,
                emb_state,
                np.concatenate((state.relative_state.position_m, state.relative_state.velocity_mps)),
                rtn @ delta,
                state.relative_state.velocity_mps - reference_v,
            )
        )
    differences = orbit_values[:, -6:-3]
    from importlib.metadata import version

    from lunarops.config.fingerprints import kernel_manifest

    source = ephemeris.source_path
    kernels = kernel_manifest(source) if source is not None and source.is_dir() else []
    common_metadata = {
        "timeScale": "TDB",
        "frame": "SSB/ICRF axes, TDB-compatible coordinates",
        "initialTdbJd1": config["initialTdbJd1"],
        "initialTdbJd2": config["initialTdbJd2"],
    }
    write_numeric_table(orbit_output, "lunarOrbit", orbit_columns, orbit_values, metadata=common_metadata)
    diagnostic_columns = ["offset_tdb_s"]
    for name, _force_names in diagnostic_force_groups:
        diagnostic_columns.extend(f"{name}_moon_{axis}_mps2" for axis in ("x", "y", "z"))
    diagnostic_values = (
        np.vstack(diagnostic_rows) if diagnostic_rows else np.empty((0, len(diagnostic_columns)), dtype=float)
    )
    write_numeric_table(
        diagnostic_output,
        "lunarAccelerationDiagnostics",
        tuple(diagnostic_columns),
        diagnostic_values,
        metadata={
            **common_metadata,
            "accelerationGroups": {name: force_names for name, force_names in diagnostic_force_groups},
            "figureGroup": "solar J2 + static Earth figure + static Moon figure",
            "lunarDegree2GravityGroup": "dynamic Moon degree-2 field minus static Moon field",
            "targets": "modeled Moon; Earth acceleration is prescribed by the ephemeris",
        },
    )
    write_lunar_orbit_metadata(
        metadata_output,
        {
            "model": f"Ephemeris-Earth/Moon-relative; external trajectories prescribed; {config['earthDynamicsFrame']} development force model",
            "timeScale": "TDB",
            "earthPoleModel": (
                config["earthDynamicsFrame"]
                if config["earthPoleModel"] == "ephemeris"
                else config["earthPoleModel"]
            ),
            "frame": "SSB/ICRF axes, TDB-compatible coordinates",
            "units": "m,s",
            "config": config,
            "initialStateSi": problem.initial_vector,
            "dynamicsBodyNames": dynamics.body_names,
            "solarGmModel": (
                "constant"
                if dynamics.solar_gm_variation is None
                else "GM_ref * (1 + fractionalRatePerYear * (JD_TDB-referenceJdTdb)/365.25); all force/torque inputs"
            ),
            "kuiperBeltRingModel": (
                None
                if dynamics.kuiper_belt_ring is None
                else "initial circular points in J2000 ecliptic; independent BCRS source-driven trajectories; Newtonian and EIH auxiliaries"
            ),
            "kernels": kernels,
            "scipyVersion": version("scipy"),
            "resolvedForceInventory": config["resolvedForceInventory"],
            "resolvedGravityFields": config["resolvedGravityFields"],
            "integrator": f"modified ABMD order {config['integratorOrder']}",
            "stepSeconds": config["stepSeconds"],
            "startupIntegrator": "Dormand-Prince 8",
            "startupStepSeconds": config["startupStepSeconds"],
            "lunarAttitudeMode": (
                (
                    "integrated variable-inertia mantle/core; delayed tidal and rotational deformation"
                    if dynamics.rotation.deformation is not None
                    else "integrated constant-inertia mantle/core"
                )
                if dynamics.rotation is not None
                else "prescribed ephemeris PA; no mantle/core rotation integration"
            ),
            "tideHistory": (
                "explicit rigid-backward joint seed; forward accepted joint state/derivative history"
                if problem.seed_duration_s is not None
                else "Ephemeris seed during startup; regular-node interpolation afterward"
            ),
            "rotationSeedDurationSeconds": problem.seed_duration_s,
            "earthTideAdvancedTranslationQueries": (
                None if trajectory.history is None else trajectory.history.advanced_translation_queries
            ),
            "earthTideMaxTranslationExtrapolationSeconds": (
                None if trajectory.history is None else trajectory.history.max_translation_extrapolation_s
            ),
            "solarForceTerms": problem.solar_force_terms,
            "referenceDifferenceRtnRmsM": np.sqrt(np.mean(differences**2, axis=0)),
            "referenceDifferenceRtnMaxAbsM": np.max(np.abs(differences), axis=0),
            "orbitFile": str(orbit_output),
            "accelerationDiagnosticsFile": str(diagnostic_output),
            "orbitSamples": len(orbit_values),
            "accelerationDiagnosticSamples": len(diagnostic_values),
            "rotationFile": None if problem.rotation_output is None else str(problem.rotation_output),
        },
    )
    print(
        f"[LunarOrbitPropagation] {len(orbit_values)} orbit samples, "
        f"{len(diagnostic_values)} diagnostic samples -> {orbit_output}"
    )
    if problem.rotation_output is not None:
        _write_rotation(problem, trajectory, offsets, common_metadata)
        return orbit_output, diagnostic_output, metadata_output, problem.rotation_output
    return orbit_output, diagnostic_output, metadata_output


def _write_rotation(problem, trajectory, offsets, metadata):
    rotation = problem.dynamics.rotation
    columns = [
        "offset_tdb_s",
        *(f"q_{axis}" for axis in ("w", "x", "y", "z")),
        *(f"mantle_omega_pa_{axis}_rad_s" for axis in "xyz"),
    ]
    if rotation.core is not None:
        columns.extend(f"core_omega_pa_{axis}_rad_s" for axis in "xyz")
    columns.append("quaternion_norm_error")
    names = [*(model.name for model in rotation.torque_group.models), "cmb_viscous", "cmb_pressure", "geodetic_rate"]
    for name in names:
        columns.extend(f"{name}_{axis}_{'rad_s' if name == 'geodetic_rate' else 'per_s2'}" for axis in "xyz")
    columns.extend(f"mantle_omega_dot_pa_{axis}_rad_s2" for axis in "xyz")
    if rotation.core is not None:
        columns.extend(f"core_omega_dot_pa_{axis}_rad_s2" for axis in "xyz")
    inertia_indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    if rotation.deformation is not None:
        for name in ("total_inertia", "mantle_inertia", "inertia_dot_per_s"):
            columns.extend(f"{name}_{axes}" for axes in ("xx", "yy", "zz", "xy", "xz", "yz"))
    rows = []
    for offset in offsets:
        epoch = problem.initial_epoch.shifted(float(offset))
        state = trajectory.vector(epoch)
        result = problem.dynamics.rotation_evaluation(epoch, state, history=trajectory.history)
        terms = [*result.torques.values(), result.viscous_torque, result.pressure_torque, result.geodetic_rate_pa_rad_s]
        inertia_terms = []
        if rotation.deformation is not None:
            for tensor in (
                result.inertia.total,
                result.inertia.total - rotation.core_inertia,
                result.inertia.derivative,
            ):
                inertia_terms.append(np.asarray([tensor[i, j] for i, j in inertia_indices]))
        rows.append(
            np.concatenate(
                (
                    [offset],
                    state[6:],
                    [np.linalg.norm(state[6:10]) - 1],
                    *terms,
                    result.derivative[4:],
                    *inertia_terms,
                )
            )
        )
    write_numeric_table(
        problem.rotation_output,
        "lunarRotation",
        tuple(columns),
        np.asarray(rows),
        metadata={
            **metadata,
            "metadataVersion": 2,
            "quaternion": "Hamilton scalar-first PA-to-inertial active rotation",
            "angularVelocity": "mantle-frame components, TDB rad/s",
            "torqueNormalization": "N/(whole Moon M R^2), s^-2",
            "inertiaModel": (
                "delayed whole-Moon tensor minus constant core; analytic mantle inertia derivative"
                if rotation.deformation is not None
                else "constant full Moon minus constant axisymmetric core"
            ),
            "pressureSpin": None if rotation.core is None else rotation.core.pressure_spin,
        },
    )
