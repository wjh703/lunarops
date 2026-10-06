"""Earth/Moon propagation and reference-ephemeris diagnostics."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

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
from lunarops.classes.dynamics.gravity import GravityField, make_j2_coefficients
from lunarops.classes.dynamics.orientation import inertial2fixed_matrix_from_pole
from lunarops.classes.ephemerides import BodyState, body_state_matrix
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.observation_factory import ensure_registered
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.schema import boolean, class_config, integer, mapping, number, sequence, string
from lunarops.fileio.mass_catalog import read_mass_catalog
from lunarops.fileio.numeric_table import write_numeric_table
from lunarops.fileio.yaml_artifact import write_structured_text
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program


def _load_configured_gravity_fields(config, context, gm_by_body):
    """Build the Earth/Moon coefficient map declared by the program config."""
    if not isinstance(config, dict):
        raise TypeError("gravityFields must be a mapping")
    fields = {}
    for body_id in ("EARTH", "MOON"):
        entry = config.get(body_id.lower(), config.get(body_id))
        if entry is None:
            continue
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"gravityFields.{body_id.lower()} requires path")
        if "gmM3S2" in entry:
            raise ValueError(f"gravityFields.{body_id.lower()}.gmM3S2 was removed; GM comes from the mass catalog.")
        fields[body_id] = load_gravity_field(
            context.resolve_path(entry["path"]),
            file_format=entry.get("format", "icgem"),
            max_degree=entry.get("maxDegree"),
            gm_m3_s2=gm_by_body[body_id],
            reference_radius_m=entry.get("radiusM"),
            name=entry.get("name", body_id),
        )
    if not fields:
        raise ValueError("gravityFields must define at least earth or moon")
    return fields


@program(
    ProgramSpec(
        name="LunarOrbitPropagation",
        summary="Propagate the Earth/Moon hybrid subsystem and compare with the input ephemeris.",
        inputs=(ArtifactSlot("inputFileMassCatalog", "MassCatalogFile"),),
        outputs=(
            ArtifactSlot("outputFileOrbit", "LunarOrbitFile"),
            ArtifactSlot("outputFileAccelerationDiagnostics", "LunarAccelerationDiagnosticsFile"),
            ArtifactSlot("outputFileMetadata", "LunarOrbitMetadataFile"),
        ),
        fields=(
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
            mapping("lunarDegree2Gravity", default=None, allow_none=True),
            mapping("gravityFields", default=None, allow_none=True),
            mapping("figurePartners", default=None, allow_none=True),
            string(
                "earthDynamicsFrame",
                default="de440",
                choices=("de440", "de430"),
                allow_none=False,
            ),
            sequence("externalBodyGroups", default=[], item_kind="string", allow_none=False),
            sequence("externalBodyIds", default=[], item_kind="string", allow_none=False),
            mapping("earthTide", default=None, allow_none=True),
            sequence("bodyForceTerms", default=[], item_kind="string", allow_none=False),
            number("solarJ2"),
            mapping("earthJ2TimeVariation", default=None, allow_none=True),
            mapping("solarRelativistic", default=None, allow_none=True),
            number("solarRadiusM", default=696000000.0, minimum=0, minimum_exclusive=True, allow_none=False),
            number("stepSeconds", default=5400.0, minimum=0, minimum_exclusive=True, allow_none=False),
            integer("integratorOrder", default=13, minimum=2, maximum=13, allow_none=False),
            integer("correctorIterations", default=2, minimum=1, allow_none=False),
            integer("historyInterpolationOrder", default=13, minimum=1, allow_none=False),
            integer("trajectoryInterpolationOrder", default=13, minimum=1, allow_none=False),
            number("startupStepSeconds", default=675.0, minimum=0, minimum_exclusive=True, allow_none=False),
        ),
    )
)
def lunar_orbit_propagation(config, context):
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
    ephemeris = context.create_class("ephemerides", config["ephemerides"], cache=True)
    epoch = Epoch(config["initialTdbJd1"], config["initialTdbJd2"], TimeScale.TDB)
    initial_states = body_state_matrix(ephemeris, ("EARTH", "MOON"), epoch)
    initial = BarycentricEarthMoonState(
        epoch,
        BodyState(initial_states[0, :3], initial_states[0, 3:]),
        BodyState(initial_states[1, :3], initial_states[1, 3:]),
    )
    earth_fixed2inertial_matrix_provider: Callable[[Epoch], np.ndarray]
    if config["earthDynamicsFrame"] == "de430":
        earth_fixed2inertial_matrix_provider = de430_earth_fixed2inertial_matrix
    else:
        earth_fixed2inertial_matrix_provider = de440_earth_fixed2inertial_matrix
    earth_inertial2fixed_matrix_provider = lambda epoch: earth_fixed2inertial_matrix_provider(epoch).T
    fields = (
        _load_configured_gravity_fields(config["gravityFields"], context, {"EARTH": earth_gravitational_parameter_m3_s2, "MOON": moon_gravitational_parameter_m3_s2})
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
            parameters=LunarDegree2GravityParameters(**(config.get("lunarDegree2Gravity") or {})),
        )
    all_bodies = (("EARTH", earth_gravitational_parameter_m3_s2), ("MOON", moon_gravitational_parameter_m3_s2)) + tuple((b.body_id, b.gravitational_parameter_m3_s2) for b in perturbing_bodies)
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
    enabled_force_names = tuple(config.get("bodyForceTerms") or default_terms["MOON"])
    if lunar_degree2_gravity is not None and "figure_moon" in enabled_force_names and "lunar_degree2_gravity" not in enabled_force_names:
        enabled_force_names += ("lunar_degree2_gravity",)
    configured_moon_terms = set(enabled_force_names)
    known_force_terms = {model.name for model in force_models}
    unknown_force_terms = configured_moon_terms - known_force_terms
    if unknown_force_terms:
        raise ValueError(f"bodyForceTerms contains unavailable force terms: {sorted(unknown_force_terms)}")
    if "eih_1pn" in configured_moon_terms and "point_mass" not in configured_moon_terms:
        raise ValueError("bodyForceTerms requires point_mass when eih_1pn is enabled")
    dynamics = LunarDynamics(
        earth_body=GravitatingBody("EARTH", earth_gravitational_parameter_m3_s2),
        moon_body=GravitatingBody("MOON", moon_gravitational_parameter_m3_s2),
        perturbing_bodies=perturbing_bodies,
        ephemeris=ephemeris,
        force_group=LunarForceGroup(force_models, enabled_force_names),
        earth_fixed2inertial_matrix_provider=earth_fixed2inertial_matrix_provider,
        moon_fixed2inertial_matrix_provider=ephemeris.lunar_orientation if fields else None,
    )
    integration_initial = MoonRelativeState.from_barycentric_state(initial)
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
    external_history_provider = lambda names, epoch: body_state_matrix(ephemeris, tuple(names), epoch)
    trajectory = integrator.integrate(
        dynamics,
        epoch,
        integration_initial.as_relative_vector(),
        duration,
        external_history_provider=(
            external_history_provider
            if earth_tide is not None or lunar_degree2_gravity is not None
            else None
        ),
        node_diagnostic=collect_diagnostics,
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
        state = MoonRelativeState.from_relative_vector(t, integration_vector).to_barycentric_state(earth_reference)
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
                (earth_gravitational_parameter_m3_s2 * state.earth.position_m + moon_gravitational_parameter_m3_s2 * state.moon.position_m) / (earth_gravitational_parameter_m3_s2 + moon_gravitational_parameter_m3_s2),
                (earth_gravitational_parameter_m3_s2 * state.earth.velocity_mps + moon_gravitational_parameter_m3_s2 * state.moon.velocity_mps) / (earth_gravitational_parameter_m3_s2 + moon_gravitational_parameter_m3_s2),
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
    kernels = []
    source = ephemeris.source_path
    if source is not None and source.is_dir():
        for path in sorted(source.iterdir()):
            if path.suffix in (".bsp", ".bpc"):
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                kernels.append({"path": str(path.resolve()), "sha256": digest})
    orbit_output = context.resolve_path(config["outputFileOrbit"])
    diagnostic_output = context.resolve_path(config["outputFileAccelerationDiagnostics"])
    metadata_output = context.resolve_path(config["outputFileMetadata"])
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
    write_structured_text(
        metadata_output,
        "lunarOrbitMetadata",
        {
            "model": "Ephemeris-Earth/Moon-relative; external trajectories prescribed; incomplete DE440 force model",
            "time_scale": "TDB",
            "frame": "SSB/ICRF axes, TDB-compatible coordinates",
            "units": "m,s",
            "config": config,
            "initial_state_si": integration_initial.as_relative_vector(),
            "kernels": kernels,
            "integrator": f"modified ABMD order {config['integratorOrder']}",
            "step_seconds": config["stepSeconds"],
            "startup_integrator": "Dormand-Prince 8",
            "startup_step_seconds": config["startupStepSeconds"],
            "lunar_attitude_mode": "prescribed ephemeris PA; no mantle/core rotation integration",
            "tide_history": "Ephemeris seed during startup; fixed-capacity regular-node interpolation afterward",
            "solar_force_terms": tuple(
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
            "reference_difference_rtn_rms_m": np.sqrt(np.mean(differences**2, axis=0)),
            "reference_difference_rtn_max_abs_m": np.max(np.abs(differences), axis=0),
            "orbit_file": str(orbit_output),
            "acceleration_diagnostics_file": str(diagnostic_output),
            "orbit_samples": len(orbit_values),
            "acceleration_diagnostic_samples": len(diagnostic_values),
        },
    )
    print(
        f"[LunarOrbitPropagation] {len(orbit_values)} orbit samples, "
        f"{len(diagnostic_values)} diagnostic samples -> {orbit_output}"
    )
    return orbit_output, diagnostic_output, metadata_output
