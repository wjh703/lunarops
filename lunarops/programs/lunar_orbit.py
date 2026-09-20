"""Earth/Moon propagation and reference-ephemeris diagnostics."""

from __future__ import annotations

import hashlib

import numpy as np

from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    BodyRole,
    DynamicalBody,
    EarthMoonBarycentricState,
    EarthMoonDynamics,
    EarthMoonIntegrationState,
    EarthTideForce,
    EarthTideModel,
    EarthTideParameters,
    EihForce,
    FigureForce,
    ForceGroup,
    ForceModel,
    FrameProviders,
    IntegratorSettings,
    LenseThirringForce,
    LunarInertiaFigureForce,
    LunarInertiaModel,
    LunarInertiaParameters,
    PointMassForce,
    SolarForceParameters,
    SolarJ2Force,
    SolarRadiationPressureForce,
    de440_earth_dynamics_frame,
    load_gravity_coefficients,
)
from lunarops.classes.dynamics.gravity import GravityField, frame_from_pole, make_j2_coefficients
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.observation_factory import ensure_registered
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.schema import boolean, class_config, integer, mapping, number, sequence
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
        fields[body_id] = load_gravity_coefficients(
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
            mapping("lunarInertia", default=None, allow_none=True),
            mapping("gravityFields", default=None, allow_none=True),
            sequence("externalBodyGroups", default=[], item_kind="string", allow_none=False),
            sequence("externalBodyIds", default=[], item_kind="string", allow_none=False),
            mapping("earthTide", default=None, allow_none=True),
            mapping("bodyForceTerms", default=None, allow_none=True),
            number("solarJ2"),
            mapping("solarRelativistic", default=None, allow_none=True),
            number("solarRadiusM", default=696000000.0, minimum=0, minimum_exclusive=True, allow_none=False),
            number("stepSeconds", default=5400.0, minimum=0, minimum_exclusive=True, allow_none=False),
            integer("integratorOrder", default=13, minimum=2, maximum=13, allow_none=False),
            integer("correctorIterations", default=2, minimum=1, allow_none=False),
            integer("interpolationOrder", default=13, minimum=1, allow_none=False),
            number("startupStepSeconds", default=675.0, minimum=0, minimum_exclusive=True, allow_none=False),
            integer("preparationBatchSize", default=64, minimum=1, allow_none=False),
            integer("preparationWorkers", default=4, minimum=1, allow_none=False),
            integer("rebuildDifferencesEvery", default=0, minimum=0, allow_none=False),
        ),
    )
)
def lunar_orbit_propagation(config, context):
    ensure_registered()
    duration = config["durationSeconds"]
    if duration == 0:
        raise ValueError("durationSeconds must be nonzero.")
    mass_catalog = read_mass_catalog(context.resolve_path(config["inputFileMassCatalog"]))
    earth_gm = mass_catalog.require("EARTH").gm_m3_s2
    moon_gm = mass_catalog.require("MOON").gm_m3_s2
    selected_external = mass_catalog.select(
        groups=config["externalBodyGroups"],
        body_ids=config["externalBodyIds"],
    )
    selected_external = tuple(body for body in selected_external if body.body_id not in {"EARTH", "MOON"})
    if not selected_external:
        raise ValueError("The mass catalog selection must contain at least one prescribed body")
    external = tuple(DynamicalBody(body.body_id, body.gm_m3_s2, BodyRole.PRESCRIBED) for body in selected_external)
    eph = context.create_class("ephemerides", config["ephemerides"], cache=True)
    epoch = Epoch(config["initialTdbJd1"], config["initialTdbJd2"], TimeScale.TDB)
    initial_states = eph.body_states_bcrs(("EARTH", "MOON"), epoch)
    initial = EarthMoonBarycentricState(epoch, initial_states["EARTH"], initial_states["MOON"])
    earth_frame_provider = de440_earth_dynamics_frame
    fields = (
        _load_configured_gravity_fields(config["gravityFields"], context, {"EARTH": earth_gm, "MOON": moon_gm})
        if config.get("gravityFields")
        else None
    )
    earth_tide = None
    if config.get("earthTide") is not None:
        tide_config = config.get("earthTide") or {}
        external_gm = {body.body_id: body.mu_m3_s2 for body in external}
        raisers = tuple(body_name(name) for name in tide_config.get("tideRaisers", ("MOON", "SUN")))
        raiser_gm = {"MOON": moon_gm}
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
            earth_gm,
            moon_gm,
            eph.body_state_vectors_bcrs,
            earth_frame_provider,
            earth_radius_m=float(tide_config.get("earthRadiusM", 6_378_136.6)),
            parameters=params,
            tide_raisers=raisers,
            tide_raiser_gm=raiser_gm,
        )
    lunar_inertia = None
    if config.get("lunarInertia") is not None:
        if fields is None or "MOON" not in fields:
            raise ValueError("lunarInertia requires a configured Moon gravity field.")
        lunar_inertia = LunarInertiaModel(
            earth_gm,
            moon_gm,
            eph.body_state_vectors_bcrs,
            eph.pa2lcrs_matrix,
            parameters=LunarInertiaParameters(**(config.get("lunarInertia") or {})),
        )
    all_bodies = (("EARTH", earth_gm), ("MOON", moon_gm)) + tuple((b.body_id, b.mu_m3_s2) for b in external)
    force_models: list[ForceModel] = [PointMassForce(all_bodies)]
    if config["includeEih"]:
        force_models.append(EihForce(all_bodies))
    ra, dec = np.deg2rad([286.13, 63.87])
    solar_pole = (float(np.cos(dec) * np.cos(ra)), float(np.cos(dec) * np.sin(ra)), float(np.sin(dec)))
    if config.get("solarJ2") is not None:
        if "SUN" not in dict(all_bodies):
            raise ValueError("solarJ2 requires a prescribed SUN body.")
        solar_j2_field = GravityField(
            make_j2_coefficients(
                gm_m3_s2=dict(all_bodies)["SUN"], radius_m=config["solarRadiusM"], j2=config["solarJ2"], name="Solar J2"
            )
        )
        force_models.append(SolarJ2Force(solar_j2_field, frame_from_pole(solar_pole)))
    if fields:
        planetary_partners = (
            "SUN",
            "MERCURY BARYCENTER",
            "VENUS BARYCENTER",
            "MARS BARYCENTER",
            "JUPITER BARYCENTER",
            "SATURN BARYCENTER",
        )
        if "EARTH" in fields:
            force_models.append(
                FigureForce(
                    {"EARTH": fields["EARTH"]},
                    {"EARTH": ("MOON", *planetary_partners)},
                    name="figure_earth",
                )
            )
        if "MOON" in fields:
            force_models.append(
                LunarInertiaFigureForce(
                    {"MOON": fields["MOON"]},
                    {"MOON": ("EARTH", *planetary_partners)},
                    inertia=lunar_inertia,
                )
                if lunar_inertia is not None
                else FigureForce(
                    {"MOON": fields["MOON"]},
                    {"MOON": ("EARTH", *planetary_partners)},
                    name="figure_moon",
                )
            )
    if earth_tide is not None:
        force_models.append(EarthTideForce(earth_tide))
    solar_relativistic = config.get("solarRelativistic")
    if solar_relativistic is not None:
        if not isinstance(solar_relativistic, dict):
            raise TypeError("solarRelativistic must be a mapping")
        if "SUN" not in dict(all_bodies):
            raise ValueError("solarRelativistic requires a prescribed SUN body.")
        solar_params = SolarForceParameters(dict(all_bodies)["SUN"], config["solarRadiusM"], spin_axis=solar_pole)
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
    default_terms: dict[str, list[str]] = {name: [] for name, _ in all_bodies}
    default_terms["EARTH"].append("point_mass")
    default_terms["MOON"].append("point_mass")
    if config["includeEih"]:
        for name in ("EARTH", "MOON"):
            default_terms[name].append("eih_1pn")
    # Only integrated Earth/Moon bodies receive the optional force terms in
    # this Earth--Moon program. Prescribed ephemeris bodies do not need their
    # own acceleration evaluated.
    for model in force_models[1:]:
        for name in ("EARTH", "MOON"):
            if model.name not in default_terms[name]:
                default_terms[name].append(model.name)
    body_terms = config.get("bodyForceTerms") or default_terms
    dynamics = EarthMoonDynamics(
        DynamicalBody("EARTH", earth_gm, BodyRole.INTEGRATED),
        DynamicalBody("MOON", moon_gm, BodyRole.INTEGRATED),
        external,
        eph,
        force_group=ForceGroup(force_models, body_terms),
        frames=FrameProviders(
            earth_frame_provider,
            (lambda t: eph.pa2lcrs_matrix(t).T) if fields and hasattr(eph, "pa2lcrs_matrix") else None,
        ),
    )
    integration_initial = initial.to_integration(earth_gm, moon_gm)
    diagnostic_step = config.get("accelerationDiagnosticsStepSeconds")
    if diagnostic_step is not None:
        diagnostic_ratio = float(diagnostic_step) / config["stepSeconds"]
        if not np.isclose(diagnostic_ratio, round(diagnostic_ratio), rtol=0.0, atol=1e-12):
            raise ValueError("accelerationDiagnosticsStepSeconds must be an integer multiple of stepSeconds")

    diagnostic_force_names = tuple(model.name for model in force_models)
    diagnostic_rows: list[np.ndarray] = []

    def collect_diagnostics(accepted_epoch, accepted_state, prepared, history) -> None:
        if diagnostic_step is None:
            return
        offset = float(epoch.seconds_until(accepted_epoch))
        quotient = abs(offset) / float(diagnostic_step)
        if not np.isclose(quotient, round(quotient), rtol=0.0, atol=1e-12):
            return
        breakdown = dynamics.accelerations(accepted_epoch, accepted_state, prepared, history=history)
        values = [np.array([offset])]
        total_gm = earth_gm + moon_gm
        for name in diagnostic_force_names:
            term = breakdown.terms_mps2.get(name, np.zeros((len(dynamics.body_names), 3)))
            earth_acceleration = term[0]
            moon_acceleration = term[1]
            emb_acceleration = (earth_gm * earth_acceleration + moon_gm * moon_acceleration) / total_gm
            values.extend(
                (earth_acceleration, moon_acceleration, emb_acceleration, moon_acceleration - earth_acceleration)
            )
        diagnostic_rows.append(np.concatenate(values))

    integrator = AdamsBashforthMoultonIntegrator(
        IntegratorSettings(
            step_s=config["stepSeconds"],
            order=config["integratorOrder"],
            corrector_iterations=config["correctorIterations"],
            interpolation_order=config["interpolationOrder"],
            startup_step_s=config["startupStepSeconds"],
            preparation_batch_size=config["preparationBatchSize"],
            preparation_workers=config["preparationWorkers"],
            rebuild_differences_every=config["rebuildDifferencesEvery"],
        )
    )
    initial_history = eph.body_state_vectors_bcrs
    trajectory = integrator.integrate(
        dynamics,
        epoch,
        integration_initial.to_vector(),
        duration,
        initial_history=initial_history if earth_tide is not None or lunar_inertia is not None else None,
        accepted_step_observer=collect_diagnostics,
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
        state = EarthMoonIntegrationState.from_vector(t, integration_vector).to_barycentric(earth_gm, moon_gm)
        reference_states = eph.body_state_vectors_bcrs(("EARTH", "MOON"), t)
        reference_r = reference_states[1, :3] - reference_states[0, :3]
        reference_v = reference_states[1, 3:] - reference_states[0, 3:]
        radial = reference_r / np.linalg.norm(reference_r)
        normal = np.cross(reference_r, reference_v)
        if np.linalg.norm(normal) == 0:
            raise ValueError("Reference orbit has undefined RTN frame.")
        normal /= np.linalg.norm(normal)
        rtn = np.array([radial, np.cross(normal, radial), normal])
        delta = state.relative.position_m - reference_r
        earth_state = np.concatenate((state.earth.position_m, state.earth.velocity_mps))
        moon_state = np.concatenate((state.moon.position_m, state.moon.velocity_mps))
        emb_state = np.concatenate(
            (
                (earth_gm * state.earth.position_m + moon_gm * state.moon.position_m) / (earth_gm + moon_gm),
                (earth_gm * state.earth.velocity_mps + moon_gm * state.moon.velocity_mps) / (earth_gm + moon_gm),
            )
        )
        orbit_values[row_index] = np.concatenate(
            (
                np.array([offset]),
                earth_state,
                moon_state,
                emb_state,
                np.concatenate((state.relative.position_m, state.relative.velocity_mps)),
                rtn @ delta,
                state.relative.velocity_mps - reference_v,
            )
        )
    differences = orbit_values[:, -6:-3]
    kernels = []
    source = eph.source_file_path
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
    for name in diagnostic_force_names:
        for target in ("earth", "moon", "emb", "relative"):
            diagnostic_columns.extend(f"{name}_{target}_{axis}_mps2" for axis in ("x", "y", "z"))
    diagnostic_values = (
        np.vstack(diagnostic_rows) if diagnostic_rows else np.empty((0, len(diagnostic_columns)), dtype=float)
    )
    write_numeric_table(
        diagnostic_output,
        "lunarAccelerationDiagnostics",
        tuple(diagnostic_columns),
        diagnostic_values,
        metadata=common_metadata,
    )
    write_structured_text(
        metadata_output,
        "lunarOrbitMetadata",
        {
            "model": "Earth-Moon hybrid; external trajectories prescribed; incomplete DE440 force model",
            "time_scale": "TDB",
            "frame": "SSB/ICRF axes, TDB-compatible coordinates",
            "units": "m,s",
            "config": config,
            "initial_state_si": integration_initial.to_vector(),
            "kernels": kernels,
            "integrator": f"modified ABMD order {config['integratorOrder']}",
            "step_seconds": config["stepSeconds"],
            "startup_integrator": "Dormand-Prince 8",
            "startup_step_seconds": config["startupStepSeconds"],
            "function_evaluations": trajectory.function_evaluations,
            "history_buffer_capacity": trajectory.history_buffer_capacity,
            "history_buffer_max_nodes": trajectory.history_buffer_max_nodes,
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
