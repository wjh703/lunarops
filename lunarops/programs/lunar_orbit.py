"""Lunar orbit (and optional rotation) propagation with comparison against the input ephemeris.

The ``LunarOrbitPropagation`` program integrates the geocentric state of the Moon, in BCRS (SSB/ICRF) axes
and on the TDB time scale, while the Earth and the external bodies follow the ephemeris. Optionally the
lunar mantle and core rotation is integrated together with the orbit. The propagated orbit is compared with
the ephemeris Moon, with position differences in the radial/transverse/normal (RTN) frame of the
ephemeris orbit.
"""

from __future__ import annotations

import numpy as np

from lunarops import console
from lunarops.classes.dynamics.assembly import DIAGNOSTIC_FORCE_GROUPS, build_lunar_propagation_problem
from lunarops.config.lunar_models import LUNAR_ORBIT_FIELDS
from lunarops.fileio.numeric_table import write_numeric_table
from lunarops.fileio.structured_reports import write_lunar_orbit_metadata
from lunarops.programs.registry import ArtifactSlot, program, program_spec

# Model descriptions recorded in the metadata of the output tables.
FRAME_DESCRIPTION = "SSB/ICRF axes, TDB-compatible coordinates"
SOLAR_GM_DESCRIPTION = "GM_ref * (1 + fractionalRatePerYear * (JD_TDB-referenceJdTdb)/365.25); all force/torque inputs"
RING_DESCRIPTION = "initial circular points in J2000 ecliptic; independent BCRS source-driven trajectories; Newtonian and EIH auxiliaries"
DEFORMED_ATTITUDE_DESCRIPTION = "integrated variable-inertia mantle/core; delayed tidal and rotational deformation"
RIGID_ATTITUDE_DESCRIPTION = "integrated constant-inertia mantle/core"
PRESCRIBED_ATTITUDE_DESCRIPTION = "prescribed ephemeris PA; no mantle/core rotation integration"
JOINT_HISTORY_DESCRIPTION = "explicit rigid-backward joint seed; forward accepted joint state/derivative history"
EPHEMERIS_HISTORY_DESCRIPTION = "Ephemeris seed during startup; regular-node interpolation afterward"
DEFORMED_INERTIA_DESCRIPTION = "delayed whole-Moon tensor minus constant core; analytic mantle inertia derivative"
RIGID_INERTIA_DESCRIPTION = "constant full Moon minus constant axisymmetric core"


@program(
    program_spec(
        name="LunarOrbitPropagation",
        summary="Propagate the Earth/Moon hybrid subsystem and compare with the input ephemeris.",
        inputs=(
            ArtifactSlot(
                "inputFileMassCatalog",
                "massCatalog",
                description="Native mass catalog with the GM of the Earth, the Moon and the perturbers.",
            ),
        ),
        outputs=(
            ArtifactSlot(
                "outputFileOrbit",
                "lunarOrbit",
                binary=True,
                description=(
                    "Output binary table of the propagated Moon-relative state and its difference to the ephemeris."
                ),
            ),
            ArtifactSlot(
                "outputFileAccelerationDiagnostics",
                "lunarAccelerationDiagnostics",
                binary=True,
                description="Output binary table of the Moon acceleration split by force group.",
            ),
            ArtifactSlot(
                "outputFileMetadata",
                "lunarOrbitMetadata",
                description="Output metadata: resolved configuration, sources, checksums and statistics.",
            ),
            ArtifactSlot(
                "outputFileRotation",
                "lunarRotation",
                binary=True,
                required=False,
                description="Output binary table of the integrated rotation. Required with `lunarRotation`.",
            ),
        ),
        fields=LUNAR_ORBIT_FIELDS,
    )
)
def lunar_orbit_propagation(config, context):
    """Propagate the Moon from the configured initial TDB epoch and compare it with the ephemeris.

    Reads the GMs from the ``massCatalog`` ``inputFileMassCatalog``, integrates the configured force model
    over ``durationSeconds`` with the fixed-step Adams-Bashforth-Moulton integrator (epoch data computed on
    MPI workers when the run uses ``--mpi``) and writes:

    - ``outputFileOrbit`` (``lunarOrbit``): every ``outputStepSeconds``, the BCRS states of the Earth
      (ephemeris), the propagated Moon and the Earth-Moon barycentre, the geocentric Moon state (m, m/s),
      and its difference to the ephemeris Moon (position in RTN, m; velocity in ICRF axes, m/s);
    - ``outputFileAccelerationDiagnostics`` (``lunarAccelerationDiagnostics``): the Moon acceleration per
      force group (m/s^2), every ``accelerationDiagnosticsStepSeconds``;
    - ``outputFileMetadata`` (``lunarOrbitMetadata``): the resolved config, kernels, model descriptions and
      the RMS and maximum RTN differences to the ephemeris;
    - ``outputFileRotation`` (``lunarRotation``), with ``lunarRotation``: the mantle attitude quaternion,
      mantle and core angular velocities (PA axes, rad/s), torques and, with deformation, inertia tensors.

    Times in the tables are offsets from the initial epoch, in TDB seconds. See the program reference
    (``docs/reference/programs.md``) for every configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    tuple of Path
        Paths of the orbit, acceleration-diagnostics and metadata files, followed by the rotation file when
        the rotation is integrated.

    Raises
    ------
    ValueError
        If ``accelerationDiagnosticsStepSeconds`` is not a whole multiple of ``stepSeconds``, or the RTN frame
        of the ephemeris orbit is undefined.
    """
    problem = build_lunar_propagation_problem(config, context)
    config = problem.config
    dynamics, integrator = problem.dynamics, problem.integrator
    ephemeris = dynamics.ephemeris
    epoch, duration = problem.initial_epoch, problem.duration_s
    earth_gravitational_parameter_m3_s2 = dynamics.earth_body.gravitational_parameter_m3_s2
    moon_gravitational_parameter_m3_s2 = dynamics.moon_body.gravitational_parameter_m3_s2
    orbit_output, diagnostic_output, metadata_output = problem.output_paths
    diagnostic_step = config["accelerationDiagnosticsStepSeconds"]
    diagnostic_stride = None
    if diagnostic_step is not None:
        diagnostic_ratio = diagnostic_step / config["stepSeconds"]
        if not np.isclose(diagnostic_ratio, round(diagnostic_ratio), rtol=0.0, atol=1e-12):
            raise ValueError("accelerationDiagnosticsStepSeconds must be an integer multiple of stepSeconds")
        diagnostic_stride = round(diagnostic_ratio)

    diagnostic_rows: list[np.ndarray] = []
    node_index = -1
    node_count = round(abs(duration) / config["stepSeconds"]) + 1
    bar = console.progress(total=node_count, desc="integration", unit="steps")

    def collect_diagnostics(node_epoch, node_state, batch, index, history) -> None:
        nonlocal node_index
        node_index += 1
        bar.update()
        if diagnostic_stride is None or node_index % diagnostic_stride:
            return
        offset = float(node_index * config["stepSeconds"] * np.sign(duration))
        breakdown = dynamics.terms(
            node_epoch,
            node_state,
            batch,
            index=index,
            history=history,
        )
        values = [np.array([offset])]
        for _group_name, force_names in DIAGNOSTIC_FORCE_GROUPS:
            term = sum((breakdown["forces"].get(name, np.zeros(3)) for name in force_names), start=np.zeros(3))
            values.append(term)
        diagnostic_rows.append(np.concatenate(values))

    epoch_source = None
    if context.runtime is not None:
        from lunarops.parallel.epoch_stream import MpiEpochSource

        epoch_source = MpiEpochSource(
            context.runtime,
            dynamics.epoch_inputs,
            config["ephemerides"],
            context.working_dir,
            context.next_observation_spec_id(),
        )
    with bar:
        trajectory = integrator.integrate(
            dynamics,
            epoch,
            problem.initial_vector,
            duration,
            external_history_provider=problem.external_history,
            node_diagnostic=collect_diagnostics,
            seed_duration_s=problem.seed_duration_s,
            epoch_source=epoch_source,
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
    epochs = tuple(epoch.shifted(float(offset)) for offset in offsets)
    reference_batch = ephemeris.body_state_matrices_bcrs(("EARTH", "MOON"), epochs)
    integration_vectors = np.asarray([trajectory.vector(t) for t in epochs])
    earth_states = reference_batch[:, 0]
    moon_states = earth_states + integration_vectors[:, :6]
    relative_states = moon_states - earth_states
    reference_relative = reference_batch[:, 1] - earth_states
    reference_r, reference_v = reference_relative[:, :3], reference_relative[:, 3:]
    radial = reference_r / np.linalg.norm(reference_r, axis=1)[:, None]
    normal = np.cross(reference_r, reference_v)
    normal_norm = np.linalg.norm(normal, axis=1)
    if np.any(normal_norm == 0):
        raise ValueError("Reference orbit has undefined RTN frame.")
    normal /= normal_norm[:, None]
    rtn = np.stack((radial, np.cross(normal, radial), normal), axis=1)
    delta = relative_states[:, :3] - reference_r
    emb_states = (
        earth_gravitational_parameter_m3_s2 * earth_states + moon_gravitational_parameter_m3_s2 * moon_states
    ) / (earth_gravitational_parameter_m3_s2 + moon_gravitational_parameter_m3_s2)
    orbit_values = np.column_stack(
        (
            offsets,
            earth_states,
            moon_states,
            emb_states,
            relative_states,
            (rtn @ delta[:, :, None])[:, :, 0],
            relative_states[:, 3:] - reference_v,
        )
    )
    differences = orbit_values[:, -6:-3]
    from importlib.metadata import version

    from lunarops.config.fingerprints import kernel_manifest

    source = ephemeris.source_path
    kernels = kernel_manifest(source) if source is not None and source.is_dir() else []
    common_metadata = {
        "timeScale": "TDB",
        "frame": FRAME_DESCRIPTION,
        "initialTdbJd1": config["initialTdbJd1"],
        "initialTdbJd2": config["initialTdbJd2"],
    }
    write_numeric_table(orbit_output, "lunarOrbit", orbit_columns, orbit_values, metadata=common_metadata)
    diagnostic_columns = ["offset_tdb_s"]
    for name, _force_names in DIAGNOSTIC_FORCE_GROUPS:
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
            "accelerationGroups": {name: force_names for name, force_names in DIAGNOSTIC_FORCE_GROUPS},
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
                config["earthDynamicsFrame"] if config["earthPoleModel"] == "ephemeris" else config["earthPoleModel"]
            ),
            "frame": FRAME_DESCRIPTION,
            "units": "m,s",
            "config": config,
            "initialStateSi": problem.initial_vector,
            "dynamicsBodyNames": dynamics.body_names,
            "solarGmModel": ("constant" if dynamics.solar_gm_variation is None else SOLAR_GM_DESCRIPTION),
            "kuiperBeltRingModel": (None if dynamics.kuiper_belt_ring is None else RING_DESCRIPTION),
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
                    DEFORMED_ATTITUDE_DESCRIPTION
                    if dynamics.rotation.deformation is not None
                    else RIGID_ATTITUDE_DESCRIPTION
                )
                if dynamics.rotation is not None
                else PRESCRIBED_ATTITUDE_DESCRIPTION
            ),
            "tideHistory": (
                JOINT_HISTORY_DESCRIPTION if problem.seed_duration_s is not None else EPHEMERIS_HISTORY_DESCRIPTION
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
    rms = np.sqrt(np.mean(differences**2, axis=0))
    values = {
        "orbit samples": len(orbit_values),
        "diagnostic samples": len(diagnostic_values),
        "RMS vs reference R/T/N": " / ".join(f"{value:.3g}" for value in rms) + " m",
    }
    outputs = [orbit_output, diagnostic_output, metadata_output]
    if problem.rotation_output is not None:
        _write_rotation(problem, trajectory, offsets, common_metadata)
        outputs.append(problem.rotation_output)
    console.summary(values, outputs=outputs)
    return tuple(outputs)


def _write_rotation(problem, trajectory, offsets, metadata):
    """Write the integrated rotation state, torques, angular accelerations and inertia at the output offsets."""
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
        result = problem.dynamics.terms(epoch, state, history=trajectory.history)
        terms = [*result["torques"], *result["rotation"]]
        inertia_terms = []
        if rotation.deformation is not None:
            for tensor in result["inertia"]:
                inertia_terms.append(np.asarray([tensor[i, j] for i, j in inertia_indices]))
        rows.append(
            np.concatenate(
                (
                    [offset],
                    state[6:],
                    [np.linalg.norm(state[6:10]) - 1],
                    *terms,
                    result["derivative"][10:],
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
            "quaternion": "Hamilton scalar-first PA-to-inertial active rotation",
            "angularVelocity": "mantle-frame components, TDB rad/s",
            "torqueNormalization": "N/(whole Moon M R^2), s^-2",
            "inertiaModel": (
                DEFORMED_INERTIA_DESCRIPTION if rotation.deformation is not None else RIGID_INERTIA_DESCRIPTION
            ),
        },
    )
