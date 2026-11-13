"""Lunar orbit (and optional rotation) propagation with comparison against the input ephemeris.

The ``LunarOrbitPropagation`` program integrates the geocentric state of the Moon, in BCRS (SSB/ICRF) axes
and on the TDB time scale, while the Earth-Moon barycentre and the external bodies follow the ephemeris; the
Earth and the Moon are placed about the barycentre and every force acts on both. Optionally the
lunar mantle and core rotation is integrated together with the orbit. The propagated orbit is compared with
the ephemeris Moon, with position differences in the radial/transverse/normal (RTN) frame of the
ephemeris orbit.
"""

from __future__ import annotations

import multiprocessing
import os
from contextlib import ExitStack
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from lunarops import console
from lunarops.classes.dynamics.arc_process import ArcSetup, BackwardArcProcess
from lunarops.classes.dynamics.assembly import DIAGNOSTIC_FORCE_GROUPS, build_lunar_propagation_problem
from lunarops.classes.dynamics.bodies import EMB, earth_moon_about_barycenter
from lunarops.fileio.numeric_table import write_numeric_table
from lunarops.fileio.structured_reports import write_lunar_orbit_metadata
from lunarops.programs.lunar_orbit_schema import LUNAR_ORBIT_FIELDS
from lunarops.programs.registry import ArtifactSlot, program, program_spec

if TYPE_CHECKING:
    from lunarops.classes.time import Epoch

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


class AccelerationDiagnostics:
    """Node diagnostic that records the Moon and Earth accelerations per force group.

    An :class:`~lunarops.classes.dynamics.arc_process.ArcDiagnostic`: it pickles without its dynamics, so that
    a parallel backward arc records its rows in the arc process.

    Parameters
    ----------
    initial_epoch : Epoch
        TDB epoch of offset 0.
    step_s : float
        Main-grid step, in s.
    stride : int or None
        Record every ``stride``-th node from the initial epoch on; nothing when ``None``.

    Attributes
    ----------
    rows : list of numpy.ndarray
        One row per recorded node, in call order: the offset in s, then the Moon and the Earth accelerations
        of each group of ``DIAGNOSTIC_FORCE_GROUPS``, in m/s^2.
    """

    def __init__(self, initial_epoch: Epoch, step_s: float, stride: int | None) -> None:
        self._initial_epoch = initial_epoch
        self._step_s = step_s
        self._stride = stride
        self._dynamics = None
        self.rows: list[np.ndarray] = []

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_dynamics"] = None  # compiled; the receiving process binds its own
        return state

    def bind(self, system) -> None:
        """Evaluate the accelerations with ``system``."""
        self._dynamics = system

    def for_backward_arc(self) -> AccelerationDiagnostics:
        """Return an empty, unbound diagnostic with the same sampling."""
        return AccelerationDiagnostics(self._initial_epoch, self._step_s, self._stride)

    def merge_backward(self, other: AccelerationDiagnostics) -> None:
        """Append the rows recorded on the backward arc."""
        self.rows.extend(other.rows)

    def __call__(self, node_epoch, node_state, batch, index, history) -> None:
        """Record the accelerations at a node on the sampling grid."""
        node_index = round(float(self._initial_epoch.seconds_until(node_epoch)) / self._step_s)
        if self._stride is None or node_index % self._stride:
            return
        offset = float(node_index * self._step_s)
        breakdown = self._dynamics.terms(node_epoch, node_state, batch, index=index, history=history)
        values = [np.array([offset])]
        for key in ("forces", "earth_forces"):
            for _group_name, force_names in DIAGNOSTIC_FORCE_GROUPS:
                values.append(sum((breakdown[key].get(name, np.zeros(3)) for name in force_names), start=np.zeros(3)))
        self.rows.append(np.concatenate(values))


def backward_arc_workers(workers: int, forward_s: float, backward_s: float) -> int:
    """Return how many of ``workers`` worker processes go to a parallel backward arc.

    The arcs take the workers in proportion to their lengths, at least one each; with fewer than two
    workers the backward arc builds its own epoch data.

    Parameters
    ----------
    workers : int
        Worker processes of the run (``--workers``), 0 for a serial run.
    forward_s, backward_s : float
        Lengths of the forward and backward arcs, in s.

    Returns
    -------
    int
        Workers of the backward arc, between 1 and ``workers - 1``, or 0 when ``workers`` < 2.
    """
    if workers < 2:
        return 0
    return min(max(round(workers * backward_s / (forward_s + backward_s)), 1), workers - 1)


def plan_backward_arc(config, runtime) -> int | None:
    """Decide how the backward arc of a run is propagated, and log the decision in one line.

    The backward arc of a two-sided run runs in a process of its own unless ``parallelArcs`` is false, the run
    uses ``--mpi`` or this process is daemonic. A warning is logged when the process may use fewer CPUs than
    the run keeps busy.

    Parameters
    ----------
    config : Mapping
        Resolved ``LunarOrbitPropagation`` config.
    runtime : TaskRuntime or None
        Task runtime of the run.

    Returns
    -------
    int or None
        Worker processes of the backward-arc process (see :func:`backward_arc_workers`); ``None`` when there is
        no backward arc or it follows the forward arc in this process.
    """
    backward = config["backwardDurationSeconds"]
    sequential = "Backward arc: propagated after the forward arc"
    if not backward:
        console.info("Propagation: one-sided, no backward arc")
        return None
    if not config["parallelArcs"]:
        console.info(f"{sequential} (parallelArcs is false)")
        return None
    if runtime is not None and runtime.name == "mpi":
        console.info(f"{sequential} (parallel arcs are not used with --mpi)")
        return None
    if multiprocessing.current_process().daemon:
        console.info(f"{sequential} (a daemonic process cannot start the backward-arc process)")
        return None
    workers = 0 if runtime is None else runtime.workers
    arc_workers = backward_arc_workers(workers, abs(config["durationSeconds"]), backward)
    if workers == 0:
        where = "with its own epoch data"
    elif arc_workers == 0:
        where = "with its own epoch data; the forward arc uses the worker process"
    else:
        where = (
            f"on {arc_workers} of the {workers} worker processes, the forward arc on the other {workers - arc_workers}"
        )
    console.info(f"Backward arc: propagated in parallel, in a process of its own {where}")
    busy = workers + 2
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    if cpus < busy:
        console.warning(
            f"The run keeps {busy} processes busy on {cpus} usable CPUs, so the arcs slow each other down; "
            "use fewer --workers or set parallelArcs: false"
        )
    return arc_workers


def _arc_setup(context_type, model_configs, working_dir: str, config: dict) -> ArcSetup:
    """Build the propagation problem in the backward-arc process (the factory of ``BackwardArcProcess``)."""
    context = context_type(model_configs=model_configs, working_dir=working_dir)
    try:
        problem = build_lunar_propagation_problem(config, context)
    except BaseException:
        context.close()
        raise
    return ArcSetup(
        problem.dynamics,
        problem.integrator,
        epoch_source_spec={"ephemerides": config["ephemerides"], "workingDir": working_dir},
        close=context.close,
    )


def _output_offsets(duration: float, step: float) -> np.ndarray:
    """Return the output offsets ``0, ±step, ...`` ending exactly at ``duration``, without a near-duplicate end."""
    count = int(np.floor(abs(duration) / step + 1e-9))
    offsets = np.arange(count + 1, dtype=float) * step
    if abs(duration) - offsets[-1] > 1e-9 * step:
        offsets = np.append(offsets, abs(duration))
    else:
        offsets[-1] = abs(duration)
    return offsets * np.sign(duration) if duration else offsets


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
    over ``durationSeconds`` (and, for a two-sided run, backward over ``backwardDurationSeconds``) with the
    fixed-step Adams-Bashforth-Moulton integrator (epoch data computed on the runtime workers when the run
    uses ``--workers`` or ``--mpi``; with ``parallelArcs``, the backward arc in a process of its own, see
    :func:`plan_backward_arc`) and writes:

    - ``outputFileOrbit`` (``lunarOrbit``): every ``outputStepSeconds``, the BCRS states of the Earth
      (ephemeris), the propagated Moon and the Earth-Moon barycentre, the geocentric Moon state (m, m/s),
      and its difference to the ephemeris Moon (position in RTN, m; velocity in ICRF axes, m/s);
    - ``outputFileAccelerationDiagnostics`` (``lunarAccelerationDiagnostics``): the Moon acceleration per
      force group (m/s^2), every ``accelerationDiagnosticsStepSeconds``;
    - ``outputFileMetadata`` (``lunarOrbitMetadata``): the resolved config, kernels, model descriptions and
      the RMS and maximum RTN differences to the ephemeris;
    - ``outputFileRotation`` (``lunarRotation``), with ``lunarRotation``: the mantle 3-1-3 Euler angles (rad),
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
    RuntimeError
        If the backward-arc process exits early or stalls; an error raised there is raised here with its
        remote traceback.
    """
    with ExitStack() as stack:
        arc_workers = plan_backward_arc(config, context.runtime)
        backward_arc = None
        if arc_workers is not None:
            # Started first, so that the arc process builds its problem while this process builds its own.
            backward_arc = stack.enter_context(
                BackwardArcProcess(
                    partial(_arc_setup, type(context), context.model_configs, str(context.working_dir), dict(config)),
                    workers=arc_workers,
                    initializer=getattr(context.runtime, "initializer", None),
                )
            )
            backward_arc.start()
        problem = build_lunar_propagation_problem(config, context)
        config = problem.config
        dynamics, integrator = problem.dynamics, problem.integrator
        epoch, duration, backward_duration = problem.initial_epoch, problem.duration_s, problem.backward_duration_s
        diagnostic_step = config["accelerationDiagnosticsStepSeconds"]
        diagnostic_stride = None
        if diagnostic_step is not None:
            diagnostic_ratio = diagnostic_step / config["stepSeconds"]
            if not np.isclose(diagnostic_ratio, round(diagnostic_ratio), rtol=0.0, atol=1e-12):
                raise ValueError("accelerationDiagnosticsStepSeconds must be an integer multiple of stepSeconds")
            diagnostic_stride = round(diagnostic_ratio)
        diagnostics = AccelerationDiagnostics(epoch, config["stepSeconds"], diagnostic_stride)
        diagnostics.bind(dynamics)

        epoch_source = None
        if context.runtime is not None:
            from lunarops.parallel.epoch_stream import RuntimeEpochSource

            epoch_source = RuntimeEpochSource(
                context.runtime,
                dynamics.epoch_inputs,
                config["ephemerides"],
                context.working_dir,
                context.next_observation_spec_id(),
            )
            if arc_workers:
                # The workers of the backward arc are its own; these stay idle meanwhile.
                stack.enter_context(context.runtime.limit(context.runtime.workers - arc_workers))
        node_count = round((abs(duration) + backward_duration) / config["stepSeconds"]) + 1
        with console.progress(total=node_count, desc="integration", unit="steps") as bar:
            trajectory = integrator.integrate(
                dynamics,
                epoch,
                problem.initial_vector,
                duration,
                backward_duration_s=backward_duration,
                external_history_provider=problem.external_history,
                node_diagnostic=diagnostics,
                seed_duration_s=problem.seed_duration_s,
                epoch_source=epoch_source,
                backward_arc=backward_arc,
                progress=bar.update,
            )
    ephemeris = dynamics.ephemeris
    earth_gravitational_parameter_m3_s2 = dynamics.earth_body.gravitational_parameter_m3_s2
    moon_gravitational_parameter_m3_s2 = dynamics.moon_body.gravitational_parameter_m3_s2
    orbit_output, diagnostic_output, metadata_output = problem.output_paths
    diagnostic_rows = diagnostics.rows
    offsets = _output_offsets(duration, config["outputStepSeconds"])
    if backward_duration:
        offsets = np.concatenate((_output_offsets(-backward_duration, config["outputStepSeconds"])[:0:-1], offsets))
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
    emb_states = ephemeris.body_state_matrices_bcrs((EMB,), epochs)[:, 0]
    reference_relative = ephemeris.relative_state_matrices("MOON", "EARTH", epochs)
    integration_vectors = np.asarray([trajectory.vector(t) for t in epochs])
    relative_states = integration_vectors[:, :6]
    earth_states, moon_states = earth_moon_about_barycenter(
        emb_states, relative_states, earth_gravitational_parameter_m3_s2, moon_gravitational_parameter_m3_s2
    )
    reference_r, reference_v = reference_relative[:, :3], reference_relative[:, 3:]
    radial = reference_r / np.linalg.norm(reference_r, axis=1)[:, None]
    normal = np.cross(reference_r, reference_v)
    normal_norm = np.linalg.norm(normal, axis=1)
    if np.any(normal_norm == 0):
        raise ValueError("Reference orbit has undefined RTN frame.")
    normal /= normal_norm[:, None]
    rtn = np.stack((radial, np.cross(normal, radial), normal), axis=1)
    delta = relative_states[:, :3] - reference_r
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

    from lunarops.classes.ephemerides.kernels import kernel_manifest

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
    for body in ("moon", "earth"):
        for name, _force_names in DIAGNOSTIC_FORCE_GROUPS:
            diagnostic_columns.extend(f"{name}_{body}_{axis}_mps2" for axis in ("x", "y", "z"))
    diagnostic_values = (
        np.vstack(diagnostic_rows) if diagnostic_rows else np.empty((0, len(diagnostic_columns)), dtype=float)
    )
    if backward_duration:
        diagnostic_values = diagnostic_values[np.argsort(diagnostic_values[:, 0])]
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
            "targets": "modeled Moon and Earth about the ephemeris Earth-Moon barycentre; the integrated "
            "acceleration is moon minus earth",
        },
    )
    write_lunar_orbit_metadata(
        metadata_output,
        {
            "model": f"Earth-Moon relative about the ephemeris EMB; external trajectories prescribed; {config['earthDynamicsFrame']} development force model",
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
            "earthTideExtrapolatedTranslationQueries": (
                None if trajectory.history is None else trajectory.history.extrapolated_translation_queries
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
        *(f"euler_{angle}_rad" for angle in ("phi", "theta", "psi")),
        *(f"mantle_omega_pa_{axis}_rad_s" for axis in "xyz"),
    ]
    if rotation.core is not None:
        columns.extend(f"core_omega_pa_{axis}_rad_s" for axis in "xyz")
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
        result = problem.dynamics.terms(epoch, state, history=trajectory.history_at(epoch))
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
                    *terms,
                    result["derivative"][9:],
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
            "eulerAngles": "PA-to-inertial Rz(phi) Rx(theta) Rz(psi), rad; phi and psi are not wrapped",
            "angularVelocity": "mantle-frame components, TDB rad/s",
            "torqueNormalization": "N/(whole Moon M R^2), s^-2",
            "inertiaModel": (
                DEFORMED_INERTIA_DESCRIPTION if rotation.deformation is not None else RIGID_INERTIA_DESCRIPTION
            ),
        },
    )
