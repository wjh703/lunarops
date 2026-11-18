"""Lunar orbit (and optional rotation) propagation with comparison against the input ephemeris.

The ``LunarOrbitPropagation`` program integrates the geocentric state of the Moon, in BCRS (SSB/ICRF) axes
and on the TDB time scale, while the Earth-Moon barycentre and the external bodies follow the ephemeris; the
Earth and the Moon are placed about the barycentre and every force acts on both. Optionally the
lunar mantle and core rotation is integrated together with the orbit. The propagated orbit is compared with
the ephemeris Moon, with position differences in the radial/transverse/normal (RTN) frame of the
ephemeris orbit.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import os
from contextlib import ExitStack
from functools import partial
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

from lunarops import console
from lunarops.classes.dynamics.arc_process import ArcSetup, BackwardArcProcess
from lunarops.classes.dynamics.assembly import (
    DIAGNOSTIC_FORCE_GROUPS,
    LunarPropagationProblem,
    build_lunar_propagation_problem,
)
from lunarops.classes.dynamics.bodies import EMB, earth_moon_about_barycenter
from lunarops.classes.dynamics.epoch_data import SerialEpochSource
from lunarops.classes.dynamics.fingerprint import model_fingerprint
from lunarops.classes.dynamics.force_models import enabled_force_terms
from lunarops.classes.dynamics.forces import EarthTideParameters
from lunarops.classes.dynamics.integrators import IntegratorSettings, check_durations
from lunarops.classes.dynamics.parameters import (
    INITIAL_VALUE_NAMES,
    build_parameter_variations,
    parameter_definition,
)
from lunarops.classes.dynamics.variational import VariationalEquations
from lunarops.classes.ephemerides.kernels import kernel_manifest
from lunarops.fileio.lunar_partials import LunarDynamicsPartials, write_lunar_dynamics_partials
from lunarops.fileio.numeric_table import write_numeric_table
from lunarops.fileio.structured_reports import write_lunar_orbit_metadata
from lunarops.parallel.epoch_stream import RuntimeEpochSource
from lunarops.parallel.process import ProcessRuntime
from lunarops.programs.lunar_orbit_schema import LUNAR_ORBIT_FIELDS, LUNAR_ORBIT_REMOVED_KEYS
from lunarops.programs.registry import ArtifactSlot, program, program_spec

if TYPE_CHECKING:
    from lunarops.classes.dynamics.lunar_dynamics import LunarDynamics
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
        self._dynamics: LunarDynamics | None = None
        self.rows: list[np.ndarray] = []

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_dynamics"] = None  # compiled; the receiving process binds its own
        return state

    def bind(self, system: LunarDynamics) -> None:
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
        assert self._dynamics is not None, "bind the dynamics first"
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


def _advanced_delay_s(config) -> float:
    """Return the ``LunarDynamics.advanced_delay_s`` of the dynamics that ``config`` builds, in s."""
    if "tide" not in enabled_force_terms(config):
        return 0.0
    return EarthTideParameters(tau_orb_days=config["earthTide"]["tauOrbDays"]).advanced_delay_s


def plan_backward_arc(config, runtime) -> int | None:
    """Decide how the backward arc of a run is propagated, and log the decision in one line.

    The backward arc of a two-sided run runs in a process of its own unless ``parallelArcs`` is false, the run
    uses a task runtime other than ``--workers`` (a :class:`~lunarops.parallel.process.ProcessRuntime`), for
    example ``--mpi``, or this process is daemonic. A warning is logged when the process may use fewer CPUs
    than the run keeps busy. The arcs of a two-sided run are checked first, as the integrator checks them, so
    that an invalid run fails before any process is started.

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

    Raises
    ------
    ValueError
        If the run is two-sided and ``durationSeconds`` is not positive, the forward arc is too short to seed
        the backward arc, or either duration is not a multiple of ``stepSeconds`` (see
        :func:`~lunarops.classes.dynamics.integrators.check_durations`).
    """
    backward = config["backwardDurationSeconds"]
    sequential = "Backward arc: propagated after the forward arc"
    if not backward:
        console.info("Propagation: one-sided, no backward arc")
        return None
    settings = IntegratorSettings(step_s=config["stepSeconds"], order=config["integratorOrder"])
    try:
        check_durations(settings, config["durationSeconds"], backward, _advanced_delay_s(config))
    except ValueError as error:
        raise ValueError(f"durationSeconds and backwardDurationSeconds: {error}") from error
    if not config["parallelArcs"]:
        console.info(f"{sequential} (parallelArcs is false)")
        return None
    if runtime is not None and not isinstance(runtime, ProcessRuntime):
        console.info(f"{sequential} (parallel arcs need --workers or no runtime; this run uses {runtime.name})")
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


def _stride(config, key: str) -> int | None:
    """Return the node stride of the sampling interval at ``key``: the interval divided by ``stepSeconds``.

    ``key`` is a top-level key or ``mapping.key`` for an entry of a nested mapping; the stride is ``None`` when
    the interval or its mapping is null.

    Raises
    ------
    ValueError
        If the interval is not a whole multiple of ``stepSeconds``.
    """
    *parents, name = key.split(".")
    holder = config
    for parent in parents:
        holder = holder[parent]
        if holder is None:
            return None
    interval = holder[name]
    if interval is None:
        return None
    step = config["stepSeconds"]
    ratio = interval / step
    stride = round(ratio)
    if stride < 1 or not np.isclose(ratio, stride, rtol=0.0, atol=1e-12):
        raise ValueError(f"{key} ({interval:g} s) must be a whole multiple of stepSeconds ({step:g} s)")
    return stride


def variational_output_stride(config) -> int | None:
    """Return the node stride of the partials table of a run.

    Parameters
    ----------
    config : Mapping
        Resolved ``LunarOrbitPropagation`` config.

    Returns
    -------
    int or None
        ``variationalEquations.outputStepSeconds / stepSeconds``; ``None`` without ``variationalEquations``.

    Raises
    ------
    ValueError
        If ``variationalEquations.outputStepSeconds`` is not a whole multiple of ``stepSeconds``.
    """
    return _stride(config, "variationalEquations.outputStepSeconds")


#: Output slots that a model setting needs, and that are allowed only with it.
_OUTPUTS_OF_SETTINGS = (("lunarRotation", "outputFileRotation"), ("variationalEquations", "outputFilePartials"))
#: Sampling intervals that must be whole multiples of ``stepSeconds``.
_SAMPLING_INTERVALS = ("accelerationDiagnosticsStepSeconds", "variationalEquations.outputStepSeconds")


def _validate_config(config, path):
    """Check the force terms, the rotation and partials output slots, and the sampling intervals."""
    try:
        enabled_force_terms(config)
    except ValueError as error:
        raise ValueError(f"{path}.{error}.") from None
    for setting, output in _OUTPUTS_OF_SETTINGS:
        if config.get(setting) is None and config.get(output) is not None:
            raise ValueError(f"{path}.{output} is given, but {setting} is null.")
        if config.get(setting) is not None and config.get(output) is None:
            raise ValueError(f"{path}.{setting} requires {output}.")
    for key in _SAMPLING_INTERVALS:
        try:
            _stride(config, key)
        except ValueError as error:
            raise ValueError(f"{path}.{error}.") from None
    return config


def _parameter_variations(problem, context):
    """Return the parameter variations of the configured variational equations; ``None`` without them."""
    variational = problem.config.get("variationalEquations")
    if variational is None:
        return None
    return build_parameter_variations(problem, context, variational["parameters"])


def _arc_setup(context_type, model_configs, working_dir: str, config: dict) -> ArcSetup:
    """Build the propagation problem in the backward-arc process (the factory of ``BackwardArcProcess``)."""
    context = context_type(model_configs=model_configs, working_dir=working_dir)
    try:
        problem = build_lunar_propagation_problem(config, context)
        variations = _parameter_variations(problem, context)
    except BaseException:
        context.close()
        raise
    return ArcSetup(
        problem.dynamics,
        problem.integrator,
        epoch_source_spec={"ephemerides": config["ephemerides"], "workingDir": working_dir},
        close=context.close,
        parameter_variations=variations,
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
            ArtifactSlot(
                "inputFileLunarDynamicsParameters",
                "lunarDynamicsParameters",
                required=False,
                description=(
                    "Initial values and dynamical parameters of the lunar orbit and rotation at the initial epoch. "
                    "Each listed quantity replaces its value from the ephemeris, this configuration, the mass "
                    "catalog or the Moon gravity field; quantities that are not listed keep those values."
                ),
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
            ArtifactSlot(
                "outputFilePartials",
                "lunarDynamicsPartials",
                binary=True,
                required=False,
                description=(
                    "Output binary table of the state and its partials with respect to the initial values and the "
                    "dynamical parameters. Required with `variationalEquations`, and only then allowed."
                ),
            ),
        ),
        fields=LUNAR_ORBIT_FIELDS,
        validator=_validate_config,
        removed_keys=LUNAR_ORBIT_REMOVED_KEYS,
    )
)
def lunar_orbit_propagation(config, context):
    """Propagate the Moon from the configured initial TDB epoch and compare it with the ephemeris.

    Reads the GMs from the ``massCatalog`` ``inputFileMassCatalog`` and, when given, initial values and
    dynamical parameters from the ``lunarDynamicsParameters`` ``inputFileLunarDynamicsParameters`` (see
    :mod:`~lunarops.classes.dynamics.parameters`), integrates the configured force model
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
      mantle and core angular velocities (PA axes, rad/s), torques and, with deformation, inertia tensors;
    - ``outputFilePartials`` (``lunarDynamicsPartials``), with ``variationalEquations``: every
      ``variationalEquations.outputStepSeconds`` from the initial epoch, the integration state and its partials
      with respect to the initial values and the configured dynamical parameters (see
      :class:`~lunarops.classes.dynamics.variational.VariationalEquations` and
      :mod:`~lunarops.fileio.lunar_partials`). The partials are integrated beside the state and do not change
      it; the parameter variations are built in the backward-arc process too, so a parallel backward arc
      gives the partials of a sequential one bit for bit.

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
        the rotation is integrated and the partials file with ``variationalEquations``.

    Raises
    ------
    ValueError
        If ``accelerationDiagnosticsStepSeconds`` or ``variationalEquations.outputStepSeconds`` is not a whole
        multiple of ``stepSeconds``, a listed variational parameter is not a parameter of the model, or the RTN
        frame of the ephemeris orbit is undefined.
    RuntimeError
        If the backward-arc process exits early or stalls; an error raised there is raised here with its
        remote traceback.
    """
    run = _integrate(config, context)
    problem, trajectory, variational = run.problem, run.trajectory, run.variational
    config = problem.config
    orbit_output, diagnostic_output, metadata_output = problem.output_paths
    offsets = _output_offsets(problem.duration_s, config["outputStepSeconds"])
    if problem.backward_duration_s:
        offsets = np.concatenate(
            (_output_offsets(-problem.backward_duration_s, config["outputStepSeconds"])[:0:-1], offsets)
        )
    orbit = _orbit_table(problem, trajectory, offsets)
    common_metadata = {
        "timeScale": "TDB",
        "frame": FRAME_DESCRIPTION,
        "initialTdbJd1": config["initialTdbJd1"],
        "initialTdbJd2": config["initialTdbJd2"],
    }
    write_numeric_table(orbit_output, "lunarOrbit", ORBIT_COLUMNS, orbit.values, metadata=common_metadata)
    partials_output = None
    if variational is not None:
        partials_output = context.resolve_path(config["outputFilePartials"])
        _write_partials(partials_output, problem, trajectory, variational, run.increments)
    diagnostic_count = _write_diagnostics(
        diagnostic_output, run.diagnostics, problem.backward_duration_s, common_metadata
    )
    write_lunar_orbit_metadata(
        metadata_output, _metadata(problem, trajectory, orbit, diagnostic_count, partials_output)
    )
    values = {
        "orbit samples": len(orbit.values),
        "diagnostic samples": diagnostic_count,
        "RMS vs reference R/T/N": " / ".join(f"{value:.3g}" for value in orbit.rtn_rms_m) + " m",
    }
    outputs = [orbit_output, diagnostic_output, metadata_output]
    if problem.rotation_output is not None:
        rotation_source = _runtime_epoch_source(context, _rotation_epoch_inputs(problem.dynamics), config)
        _write_rotation(problem, trajectory, offsets, orbit.integration_vectors, common_metadata, rotation_source)
        outputs.append(problem.rotation_output)
    if partials_output is not None:
        assert variational is not None
        values["partials samples"] = len(variational.times)
        outputs.append(partials_output)
    console.summary(values, outputs=outputs)
    return tuple(outputs)


@dataclasses.dataclass(frozen=True)
class _Integration:
    """Propagation problem and integrated trajectory of a run, with what was recorded beside the state."""

    problem: LunarPropagationProblem
    trajectory: object
    diagnostics: AccelerationDiagnostics
    variational: VariationalEquations | None
    increments: dict[str, float]


def _runtime_epoch_source(context, epoch_inputs, config) -> RuntimeEpochSource | None:
    """Return a source that builds ``epoch_inputs`` on the runtime workers; ``None`` for a serial run."""
    if context.runtime is None:
        return None
    return RuntimeEpochSource(
        context.runtime,
        epoch_inputs,
        config["ephemerides"],
        context.working_dir,
        context.next_observation_spec_id(),
    )


def _integrate(config, context) -> _Integration:
    """Build the propagation problem and integrate it, with a parallel backward arc when planned."""
    with ExitStack() as stack:
        arc_workers = plan_backward_arc(config, context.runtime)
        backward_arc = None
        if arc_workers is not None:
            # Started first, so that the arc process builds its problem while this process builds its own.
            backward_arc = stack.enter_context(
                BackwardArcProcess(
                    partial(_arc_setup, type(context), context.model_configs, str(context.working_dir), dict(config)),
                    workers=arc_workers,
                    initializer=context.runtime.initializer if isinstance(context.runtime, ProcessRuntime) else None,
                )
            )
            backward_arc.start()
        problem = build_lunar_propagation_problem(config, context)
        config = problem.config
        dynamics, integrator = problem.dynamics, problem.integrator
        diagnostics = AccelerationDiagnostics(
            problem.initial_epoch, config["stepSeconds"], _stride(config, "accelerationDiagnosticsStepSeconds")
        )
        diagnostics.bind(dynamics)
        variational, increments = None, {}
        partials_stride = variational_output_stride(config)
        if partials_stride is not None:
            variations = _parameter_variations(problem, context)
            increments = {variation.name: variation.increment for variation in variations}
            variational = VariationalEquations(
                dynamics, integrator.settings, parameters=variations, output_stride=partials_stride
            )
            console.info(
                f"Variational equations: {variational.dimension} initial values and {len(variations)} dynamical "
                f"parameters, every {partials_stride} steps"
            )

        epoch_source = _runtime_epoch_source(context, dynamics.epoch_inputs, config)
        if epoch_source is not None and arc_workers and isinstance(context.runtime, ProcessRuntime):
            # The workers of the backward arc are its own; these stay idle meanwhile.
            stack.enter_context(context.runtime.limit(context.runtime.workers - arc_workers))
        node_count = round((abs(problem.duration_s) + problem.backward_duration_s) / config["stepSeconds"]) + 1
        with console.progress(total=node_count, desc="integration", unit="steps") as bar:
            trajectory = integrator.integrate(
                dynamics,
                problem.initial_epoch,
                problem.initial_vector,
                problem.duration_s,
                backward_duration_s=problem.backward_duration_s,
                external_history_provider=problem.external_history,
                node_diagnostic=diagnostics,
                seed_duration_s=problem.seed_duration_s,
                epoch_source=epoch_source,
                backward_arc=backward_arc,
                progress=bar.update,
                variational=variational,
            )
    return _Integration(problem, trajectory, diagnostics, variational, increments)


ORBIT_COLUMNS = (
    "offset_tdb_s",
    *(f"earth_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")),
    *(f"moon_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")),
    *(f"emb_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")),
    *(f"relative_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")),
    *(f"reference_difference_rtn_{axis}_m" for axis in ("r", "t", "n")),
    *(f"reference_difference_velocity_{axis}_mps" for axis in ("x", "y", "z")),
)
"""Columns of the ``lunarOrbit`` table."""


@dataclasses.dataclass(frozen=True)
class _OrbitTable:
    """Rows of the ``lunarOrbit`` table (``ORBIT_COLUMNS``), the integration states at its offsets and RTN stats."""

    values: np.ndarray
    integration_vectors: np.ndarray
    rtn_rms_m: np.ndarray
    rtn_max_abs_m: np.ndarray


def _orbit_table(problem, trajectory, offsets) -> _OrbitTable:
    """Evaluate the propagated and the ephemeris orbit at ``offsets`` and difference them in RTN.

    Raises
    ------
    ValueError
        If the RTN frame of the ephemeris orbit is undefined.
    """
    dynamics = problem.dynamics
    ephemeris = dynamics.ephemeris
    epochs = tuple(problem.initial_epoch.shifted(float(offset)) for offset in offsets)
    emb_states = ephemeris.body_state_matrices_bcrs((EMB,), epochs)[:, 0]
    reference_relative = ephemeris.relative_state_matrices("MOON", "EARTH", epochs)
    integration_vectors = np.asarray([trajectory.vector(t) for t in epochs])
    relative_states = integration_vectors[:, :6]
    earth_states, moon_states = earth_moon_about_barycenter(
        emb_states,
        relative_states,
        dynamics.earth_body.gravitational_parameter_m3_s2,
        dynamics.moon_body.gravitational_parameter_m3_s2,
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
    values = np.column_stack(
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
    differences = values[:, -6:-3]
    return _OrbitTable(
        values,
        integration_vectors,
        np.sqrt(np.mean(differences**2, axis=0)),
        np.max(np.abs(differences), axis=0),
    )


def _write_diagnostics(path, diagnostics, backward_duration_s, common_metadata) -> int:
    """Write the ``lunarAccelerationDiagnostics`` table in offset order; return its row count."""
    columns = ["offset_tdb_s"]
    for body in ("moon", "earth"):
        for name, _force_names in DIAGNOSTIC_FORCE_GROUPS:
            columns.extend(f"{name}_{body}_{axis}_mps2" for axis in ("x", "y", "z"))
    rows = diagnostics.rows
    values = np.vstack(rows) if rows else np.empty((0, len(columns)), dtype=float)
    if backward_duration_s:
        values = values[np.argsort(values[:, 0])]
    write_numeric_table(
        path,
        "lunarAccelerationDiagnostics",
        tuple(columns),
        values,
        metadata={
            **common_metadata,
            "accelerationGroups": {name: force_names for name, force_names in DIAGNOSTIC_FORCE_GROUPS},
            "figureGroup": "solar J2 + static Earth figure + static Moon figure",
            "lunarDegree2GravityGroup": "dynamic Moon degree-2 field minus static Moon field",
            "targets": "modeled Moon and Earth about the ephemeris Earth-Moon barycentre; the integrated "
            "acceleration is moon minus earth",
        },
    )
    return len(values)


def _metadata(problem, trajectory, orbit: _OrbitTable, diagnostic_count: int, partials_output) -> dict[str, object]:
    """Return the ``lunarOrbitMetadata`` payload of a run."""
    config, dynamics = problem.config, problem.dynamics
    orbit_output, diagnostic_output, _ = problem.output_paths
    source = dynamics.ephemeris.source_path
    pole_model = config["earthPoleModel"]
    if dynamics.rotation is None:
        attitude = PRESCRIBED_ATTITUDE_DESCRIPTION
    elif dynamics.rotation.deformation is not None:
        attitude = DEFORMED_ATTITUDE_DESCRIPTION
    else:
        attitude = RIGID_ATTITUDE_DESCRIPTION
    history = trajectory.history
    return {
        "model": (
            "Earth-Moon relative about the ephemeris EMB; external trajectories prescribed; "
            f"{pole_model} development force model"
        ),
        "timeScale": "TDB",
        "earthPoleModel": pole_model,
        "frame": FRAME_DESCRIPTION,
        "units": "m,s",
        "config": config,
        "initialStateSi": problem.initial_vector,
        "dynamicsBodyNames": dynamics.body_names,
        "solarGmModel": "constant" if dynamics.solar_gm_variation is None else SOLAR_GM_DESCRIPTION,
        "kuiperBeltRingModel": None if dynamics.kuiper_belt_ring is None else RING_DESCRIPTION,
        "kernels": kernel_manifest(source) if source is not None and source.is_dir() else [],
        "scipyVersion": version("scipy"),
        "resolvedForceInventory": config["resolvedForceInventory"],
        "resolvedGravityFields": config["resolvedGravityFields"],
        "integrator": f"modified ABMD order {config['integratorOrder']}",
        "stepSeconds": config["stepSeconds"],
        "startupIntegrator": "Dormand-Prince 8",
        "startupStepSeconds": config["startupStepSeconds"],
        "lunarAttitudeMode": attitude,
        "tideHistory": (
            JOINT_HISTORY_DESCRIPTION if problem.seed_duration_s is not None else EPHEMERIS_HISTORY_DESCRIPTION
        ),
        "rotationSeedDurationSeconds": problem.seed_duration_s,
        "earthTideExtrapolatedTranslationQueries": None
        if history is None
        else history.extrapolated_translation_queries,
        "earthTideMaxTranslationExtrapolationSeconds": (
            None if history is None else history.max_translation_extrapolation_s
        ),
        "solarForceTerms": problem.solar_force_terms,
        "referenceDifferenceRtnRmsM": orbit.rtn_rms_m,
        "referenceDifferenceRtnMaxAbsM": orbit.rtn_max_abs_m,
        "orbitFile": str(orbit_output),
        "accelerationDiagnosticsFile": str(diagnostic_output),
        "orbitSamples": len(orbit.values),
        "accelerationDiagnosticSamples": diagnostic_count,
        "rotationFile": None if problem.rotation_output is None else str(problem.rotation_output),
        "partialsFile": None if partials_output is None else str(partials_output),
    }


def _write_partials(path, problem, trajectory, variational, increments):
    """Write the state and the partials at the epochs where the variational equations stored them."""
    offsets = variational.times
    epoch = problem.initial_epoch
    states = np.asarray([trajectory.vector(epoch.shifted(float(offset))) for offset in offsets])
    dimension = variational.dimension
    quantities = (*INITIAL_VALUE_NAMES[:dimension], *variational.parameter_names)
    config = problem.config
    write_lunar_dynamics_partials(
        path,
        LunarDynamicsPartials(
            config["initialTdbJd1"],
            config["initialTdbJd2"],
            offsets,
            states,
            variational.partials,
            quantities,
            tuple(parameter_definition(name).unit for name in quantities),
            producer="variational",
            metadata={
                "frame": FRAME_DESCRIPTION,
                "modelFingerprint": model_fingerprint(problem.dynamics, problem.integrator.settings),
                "stepSeconds": config["stepSeconds"],
                "outputStepSeconds": config["variationalEquations"]["outputStepSeconds"],
                "parameterIncrements": increments,
            },
        ),
    )


_ROTATION_OUTPUT_BLOCK = 512
"""Output epochs per epoch-inputs batch of the rotation table."""


def _rotation_epoch_inputs(dynamics):
    """Return the epoch inputs of the rotation table: those of the integration without the source/source terms.

    The table holds torques, core-mantle terms, angular accelerations and inertia, none of which depends on
    the accelerations and potentials of the source bodies at each other (``EpochInputs.point_mass``), which
    only the EIH terms of the translation use; their N² body pairs are four fifths of the cost of the inputs.
    """
    return dataclasses.replace(dynamics.epoch_inputs, point_mass=False)


def _blocked_rows(problem, trajectory, blocks, batches):
    """Yield ``(row, offset, epoch, batch, index, step)`` for every offset of ``blocks``.

    ``batch`` is the epoch inputs of the block, ``step`` the step data of the epoch, taken together for the
    epochs of a block that share a history (``LunarDynamics.prepare_steps``).
    """
    dynamics = problem.dynamics
    row = 0
    for block, batch in zip(blocks, batches, strict=True):
        epochs = [problem.initial_epoch.shifted(float(offset)) for offset in block]
        histories = [trajectory.history_at(epoch) for epoch in epochs]
        steps = []
        start = 0
        while start < len(epochs):
            stop = start + 1
            while stop < len(epochs) and histories[stop] is histories[start]:
                stop += 1
            steps += dynamics.prepare_steps(epochs[start:stop], batch, histories[start], range(start, stop))
            start = stop
        for index, (offset, epoch, step) in enumerate(zip(block, epochs, steps, strict=True)):
            yield row, offset, epoch, batch, index, step
            row += 1


def _write_rotation(problem, trajectory, offsets, states, metadata, epoch_source=None):
    """Write the integrated rotation state, torques, angular accelerations and inertia at the output offsets.

    ``states`` are the integration states at ``offsets``; ``epoch_source`` builds the epoch inputs of
    :func:`_rotation_epoch_inputs`, in this process when ``None``.
    """
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
    dynamics = problem.dynamics
    if epoch_source is None:
        epoch_source = SerialEpochSource(_rotation_epoch_inputs(dynamics), dynamics.ephemeris)
    # Epoch inputs in blocks, built by the workers when there are any: one inputs build per epoch would cost
    # more than the terms themselves.
    blocks = [
        offsets[start : start + _ROTATION_OUTPUT_BLOCK] for start in range(0, len(offsets), _ROTATION_OUTPUT_BLOCK)
    ]
    batches = epoch_source.stream(problem.initial_epoch, [[float(offset) for offset in block] for block in blocks])
    rows = []
    for row, offset, epoch, batch, index, step in _blocked_rows(problem, trajectory, blocks, batches):
        state = states[row]
        result = dynamics.terms(epoch, state, batch, index=index, step=step, rotation_only=True)
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
