"""Shared building blocks of the LLR residual and processing programs.

The functions here read the normal points of a program config, build the observation options and the
parametrization, and assemble the equation source that evaluates the linearized observation equations of all
datasets at each iteration, either serially in this process or distributed over the workers of a task runtime
(MPI ranks or local worker processes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lunarops.config.context import RunContext

if TYPE_CHECKING:
    from lunarops.classes.observation import LlrObservationProcessor, ObservationOptions
    from lunarops.parallel.runtime import TaskRuntime


def load_datasets(config: dict, context: RunContext):
    """Read the normal-point files of a program config.

    Every supported file found under ``inputFilesNormalPoints`` is read and restricted to the time window
    given by ``startTime`` and ``endTime`` (civil time with ``utcOffsetHours``, converted to UTC). Files with
    no remaining records are dropped. Records receive consecutive global indices across all datasets in
    file order.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context used to resolve relative paths.

    Returns
    -------
    dict of str to NptDataset
        Datasets keyed by file stem.

    Raises
    ------
    FileNotFoundError
        If no supported normal-point file is found.
    ValueError
        If no normal point remains after time filtering.
    """
    from lunarops.fileio.formats.normal_point_sources import resolve_normal_point_inputs
    from lunarops.fileio.normal_points import read_normal_points

    inputs = config["inputFilesNormalPoints"]
    input_files = resolve_normal_point_inputs([context.resolve_path(item) for item in inputs])
    if not input_files:
        raise FileNotFoundError(f"No supported normal-point files found under {inputs!r}")

    datasets = {}
    from lunarops.programs.observation_schema import observation_time_window

    start, end = observation_time_window(config)

    for path in input_files:
        dataset = read_normal_points(path)
        if start or end:
            dataset = dataset.filter_time(start, end)
        if dataset.records:
            datasets[Path(path).stem] = dataset

    next_index = 0
    for dataset in datasets.values():
        dataset.assign_indices(start=next_index)
        next_index += len(dataset.records)
    if not datasets:
        raise ValueError("No normal points remain after time filtering.")
    return datasets


def make_processing_options(
    config: dict,
    *,
    include_design: bool = False,
    include_station_design: bool = False,
    include_orientation_design: bool = False,
    include_moon_position_design: bool = False,
    include_lunar_tide_design: bool = False,
):
    """Build the observation options from a program config.

    Reads ``overrideStation``, ``overrideReflector``, ``minElevationDeg``, ``lowElevationPolicy``,
    ``utcOffsetHours`` and ``showProgress``.

    Parameters
    ----------
    config : dict
        Resolved program config.
    include_design : bool, optional
        Compute partial derivatives with respect to the reflector positions.
    include_station_design : bool, optional
        Compute partial derivatives with respect to the station positions.
    include_orientation_design : bool, optional
        Compute partial derivatives with respect to the lunar orientation.
    include_moon_position_design : bool, optional
        Compute partial derivatives with respect to the Moon-minus-Earth position.
    include_lunar_tide_design : bool, optional
        Compute the partial derivative with respect to the lunar Love number h2.

    Returns
    -------
    ObservationOptions
        Options for the observation processor.
    """
    from lunarops.classes.observation import ObservationOptions

    return ObservationOptions(
        station_identifier=config.get("overrideStation"),
        reflector_identifier=config.get("overrideReflector"),
        min_elevation_deg=float(config["minElevationDeg"]),
        low_elevation_policy=config["lowElevationPolicy"],
        utc_offset_hours=float(config["utcOffsetHours"]),
        include_reflector_position_partials=bool(include_design),
        include_station_position_partials=bool(include_station_design),
        include_lunar_orientation_partials=bool(include_orientation_design),
        include_moon_position_partials=bool(include_moon_position_design),
        include_lunar_tide_partials=bool(include_lunar_tide_design),
        show_progress=bool(config["showProgress"]),
    )


def output_level(config: dict):
    """Return the detail level of the observation results.

    Parameters
    ----------
    config : dict
        Resolved program config; ``outputLevel`` selects the level.

    Returns
    -------
    ObservationResultDetail
        The parsed ``outputLevel``.
    """
    from lunarops.classes.observation import ObservationResultDetail

    return ObservationResultDetail.parse(config["outputLevel"])


def build_parametrization(config: dict, context: RunContext, model_state, observation_model=None):
    """Create the parametrization blocks listed under ``parametrization``.

    ``stationPosition`` and ``reflectorPosition`` blocks also receive the model state, and a
    ``lunarLoveNumber`` block the lunar solid tide model of the observation model.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context passed to the class factories.
    model_state : object
        Observation model state holding the station and reflector catalogs.
    observation_model : LlrObservationModel, optional
        Observation model of the master process; needed by a ``lunarLoveNumber`` block.

    Returns
    -------
    ParametrizationList
        The blocks in config order.

    Raises
    ------
    ValueError
        If a ``lunarLoveNumber`` block is configured and the observation model has no lunar solid tide.
    """
    from lunarops.classes.parametrization.base import ParametrizationList
    from lunarops.config.registry import create

    def dependencies(kind: str) -> dict:
        if kind in ("stationPosition", "reflectorPosition"):
            return {"model_state": model_state}
        if kind == "lunarLoveNumber":
            models = (
                () if observation_model is None else observation_model.light_time_solver.reflector_displacement_models
            )
            tides = [model for model in models if hasattr(model, "displacement_per_h2_lcrs_m")]
            return {"lunar_solid_tide": tides[0] if tides else None}
        return {}

    blocks = [
        create("parametrization", item, context, **dependencies(item["type"])) for item in config["parametrization"]
    ]
    return ParametrizationList(blocks)


@dataclass(frozen=True, slots=True)
class WorkerTasks:
    """Task runtime of a parallel run, with the observation spec from which its workers build their models.

    Attributes
    ----------
    runtime : TaskRuntime
        Task runtime of the master (``MpiRuntime`` or ``ProcessRuntime``).
    spec : dict
        Observation spec from ``make_observation_spec``.
    chunksize : int
        Number of items (normal points or epochs) per worker task, from ``tasks.chunksize``.
    """

    runtime: TaskRuntime
    spec: dict
    chunksize: int


def worker_tasks(config: dict, context: RunContext, *, processor=None) -> WorkerTasks | None:
    """Return the worker set-up of a program run, or None when the run is serial.

    A run is parallel when its context has a task runtime (``--mpi`` or ``--workers``): the programs then
    evaluate their work on the workers, which build their models from the observation spec built here.
    Otherwise they evaluate it in this process.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context, providing the optional task runtime.
    processor : LlrObservationProcessor, optional
        Observation processor of the master; the station and reflector catalogs of its model state are sent
        to the workers instead of those of the config.

    Returns
    -------
    WorkerTasks or None
        The runtime, spec and task size; None without a task runtime.
    """
    if context.runtime is None:
        return None
    from lunarops.parallel.observation_spec import make_observation_spec

    catalogs = {}
    if processor is not None:
        state = processor.model_state
        catalogs = {"station_catalog": state.station_catalog, "reflector_catalog": state.reflector_catalog}
    return WorkerTasks(
        context.runtime, make_observation_spec(config, context, **catalogs), int(config["tasks"]["chunksize"])
    )


@dataclass(frozen=True, slots=True)
class SerialEquationBackend:
    """Evaluate the observation equations in this process.

    Attributes
    ----------
    processor : LlrObservationProcessor
        Processor that evaluates the observation equations of one dataset.
    """

    processor: LlrObservationProcessor

    def evaluate(self, datasets, options, state):
        """Return the observation equations of each dataset.

        Parameters
        ----------
        datasets : dict of str to NptDataset
            Datasets keyed by source name.
        options : ObservationOptions
            Observation options.
        state : object
            Unused; present for interface compatibility with ``RuntimeEquationBackend``.

        Returns
        -------
        dict of str to EquationBatch
            Equations keyed by source name.
        """
        return {source: self.processor.equations(dataset, options=options) for source, dataset in datasets.items()}


@dataclass(frozen=True, slots=True)
class RuntimeEquationBackend:
    """Evaluate the observation equations on the workers of a task runtime.

    Attributes
    ----------
    tasks : WorkerTasks
        Task runtime, observation spec and number of normal points per worker task.
    """

    tasks: WorkerTasks

    def evaluate(self, datasets, options, state):
        """Return the observation equations of each dataset, computed by the workers.

        Parameters
        ----------
        datasets : dict of str to NptDataset
            Datasets keyed by source name.
        options : ObservationOptions
            Observation options.
        state : object
            Observation model state whose current station and reflector catalogs are sent to the workers.

        Returns
        -------
        dict of str to EquationBatch
            Equations keyed by source name.
        """
        from lunarops.parallel.observation_spec import snapshot_catalog_state
        from lunarops.parallel.observation_tasks import distributed_observations

        return distributed_observations(
            self.tasks.runtime,
            self.tasks.spec,
            datasets,
            options,
            rows=False,
            chunksize=self.tasks.chunksize,
            catalog_state=snapshot_catalog_state(state),
            progress_desc=options.progress_description,
            quiet=not options.show_progress,
        )


@dataclass(frozen=True, slots=True)
class EquationSource:
    """Callable that evaluates the observation equations of all datasets for one iteration.

    Attributes
    ----------
    backend : SerialEquationBackend or RuntimeEquationBackend
        Evaluation backend.
    datasets : dict of str to NptDataset
        Datasets keyed by source name.
    options : ObservationOptions
        Observation options.
    state : object
        Observation model state for the runtime backend, None for the serial backend.
    """

    backend: SerialEquationBackend | RuntimeEquationBackend
    datasets: dict
    options: ObservationOptions
    state: object

    def __call__(self, iteration: int):
        """Evaluate the equations of all datasets and concatenate them into one batch.

        Parameters
        ----------
        iteration : int
            Evaluation number, shown in the progress label; 0 is the evaluation for the initial screening.

        Returns
        -------
        EquationBatch
            Equations of all datasets.
        """
        label = "residuals for screening" if iteration == 0 else f"linearization (pass {iteration})"
        options = self.options.with_progress(label)
        by_source = self.backend.evaluate(self.datasets, options, self.state)
        from lunarops.classes.observation.equations import EquationBatch

        return EquationBatch.concatenate(by_source.values())


def build_equation_source(config, context, datasets, processor):
    """Build the equation source of a processing program.

    The runtime backend is used when the run context has a task runtime (``--mpi`` or ``--workers``), with
    ``tasks.chunksize`` normal points per task; otherwise the equations are evaluated serially. Partial
    derivatives are requested for the reflector positions, station positions and lunar orientation when
    ``parametrization`` contains a ``reflectorPosition``, ``stationPosition`` or ``empiricalLibration`` block,
    respectively. A ``lunarDynamics`` block requests the partials for the Moon-minus-Earth position and the
    lunar orientation, a ``lunarLoveNumber`` block the partial for the lunar Love number h2.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context, providing the optional task runtime.
    datasets : dict of str to NptDataset
        Datasets keyed by source name.
    processor : LlrObservationProcessor
        Observation processor of the master process.

    Returns
    -------
    EquationSource
        Callable returning the equations of all datasets for one iteration.
    """
    tasks = worker_tasks(config, context, processor=processor)
    backend: SerialEquationBackend | RuntimeEquationBackend
    if tasks is not None:
        backend, state = RuntimeEquationBackend(tasks), processor.model_state
    else:
        backend, state = SerialEquationBackend(processor), None
    parametrization_types = {item["type"] for item in config["parametrization"]}
    options = make_processing_options(
        config,
        include_design="reflectorPosition" in parametrization_types,
        include_station_design="stationPosition" in parametrization_types,
        include_orientation_design=bool({"empiricalLibration", "lunarDynamics"} & parametrization_types),
        include_moon_position_design="lunarDynamics" in parametrization_types,
        include_lunar_tide_design="lunarLoveNumber" in parametrization_types,
    )
    return EquationSource(backend, datasets, options, state)


__all__ = [
    "WorkerTasks",
    "build_equation_source",
    "build_parametrization",
    "load_datasets",
    "make_processing_options",
    "output_level",
    "worker_tasks",
]
