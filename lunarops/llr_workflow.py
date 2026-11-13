"""Shared building blocks of the LLR residual and processing programs.

The functions here read the normal points of a program config, build the observation options and the
parametrization, and assemble the equation source that evaluates the linearized observation equations of all
datasets at each iteration, either serially in this process or distributed over the workers of a task runtime (MPI ranks or local
worker processes).
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
    from lunarops.programs.specs import observation_time_window

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
        show_progress=bool(config["showProgress"]),
    )


def output_level(config: dict, *, include_design: bool = False):
    """Return the detail level of the observation results.

    Parameters
    ----------
    config : dict
        Resolved program config; ``outputLevel`` selects the level.
    include_design : bool, optional
        Force the full level, regardless of ``outputLevel``.

    Returns
    -------
    ObservationResultDetail
        ``FULL`` when ``include_design`` is set, otherwise the parsed ``outputLevel``.
    """
    from lunarops.classes.observation import ObservationResultDetail

    if include_design:
        return ObservationResultDetail.FULL
    return ObservationResultDetail.parse(config["outputLevel"])


def build_parametrization(config: dict, context: RunContext, model_state):
    """Create the parametrization blocks listed under ``parametrization``.

    ``stationPosition`` and ``reflectorPosition`` blocks also receive the model state.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context passed to the class factories.
    model_state : object
        Observation model state holding the station and reflector catalogs.

    Returns
    -------
    ParametrizationList
        The blocks in config order.
    """
    from lunarops.classes.parametrization.base import ParametrizationList
    from lunarops.config.registry import create

    blocks = [
        create(
            "parametrization",
            item,
            context,
            **({"model_state": model_state} if item["type"] in ("stationPosition", "reflectorPosition") else {}),
        )
        for item in config["parametrization"]
    ]
    return ParametrizationList(blocks)


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
    runtime : TaskRuntime
        Task runtime of the master (``MpiRuntime`` or ``ProcessRuntime``).
    spec : dict
        Observation spec from which each worker builds its observation processor.
    chunksize : int
        Number of normal points per worker task.
    """

    runtime: TaskRuntime
    spec: dict
    chunksize: int

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
        from lunarops.parallel.mpi import mpi_observations
        from lunarops.parallel.observation_spec import snapshot_catalog_state

        return mpi_observations(
            self.runtime,
            self.spec,
            datasets,
            options,
            rows=False,
            chunksize=self.chunksize,
            catalog_state=snapshot_catalog_state(state),
            progress_desc=options.progress_description,
            quiet=not options.show_progress,
        )


MpiEquationBackend = RuntimeEquationBackend
"""Former name of ``RuntimeEquationBackend``."""


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

    The runtime backend is used when the run context has a task runtime (``--mpi`` or ``--workers``), with ``mpi.chunksize`` normal points
    per task; otherwise the equations are evaluated serially. Partial derivatives are requested for the
    reflector positions, station positions and lunar orientation when ``parametrization`` contains a
    ``reflectorPosition``, ``stationPosition`` or ``empiricalLibration`` block, respectively.

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
    runtime = context.runtime
    backend: SerialEquationBackend | RuntimeEquationBackend
    if runtime is not None:
        from lunarops.parallel.observation_spec import make_observation_spec

        state = processor.model_state
        backend = RuntimeEquationBackend(
            runtime,
            make_observation_spec(
                config,
                context,
                station_catalog=state.station_catalog,
                reflector_catalog=state.reflector_catalog,
            ),
            int(config["mpi"]["chunksize"]),
        )
    else:
        backend = SerialEquationBackend(processor)
        state = None
    parametrization_types = {item["type"] for item in config["parametrization"]}
    options = make_processing_options(
        config,
        include_design="reflectorPosition" in parametrization_types,
        include_station_design="stationPosition" in parametrization_types,
        include_orientation_design="empiricalLibration" in parametrization_types,
    )
    return EquationSource(backend, datasets, options, state)


__all__ = [
    "build_equation_source",
    "build_parametrization",
    "load_datasets",
    "make_processing_options",
    "output_level",
]
