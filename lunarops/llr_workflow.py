from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lunarops.config.context import RunContext

if TYPE_CHECKING:
    from lunarops.classes.observation import LlrObservationProcessor, ObservationProcessingOptions
    from lunarops.parallel.mpi import MpiRuntime


def load_datasets(config: dict, context: RunContext):
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
    from lunarops.classes.observation import ObservationProcessingOptions

    return ObservationProcessingOptions(
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
    from lunarops.classes.observation import ObservationResultDetail

    if include_design:
        return ObservationResultDetail.FULL
    return ObservationResultDetail.parse(config["outputLevel"])


def build_parametrization(config: dict, context: RunContext):
    from lunarops.classes.parametrization.base import ParametrizationList

    blocks = [context.create_class("parametrization", item, cache=False) for item in config["parametrization"]]
    return ParametrizationList(blocks)


def model_compatibility_fingerprint(config: dict, context: RunContext) -> str:
    """Fingerprint model conventions while allowing independent data arcs."""
    from lunarops.config.fingerprints import scientific_fingerprint

    operational_keys = {
        "inputFilesNormalPoints",
        "inputFileProcessingState",
        "startTime",
        "endTime",
        "utcOffsetHours",
        "overrideStation",
        "overrideReflector",
        "minElevationDeg",
        "showProgress",
        "mpi",
        "outputLevel",
        "includeReflectorDesign",
    }
    output_keys = {key for key in config if key.startswith("outputFile")}
    return scientific_fingerprint(
        config,
        context,
        excluded_keys=operational_keys | output_keys,
    )


@dataclass(frozen=True, slots=True)
class SerialEquationBackend:
    processor: LlrObservationProcessor

    def evaluate(self, datasets, options, state):
        return {source: self.processor.equations(dataset, options=options) for source, dataset in datasets.items()}


@dataclass(frozen=True, slots=True)
class MpiEquationBackend:
    runtime: MpiRuntime
    spec: dict
    chunksize: int

    def evaluate(self, datasets, options, state):
        from lunarops.parallel.mpi import mpi_observation_equations
        from lunarops.parallel.observation_spec import snapshot_catalog_state

        return mpi_observation_equations(
            self.runtime,
            self.spec,
            datasets,
            options,
            chunksize=self.chunksize,
            catalog_state=snapshot_catalog_state(state),
            progress_desc=options.progress_description,
            quiet=not options.show_progress,
        )


@dataclass(frozen=True, slots=True)
class EquationSource:
    backend: SerialEquationBackend | MpiEquationBackend
    datasets: dict
    options: ObservationProcessingOptions
    state: object

    def __call__(self, iteration: int):
        options = self.options.with_progress(f"linearization {iteration}")
        by_source = self.backend.evaluate(self.datasets, options, self.state)
        return [equation for equations in by_source.values() for equation in equations]


def build_equation_source(config, context, datasets, processor):
    runtime = context.runtime
    backend: SerialEquationBackend | MpiEquationBackend
    if runtime is not None:
        from lunarops.parallel.observation_spec import make_observation_spec

        state = processor.model_state
        backend = MpiEquationBackend(
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
    "model_compatibility_fingerprint",
    "output_level",
]
