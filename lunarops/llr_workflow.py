"""Shared application workflow for LLR programs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lunarops.config.context import RunContext

if TYPE_CHECKING:
    from lunarops.classes.observation import LlrObservationProcessor, ObservationProcessingOptions
    from lunarops.parallel.mpi import MpiRuntime


def load_datasets(config: dict, context: RunContext):
    from lunarops.fileio.formats.normal_point_sources import (
        read_normal_points,
        resolve_normal_point_inputs,
    )

    inputs = config.get("inputFilesNormalPoints")
    if not inputs:
        raise ValueError("inputFilesNormalPoints is required")
    if isinstance(inputs, (str, bytes)):
        raise TypeError("inputFilesNormalPoints must be a list of native normal-point files.")
    input_values = list(inputs)
    input_files = resolve_normal_point_inputs([context.resolve_path(item) for item in input_values])
    if not input_files:
        raise FileNotFoundError(f"No supported normal-point files found under {inputs!r}")

    datasets = {}
    utc_offset_hours = float(config.get("utcOffsetHours", 0.0))
    from lunarops.classes.observation.normal_points import parse_time_filter

    for path in input_files:
        dataset = read_normal_points(path)
        start, end = config.get("startTime"), config.get("endTime")
        if start or end:
            if utc_offset_hours == 0.0:
                dataset = dataset.filter_time(start, end)
            else:
                dataset = dataset.filter_time(
                    parse_time_filter(start, utc_offset_hours=utc_offset_hours),
                    parse_time_filter(end, utc_offset_hours=utc_offset_hours),
                )
        if dataset.records:
            datasets[Path(path).stem] = dataset

    next_index = 0
    for dataset in datasets.values():
        dataset.assign_indices(start=next_index)
        next_index += len(dataset.records)
    if not datasets:
        raise ValueError("No normal points remain after time filtering.")
    return datasets


def build_processor(config: dict, context: RunContext):
    from lunarops.classes.observation_factory import build_observation_processor

    return build_observation_processor(context, config)


def make_processing_options(config: dict, *, include_design: bool = False):
    from lunarops.classes.observation import ObservationProcessingOptions

    return ObservationProcessingOptions(
        station_identifier=config.get("stationName"),
        reflector_identifier=config.get("reflectorName"),
        min_elevation_deg=float(config.get("minElevationDeg", 0.0)),
        utc_offset_hours=float(config.get("utcOffsetHours", 0.0)),
        include_reflector_position_partials=bool(include_design or config.get("includeReflectorDesign", False)),
        show_progress=bool(config.get("showProgress", True)),
    )


def output_level(config: dict, *, include_design: bool = False):
    from lunarops.classes.observation import ObservationResultDetail

    if include_design:
        return ObservationResultDetail.FULL
    return ObservationResultDetail.parse(config.get("outputLevel", "standard"))


def build_parametrization(config: dict, context: RunContext):
    from lunarops.classes.observation_factory import ensure_registered
    from lunarops.classes.parametrization.base import ParametrizationList
    from lunarops.config.registry import create_list

    ensure_registered()
    blocks = create_list("parametrization", config.get("parametrization"), context)
    if not blocks:
        raise ValueError("At least one parametrization block is required.")
    return ParametrizationList(blocks)


def model_compatibility_fingerprint(config: dict, context: RunContext) -> str:
    """Fingerprint model conventions while allowing independent data arcs."""
    from lunarops.config.fingerprints import scientific_fingerprint

    operational_keys = {
        "inputFileNormalPoints",
        "inputFilesNormalPoints",
        "inputFileAdjustmentState",
        "startTime",
        "endTime",
        "utcOffsetHours",
        "stationName",
        "reflectorName",
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
        from lunarops.parallel.mpi import mpi_observation_equations, snapshot_catalog_state

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
    """Bind a backend once; iteration execution uses the same interface."""
    runtime = context.runtime
    backend: SerialEquationBackend | MpiEquationBackend
    if runtime is not None and runtime.has_workers:
        from lunarops.parallel.mpi import make_observation_spec

        state = processor.model_state
        backend = MpiEquationBackend(
            runtime,
            make_observation_spec(
                config,
                context,
                station_catalog=state.station_catalog,
                reflector_catalog=state.reflector_catalog,
            ),
            int((config.get("mpi") or {}).get("chunksize", 8)),
        )
    else:
        backend = SerialEquationBackend(processor)
        state = None
    return EquationSource(backend, datasets, make_processing_options(config, include_design=True), state)


__all__ = [
    "build_equation_source",
    "build_parametrization",
    "build_processor",
    "load_datasets",
    "make_processing_options",
    "model_compatibility_fingerprint",
    "output_level",
]
