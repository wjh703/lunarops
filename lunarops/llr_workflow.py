"""Shared application workflow for LLR programs."""

from __future__ import annotations

from pathlib import Path

from lunarops.config.context import RunContext


def load_datasets(config: dict, context: RunContext):
    from lunarops.fileio.normal_point_inputs import (
        read_normal_points,
        resolve_normal_point_inputs,
    )
    from lunarops.classes.observation.normal_points import combine_npt_datasets

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
    for path in input_files:
        dataset = read_normal_points(path)
        start, end = config.get("startTime"), config.get("endTime")
        if start or end:
            dataset = dataset.filter_time(start, end)
        if dataset.records:
            datasets[Path(path).stem] = dataset

    if config.get("combineInputs"):
        datasets = {config.get("combinedName", "combined"): combine_npt_datasets(list(datasets.values()))}

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
    from lunarops.classes.observation_factory import validate_observation_config

    validate_observation_config(config)
    return ObservationProcessingOptions(
        station_identifier=config.get("stationName"),
        reflector_identifier=config.get("reflectorName"),
        min_elevation_deg=float(config.get("minElevationDeg", 0.0)),
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
    from lunarops.fileio.fingerprints import scientific_fingerprint

    operational_keys = {
        "inputFileNormalPoints",
        "inputFilesNormalPoints",
        "inputFileAdjustmentState",
        "combineInputs",
        "combinedName",
        "startTime",
        "endTime",
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


def build_equation_source(config, context, datasets, processor):
    """Return a closure that relinearizes all observations per iteration."""
    options = make_processing_options(config, include_design=True)
    runtime = context.runtime
    use_mpi = runtime is not None and runtime.has_workers
    spec: dict | None = None
    chunksize = 8
    if use_mpi:
        assert runtime is not None
        from lunarops.parallel.mpi import make_observation_spec

        spec = make_observation_spec(
            config,
            context,
            station_catalog=processor.model_state.station_catalog,
            reflector_catalog=processor.model_state.reflector_catalog,
        )
        chunksize = int((config.get("mpi") or {}).get("chunksize", 8))

    def equation_source(iteration: int):
        if use_mpi:
            assert runtime is not None
            assert spec is not None
            from lunarops.parallel.mpi import mpi_observation_equations, snapshot_catalog_state

            equations_by_source = mpi_observation_equations(
                runtime,
                spec,
                datasets,
                options,
                chunksize=chunksize,
                catalog_state=snapshot_catalog_state(processor.model_state),
                progress_desc=f"linearization {iteration}",
                quiet=not bool(config.get("showProgress", True)),
            )
        else:
            iteration_options = options.with_progress(f"linearization {iteration}")
            equations_by_source = {
                source_name: processor.equations(dataset, options=iteration_options)
                for source_name, dataset in datasets.items()
            }
        return [equation for equations in equations_by_source.values() for equation in equations]

    return equation_source


__all__ = [
    "build_equation_source",
    "build_parametrization",
    "build_processor",
    "load_datasets",
    "make_processing_options",
    "model_compatibility_fingerprint",
    "output_level",
]
