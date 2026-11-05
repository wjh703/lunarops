"""Observed-minus-computed (O-C) residuals of LLR normal points.

The ``LlrResiduals`` program evaluates the two-way light-time observation model at each normal point and
writes an ``observationResult`` table with the residuals and model diagnostics.
"""

from __future__ import annotations

from lunarops.classes.observation.assembly import build_observation_processor
from lunarops.config.context import RunContext
from lunarops.llr_workflow import (
    load_datasets,
    make_processing_options,
    output_level,
)
from lunarops.programs.registry import ArtifactSlot, program, program_spec
from lunarops.programs.specs import observation_fields, validate_observation_time_config


def _validate_config(config: dict, path: str) -> dict:
    return validate_observation_time_config(config, path)


@program(
    program_spec(
        name="LlrResiduals",
        summary="Evaluate LLR O-C residuals and diagnostics.",
        inputs=(
            ArtifactSlot(
                "inputFilesNormalPoints",
                "normalPoint",
                many=True,
                description="Canonical normal-point files, or directories searched recursively for them.",
            ),
            ArtifactSlot("inputFileStationCatalog", "stationCatalog", description="Native ITRF station catalog."),
            ArtifactSlot(
                "inputFileReflectorCatalog",
                "reflectorCatalog",
                description="Native Moon-PA reflector catalog.",
            ),
        ),
        outputs=(
            ArtifactSlot(
                "outputFileObservationResults",
                "observationResult",
                description="Output observation-result table with one row per normal point.",
            ),
        ),
        fields=observation_fields(residual=True),
        validator=_validate_config,
    )
)
def llr_residuals(config: dict, context: RunContext):
    """Evaluate the O-C residuals of normal points with the configured observation models.

    Reads the ``normalPoint`` files ``inputFilesNormalPoints`` within the ``startTime``/``endTime`` window and
    the ``stationCatalog`` and ``reflectorCatalog`` inputs, evaluates every normal point serially or on the
    MPI workers, and writes the ``observationResult`` table ``outputFileObservationResults`` with one row per
    normal point. ``outputLevel`` selects the column set and ``includeReflectorDesign`` adds the partial
    derivatives with respect to the reflector PA coordinates. See the program reference
    (``docs/reference/programs.md``) for every configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    dict of str to list
        Result rows keyed by source file stem.
    """
    from lunarops.fileio.observation_results import write_observation_results

    datasets = load_datasets(config, context)
    options = make_processing_options(config, include_design=config["includeReflectorDesign"])
    table_level = output_level(config)

    results_by_file: dict[str, list]
    runtime = context.runtime
    if runtime is not None:
        from lunarops.parallel.mpi import mpi_observations
        from lunarops.parallel.observation_spec import make_observation_spec

        spec = make_observation_spec(config, context)
        results_by_file = mpi_observations(
            runtime,
            spec,
            datasets,
            options,
            rows=True,
            output_level=table_level.value,
            chunksize=int(config["mpi"]["chunksize"]),
            progress_desc="O-C normal points",
            quiet=not bool(config["showProgress"]),
        )
    else:
        processor = build_observation_processor(context, config)
        results_by_file = {
            source_name: processor.rows(dataset, options=options, detail=table_level)
            for source_name, dataset in datasets.items()
        }

    output = context.resolve_path(config["outputFileObservationResults"])
    write_observation_results(results_by_file, output)
    total = sum(len(rows) for rows in results_by_file.values())
    print(f"[LlrResiduals] {total} normal point(s) over {len(results_by_file)} source(s) -> {output}")
    return results_by_file


__all__ = ["llr_residuals"]
