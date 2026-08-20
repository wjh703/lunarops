"""Compute typed LLR observation-result tables from canonical normal points."""

from __future__ import annotations

from typing import Dict

from lunarops.config.context import RunContext
from lunarops.llr_workflow import (
    build_processor,
    load_datasets,
    make_processing_options,
    output_level,
)
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program
from lunarops.programs.specs import observation_fields


@program(
    ProgramSpec(
        name="LlrResiduals",
        summary="Evaluate LLR O-C residuals and diagnostics.",
        inputs=(
            ArtifactSlot("inputFilesNormalPoints", "NormalPointFile", many=True),
            ArtifactSlot("inputFileStationCatalog", "StationCatalogFile"),
            ArtifactSlot("inputFileReflectorCatalog", "ReflectorCatalogFile"),
        ),
        outputs=(ArtifactSlot("outputFileObservationResults", "ObservationResultFile"),),
        fields=observation_fields(residual=True),
    )
)
def llr_residuals(config: dict, context: RunContext):
    from lunarops.fileio.observation_results import write_observation_results

    datasets = load_datasets(config, context)
    options = make_processing_options(config)
    table_level = output_level(config)

    results_by_file: Dict[str, list]
    runtime = context.runtime
    if runtime is not None and runtime.has_workers:
        from lunarops.parallel.mpi import make_observation_spec, mpi_observation_rows

        spec = make_observation_spec(config, context)
        results_by_file = mpi_observation_rows(
            runtime,
            spec,
            datasets,
            options,
            output_level=table_level.value,
            chunksize=int((config.get("mpi") or {}).get("chunksize", 8)),
            progress_desc="O-C normal points",
            quiet=not bool(config.get("showProgress", True)),
        )
    else:
        processor = build_processor(config, context)
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
