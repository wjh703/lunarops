"""Import external normal-point sources into the canonical LunarOps format."""

from __future__ import annotations

from lunarops.config.context import RunContext
from lunarops.config.schema import string
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program


@program(
    ProgramSpec(
        name="NormalPointsConvert",
        summary="Import CRD, MINI, or native normal points into one canonical file.",
        inputs=(ArtifactSlot("inputFilesNormalPoints", "ExternalNormalPointFile", many=True),),
        outputs=(
            ArtifactSlot("outputFileNormalPoints", "NormalPointFile"),
            ArtifactSlot("outputFileImportReport", "ImportReportFile"),
        ),
        fields=(string("datasetName", non_empty=True),),
    )
)
def normal_points_convert(config: dict, context: RunContext):
    from lunarops.classes.observation.normal_points import combine_npt_datasets
    from lunarops.fileio.formats.normal_point_sources import (
        read_normal_point_source,
        resolve_normal_point_sources,
    )
    from lunarops.fileio.normal_points import write_normal_points
    from lunarops.fileio.yaml_artifact import write_structured_text

    output = context.resolve_path(config["outputFileNormalPoints"])
    sources = [
        path
        for path in resolve_normal_point_sources(
            [context.resolve_path(value) for value in config["inputFilesNormalPoints"]]
        )
        if path.resolve() != output.resolve()
    ]
    if not sources:
        raise FileNotFoundError("NormalPointsConvert found no supported input files.")
    datasets = [read_normal_point_source(path) for path in sources]
    combined = combine_npt_datasets(
        datasets,
        name=str(config.get("datasetName", "normal-points")),
    )
    write_normal_points(combined, output)
    report = {
        "sourceCount": len(sources),
        "recordCount": len(combined),
        "inputRecordCount": combined.n_input_records,
        "invalidRecordCount": combined.n_invalid_records,
        "sources": [
            {
                "path": str(path),
                "recordCount": len(dataset),
                "inputRecordCount": dataset.n_input_records,
                "invalidRecordCount": dataset.n_invalid_records,
                "issues": dataset.import_issues,
            }
            for path, dataset in zip(sources, datasets)
        ],
    }
    write_structured_text(
        context.resolve_path(config["outputFileImportReport"]),
        "normalPointImportReport",
        report,
    )
    print(f"[NormalPointsConvert] {len(combined)} record(s) from {len(sources)} source(s) -> {output}")
    return output


__all__ = ["normal_points_convert"]
