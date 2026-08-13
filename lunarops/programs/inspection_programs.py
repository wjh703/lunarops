"""Inspection and encoding-conversion programs for typed artifacts."""

from __future__ import annotations

from typing import cast

from lunarops.config.context import RunContext
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program


@program(
    ProgramSpec(
        name="MatrixConvert",
        summary="Convert a generic matrix between LunarOps ASCII and binary encodings.",
        inputs=(ArtifactSlot("inputFileMatrix", "MatrixFile"),),
        outputs=(ArtifactSlot("outputFileMatrix", "MatrixFile"),),
    )
)
def matrix_convert(config: dict, context: RunContext):
    from lunarops.fileio.matrix import matrix_kind, read_matrix, write_matrix

    source = context.resolve_path(config["inputFileMatrix"])
    target = context.resolve_path(config["outputFileMatrix"])
    if source.resolve() == target.resolve():
        raise ValueError("MatrixConvert input and output must differ.")
    kind = matrix_kind(source)
    matrix = read_matrix(source)
    write_matrix(target, matrix, kind=kind)
    print(f"[MatrixConvert] {matrix.shape} {kind} -> {target}")
    return target


@program(
    ProgramSpec(
        name="ObservationResultsStatistics",
        summary="Summarize numeric fields and status counts in observation results.",
        inputs=(ArtifactSlot("inputFileObservationResults", "ObservationResultFile"),),
        outputs=(ArtifactSlot("outputFileStatistics", "ObservationResultStatisticsFile"),),
    )
)
def observation_results_statistics(config: dict, context: RunContext):
    from collections import Counter

    import numpy as np

    from lunarops.fileio.observation_results import read_observation_results
    from lunarops.fileio.yaml_artifact import write_structured_text

    rows = read_observation_results(context.resolve_path(config["inputFileObservationResults"]))
    numeric: dict[str, dict[str, float | int]] = {}
    for field in rows[0] if rows else ():
        values = [
            float(cast(int | float, row[field]))
            for row in rows
            if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
        ]
        if values:
            array = np.asarray(values)
            numeric[field] = {
                "count": len(values),
                "minimum": float(np.min(array)),
                "median": float(np.median(array)),
                "maximum": float(np.max(array)),
                "rms": float(np.sqrt(np.mean(array * array))),
            }
    payload = {
        "recordCount": len(rows),
        "sourceCounts": dict(sorted(Counter(str(row.get("source", "")) for row in rows).items())),
        "statusCounts": dict(sorted(Counter(str(row.get("status", "")) for row in rows).items())),
        "numericFields": numeric,
    }
    output = context.resolve_path(config["outputFileStatistics"])
    write_structured_text(output, "observationResultStatistics", payload)
    print(f"[ObservationResultsStatistics] {len(rows)} row(s) -> {output}")
    return payload


__all__ = ["matrix_convert", "observation_results_statistics"]
