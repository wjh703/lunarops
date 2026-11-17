"""Merging of LLR prediction tables with disjoint epochs into one chronological table."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from lunarops import console
from lunarops.classes.observation import build_visibility_windows
from lunarops.classes.time import Epoch, parse_time_with_utc_offset
from lunarops.config.context import RunContext
from lunarops.config.schema import number
from lunarops.fileio.prediction_results import (
    PREDICTION_ARTIFACT_TYPE,
    PREDICTION_FIELDS,
    write_prediction_results,
    write_prediction_windows,
)
from lunarops.fileio.result_table import read_records
from lunarops.programs.registry import ArtifactSlot, program, program_spec


def _row_epoch(row: dict[str, object], source: Path, row_index: int) -> Epoch:
    try:
        epoch = parse_time_with_utc_offset(
            row["utc_t1"],
            name=f"{source} prediction row {row_index} utc_t1",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid prediction timestamp in {source} row {row_index}.") from exc
    if epoch is None:
        raise ValueError(f"Prediction timestamp in {source} row {row_index} must not be empty.")
    return epoch


def _merge_rows(
    tables: list[tuple[Path, list[dict[str, object]]]],
) -> list[tuple[Epoch, Path, int, dict[str, object]]]:
    """Check the rows of all files for one station/reflector pair, sort them by ``utc_t1``, reject duplicates.

    Parameters
    ----------
    tables : list of (Path, list of dict)
        Rows of every input file.

    Returns
    -------
    list of (Epoch, Path, int, dict)
        Epoch, source file, 1-based row number in that file and row, in chronological order.
    """
    indexed: list[tuple[Epoch, Path, int, dict[str, object]]] = []
    targets: set[tuple[str, str]] = set()
    for source, table_rows in tables:
        for row_index, row in enumerate(table_rows, start=1):
            try:
                targets.add((str(row["station"]), str(row["reflector"])))
            except KeyError as exc:
                raise ValueError(f"Prediction row {row_index} in {source} has no station or reflector.") from exc
            indexed.append((_row_epoch(row, source, row_index), source, row_index, row))
    if not indexed:
        raise ValueError("LlrObservationPredictionMerge requires at least one prediction row.")
    if len(targets) != 1:
        raise ValueError("LlrObservationPredictionMerge requires all inputs to use one station and reflector.")

    indexed.sort(key=lambda item: (item[0].jd1, item[0].jd2))
    for (previous_epoch, previous_source, _, _), (epoch, source, row_index, row) in pairwise(indexed):
        if abs(previous_epoch.seconds_until(epoch)) < 1.0e-9:
            raise ValueError(
                f"LlrObservationPredictionMerge found duplicate utc_t1 {row['utc_t1']!r} "
                f"in {previous_source} and {source} row {row_index}."
            )
    return indexed


@program(
    program_spec(
        name="LlrObservationPredictionMerge",
        summary="Merge disjoint native LLR prediction grids and rebuild visibility windows.",
        inputs=(
            ArtifactSlot(
                "inputFilesPrediction",
                "observationPrediction",
                many=True,
                description="Prediction files of one station/reflector pair with disjoint epochs.",
            ),
        ),
        outputs=(
            ArtifactSlot(
                "outputFilePrediction",
                "observationPrediction",
                description="Merged prediction table in chronological order.",
            ),
            ArtifactSlot(
                "outputFileWindows",
                "predictionWindow",
                description="Visibility windows rebuilt from the merged table.",
            ),
        ),
        fields=(
            number(
                "stepSeconds",
                required=True,
                gt=0.0,
                description="Grid spacing of the inputs, in seconds; samples further apart start a new window.",
            ),
        ),
    )
)
def llr_observation_prediction_merge(config: dict, context: RunContext):
    """Merge prediction tables of one station/reflector pair and rebuild the visibility windows.

    Reads the ``observationPrediction`` tables ``inputFilesPrediction``, sorts their rows by UTC epoch
    (``utc_t1``) and checks that no epoch repeats, that every table has the configured ``stepSeconds`` and
    that all epochs lie on one grid of that spacing. Visibility windows are rebuilt from the merged rows,
    so a gap between inputs splits a window. Writes the ``observationPrediction`` table
    ``outputFilePrediction`` and the ``predictionWindow`` table ``outputFileWindows``. See the program
    reference (``docs/reference/programs.md``) for every configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    dict
        ``rows`` (merged prediction rows) and ``windows`` (visibility windows).

    Raises
    ------
    ValueError
        If the inputs are empty, mix station/reflector pairs, repeat an epoch, or do not share the grid.
    """
    sources = tuple(context.resolve_path(value) for value in config["inputFilesPrediction"])
    tables = [
        (path, *read_records(path, artifact_type=PREDICTION_ARTIFACT_TYPE, expected_fields=PREDICTION_FIELDS))
        for path in sources
    ]
    merged = _merge_rows([(path, table_rows) for path, _, table_rows in tables])
    rows = [row for _, _, _, row in merged]

    step = float(config["stepSeconds"])
    metadata_sources = {str(path): table_metadata for path, table_metadata, _ in tables}
    for path, table_metadata in metadata_sources.items():
        if "stepSeconds" in table_metadata and abs(float(table_metadata["stepSeconds"]) - step) > 1e-9:
            raise ValueError(f"Prediction stepSeconds mismatch in {path}")
    anchor = merged[0][0]
    for epoch, source, row_index, _ in merged[1:]:
        ratio = anchor.seconds_until(epoch) / step
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                f"Prediction timestamp in {source} row {row_index} does not lie on the configured sampling grid"
            )
    windows = build_visibility_windows(rows, step_seconds=step)
    metadata = {
        "gridDefined": True,
        "stepSeconds": step,
        "sources": metadata_sources,
        "windowBoundaryConvention": "inclusive first/last observable samples; gaps split windows",
    }
    prediction_path = write_prediction_results(
        rows, context.resolve_path(config["outputFilePrediction"]), metadata=metadata
    )
    windows_path = write_prediction_windows(
        windows, context.resolve_path(config["outputFileWindows"]), metadata=metadata
    )
    console.summary(
        {"epochs": len(rows), "input files": len(sources), "visibility windows": len(windows)},
        outputs=[prediction_path, windows_path],
    )
    return {"rows": rows, "windows": windows}


__all__ = ["llr_observation_prediction_merge"]
