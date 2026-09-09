"""Merge disjoint LLR prediction grids into one chronological campaign."""

from __future__ import annotations

from pathlib import Path

from lunarops.classes.observation import build_visibility_windows
from lunarops.classes.time import Epoch, parse_time_with_utc_offset
from lunarops.config.context import RunContext
from lunarops.config.schema import number
from lunarops.fileio.prediction_results import (
    read_prediction_results,
    write_prediction_results,
    write_prediction_windows,
)
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program


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


def _merge_rows(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    indexed: list[tuple[Epoch, Path, int, dict[str, object]]] = []
    targets: set[tuple[str, str]] = set()
    for source in paths:
        for row_index, row in enumerate(read_prediction_results(source), start=1):
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
    rows: list[dict[str, object]] = []
    previous_epoch: Epoch | None = None
    previous_source: Path | None = None
    for epoch, source, row_index, row in indexed:
        if previous_epoch is not None and abs(previous_epoch.seconds_until(epoch)) < 1.0e-9:
            raise ValueError(
                f"LlrObservationPredictionMerge found duplicate utc_t1 {row['utc_t1']!r} "
                f"in {previous_source} and {source} row {row_index}."
            )
        rows.append(row)
        previous_epoch = epoch
        previous_source = source
    return rows


@program(
    ProgramSpec(
        name="LlrObservationPredictionMerge",
        summary="Merge disjoint native LLR prediction grids and rebuild visibility windows.",
        inputs=(ArtifactSlot("inputFilesPrediction", "PredictionResultFile", many=True),),
        outputs=(
            ArtifactSlot("outputFilePrediction", "PredictionResultFile"),
            ArtifactSlot("outputFileWindows", "PredictionWindowFile"),
        ),
        fields=(
            number(
                "stepSeconds",
                required=True,
                minimum=0.0,
                minimum_exclusive=True,
                allow_none=False,
            ),
        ),
    )
)
def llr_observation_prediction_merge(config: dict, context: RunContext):
    sources = tuple(context.resolve_path(value) for value in config["inputFilesPrediction"])
    rows = _merge_rows(sources)
    windows = build_visibility_windows(rows, step_seconds=float(config["stepSeconds"]))
    prediction_path = write_prediction_results(rows, context.resolve_path(config["outputFilePrediction"]))
    windows_path = write_prediction_windows(windows, context.resolve_path(config["outputFileWindows"]))
    print(
        f"[LlrObservationPredictionMerge] {len(rows)} epoch(s) from {len(sources)} file(s), "
        f"{len(windows)} window(s) -> {prediction_path}, {windows_path}"
    )
    return {"rows": rows, "windows": windows}


__all__ = ["llr_observation_prediction_merge"]
