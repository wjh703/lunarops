"""Conversion, concatenation, filtering, and inspection of normal points."""

from __future__ import annotations

from collections import Counter

from lunarops.config.context import RunContext
from lunarops.config.schema import string
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program
from lunarops.programs.specs import NORMAL_POINT_FILTER_FIELDS


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
    from lunarops.fileio.normal_point_file import write_normal_point_file
    from lunarops.fileio.normal_point_inputs import (
        read_normal_point_source,
        resolve_normal_point_sources,
    )
    from lunarops.classes.observation.normal_points import combine_npt_datasets
    from lunarops.fileio.structured_text import write_structured_text

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
    write_normal_point_file(combined, output)
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


@program(
    ProgramSpec(
        name="NormalPointsConcatenate",
        summary="Concatenate canonical normal-point files in declared order.",
        inputs=(ArtifactSlot("inputFilesNormalPoints", "NormalPointFile", many=True),),
        outputs=(ArtifactSlot("outputFileNormalPoints", "NormalPointFile"),),
        fields=(string("datasetName", non_empty=True),),
    )
)
def normal_points_concatenate(config: dict, context: RunContext):
    from lunarops.fileio.normal_point_file import (
        read_normal_point_file,
        write_normal_point_file,
    )
    from lunarops.classes.observation.normal_points import combine_npt_datasets

    output = context.resolve_path(config["outputFileNormalPoints"])
    paths = [context.resolve_path(value) for value in config["inputFilesNormalPoints"]]
    if any(path.resolve() == output.resolve() for path in paths):
        raise ValueError("Concatenation output must not also be an input.")
    dataset = combine_npt_datasets(
        [read_normal_point_file(path) for path in paths],
        name=str(config.get("datasetName", "concatenated")),
    )
    write_normal_point_file(dataset, output)
    print(f"[NormalPointsConcatenate] {len(dataset)} record(s) -> {output}")
    return output


@program(
    ProgramSpec(
        name="NormalPointsFilter",
        summary="Select canonical normal points by time, station, reflector, and uncertainty.",
        inputs=(ArtifactSlot("inputFileNormalPoints", "NormalPointFile"),),
        outputs=(ArtifactSlot("outputFileNormalPoints", "NormalPointFile"),),
        fields=NORMAL_POINT_FILTER_FIELDS,
    )
)
def normal_points_filter(config: dict, context: RunContext):
    from lunarops.fileio.normal_point_file import (
        read_normal_point_file,
        write_normal_point_file,
    )
    from lunarops.classes.observation.normal_points import NptDataset, parse_time_filter

    dataset = read_normal_point_file(context.resolve_path(config["inputFileNormalPoints"]))
    start = parse_time_filter(config.get("startTime"))
    end = parse_time_filter(config.get("endTime"))
    stations = None if config.get("stationNames") is None else {str(value) for value in config["stationNames"]}
    reflectors = None if config.get("reflectorNames") is None else {str(value) for value in config["reflectorNames"]}
    maximum_sigma = None if config.get("maximumOneWaySigmaM") is None else float(config["maximumOneWaySigmaM"])
    if maximum_sigma is not None and maximum_sigma <= 0.0:
        raise ValueError("maximumOneWaySigmaM must be positive.")
    records = [
        record
        for record in dataset.records
        if (start is None or record.transmit_epoch >= start)
        and (end is None or record.transmit_epoch < end)
        and (stations is None or record.station_name in stations or record.station_code in stations)
        and (reflectors is None or record.reflector_name in reflectors or record.reflector_code in reflectors)
        and (maximum_sigma is None or record.range_uncertainty_one_way_m <= maximum_sigma)
    ]
    if not records:
        raise ValueError("NormalPointsFilter selected no records.")
    filtered = NptDataset(
        records,
        name=str(config.get("datasetName", dataset.name or "filtered")),
        n_input_records=dataset.n_input_records,
        n_invalid_records=dataset.n_invalid_records,
    ).assign_indices()
    output = context.resolve_path(config["outputFileNormalPoints"])
    write_normal_point_file(filtered, output)
    print(f"[NormalPointsFilter] {len(dataset)} -> {len(filtered)} record(s) -> {output}")
    return output


@program(
    ProgramSpec(
        name="NormalPointsStatistics",
        summary="Summarize canonical normal-point coverage and uncertainty.",
        inputs=(ArtifactSlot("inputFilesNormalPoints", "NormalPointFile", many=True),),
        outputs=(ArtifactSlot("outputFileStatistics", "NormalPointStatisticsFile"),),
    )
)
def normal_points_statistics(config: dict, context: RunContext):
    import numpy as np

    from lunarops.fileio.normal_point_file import read_normal_point_file
    from lunarops.fileio.structured_text import write_structured_text

    datasets = [read_normal_point_file(context.resolve_path(value)) for value in config["inputFilesNormalPoints"]]
    records = [record for dataset in datasets for record in dataset.records]
    if not records:
        raise ValueError("NormalPointsStatistics has no records.")
    epochs = sorted(record.transmit_epoch for record in records)
    sigmas = np.asarray([record.range_uncertainty_one_way_m for record in records])
    payload = {
        "artifact": "NormalPointFile",
        "fileCount": len(datasets),
        "recordCount": len(records),
        "startTimeUtc": epochs[0].isot(precision=6),
        "endTimeUtc": epochs[-1].isot(precision=6),
        "recordsByStation": dict(sorted(Counter(record.station_name for record in records).items())),
        "recordsByReflector": dict(sorted(Counter(record.reflector_name for record in records).items())),
        "oneWaySigmaM": {
            "minimum": float(np.min(sigmas)),
            "median": float(np.median(sigmas)),
            "maximum": float(np.max(sigmas)),
        },
        "invalidInputRecordCount": sum(dataset.n_invalid_records for dataset in datasets),
    }
    output = context.resolve_path(config["outputFileStatistics"])
    write_structured_text(output, "normalPointStatistics", payload)
    print(f"[NormalPointsStatistics] {len(records)} record(s) -> {output}")
    return payload


__all__ = [
    "normal_points_concatenate",
    "normal_points_convert",
    "normal_points_filter",
    "normal_points_statistics",
]
