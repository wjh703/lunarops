"""Shared declarative schema fragments for LLR programs."""

from __future__ import annotations

from lunarops.config.schema import (
    ConfigSchema,
    FieldSpec,
    boolean,
    class_config,
    class_list,
    integer,
    mapping,
    number,
    sequence,
    string,
    time,
    UiHints,
)


_MPI_SCHEMA = ConfigSchema(
    fields=(
        integer("chunksize", default=8, minimum=1, allow_none=False),
    ),
    description="MPI task scheduling options.",
)


# Catalog input is deliberately small and explicit.  The native artifact
# writers use the same fields, so inline and file-backed runs produce the
# same internal records.
STATION_COORDINATE_SCHEMA = ConfigSchema(
    fields=(
        string("key", required=True, non_empty=True, allow_none=False),
        sequence(
            "xyzM",
            required=True,
            item_kind="number",
            min_items=3,
            max_items=3,
            allow_none=False,
        ),
        sequence(
            "velocityMPerYear",
            default=[0.0, 0.0, 0.0],
            item_kind="number",
            min_items=3,
            max_items=3,
            allow_none=False,
        ),
        string(
            "positionEpochUtc",
            default="2010-01-01T00:00:00",
            non_empty=True,
            allow_none=False,
        ),
    ),
    description="One station ITRF position with an optional linear velocity model.",
)


REFLECTOR_COORDINATE_SCHEMA = ConfigSchema(
    fields=(
        string("key", required=True, non_empty=True, allow_none=False),
        sequence(
            "xyzM",
            required=True,
            item_kind="number",
            min_items=3,
            max_items=3,
            allow_none=False,
        ),
    ),
    description="One reflector position in the selected ephemeris Moon PA frame.",
)


OBSERVATION_FIELDS = (
    time("startTime", ui=UiHints(group="Time window", widget="datetime-range-start")),
    time("endTime", ui=UiHints(group="Time window", widget="datetime-range-end")),
    string("stationName", non_empty=True, ui=UiHints(group="Selection")),
    string("reflectorName", non_empty=True, ui=UiHints(group="Selection")),
    number(
        "minElevationDeg",
        default=0.0,
        minimum=0.0,
        allow_none=False,
        ui=UiHints(group="Observation", unit="deg"),
    ),
    boolean("showProgress", default=True, allow_none=False, ui=UiHints(group="Runtime", advanced=True)),
    mapping("mpi", nested=_MPI_SCHEMA),
    class_config("ephemerides", "ephemerides"),
    class_config("earthRotation", "earthRotation"),
    class_config("troposphere", "troposphere"),
    class_config("relativity", "relativity"),
    class_list("stationDisplacement", "stationDisplacement", min_items=1),
    class_config("reflectorDisplacement", "reflectorDisplacement"),
    class_config("rangeBias", "rangeBias"),
    sequence(
        "stationCoordinates",
        item_kind="mapping",
        item_nested=STATION_COORDINATE_SCHEMA,
        min_items=1,
        allow_none=True,
        ui=UiHints(group="Catalogs", widget="table"),
    ),
    sequence(
        "reflectorCoordinates",
        item_kind="mapping",
        item_nested=REFLECTOR_COORDINATE_SCHEMA,
        min_items=1,
        allow_none=True,
        ui=UiHints(group="Catalogs", widget="table"),
    ),
)


RESIDUAL_FIELDS = (
    string("outputLevel", default="standard", choices=("standard", "full"), allow_none=False),
    boolean("includeReflectorDesign", default=False, allow_none=False),
)


PARAMETRIZATION_FIELD = class_list(
    "parametrization",
    "parametrization",
    required=True,
    min_items=1,
    allow_none=False,
)


PROCESSING_FIELDS = (
    sequence("varianceComponents", required=True, item_kind="mapping", min_items=1, allow_none=False),
    sequence("processingSteps", required=True, item_kind="mapping", min_items=1, allow_none=False),
)


def observation_fields(
    *,
    parametrized: bool = False,
    processing: bool = False,
    residual: bool = False,
) -> tuple[FieldSpec, ...]:
    """Return the shared fields for one observation program."""
    fields = list(OBSERVATION_FIELDS)
    if parametrized:
        fields.append(PARAMETRIZATION_FIELD)
    if processing:
        fields.extend(PROCESSING_FIELDS)
    if residual:
        fields.extend(RESIDUAL_FIELDS)
    return tuple(fields)


def validate_observation_config(config: dict, path: str) -> dict:
    """Require exactly one file or inline source for each observation catalog."""
    for label, file_key, inline_key in (
        ("station", "inputFileStationCatalog", "stationCoordinates"),
        ("reflector", "inputFileReflectorCatalog", "reflectorCoordinates"),
    ):
        has_file = config.get(file_key) is not None
        has_inline = config.get(inline_key) is not None
        if has_file == has_inline:
            raise ValueError(
                f"{path} requires exactly one of {file_key!r} or {inline_key!r} for the {label} catalog."
            )
    return config


def validate_processing_config(config: dict, path: str) -> dict:
    """Validate catalog sources and run the scientific processing parser."""
    from lunarops.estimation.adjustment_config import parse_adjustment_plan

    validate_observation_config(config, path)
    parse_adjustment_plan(config)
    return config


__all__ = [
    "PROCESSING_FIELDS",
    "REFLECTOR_COORDINATE_SCHEMA",
    "OBSERVATION_FIELDS",
    "PARAMETRIZATION_FIELD",
    "RESIDUAL_FIELDS",
    "STATION_COORDINATE_SCHEMA",
    "observation_fields",
    "validate_observation_config",
    "validate_processing_config",
]
