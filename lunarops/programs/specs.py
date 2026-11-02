from __future__ import annotations

from lunarops.classes.time import Epoch, parse_time_with_utc_offset
from lunarops.config.schema import (
    ConfigSchema,
    FieldSpec,
    boolean,
    class_config,
    class_list,
    integer,
    mapping,
    number,
    string,
    time,
)

MPI_SCHEMA = ConfigSchema(
    fields=(
        integer("chunksize", default=8, ge=1),
    ),
    description="MPI task scheduling options.",
)


OBSERVATION_FIELDS = (
    time("startTime", allow_none=True),
    time("endTime", allow_none=True),
    number("utcOffsetHours", default=0.0, ge=-24.0, le=24.0),
    string("overrideStation", allow_none=True),
    string("overrideReflector", allow_none=True),
    number("minElevationDeg", default=0.0, ge=0.0),
    string("lowElevationPolicy", default="rejectClamped", choices=("rejectClamped", "allowFlagged")),
    boolean("showProgress", default=True),
    mapping("mpi", default={}, nested=MPI_SCHEMA),
    class_config("ephemerides", "ephemerides"),
    class_config("earthRotation", "earthRotation"),
    class_config("troposphere", "troposphere"),
    class_config("relativity", "relativity"),
    class_list("stationDisplacement", "stationDisplacement", min_items=1),
    class_config("reflectorDisplacement", "reflectorDisplacement"),
    class_config("rangeBias", "rangeBias"),
)


RESIDUAL_FIELDS = (
    string("outputLevel", default="standard", choices=("standard", "full")),
    boolean("includeReflectorDesign", default=False),
)


PARAMETRIZATION_FIELD = class_list("parametrization", "parametrization", required=True, min_items=1)


from lunarops.config.processing import PROCESSING_FIELDS, validate_processing_plan


def observation_fields(
    *,
    parametrized: bool = False,
    processing: bool = False,
    residual: bool = False,
) -> tuple[FieldSpec, ...]:
    fields = list(OBSERVATION_FIELDS)
    if parametrized:
        fields.append(PARAMETRIZATION_FIELD)
    if processing:
        fields.extend(PROCESSING_FIELDS)
    if residual:
        fields.extend(RESIDUAL_FIELDS)
    return tuple(fields)


def validate_processing_config(config: dict, path: str) -> dict:
    config = validate_observation_time_config(config, path)
    validate_processing_plan(config, path)
    return config


def validate_observation_time_config(config: dict, path: str) -> dict:
    offset = float(config["utcOffsetHours"])
    if abs(offset * 60.0 - round(offset * 60.0)) > 1.0e-9:
        raise ValueError(f"{path}.utcOffsetHours must represent a whole number of minutes.")
    config["utcOffsetHours"] = offset
    observation_time_window(config)
    return config


def observation_time_window(config: dict) -> tuple[Epoch | None, Epoch | None]:
    offset = config["utcOffsetHours"]
    return (
        None if config.get("startTime") is None else parse_time_with_utc_offset(config["startTime"], utc_offset_hours=offset, name="startTime"),
        None if config.get("endTime") is None else parse_time_with_utc_offset(config["endTime"], utc_offset_hours=offset, name="endTime"),
    )


__all__ = [
    "MPI_SCHEMA",
    "OBSERVATION_FIELDS",
    "PARAMETRIZATION_FIELD",
    "PROCESSING_FIELDS",
    "RESIDUAL_FIELDS",
    "observation_fields",
    "observation_time_window",
    "validate_observation_time_config",
    "validate_processing_config",
]
