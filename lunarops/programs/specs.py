from __future__ import annotations

import math

from lunarops.classes.displacement.terrestrial_geometry import geodetic2itrf
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
    sequence,
    string,
    time,
)

MPI_SCHEMA = ConfigSchema(
    fields=(
        integer(
            "chunksize",
            default=8,
            ge=1,
        ),
    ),
    description="MPI task scheduling options.",
)


def _validate_station_coordinate(config: dict, path: str) -> dict:
    """Normalize either XYZ or WGS84 geodetic coordinates to XYZ."""
    geodetic_fields = ("longitudeDeg", "latitudeDeg", "heightM")
    has_xyz = config.get("xyzM") is not None
    supplied_geodetic = [name for name in geodetic_fields if config.get(name) is not None]
    if has_xyz and supplied_geodetic:
        raise ValueError(f"{path} must specify either xyzM or all geodetic fields, not both.")
    if not has_xyz:
        missing = [name for name in geodetic_fields if config.get(name) is None]
        if missing:
            raise ValueError(f"{path} requires xyzM or all geodetic fields; missing {missing}.")
        config["xyzM"] = geodetic2itrf(
            latitude_rad=math.radians(float(config["latitudeDeg"])),
            longitude_rad=math.radians(float(config["longitudeDeg"])),
            ellipsoidal_height_m=float(config["heightM"]),
        ).tolist()
        for field in geodetic_fields:
            config.pop(field, None)
    return config


STATION_COORDINATE_SCHEMA = ConfigSchema(
    fields=(
        string(
            "key",
            required=True,
            non_empty=True,
        ),
        sequence(
            "xyzM",
            item_kind="number",
            min_items=3,
            max_items=3,
            allow_none=True,
        ),
        number(
            "longitudeDeg",
            ge=-180.0,
            le=180.0,
            allow_none=True,
        ),
        number(
            "latitudeDeg",
            ge=-90.0,
            le=90.0,
            allow_none=True,
        ),
        number(
            "heightM",
            allow_none=True,
        ),
        sequence(
            "velocityMPerYear",
            default=[0.0, 0.0, 0.0],
            item_kind="number",
            min_items=3,
            max_items=3,
        ),
        string(
            "positionEpochUtc",
            default="2010-01-01T00:00:00",
            non_empty=True,
        ),
    ),
    description=(
        "One station position given as ITRF XYZ or WGS84 geodetic longitude, "
        "latitude, and ellipsoidal height, with an optional linear velocity model."
    ),
    validator=_validate_station_coordinate,
)


REFLECTOR_COORDINATE_SCHEMA = ConfigSchema(
    fields=(
        string(
            "key",
            required=True,
            non_empty=True,
        ),
        sequence(
            "xyzM",
            required=True,
            item_kind="number",
            min_items=3,
            max_items=3,
        ),
    ),
    description="One reflector position in the selected ephemeris Moon PA frame.",
)


OBSERVATION_FIELDS = (
    time("startTime", allow_none=True),
    time("endTime", allow_none=True),
    number(
        "utcOffsetHours",
        default=0.0,
        ge=-24.0,
        le=24.0,
    ),
    string("overrideStation", allow_none=True),
    string("overrideReflector", allow_none=True),
    number(
        "minElevationDeg",
        default=0.0,
        ge=0.0,
    ),
    string(
        "lowElevationPolicy",
        default="rejectClamped",
        choices=("rejectClamped", "allowFlagged"),
    ),
    boolean(
        "showProgress",
        default=True,
    ),
    mapping(
        "mpi",
        default={},
        nested=MPI_SCHEMA,
    ),
    class_config("ephemerides", "ephemerides"),
    class_config("earthRotation", "earthRotation"),
    class_config("troposphere", "troposphere"),
    class_config("relativity", "relativity"),
    class_list("stationDisplacement", "stationDisplacement", min_items=1),
    class_config("reflectorDisplacement", "reflectorDisplacement"),
    class_config("rangeBias", "rangeBias"),
)


RESIDUAL_FIELDS = (
    string(
        "outputLevel",
        default="standard",
        choices=("standard", "full"),
    ),
    boolean(
        "includeReflectorDesign",
        default=False,
    ),
)


PARAMETRIZATION_FIELD = class_list(
    "parametrization",
    "parametrization",
    required=True,
    min_items=1,
)


PROCESSING_FIELDS = (
    sequence(
        "varianceComponents",
        required=True,
        item_kind="mapping",
        min_items=1,
    ),
    sequence(
        "processingSteps",
        required=True,
        item_kind="mapping",
        min_items=1,
    ),
)


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
    from lunarops.estimation.adjustment_plan import parse_adjustment_plan

    config = validate_observation_time_config(config, path)
    parse_adjustment_plan(config)
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
        parse_time_with_utc_offset(config.get("startTime"), utc_offset_hours=offset, name="startTime"),
        parse_time_with_utc_offset(config.get("endTime"), utc_offset_hours=offset, name="endTime"),
    )


__all__ = [
    "MPI_SCHEMA",
    "OBSERVATION_FIELDS",
    "PARAMETRIZATION_FIELD",
    "PROCESSING_FIELDS",
    "REFLECTOR_COORDINATE_SCHEMA",
    "RESIDUAL_FIELDS",
    "STATION_COORDINATE_SCHEMA",
    "observation_fields",
    "observation_time_window",
    "validate_observation_time_config",
    "validate_processing_config",
]
