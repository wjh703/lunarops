"""Shared declarative schema fragments for LLR programs."""

from __future__ import annotations

import math

from lunarops.classes.displacement.terrestrial_geometry import geodetic2itrf
from lunarops.classes.time import parse_time_with_utc_offset, validate_utc_offset_hours
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
            raise ValueError(
                f"{path} requires xyzM or all geodetic fields; missing {missing}."
            )
        config["xyzM"] = geodetic2itrf(
            latitude_rad=math.radians(float(config["latitudeDeg"])),
            longitude_rad=math.radians(float(config["longitudeDeg"])),
            ellipsoidal_height_m=float(config["heightM"]),
        ).tolist()
        for field in geodetic_fields:
            config.pop(field, None)
    return config


# Catalog creation uses small, explicit coordinate records.  Observation
# programs consume only the native catalog artifacts produced from them.
STATION_COORDINATE_SCHEMA = ConfigSchema(
    fields=(
        string("key", required=True, non_empty=True, allow_none=False),
        sequence(
            "xyzM",
            item_kind="number",
            min_items=3,
            max_items=3,
            allow_none=True,
        ),
        number(
            "longitudeDeg",
            minimum=-180.0,
            maximum=180.0,
            allow_none=True,
            ui=UiHints(group="Geodetic position", unit="deg"),
        ),
        number(
            "latitudeDeg",
            minimum=-90.0,
            maximum=90.0,
            allow_none=True,
            ui=UiHints(group="Geodetic position", unit="deg"),
        ),
        number(
            "heightM",
            allow_none=True,
            ui=UiHints(group="Geodetic position", unit="m"),
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
    description=(
        "One station position given as ITRF XYZ or WGS84 geodetic longitude, "
        "latitude, and ellipsoidal height, with an optional linear velocity model."
    ),
    validator=_validate_station_coordinate,
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
    number(
        "utcOffsetHours",
        default=0.0,
        minimum=-24.0,
        maximum=24.0,
        allow_none=False,
        ui=UiHints(group="Time window", unit="h"),
    ),
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


def validate_processing_config(config: dict, path: str) -> dict:
    """Run the scientific processing parser as the program schema validator."""
    from lunarops.estimation.adjustment_config import parse_adjustment_plan

    parse_adjustment_plan(config)
    return validate_observation_time_config(config, path)


def validate_observation_time_config(config: dict, path: str) -> dict:
    """Validate shared observation time inputs and normalize the fixed offset."""
    offset = validate_utc_offset_hours(config.get("utcOffsetHours", 0.0))
    config["utcOffsetHours"] = offset
    for field in ("startTime", "endTime"):
        value = config.get(field)
        if value is not None:
            parse_time_with_utc_offset(value, utc_offset_hours=offset, name=f"{path}.{field}")
    return config


__all__ = [
    "PROCESSING_FIELDS",
    "REFLECTOR_COORDINATE_SCHEMA",
    "OBSERVATION_FIELDS",
    "PARAMETRIZATION_FIELD",
    "RESIDUAL_FIELDS",
    "STATION_COORDINATE_SCHEMA",
    "observation_fields",
    "validate_processing_config",
    "validate_observation_time_config",
]
