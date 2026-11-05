"""Config fields shared by the LLR observation programs.

``OBSERVATION_FIELDS`` select the normal points (transmit-time window, elevation limit, station and reflector
overrides) and the observation models; model entries fall back to ``observationModel:`` and then to
``shared:``. ``RESIDUAL_FIELDS``, ``PARAMETRIZATION_FIELD`` and ``PROCESSING_FIELDS`` add the result
columns, the estimated parameter blocks and the processing plan.
"""

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
        integer(
            "chunksize",
            default=8,
            ge=1,
            description=(
                "Number of tasks sent to an MPI worker per message. Larger values reduce messaging overhead; smaller "
                "values balance load."
            ),
        ),
    ),
    description="MPI task scheduling options.",
)


OBSERVATION_FIELDS = (
    time(
        "startTime",
        allow_none=True,
        description=(
            "First transmit epoch to process (inclusive), as an ISO date or timestamp. A naive timestamp is read in "
            "the `utcOffsetHours` civil time; null means no lower bound."
        ),
    ),
    time(
        "endTime",
        allow_none=True,
        description=(
            "Transmit epoch at which processing stops (exclusive), in the same form as `startTime`. Null means no "
            "upper bound."
        ),
    ),
    number(
        "utcOffsetHours",
        default=0.0,
        ge=-24.0,
        le=24.0,
        description=(
            "Fixed civil-time offset from UTC in hours (whole minutes, -24 to 24). Applies to naive "
            "`startTime`/`endTime` values and to the local-time columns of the output; the model is always evaluated "
            "in UTC."
        ),
    ),
    string(
        "overrideStation",
        allow_none=True,
        description=(
            "Station catalog key used for every normal point instead of the station identity recorded in the file."
        ),
    ),
    string(
        "overrideReflector",
        allow_none=True,
        description=(
            "Reflector catalog key used for every normal point instead of the reflector identity recorded in the file."
        ),
    ),
    number(
        "minElevationDeg",
        default=0.0,
        ge=0.0,
        description=(
            "Minimum vacuum elevation of the uplink and downlink at the station, in degrees. Normal points below it "
            "are excluded."
        ),
    ),
    string(
        "lowElevationPolicy",
        default="rejectClamped",
        choices=("rejectClamped", "allowFlagged"),
        description=(
            "Handling of normal points whose elevation lies below the validity floor of the troposphere model: "
            "`rejectClamped` excludes them, `allowFlagged` keeps them with the delay evaluated at the floor and flags"
            " them."
        ),
    ),
    boolean("showProgress", default=True, description="Show a progress bar while normal points are evaluated."),
    mapping(
        "mpi",
        default={},
        nested=MPI_SCHEMA,
        description="MPI scheduling options, used when the run is started with `--mpi`.",
    ),
    class_config(
        "ephemerides",
        "ephemerides",
        description="Ephemeris model. Falls back to `observationModel.ephemerides`, then `shared.ephemerides`.",
    ),
    class_config(
        "earthRotation",
        "earthRotation",
        description=(
            "Earth-orientation source. Falls back to `observationModel.earthRotation`, then `shared.earthRotation`."
        ),
    ),
    class_config(
        "troposphere",
        "troposphere",
        description=(
            "Tropospheric delay model. Falls back to `observationModel.troposphere`, then `shared.troposphere`."
        ),
    ),
    class_config(
        "relativity",
        "relativity",
        description=(
            "Relativistic (Shapiro) delay model. Falls back to `observationModel.relativity`, then `shared.relativity`."
        ),
    ),
    class_list(
        "stationDisplacement",
        "stationDisplacement",
        min_items=1,
        description=(
            "Station displacement models; their displacements are summed. Falls back to "
            "`observationModel.stationDisplacement`, then `shared.stationDisplacement`."
        ),
    ),
    class_config(
        "reflectorDisplacement",
        "reflectorDisplacement",
        description=(
            "Reflector displacement model. Falls back to `observationModel.reflectorDisplacement`, then "
            "`shared.reflectorDisplacement`."
        ),
    ),
    class_config(
        "rangeBias",
        "rangeBias",
        description=(
            "Deterministic range-bias correction applied to the computed observable. Falls back to "
            "`observationModel.rangeBias`, then `shared.rangeBias`."
        ),
    ),
)


RESIDUAL_FIELDS = (
    string(
        "outputLevel",
        default="standard",
        choices=("standard", "full"),
        description=(
            "Column set of the observation-result table: `standard` (residuals and main diagnostics) or `full` (every"
            " model component)."
        ),
    ),
    boolean(
        "includeReflectorDesign",
        default=False,
        description=(
            "Add the columns `design_reflector_pa_x/y/z`: partial derivatives of the one-way residual with respect to"
            " the reflector PA coordinates."
        ),
    ),
)


PARAMETRIZATION_FIELD = class_list(
    "parametrization",
    "parametrization",
    required=True,
    min_items=1,
    description=(
        "Parameter blocks available to the estimation. Each entry is a class declaration such as `{type: "
        "reflectorPosition}`."
    ),
)


from lunarops.config.processing import PROCESSING_FIELDS, validate_processing_plan


def observation_fields(
    *,
    parametrized: bool = False,
    processing: bool = False,
    residual: bool = False,
) -> tuple[FieldSpec, ...]:
    """Return the config fields of an observation program.

    Parameters
    ----------
    parametrized : bool, optional
        Add the ``parametrization`` field.
    processing : bool, optional
        Add the processing-plan fields ``varianceComponents`` and ``processingSteps``.
    residual : bool, optional
        Add the result-table fields ``outputLevel`` and ``includeReflectorDesign``.

    Returns
    -------
    tuple of FieldSpec
        ``OBSERVATION_FIELDS`` followed by the selected groups.
    """
    fields = list(OBSERVATION_FIELDS)
    if parametrized:
        fields.append(PARAMETRIZATION_FIELD)
    if processing:
        fields.extend(PROCESSING_FIELDS)
    if residual:
        fields.extend(RESIDUAL_FIELDS)
    return tuple(fields)


def validate_processing_config(config: dict, path: str) -> dict:
    """Validate the time window and the processing plan of a processing config.

    Parameters
    ----------
    config : dict
        Program config after field validation.
    path : str
        Config path used in error messages.

    Returns
    -------
    dict
        The validated config.

    Raises
    ------
    ValueError
        If the time settings or the processing plan are invalid.
    """
    config = validate_observation_time_config(config, path)
    validate_processing_plan(config, path)
    return config


def validate_observation_time_config(config: dict, path: str) -> dict:
    """Validate ``utcOffsetHours``, ``startTime`` and ``endTime`` of an observation config.

    Parameters
    ----------
    config : dict
        Program config after field validation.
    path : str
        Config path used in error messages.

    Returns
    -------
    dict
        The config, unchanged.

    Raises
    ------
    ValueError
        If ``utcOffsetHours`` is not a whole number of minutes or a time cannot be parsed.
    """
    offset = config["utcOffsetHours"]
    if abs(offset * 60.0 - round(offset * 60.0)) > 1.0e-9:
        raise ValueError(f"{path}.utcOffsetHours must represent a whole number of minutes.")
    config["utcOffsetHours"] = offset
    observation_time_window(config)
    return config


def observation_time_window(config: dict) -> tuple[Epoch | None, Epoch | None]:
    """Return the transmit-time window of an observation config as UTC epochs.

    Times without a zone suffix are read as civil time with offset ``utcOffsetHours`` (UTC = local time -
    offset).

    Parameters
    ----------
    config : dict
        Program config with ``utcOffsetHours`` and optional ``startTime`` and ``endTime``.

    Returns
    -------
    tuple of (Epoch or None, Epoch or None)
        Inclusive start and exclusive end on the UTC scale; None where unbounded.
    """
    offset = config["utcOffsetHours"]
    return (
        None
        if config.get("startTime") is None
        else parse_time_with_utc_offset(config["startTime"], utc_offset_hours=offset, name="startTime"),
        None
        if config.get("endTime") is None
        else parse_time_with_utc_offset(config["endTime"], utc_offset_hours=offset, name="endTime"),
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
