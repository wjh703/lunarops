"""LLR observation prediction on a UTC epoch grid.

For one station and one reflector, the ``LlrObservationPrediction`` program evaluates on an evenly spaced
UTC grid the uplink pointing, the reflector elevation above its local lunar horizon, the Sun elevation at
the station and the mean elongation of the Moon from the Sun, and flags each epoch as observable or not.
Visibility windows are runs of adjacent observable grid epochs; their boundaries are the first and last
observable samples, without a search for the exact crossing times.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from lunarops import console
from lunarops.classes.observation import (
    LlrObservationPredictor,
    PredictionCriteria,
    PredictionMeteorology,
    build_visibility_windows,
    resolve_catalog_key,
)
from lunarops.classes.observation.assembly import build_observation_runtime
from lunarops.classes.time import (
    Epoch,
)
from lunarops.config.context import RunContext
from lunarops.config.schema import (
    ConfigSchema,
    boolean,
    mapping,
    number,
    sequence,
    string,
    time,
)
from lunarops.fileio.prediction_results import (
    write_prediction_results,
    write_prediction_windows,
)
from lunarops.programs.registry import ArtifactSlot, program, program_spec
from lunarops.programs.specs import (
    MPI_SCHEMA,
    model_override_fields,
    observation_time_window,
    validate_observation_time_config,
)

_ELONGATION_RANGE_SCHEMA = ConfigSchema(
    fields=(
        number(
            "startDeg",
            required=True,
            ge=0.0,
            le=360.0,
            description="Start of the allowed mean-elongation interval, in degrees (inclusive).",
        ),
        number(
            "endDeg",
            required=True,
            ge=0.0,
            le=360.0,
            description=(
                "End of the allowed mean-elongation interval, in degrees (inclusive). A value below `startDeg` wraps "
                "through 0."
            ),
        ),
    ),
    description="One inclusive mean-elongation interval; start greater than end wraps through zero.",
)


def _require_millisecond_aligned(epoch: Epoch, *, name: str) -> None:
    rounded = Epoch.from_isot(epoch.isot(precision=3))
    if abs(epoch.seconds_until(rounded)) > 1.0e-9:
        raise ValueError(f"{name} must be aligned to a whole millisecond for three-digit prediction timestamps.")


def _validate_config(config: dict, path_name: str) -> dict:
    nominal_meteorology = PredictionMeteorology()
    fields = {
        "pressureHpa": nominal_meteorology.pressure_hpa,
        "temperatureK": nominal_meteorology.temperature_k,
        "relativeHumidityPercent": nominal_meteorology.relative_humidity_percent,
        "wavelengthNm": nominal_meteorology.wavelength_nm,
    }
    for key, nominal in fields.items():
        if key not in config:
            if config["meteorologyPolicy"] != "nominal":
                raise ValueError(f"{path_name}.{key} is required for measured meteorology")
            config[key] = nominal
    config = validate_observation_time_config(config, path_name)
    start, end = observation_time_window(config)
    start, end = cast(Epoch, start), cast(Epoch, end)
    if start.seconds_until(end) < 0.0:
        raise ValueError(f"{path_name}.endTime must not precede startTime.")
    _require_millisecond_aligned(start, name=f"{path_name}.startTime")
    milliseconds = float(config["stepSeconds"]) * 1000.0
    rounded_milliseconds = round(milliseconds)
    if rounded_milliseconds < 1 or abs(milliseconds - rounded_milliseconds) > 1.0e-9:
        raise ValueError(f"{path_name}.stepSeconds must be a positive whole number of milliseconds.")
    config["stepSeconds"] = rounded_milliseconds / 1000.0
    return config


def _utc_grid(start: Epoch, end: Epoch, step_seconds: float) -> tuple[Iterator[Epoch], int]:
    """Return the grid ``start + k * step_seconds`` up to ``end`` (included within a small tolerance)."""
    duration = start.seconds_until(end)
    tolerance_s = max(1.0e-9, step_seconds * 1.0e-12)
    count = math.floor((duration + tolerance_s) / step_seconds) + 1
    return (start.shifted(index * step_seconds) for index in range(count)), count


_PREDICTION_FIELDS = (
    time(
        "startTime",
        required=True,
        non_empty=True,
        description=(
            "First epoch of the prediction grid, as an ISO timestamp aligned to a whole millisecond. A naive "
            "timestamp is read in the `utcOffsetHours` civil time."
        ),
    ),
    time(
        "endTime",
        required=True,
        non_empty=True,
        description=(
            "Last epoch of the prediction grid (inclusive when it falls on the grid), in the same form as `startTime`."
        ),
    ),
    number(
        "utcOffsetHours",
        default=0.0,
        ge=-24.0,
        le=24.0,
        description=(
            "Fixed civil-time offset from UTC in hours (whole minutes). Applies to naive time inputs and to the "
            "local-time columns."
        ),
    ),
    number(
        "stepSeconds",
        default=60.0,
        gt=0.0,
        description="Spacing of the prediction grid, in seconds (a whole number of milliseconds).",
    ),
    mapping(
        "mpi",
        default={},
        nested=MPI_SCHEMA,
        description="Worker task scheduling options, used when the run is started with `--mpi` or `--workers`.",
    ),
    string("stationName", required=True, non_empty=True, description="Station catalog key of the observing station."),
    string("reflectorName", required=True, non_empty=True, description="Reflector catalog key of the target."),
    number(
        "minElevationDeg",
        default=20.0,
        ge=0.0,
        le=90.0,
        description="Minimum topocentric elevation of the reflector at the station, in degrees.",
    ),
    number(
        "minReflectorElevationDeg",
        default=0.0,
        ge=-90.0,
        le=90.0,
        description="Minimum elevation of the incoming beam above the reflector's local lunar horizon, in degrees.",
    ),
    number(
        "maxSunElevationDeg",
        default=-6.0,
        ge=-90.0,
        le=90.0,
        description="Maximum Sun elevation at the station, in degrees (-6 is civil twilight).",
    ),
    sequence(
        "allowedElongationRangesDeg",
        default=[{"startDeg": 0.0, "endDeg": 360.0}],
        item_kind="mapping",
        item_nested=_ELONGATION_RANGE_SCHEMA,
        min_items=1,
        description=(
            "Allowed intervals of the mean elongation D of the Moon from the Sun, in degrees. A sample is observable "
            "if D lies in any interval."
        ),
    ),
    string(
        "meteorologyPolicy",
        required=True,
        choices=("nominal", "measured"),
        description=(
            "`nominal` uses representative weather (900 hPa, 285 K, 25 %, 532 nm) for any value not given; `measured`"
            " requires all four values."
        ),
    ),
    number("pressureHpa", gt=0.0, description="Station surface pressure for the troposphere model, in hPa."),
    number("temperatureK", gt=0.0, description="Station surface temperature, in kelvin."),
    number("relativeHumidityPercent", ge=0.0, le=100.0, description="Station relative humidity, in percent."),
    number("wavelengthNm", gt=0.0, description="Laser wavelength, in nanometres."),
    boolean("showProgress", default=True, description="Report progress while the grid is evaluated."),
    *model_override_fields(without=("rangeBias",)),
)


@dataclass(frozen=True, slots=True)
class PredictionProblem:
    """Prepared prediction run.

    Attributes
    ----------
    epochs : iterator of Epoch
        UTC grid epochs.
    count : int
        Number of grid epochs.
    step_seconds : float
        Grid spacing, in seconds.
    evaluate : callable
        Function mapping an iterable of epochs to the list of prediction rows, serially or on the runtime workers.
    output_paths : tuple of (Path, Path)
        Paths of the prediction table and of the visibility-window table.
    show_serial_progress : bool
        Whether the caller reports progress (serial runs only; parallel runs report their own progress).
    """

    epochs: Iterator[Epoch]
    count: int
    step_seconds: float
    evaluate: Callable
    output_paths: tuple[Path, Path]
    show_serial_progress: bool


def build_prediction_problem(config, context) -> PredictionProblem:
    """Build the epoch grid, the observability criteria and the evaluation function of a prediction config.

    With a task runtime in the context the rows are evaluated by its workers; otherwise a predictor is
    built in this process from the observation models of the config.

    Parameters
    ----------
    config : dict
        Resolved ``LlrObservationPrediction`` config.
    context : RunContext
        Run context.

    Returns
    -------
    PredictionProblem
        The prepared prediction run.
    """
    criteria = PredictionCriteria(
        minimum_elevation_deg=float(config["minElevationDeg"]),
        minimum_reflector_elevation_deg=float(config["minReflectorElevationDeg"]),
        maximum_sun_elevation_deg=float(config["maxSunElevationDeg"]),
        allowed_elongation_ranges_deg=tuple(
            (float(item["startDeg"]), float(item["endDeg"])) for item in config["allowedElongationRangesDeg"]
        ),
    )
    meteorology = PredictionMeteorology(
        pressure_hpa=float(config["pressureHpa"]),
        temperature_k=float(config["temperatureK"]),
        relative_humidity_percent=float(config["relativeHumidityPercent"]),
        wavelength_nm=float(config["wavelengthNm"]),
    )
    utc_offset_hours = config["utcOffsetHours"]
    start, end = observation_time_window(config)
    step_seconds = float(config["stepSeconds"])
    epochs, count = _utc_grid(cast(Epoch, start), cast(Epoch, end), step_seconds)
    task_runtime = context.runtime
    if task_runtime is not None:
        from lunarops.parallel.mpi import mpi_prediction_rows
        from lunarops.parallel.observation_spec import make_observation_spec

        spec = make_observation_spec(config, context)
        station_key = resolve_catalog_key(config["stationName"], spec["stationCatalog"], "Station")
        reflector_key = resolve_catalog_key(config["reflectorName"], spec["reflectorCatalog"], "Reflector")

        def evaluate(epochs):
            return mpi_prediction_rows(
                task_runtime,
                spec,
                list(epochs),
                step_seconds=step_seconds,
                station=station_key,
                reflector=reflector_key,
                criteria=asdict(criteria),
                meteorology=asdict(meteorology),
                utc_offset_hours=utc_offset_hours,
                chunksize=int(config["mpi"]["chunksize"]),
                progress_desc="prediction",
                quiet=not bool(config["showProgress"]),
            )
    else:
        runtime = build_observation_runtime(context, config)
        station_key = resolve_catalog_key(config["stationName"], runtime.assembly.station_catalog, "Station")
        reflector_key = resolve_catalog_key(config["reflectorName"], runtime.assembly.reflector_catalog, "Reflector")
        predictor = LlrObservationPredictor(
            runtime.frames,
            runtime.light_time_solver,
            runtime.assembly.station_catalog[station_key],
            runtime.assembly.reflector_catalog[reflector_key],
            station_key=station_key,
            reflector_key=reflector_key,
            criteria=criteria,
            meteorology=meteorology,
            utc_offset_hours=utc_offset_hours,
        )

        def evaluate(epochs):
            return [predictor.evaluate(epoch) for epoch in epochs]

    return PredictionProblem(
        epochs,
        count,
        step_seconds,
        evaluate,
        (context.resolve_path(config["outputFilePrediction"]), context.resolve_path(config["outputFileWindows"])),
        bool(config["showProgress"] and not (task_runtime is not None)),
    )


@program(
    program_spec(
        name="LlrObservationPrediction",
        summary="Predict LLR uplink pointing, Sun elevation, mean elongation, and visibility windows.",
        inputs=(
            ArtifactSlot(
                "inputFileStationCatalog",
                "stationCatalog",
                description="Native station catalog that contains `stationName`.",
            ),
            ArtifactSlot(
                "inputFileReflectorCatalog",
                "reflectorCatalog",
                description="Native reflector catalog that contains `reflectorName`.",
            ),
        ),
        outputs=(
            ArtifactSlot(
                "outputFilePrediction",
                "observationPrediction",
                description="Output table with one row per grid epoch.",
            ),
            ArtifactSlot(
                "outputFileWindows",
                "predictionWindow",
                description="Output table of visibility windows built from adjacent observable samples.",
            ),
        ),
        fields=_PREDICTION_FIELDS,
        validator=_validate_config,
    )
)
def llr_observation_prediction(config: dict, context: RunContext):
    """Predict pointing and visibility of one reflector from one station on a UTC grid.

    Reads the station and reflector from ``inputFileStationCatalog`` and ``inputFileReflectorCatalog``,
    evaluates every epoch from ``startTime`` to ``endTime`` with spacing ``stepSeconds`` against the
    elevation, Sun-elevation and elongation criteria, and writes the per-epoch ``observationPrediction``
    table ``outputFilePrediction`` and the ``predictionWindow`` table ``outputFileWindows``. Both files record
    the meteorology used and the number of epochs failing each criterion. See the program reference
    (``docs/reference/programs.md``) for every configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    dict
        ``rows`` (prediction rows, one per epoch) and ``windows`` (visibility windows).
    """
    problem = build_prediction_problem(config, context)
    epochs = problem.epochs
    if problem.show_serial_progress:
        epochs = iter(console.progress(epochs, total=problem.count, desc="prediction", unit="epochs"))
    rows = problem.evaluate(epochs)
    step_seconds = problem.step_seconds
    windows = build_visibility_windows(rows, step_seconds=step_seconds)

    metadata = {
        "meteorologyPolicy": config["meteorologyPolicy"],
        "meteorology": {
            key: config[key] for key in ("pressureHpa", "temperatureK", "relativeHumidityPercent", "wavelengthNm")
        },
        "gridDefined": True,
        "stepSeconds": step_seconds,
        "windowBoundaryConvention": "inclusive first/last observable samples; no continuous crossing search",
        "conditionFailures": {
            key: sum(not row[key] for row in rows)
            for key in ("station_elevation_ok", "reflector_elevation_ok", "sun_elevation_ok", "elongation_ok")
        },
        "executionMode": context.runtime.name if context.runtime is not None else "serial",
    }
    prediction_path = write_prediction_results(
        rows,
        problem.output_paths[0],
        metadata=metadata,
    )
    windows_path = write_prediction_windows(
        windows,
        problem.output_paths[1],
        metadata=metadata,
    )
    console.summary({"epochs": len(rows), "visibility windows": len(windows)}, outputs=[prediction_path, windows_path])
    return {"rows": rows, "windows": windows}


__all__ = ["llr_observation_prediction"]
