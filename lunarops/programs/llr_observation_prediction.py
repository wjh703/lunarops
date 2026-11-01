"""Predict LLR geometric pointing and coarse visibility windows on a UTC grid."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]

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
    class_config,
    class_list,
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
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program
from lunarops.programs.specs import MPI_SCHEMA, observation_time_window, validate_observation_time_config

_ELONGATION_RANGE_SCHEMA = ConfigSchema(
    fields=(
        number("startDeg", required=True, ge=0.0, le=360.0),
        number("endDeg", required=True, ge=0.0, le=360.0),
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
    duration = start.seconds_until(end)
    tolerance_s = max(1.0e-9, step_seconds * 1.0e-12)
    count = math.floor((duration + tolerance_s) / step_seconds) + 1
    return (start.shifted(index * step_seconds) for index in range(count)), count


_PREDICTION_FIELDS = (
    time("startTime", required=True, non_empty=True),
    time("endTime", required=True, non_empty=True),
    number("utcOffsetHours", default=0.0, ge=-24.0, le=24.0),
    number("stepSeconds", default=60.0, gt=0.0),
    mapping("mpi", default={}, nested=MPI_SCHEMA),
    string("stationName", required=True, non_empty=True),
    string("reflectorName", required=True, non_empty=True),
    number("minElevationDeg", default=20.0, ge=0.0, le=90.0),
    number("minReflectorElevationDeg", default=0.0, ge=-90.0, le=90.0),
    number("maxSunElevationDeg", default=-6.0, ge=-90.0, le=90.0),
    sequence(
        "allowedElongationRangesDeg",
        default=[{"startDeg": 0.0, "endDeg": 360.0}],
        item_kind="mapping",
        item_nested=_ELONGATION_RANGE_SCHEMA,
        min_items=1,
    ),
    string("meteorologyPolicy", required=True, choices=("nominal", "measured")),
    number("pressureHpa", gt=0.0),
    number("temperatureK", gt=0.0),
    number("relativeHumidityPercent", ge=0.0, le=100.0),
    number("wavelengthNm", gt=0.0),
    boolean("showProgress", default=True),
    class_config("ephemerides", "ephemerides"),
    class_config("earthRotation", "earthRotation"),
    class_config("troposphere", "troposphere"),
    class_config("relativity", "relativity"),
    class_list("stationDisplacement", "stationDisplacement", min_items=1),
    class_config("reflectorDisplacement", "reflectorDisplacement"),
)


@dataclass(frozen=True, slots=True)
class PredictionProblem:
    epochs: Iterator[Epoch]
    count: int
    step_seconds: float
    evaluate: Callable
    output_paths: tuple[Path, Path]
    show_serial_progress: bool


def build_prediction_problem(config, context) -> PredictionProblem:
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
    runtime_mpi = context.runtime
    if runtime_mpi is not None:
        from lunarops.parallel.mpi import mpi_prediction_rows
        from lunarops.parallel.observation_spec import make_observation_spec

        spec = make_observation_spec(config, context)
        station_key = resolve_catalog_key(config["stationName"], spec["stationCatalog"], "Station")
        reflector_key = resolve_catalog_key(config["reflectorName"], spec["reflectorCatalog"], "Reflector")

        def evaluate(epochs):
            return mpi_prediction_rows(
                runtime_mpi,
                spec,
                list(epochs),
                step_seconds=step_seconds,
                station=station_key,
                reflector=reflector_key,
                criteria=asdict(criteria),
                meteorology=asdict(meteorology),
                utc_offset_hours=utc_offset_hours,
                chunksize=int(config["mpi"]["chunksize"]),
                progress_desc="LLR prediction",
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
        bool(config["showProgress"] and not (runtime_mpi is not None)),
    )


@program(
    ProgramSpec(
        name="LlrObservationPrediction",
        summary="Predict LLR uplink pointing, Sun elevation, mean elongation, and visibility windows.",
        inputs=(
            ArtifactSlot("inputFileStationCatalog", "stationCatalog"),
            ArtifactSlot("inputFileReflectorCatalog", "reflectorCatalog"),
        ),
        outputs=(
            ArtifactSlot("outputFilePrediction", "observationPrediction"),
            ArtifactSlot("outputFileWindows", "predictionWindow"),
        ),
        fields=_PREDICTION_FIELDS,
        validator=_validate_config,
    )
)
def llr_observation_prediction(config: dict, context: RunContext):
    problem = build_prediction_problem(config, context)
    epochs = problem.epochs
    if problem.show_serial_progress:
        epochs = iter(_tqdm(epochs, total=problem.count, desc="LLR prediction", unit="epoch"))
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
        "executionMode": "mpi" if context.runtime is not None else "serial",
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
    print(
        f"[LlrObservationPrediction] {len(rows)} epoch(s), {len(windows)} window(s) "
        f"-> {prediction_path}, {windows_path}"
    )
    return {"rows": rows, "windows": windows}


__all__ = ["llr_observation_prediction"]
