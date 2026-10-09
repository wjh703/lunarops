"""Predict LLR geometric pointing and coarse visibility windows on a UTC grid."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]

from lunarops.classes.observation import (
    LlrObservationPredictor,
    PredictionCriteria,
    PredictionMeteorology,
    build_visibility_windows,
    resolve_catalog_key,
)
from lunarops.classes.observation_factory import build_observation_runtime
from lunarops.classes.time import (
    Epoch,
    parse_time_with_utc_offset,
    validate_utc_offset_hours,
)
from lunarops.config.context import RunContext
from lunarops.config.schema import (
    ConfigSchema,
    UiHints,
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
from lunarops.programs.specs import MPI_SCHEMA

_ELONGATION_RANGE_SCHEMA = ConfigSchema(
    fields=(
        number("startDeg", required=True, minimum=0.0, maximum=360.0, allow_none=False),
        number("endDeg", required=True, minimum=0.0, maximum=360.0, allow_none=False),
    ),
    description="One inclusive mean-elongation interval; start greater than end wraps through zero.",
)


def _parse_utc(value: object, *, name: str, utc_offset_hours: object = 0.0) -> Epoch:
    try:
        epoch = parse_time_with_utc_offset(value, utc_offset_hours=utc_offset_hours, name=name)
        if epoch is None:
            raise ValueError("time must not be empty")
        return epoch
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid UTC/local ISO date or timestamp.") from exc


def _require_millisecond_aligned(epoch: Epoch, *, name: str) -> None:
    rounded = Epoch.from_isot(epoch.isot(precision=3))
    if abs(epoch.seconds_until(rounded)) > 1.0e-9:
        raise ValueError(f"{name} must be aligned to a whole millisecond for three-digit prediction timestamps.")


def _validate_config(config: dict, path_name: str) -> dict:
    offset = validate_utc_offset_hours(config.get("utcOffsetHours", 0.0))
    if abs(offset * 60.0 - round(offset * 60.0)) > 1.0e-9:
        raise ValueError(f"{path_name}.utcOffsetHours must represent a whole number of minutes.")
    config["utcOffsetHours"] = offset
    start = _parse_utc(config["startTime"], name=f"{path_name}.startTime", utc_offset_hours=offset)
    end = _parse_utc(config["endTime"], name=f"{path_name}.endTime", utc_offset_hours=offset)
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
    time(
        "startTime",
        required=True,
        non_empty=True,
        allow_none=False,
        ui=UiHints(group="Time grid", widget="datetime-range-start"),
    ),
    time(
        "endTime",
        required=True,
        non_empty=True,
        allow_none=False,
        ui=UiHints(group="Time grid", widget="datetime-range-end"),
    ),
    number(
        "utcOffsetHours",
        default=0.0,
        minimum=-24.0,
        maximum=24.0,
        allow_none=False,
        ui=UiHints(group="Time grid", unit="h"),
    ),
    number(
        "stepSeconds",
        default=60.0,
        minimum=0.0,
        minimum_exclusive=True,
        allow_none=False,
        ui=UiHints(group="Time grid", unit="s"),
    ),
    mapping("mpi", nested=MPI_SCHEMA),
    string("stationName", required=True, non_empty=True, allow_none=False, ui=UiHints(group="Target")),
    string("reflectorName", required=True, non_empty=True, allow_none=False, ui=UiHints(group="Target")),
    number(
        "minElevationDeg",
        default=20.0,
        minimum=0.0,
        maximum=90.0,
        allow_none=False,
        ui=UiHints(group="Visibility", unit="deg"),
    ),
    number(
        "minReflectorElevationDeg",
        default=0.0,
        minimum=-90.0,
        maximum=90.0,
        allow_none=False,
        ui=UiHints(group="Visibility", unit="deg"),
    ),
    number(
        "maxSunElevationDeg",
        default=-6.0,
        minimum=-90.0,
        maximum=90.0,
        allow_none=False,
        ui=UiHints(group="Visibility", unit="deg"),
    ),
    sequence(
        "allowedElongationRangesDeg",
        default=[{"startDeg": 0.0, "endDeg": 360.0}],
        item_kind="mapping",
        item_nested=_ELONGATION_RANGE_SCHEMA,
        min_items=1,
        allow_none=False,
        ui=UiHints(group="Visibility", unit="deg"),
    ),
    number(
        "pressureHpa",
        default=900.0,
        minimum=0.0,
        minimum_exclusive=True,
        allow_none=False,
        ui=UiHints(group="Meteorology", unit="hPa", advanced=True),
    ),
    number(
        "temperatureK",
        default=285.0,
        minimum=0.0,
        minimum_exclusive=True,
        allow_none=False,
        ui=UiHints(group="Meteorology", unit="K", advanced=True),
    ),
    number(
        "relativeHumidityPercent",
        default=25.0,
        minimum=0.0,
        maximum=100.0,
        allow_none=False,
        ui=UiHints(group="Meteorology", unit="%", advanced=True),
    ),
    number(
        "wavelengthNm",
        default=532.0,
        minimum=0.0,
        minimum_exclusive=True,
        allow_none=False,
        ui=UiHints(group="Meteorology", unit="nm", advanced=True),
    ),
    boolean("showProgress", default=True, allow_none=False, ui=UiHints(group="Runtime", advanced=True)),
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
    utc_offset_hours = validate_utc_offset_hours(config.get("utcOffsetHours", 0.0))
    start = _parse_utc(config["startTime"], name="startTime", utc_offset_hours=utc_offset_hours)
    end = _parse_utc(config["endTime"], name="endTime", utc_offset_hours=utc_offset_hours)
    step_seconds = float(config["stepSeconds"])
    epochs, count = _utc_grid(start, end, step_seconds)
    runtime_mpi = context.runtime
    if runtime_mpi is not None and runtime_mpi.has_workers:
        from lunarops.parallel.mpi import make_observation_spec, mpi_prediction_rows

        spec = make_observation_spec(config, context)
        station_key = resolve_catalog_key(config["stationName"], spec["stationCatalog"], "Station")
        reflector_key = resolve_catalog_key(config["reflectorName"], spec["reflectorCatalog"], "Reflector")

        def evaluate(epochs):
            return mpi_prediction_rows(
                runtime_mpi,
                spec,
                list(epochs),
                station=station_key,
                reflector=reflector_key,
                criteria=asdict(criteria),
                meteorology=asdict(meteorology),
                utc_offset_hours=utc_offset_hours,
                chunksize=int((config.get("mpi") or {}).get("chunksize", 8)),
                progress_desc="LLR prediction",
                quiet=not bool(config.get("showProgress", True)),
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
        bool(config["showProgress"] and not (runtime_mpi is not None and runtime_mpi.has_workers)),
    )


@program(
    ProgramSpec(
        name="LlrObservationPrediction",
        summary="Predict LLR uplink pointing, Sun elevation, mean elongation, and visibility windows.",
        inputs=(
            ArtifactSlot("inputFileStationCatalog", "StationCatalogFile"),
            ArtifactSlot("inputFileReflectorCatalog", "ReflectorCatalogFile"),
        ),
        outputs=(
            ArtifactSlot("outputFilePrediction", "PredictionResultFile"),
            ArtifactSlot("outputFileWindows", "PredictionWindowFile"),
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

    prediction_path = write_prediction_results(
        rows,
        problem.output_paths[0],
    )
    windows_path = write_prediction_windows(
        windows,
        problem.output_paths[1],
    )
    print(
        f"[LlrObservationPrediction] {len(rows)} epoch(s), {len(windows)} window(s) "
        f"-> {prediction_path}, {windows_path}"
    )
    return {"rows": rows, "windows": windows}


__all__ = ["llr_observation_prediction"]
