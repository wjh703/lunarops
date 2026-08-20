"""Predict LLR geometric pointing and coarse visibility windows on a UTC grid."""

from __future__ import annotations

import math
from collections.abc import Iterator

from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]

from lunarops.classes.observation import (
    LlrObservationPredictor,
    PredictionCriteria,
    PredictionMeteorology,
    build_visibility_windows,
    resolve_catalog_key,
)
from lunarops.classes.observation_factory import build_observation_runtime
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.context import RunContext
from lunarops.config.schema import (
    ConfigSchema,
    UiHints,
    boolean,
    class_config,
    class_list,
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


_ELONGATION_RANGE_SCHEMA = ConfigSchema(
    fields=(
        number("startDeg", required=True, minimum=0.0, maximum=360.0, allow_none=False),
        number("endDeg", required=True, minimum=0.0, maximum=360.0, allow_none=False),
    ),
    description="One inclusive mean-elongation interval; start greater than end wraps through zero.",
)


def _parse_utc(value: object, *, name: str) -> Epoch:
    try:
        return Epoch.from_isot(str(value), scale=TimeScale.UTC)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid UTC ISO date or timestamp.") from exc


def _validate_config(config: dict, path_name: str) -> dict:
    start = _parse_utc(config["startTime"], name=f"{path_name}.startTime")
    end = _parse_utc(config["endTime"], name=f"{path_name}.endTime")
    if start.seconds_until(end) < 0.0:
        raise ValueError(f"{path_name}.endTime must not precede startTime.")
    return config


def _utc_grid(start: Epoch, end: Epoch, step_seconds: float) -> tuple[Iterator[Epoch], int]:
    duration = start.seconds_until(end)
    tolerance_s = max(1.0e-9, step_seconds * 1.0e-12)
    count = int(math.floor((duration + tolerance_s) / step_seconds)) + 1
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
        "stepSeconds",
        default=60.0,
        minimum=0.0,
        minimum_exclusive=True,
        allow_none=False,
        ui=UiHints(group="Time grid", unit="s"),
    ),
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
    runtime = build_observation_runtime(context, config)
    station_key = resolve_catalog_key(config["stationName"], runtime.station_catalog, "Station")
    reflector_key = resolve_catalog_key(config["reflectorName"], runtime.reflector_catalog, "Reflector")
    criteria = PredictionCriteria(
        minimum_elevation_deg=float(config["minElevationDeg"]),
        minimum_reflector_elevation_deg=float(config["minReflectorElevationDeg"]),
        maximum_sun_elevation_deg=float(config["maxSunElevationDeg"]),
        allowed_elongation_ranges_deg=tuple(
            (float(item["startDeg"]), float(item["endDeg"]))
            for item in config["allowedElongationRangesDeg"]
        ),
    )
    meteorology = PredictionMeteorology(
        pressure_hpa=float(config["pressureHpa"]),
        temperature_k=float(config["temperatureK"]),
        relative_humidity_percent=float(config["relativeHumidityPercent"]),
        wavelength_nm=float(config["wavelengthNm"]),
    )
    predictor = LlrObservationPredictor(
        runtime.frames,
        runtime.light_time_solver,
        runtime.station_catalog[station_key],
        runtime.reflector_catalog[reflector_key],
        station_key=station_key,
        reflector_key=reflector_key,
        criteria=criteria,
        meteorology=meteorology,
    )

    start = _parse_utc(config["startTime"], name="startTime")
    end = _parse_utc(config["endTime"], name="endTime")
    step_seconds = float(config["stepSeconds"])
    epochs, count = _utc_grid(start, end, step_seconds)
    if config["showProgress"]:
        epochs = iter(_tqdm(epochs, total=count, desc="LLR prediction", unit="epoch"))
    rows = [predictor.evaluate(epoch) for epoch in epochs]
    windows = build_visibility_windows(rows, step_seconds=step_seconds)

    prediction_path = write_prediction_results(
        rows,
        context.resolve_path(config["outputFilePrediction"]),
    )
    windows_path = write_prediction_windows(
        windows,
        context.resolve_path(config["outputFileWindows"]),
    )
    print(
        f"[LlrObservationPrediction] {len(rows)} epoch(s), {len(windows)} window(s) "
        f"-> {prediction_path}, {windows_path}"
    )
    return {"rows": rows, "windows": windows}


__all__ = ["llr_observation_prediction"]
