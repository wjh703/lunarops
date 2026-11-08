"""Schemas of the observation programs and their fixed UTC offsets."""

import pytest

from lunarops.config.class_configs import validate_model_configs
from lunarops.config.schema import ConfigSchema
from lunarops.programs.registry import ensure_builtin_programs, resolve_program_config
from lunarops.programs.specs import observation_fields, observation_time_window, validate_observation_time_config

PREDICTION = {
    "inputFileStationCatalog": "stations.txt",
    "inputFileReflectorCatalog": "reflectors.txt",
    "outputFilePrediction": "prediction.txt",
    "outputFileWindows": "windows.txt",
    "meteorologyPolicy": "nominal",
    "startTime": "2025-01-01T00:00:00",
    "endTime": "2025-01-01T00:02:00",
    "stationName": "station",
    "reflectorName": "reflector",
    "ephemerides": {"type": "calceph", "directory": "kernels", "lunarRelativisticScaleConvention": "alreadyScaled"},
    "earthRotation": {"type": "file", "file": "eop.txt"},
    "troposphere": {"type": "none"},
    "relativity": {"type": "none"},
    "stationDisplacement": [{"type": "none"}],
    "reflectorDisplacement": {"type": "none"},
}


@pytest.fixture(autouse=True)
def _programs():
    ensure_builtin_programs()


def _prediction(**changes):
    return resolve_program_config("LlrObservationPrediction", {**PREDICTION, **changes})


@pytest.mark.parametrize("key", ["uncertainty", "uncertaintyModel"])
def test_undeclared_options_are_rejected(key):
    with pytest.raises(ValueError, match="unknown configuration key"):
        resolve_program_config("LlrResiduals", {key: "obsolete"})
    with pytest.raises(ValueError, match="unknown configuration key"):
        validate_model_configs({key: "obsolete"})
    with pytest.raises(ValueError, match="unknown configuration key"):
        _prediction(cpfFile="x")


def test_prediction_defaults_are_resolved():
    resolved = _prediction()

    assert (resolved["stepSeconds"], resolved["utcOffsetHours"]) == (60.0, 0.0)
    assert (resolved["minReflectorElevationDeg"], resolved["maxSunElevationDeg"]) == (0.0, -6.0)
    assert resolved["allowedElongationRangesDeg"] == [{"startDeg": 0.0, "endDeg": 360.0}]
    with pytest.raises(TypeError, match="integer"):
        _prediction(mpi={"chunksize": 1.5})


def test_fixed_utc_offset_converts_the_local_window_to_utc():
    resolved = _prediction(startTime="2025-01-01T08:00:00", endTime="2025-01-01T08:02:00", utcOffsetHours=8.0)
    start, end = observation_time_window(resolved)

    assert resolved["startTime"] == "2025-01-01T08:00:00"  # the resolved config keeps the user's text
    assert (start.isot(precision=0), end.isot(precision=0)) == ("2025-01-01T00:00:00", "2025-01-01T00:02:00")


def test_reversed_time_ranges_and_sub_millisecond_grids_are_rejected():
    with pytest.raises(ValueError, match="endTime must not precede"):
        _prediction(startTime="2025-01-02T00:00:00", endTime="2025-01-01T00:00:00")
    with pytest.raises(ValueError, match="whole number of milliseconds"):
        _prediction(endTime="2025-01-01T00:00:01.000", stepSeconds=0.0015)
    with pytest.raises(ValueError, match="aligned to a whole millisecond"):
        _prediction(startTime="2025-01-01T00:00:00.0005", endTime="2025-01-01T00:00:01.000")


@pytest.mark.parametrize("offset", [-24.01, 24.01, 0.001, float("nan"), True])
def test_invalid_fixed_offsets_are_rejected(offset):
    schema = ConfigSchema(observation_fields(), validator=validate_observation_time_config)
    with pytest.raises((TypeError, ValueError)):
        schema.resolve({"utcOffsetHours": offset})
