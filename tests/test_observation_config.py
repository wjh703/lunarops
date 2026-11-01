import pytest

from lunarops import cli
from lunarops.config.registry import validate_global_class_configs
from lunarops.programs.registry import resolve_program_config


def _register_observation_contracts():
    cli._import_programs()


@pytest.mark.parametrize("key", ["uncertainty", "uncertaintyModel"])
def test_observation_program_schema_rejects_undeclared_options(key):
    _register_observation_contracts()
    with pytest.raises(ValueError, match="unknown configuration key"):
        resolve_program_config("LlrResiduals", {key: "obsolete"})


def test_global_schema_rejects_undeclared_options():
    _register_observation_contracts()
    with pytest.raises(ValueError, match="unknown configuration key"):
        validate_global_class_configs({"uncertaintyModel": "obsolete"})


def test_observation_prediction_schema_resolves_defaults_and_rejects_unknown_keys():
    _register_observation_contracts()
    config = {
        "inputFileStationCatalog": "stations.txt",
        "inputFileReflectorCatalog": "reflectors.txt",
        "outputFilePrediction": "prediction.txt",
        "meteorologyPolicy": "nominal",
        "outputFileWindows": "windows.txt",
        "startTime": "2025-01-01T00:00:00",
        "endTime": "2025-01-01T00:02:00",
        "stationName": "station",
        "reflectorName": "reflector",
        "ephemerides": {
            "type": "calceph",
            "directory": "kernels",
            "lunarRelativisticScaleConvention": "alreadyScaled",
        },
        "earthRotation": {"type": "file", "file": "eop.txt"},
        "troposphere": {"type": "none"},
        "relativity": {"type": "none"},
        "stationDisplacement": [{"type": "none"}],
        "reflectorDisplacement": {"type": "none"},
    }
    from lunarops.programs.registry import resolve_program_config

    resolved = resolve_program_config("LlrObservationPrediction", config)
    assert resolved["stepSeconds"] == 60.0
    assert resolved["utcOffsetHours"] == 0.0
    assert resolved["minReflectorElevationDeg"] == 0.0
    assert resolved["maxSunElevationDeg"] == -6.0
    assert resolved["allowedElongationRangesDeg"] == [{"startDeg": 0.0, "endDeg": 360.0}]
    with pytest.raises(ValueError, match="unknown configuration key"):
        resolve_program_config("LlrObservationPrediction", {**config, "cpfFile": "x"})
    with pytest.raises(TypeError, match="integer"):
        resolve_program_config("LlrObservationPrediction", {**config, "mpi": {"chunksize": 1.5}})


def test_observation_time_configuration_applies_fixed_offset():
    _register_observation_contracts()
    config = {
        "inputFileStationCatalog": "stations.txt",
        "inputFileReflectorCatalog": "reflectors.txt",
        "outputFilePrediction": "prediction.txt",
        "meteorologyPolicy": "nominal",
        "outputFileWindows": "windows.txt",
        "startTime": "2025-01-01T08:00:00",
        "endTime": "2025-01-01T08:02:00",
        "utcOffsetHours": 8.0,
        "stationName": "station",
        "reflectorName": "reflector",
        "ephemerides": {
            "type": "calceph",
            "directory": "kernels",
            "lunarRelativisticScaleConvention": "alreadyScaled",
        },
        "earthRotation": {"type": "file", "file": "eop.txt"},
        "troposphere": {"type": "none"},
        "relativity": {"type": "none"},
        "stationDisplacement": [{"type": "none"}],
        "reflectorDisplacement": {"type": "none"},
    }
    resolved = resolve_program_config("LlrObservationPrediction", config)
    assert resolved["utcOffsetHours"] == 8.0
    assert resolved["startTime"] == config["startTime"]
    from lunarops.programs.specs import observation_time_window

    start, end = observation_time_window(resolved)
    assert start.isot(precision=0) == "2025-01-01T00:00:00"
    assert end.isot(precision=0) == "2025-01-01T00:02:00"


def test_observation_prediction_schema_rejects_reversed_time_range():
    _register_observation_contracts()
    from lunarops.programs.registry import resolve_program_config

    config = {
        "inputFileStationCatalog": "stations.txt",
        "inputFileReflectorCatalog": "reflectors.txt",
        "outputFilePrediction": "prediction.txt",
        "meteorologyPolicy": "nominal",
        "outputFileWindows": "windows.txt",
        "startTime": "2025-01-02T00:00:00",
        "endTime": "2025-01-01T00:00:00",
        "stationName": "station",
        "reflectorName": "reflector",
    }
    with pytest.raises(ValueError, match="endTime must not precede"):
        resolve_program_config("LlrObservationPrediction", config)


def test_prediction_time_grid_requires_millisecond_representable_samples():
    _register_observation_contracts()
    config = {
        "inputFileStationCatalog": "stations.txt",
        "inputFileReflectorCatalog": "reflectors.txt",
        "outputFilePrediction": "prediction.txt",
        "meteorologyPolicy": "nominal",
        "outputFileWindows": "windows.txt",
        "startTime": "2025-01-01T00:00:00.000",
        "endTime": "2025-01-01T00:00:01.000",
        "stationName": "station",
        "reflectorName": "reflector",
    }
    with pytest.raises(ValueError, match="whole number of milliseconds"):
        resolve_program_config("LlrObservationPrediction", {**config, "stepSeconds": 0.0015})
    with pytest.raises(ValueError, match="aligned to a whole millisecond"):
        resolve_program_config(
            "LlrObservationPrediction",
            {**config, "startTime": "2025-01-01T00:00:00.0005"},
        )


@pytest.mark.parametrize("offset", [-24.01, 24.01, 0.001, float("nan"), True])
def test_observation_schema_rejects_invalid_fixed_offsets(offset):
    from lunarops.config.schema import ConfigSchema
    from lunarops.programs.specs import observation_fields, validate_observation_time_config

    schema = ConfigSchema(observation_fields(), validator=validate_observation_time_config)
    with pytest.raises((TypeError, ValueError)):
        schema.resolve({"utcOffsetHours": offset})
