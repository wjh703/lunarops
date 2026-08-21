import pytest

import lunarops.cli as cli
from lunarops.classes.observation_factory import ensure_registered
from lunarops.config.registry import validate_global_class_configs
from lunarops.programs.registry import validate_program_config


def _register_observation_contracts():
    cli._import_programs()
    ensure_registered()


@pytest.mark.parametrize("key", ["uncertainty", "uncertaintyModel"])
def test_observation_program_schema_rejects_undeclared_options(key):
    _register_observation_contracts()
    with pytest.raises(ValueError, match="unknown configuration key"):
        validate_program_config("LlrResiduals", {key: "obsolete"})


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
        "troposphere": "none",
        "relativity": "none",
        "stationDisplacement": ["none"],
        "reflectorDisplacement": "none",
    }
    from lunarops.programs.registry import validate_program_config

    resolved = validate_program_config("LlrObservationPrediction", config)
    assert resolved["stepSeconds"] == 60.0
    assert resolved["minReflectorElevationDeg"] == 0.0
    assert resolved["maxSunElevationDeg"] == -6.0
    assert resolved["allowedElongationRangesDeg"] == [{"startDeg": 0.0, "endDeg": 360.0}]
    with pytest.raises(ValueError, match="unknown configuration key"):
        validate_program_config("LlrObservationPrediction", {**config, "cpfFile": "x"})


def test_observation_prediction_schema_rejects_reversed_time_range():
    _register_observation_contracts()
    from lunarops.programs.registry import validate_program_config

    config = {
        "inputFileStationCatalog": "stations.txt",
        "inputFileReflectorCatalog": "reflectors.txt",
        "outputFilePrediction": "prediction.txt",
        "outputFileWindows": "windows.txt",
        "startTime": "2025-01-02T00:00:00",
        "endTime": "2025-01-01T00:00:00",
        "stationName": "station",
        "reflectorName": "reflector",
    }
    with pytest.raises(ValueError, match="endTime must not precede"):
        validate_program_config("LlrObservationPrediction", config)
