"""Checks made before a run starts: every program call, shared outputs and outputs of nested steps."""

from types import SimpleNamespace

import pytest
import yaml

from lunarops import cli
from lunarops.config.context import RunContext
from lunarops.programs.registry import ensure_builtin_programs, program_outputs, validate_program_calls


def _reflector_call(path, **extra):
    return {
        "program": "ReflectorCatalogCreate",
        "outputFileReflectorCatalog": str(path),
        "reflectorCoordinates": [{"key": "A", "xyzM": [1.0, 2.0, 3.0]}],
        **extra,
    }


def _body(call):
    return {key: value for key, value in call.items() if key != "program"}


def test_run_rejects_a_bad_later_call_before_any_program_runs(tmp_path):
    first_output = tmp_path / "first.txt"
    config = tmp_path / "run.yml"
    config.write_text(
        yaml.safe_dump({"programs": [_reflector_call(first_output), _reflector_call(tmp_path / "b.txt", typo=1)]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown configuration key"):
        cli.cmd_run(SimpleNamespace(mpi=False, workers=None, config=str(config), set=[], working_dir=str(tmp_path)))
    assert not first_output.exists()


def test_two_calls_may_not_write_the_same_file(tmp_path):
    ensure_builtin_programs()
    output = tmp_path / "same.txt"
    calls = [("ReflectorCatalogCreate", _body(_reflector_call(output)))] * 2

    with pytest.raises(ValueError, match="more than one artifact"):
        validate_program_calls(calls, RunContext(working_dir=tmp_path))


def test_files_written_by_processing_steps_count_as_outputs():
    ensure_builtin_programs()
    config = {
        "processingSteps": [
            {"type": "screenObservations"},
            {"type": "writeResiduals", "outputFile": "residuals.txt"},
            {"type": "writeNormalEquations", "outputFile": "normals"},
            {
                "type": "writeResults",
                "outputFileStationCatalog": "stations.txt",
                "outputFileCovariance": "covariance",
                "outputFileReport": None,
            },
        ]
    }

    assert program_outputs("LlrProcessing", config) == [
        ("residuals.txt", "observationResult"),
        ("normals", "normalEquations"),
        ("covariance", "covariance"),
        ("stations.txt", "stationCatalog"),
    ]


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"type": "writeResiduals", "outputFile": "residuals.csv"}, r"processingSteps\[2\]\.outputFile must use \.txt"),
        ({"type": "writeNormalEquations", "outputFile": "normals.txt"}, "must be an extensionless directory"),
        ({"type": "writeResults", "outputFileCovariance": "covariance.dat"}, "must be an extensionless directory"),
        ({"type": "writeResults", "outputFileReport": "report.yml"}, r"outputFileReport must use \.txt"),
    ],
)
def test_processing_step_outputs_are_checked_before_the_run(tmp_path, step, message):
    ensure_builtin_programs()
    config = {
        "inputFilesNormalPoints": ["normal.txt"],
        "inputFileStationCatalog": "stations.txt",
        "inputFileReflectorCatalog": "reflectors.txt",
        "parametrization": [{"type": "stationRangeBias"}],
        "varianceComponents": [{"id": "A", "station": "APOLLO", "start": "2020-01-01", "endExclusive": None}],
        "processingSteps": [{"type": "screenObservations"}, {"type": "estimate", "name": "bias"}, step],
    }

    with pytest.raises(ValueError, match=message):
        validate_program_calls([("LlrProcessing", config)], RunContext(working_dir=tmp_path), require_inputs=False)


@pytest.mark.parametrize("name", ["LlrResiduals", "LlrProcessing", "LlrObservationPrediction"])
def test_the_former_mpi_option_is_rejected_with_its_new_name(name):
    from lunarops.programs.registry import get_program

    ensure_builtin_programs()
    with pytest.raises(ValueError, match=r"config\.mpi was removed: renamed to `tasks`"):
        get_program(name)[0].schema.resolve({"mpi": {"chunksize": 4}})
