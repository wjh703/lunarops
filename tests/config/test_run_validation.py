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
        cli.cmd_run(SimpleNamespace(mpi=False, config=str(config), set=[], working_dir=str(tmp_path)))
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
            {"type": "writeResults", "outputFileStationCatalog": "stations.txt", "outputFileReport": None},
        ]
    }

    assert program_outputs("LlrProcessing", config) == [
        ("residuals.txt", "observationResult"),
        ("stations.txt", "stationCatalog"),
    ]
