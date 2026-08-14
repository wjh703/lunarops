import pytest
from typing import Any, cast

import lunarops.cli as cli
from lunarops.config.context import RunContext
from lunarops.config.schema import integer
from lunarops.fileio.archive import atomic_text_writer
from lunarops.programs.registry import (
    ProgramSpec,
    available_programs,
    program,
    run_program,
    validate_program_artifacts,
    validate_program_config,
)


def test_program_registry_is_case_insensitive():
    @program(
        ProgramSpec(
            "TestCanonicalProgram",
            "Test a small registered callable.",
            fields=(integer("value", required=True, allow_none=False),),
        )
    )
    def canonical(config, context):
        return config["value"]

    assert "TestCanonicalProgram" in available_programs()
    assert run_program("testcanonicalprogram", {"value": 3}, cast(Any, None)) == 3
    with pytest.raises(ValueError, match="unknown configuration"):
        validate_program_config("TestCanonicalProgram", {"value": 3, "legacy": True})


def test_program_discovery_registers_every_configurable_program():
    cli._import_programs()

    assert {
        "LlrResiduals",
        "NormalPointsConvert",
        "LlrProcessing",
    } <= set(available_programs())
    assert {
        "CatalogCreate",
        "CrdToMini",
        "LlrAdjustment",
        "LlrApplySolution",
        "LlrNormalEquations",
        "MatrixConvert",
        "NormalPointsToLunarOps",
        "NormalsCombineSolve",
        "NormalPointsStatistics",
    }.isdisjoint(available_programs())


def test_program_artifact_validation_rejects_wrong_type_header(tmp_path):
    cli._import_programs()
    wrong = tmp_path / "wrong.txt"
    with atomic_text_writer(wrong, "stationCatalog", version=1) as stream:
        stream.write("frame ITRF\nrecordCount 0\ndata\n")

    with pytest.raises(ValueError, match="expects 'normalPoint'"):
        validate_program_artifacts(
            "LlrResiduals",
            {
                "inputFilesNormalPoints": ["wrong.txt"],
                "outputFileObservationResults": "results.txt",
            },
            RunContext(working_dir=tmp_path),
        )


def test_validate_command_understands_outputs_produced_earlier_in_graph(tmp_path, capsys):
    source = tmp_path / "source.crd"
    source.write_text("external source placeholder\n", encoding="ascii")
    config = tmp_path / "scenario.yml"
    config.write_text(
        """
programs:
  - program: NormalPointsConvert
    inputFilesNormalPoints: [source.crd]
    outputFileNormalPoints: normalPoints.txt.gz
    outputFileImportReport: importReport.txt.gz
  - program: LlrResiduals
    inputFilesNormalPoints: [normalPoints.txt.gz]
    outputFileObservationResults: residuals.txt.gz
""".strip(),
        encoding="utf-8",
    )

    assert cli.main(["validate", str(config), "--working-dir", str(tmp_path)]) == 0
    assert "valid: 2 program call(s)" in capsys.readouterr().out
