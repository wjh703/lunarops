from typing import Any, cast

import pytest

import lunarops.cli as cli
from lunarops.config.context import RunContext
from lunarops.config.schema import integer
from lunarops.fileio.archive import atomic_text_writer
from lunarops.programs.registry import (
    ProgramSpec,
    available_programs,
    program,
    resolve_program_config,
    run_program,
    validate_program_artifacts,
)


def test_program_registry_requires_exact_spelling():
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
    assert run_program("TestCanonicalProgram", {"value": 3}, cast(Any, None)) == 3
    with pytest.raises(KeyError, match="Unknown program"):
        run_program("testcanonicalprogram", {"value": 3}, cast(Any, None))
    with pytest.raises(ValueError, match="unknown configuration"):
        resolve_program_config("TestCanonicalProgram", {"value": 3, "legacy": True})


def test_program_discovery_registers_every_configurable_program():
    cli._import_programs()

    assert {
        "LlrResiduals",
        "NormalPointsConvert",
        "LlrProcessing",
        "LlrObservationPredictionMerge",
        "MassCatalogCreate",
        "StationCatalogCreate",
        "ReflectorCatalogCreate",
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
                "inputFileStationCatalog": "stations.txt",
                "inputFileReflectorCatalog": "reflectors.txt",
            },
            RunContext(working_dir=tmp_path),
        )


def test_observation_programs_require_native_catalog_files():
    with pytest.raises(ValueError, match="inputFileStationCatalog"):
        resolve_program_config(
            "LlrResiduals",
            {
                "inputFilesNormalPoints": ["normal.txt"],
                "outputFileObservationResults": "results.txt",
            },
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
        crdUncertaintyPolicy: binRmsAsSigma
        outputFileNormalPoints: normalPoints.txt.gz
        outputFileImportReport: importReport.txt.gz
      - program: StationCatalogCreate
        outputFileStationCatalog: stations.txt
        stationCoordinates:
          - {key: S, xyzM: [1.0, 2.0, 3.0]}
      - program: ReflectorCatalogCreate
        outputFileReflectorCatalog: reflectors.txt
        reflectorCoordinates:
          - {key: R, xyzM: [4.0, 5.0, 6.0]}
      - program: LlrResiduals
        inputFilesNormalPoints: [normalPoints.txt.gz]
        outputFileObservationResults: residuals.txt.gz
        inputFileStationCatalog: stations.txt
        inputFileReflectorCatalog: reflectors.txt
""".strip(),
        encoding="utf-8",
    )

    assert cli.main(["validate", str(config), "--working-dir", str(tmp_path)]) == 0
    assert "valid: 4 program call(s)" in capsys.readouterr().out
