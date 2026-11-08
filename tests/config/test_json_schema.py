"""The JSON Schema used by editors: in step with the registries and accepting every production config."""

import json

import pytest

from lunarops.config.json_schema import render_run_config_json_schema, run_config_json_schema
from lunarops.config.loader import load_config_file
from lunarops.programs.registry import ensure_builtin_programs, program_specs
from tests.conftest import CONFIGS_DIR

CONFIGS = sorted(CONFIGS_DIR.glob("*.yml"))


def _schema():
    ensure_builtin_programs()
    return run_config_json_schema(program_specs())


def _validator():
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft7Validator(_schema())


def _as_json(document):
    return json.loads(json.dumps(document, default=str))


def test_the_committed_schema_is_current():
    ensure_builtin_programs()
    assert (CONFIGS_DIR / "lunarops.schema.json").read_text(encoding="utf-8") == render_run_config_json_schema(
        program_specs()
    )


def test_every_program_is_described_with_its_control_keys():
    variants = _schema()["properties"]["programs"]["items"]["oneOf"]

    assert {variant["properties"]["program"]["const"] for variant in variants} == {s.name for s in program_specs()}
    assert all({"loop", "when"} <= set(variant["properties"]) for variant in variants)
    assert all("program" in variant["required"] for variant in variants)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_production_configurations_match_the_schema(path):
    validator = _validator()
    errors = [f"{list(e.path)}: {e.message[:200]}" for e in validator.iter_errors(_as_json(load_config_file(path)))]
    assert not errors


def _reflector_program(**extra):
    return {
        "program": "ReflectorCatalogCreate",
        "outputFileReflectorCatalog": "out.txt",
        "reflectorCoordinates": [{"key": "A", "xyzM": [1.0, 2.0, 3.0]}],
        **extra,
    }


def test_the_schema_catches_misspelled_keys_and_accepts_placeholders():
    validator = _validator()

    assert validator.is_valid({"programs": [_reflector_program()], "models": {"troposphere": {"type": "none"}}})
    assert validator.is_valid({"programs": [_reflector_program(reflectorCoordinates="{table}")]})
    assert not validator.is_valid({"programs": [_reflector_program(typo=1)]})
    assert not validator.is_valid({"models": {"troposphere": {"type": "notAModel"}}})
    assert not validator.is_valid({"shared": {}})
