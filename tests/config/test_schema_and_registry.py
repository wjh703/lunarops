"""Configuration schemas, the class registry, run contexts and program declarations."""

import re
from typing import Any, cast
from uuid import uuid4

import pytest

from lunarops import cli
from lunarops.classes.builtin_registry import BUILTINS
from lunarops.config.class_configs import available, global_config_schema, validate_global_class_configs
from lunarops.config.context import ClassCache, RunContext
from lunarops.config.registry import create, register_factory, resolve
from lunarops.config.schema import ConfigSchema, class_config, class_list, integer, sequence, string
from lunarops.fileio.archive import atomic_text_writer
from lunarops.programs.registry import (
    ArtifactSlot,
    available_programs,
    ensure_builtin_programs,
    get_program,
    program,
    program_spec,
    program_specs,
    resolve_program_config,
    run_program,
    validate_program_artifacts,
)


def _category(prefix):
    return f"{prefix}_{uuid4().hex}"


def test_registered_models_resolve_defaults_and_share_one_instance_per_declaration():
    category = _category("contract")
    register_factory(
        category, "Model", lambda config, context: config, schema=ConfigSchema(fields=(integer("level", default=1),))
    )
    config = resolve(category, {"type": "Model"})
    context = RunContext()

    assert config == {"type": "Model", "level": 1}
    assert available(category) == {category: ["Model"]}
    assert create(category, config) == config
    assert context.create_class(category, config) is context.create_class(category, dict(config))
    with pytest.raises(TypeError, match="explicit type: none"):
        resolve(category, None)


def test_factories_without_a_schema_accept_no_extra_keys_and_later_declarations_win():
    category = _category("strict")
    register_factory(category, "model", lambda config, context: "original")
    register_factory(category, "model", lambda config, context: "replacement")

    assert create(category, resolve(category, {"type": "model"})) == "replacement"
    with pytest.raises(ValueError, match="unknown configuration key"):
        resolve(category, {"type": "model", "legacy": True})


def test_schemas_keep_exact_choices_copy_sequences_and_run_validators_once_on_defaults():
    seen = []

    def validate(config, path):
        seen.append(dict(config))
        return {**config, "mode": "Fast"}

    schema = ConfigSchema(
        fields=(
            string("mode", default="Safe", choices=("Fast", "Safe")),
            sequence("labels", item_kind="string", default=[]),
        ),
        validator=validate,
    )
    source = {"labels": ["input"]}
    resolved = schema.resolve(source)
    resolved["labels"].append("output")

    assert resolved["mode"] == "Fast" and seen == [{"mode": "Safe", "labels": ["input"]}]
    assert source == {"labels": ["input"]}
    with pytest.raises(ValueError):
        schema.resolve({"mode": "fast"})


def test_nested_class_declarations_are_resolved_once_and_reach_the_factory_with_defaults():
    calls = []
    category = _category("nested")
    register_factory(
        category,
        "model",
        lambda config, context: config,
        schema=ConfigSchema(
            fields=(integer("value", default=7),), validator=lambda config, path: calls.append(path) or config
        ),
    )
    config = ConfigSchema(fields=(class_config("model", category),)).resolve({"model": {"type": "model"}})

    assert RunContext().create_class(category, config["model"]) == {"type": "model", "value": 7}
    assert len(calls) == 1


def test_class_lists_and_required_artifacts_are_strict():
    spec = program_spec(
        name="StrictProgram", summary="A strict test contract.", outputs=(ArtifactSlot("output", "TextFile"),)
    )
    with pytest.raises(ValueError, match="must not be null"):
        spec.schema.resolve({"output": None})
    with pytest.raises(TypeError, match="must be a sequence"):
        ConfigSchema(fields=(class_list("models", "test"),)).resolve({"models": "single"})
    with pytest.raises(ValueError, match="must not be empty"):
        ConfigSchema(fields=(class_list("models", "test", non_empty=True),)).resolve({"models": []})


def test_shared_and_observation_model_defaults_are_validated_by_scope():
    context = RunContext(
        observation_model_configs={"troposphere": {"type": "mendesPavlis"}, "stationDisplacement": [{"type": "none"}]}
    )
    context.resolve_defaults()

    assert context.class_config("rangeBias", {}) is None
    assert context.observation_model_configs["stationDisplacement"] == [{"type": "none"}]
    with pytest.raises(ValueError, match="unknown configuration key"):
        RunContext(observation_model_configs={"troposphere": {"type": "none", "extra": 1}}).resolve_defaults()
    with pytest.raises(ValueError, match="unknown configuration key"):
        validate_global_class_configs({"parametrization": {"type": "reflectorPosition"}})


def test_cached_resources_belong_to_the_cache_and_transient_ones_to_the_context():
    category = _category("cache")

    class Resource:
        closed = 0

        def close(self):
            self.closed += 1

    register_factory(category, "resource", lambda config, context: Resource())
    config = resolve(category, {"type": "resource"})
    cache = ClassCache()
    context = RunContext(class_cache=cache)
    shared = context.create_class(category, config)
    transient = context.create_class(category, config, cache=False)
    context.close()

    assert (shared.closed, transient.closed) == (0, 1)
    cache.close()
    assert shared.closed == 1 and cache == {}


def test_cache_close_runs_every_callback_after_a_failure():
    closed = []
    cache = ClassCache()
    cache.resources.callback(closed.append, "first")
    cache.resources.callback(lambda: (_ for _ in ()).throw(RuntimeError("close failure")))
    cache.resources.callback(closed.append, "last")
    with pytest.raises(RuntimeError, match="close failure"):
        cache.close()
    assert closed == ["last", "first"]


def test_every_registered_field_name_is_lower_camel_case():
    from lunarops.config.registry import _REGISTRY, _declaration  # enumerating every declaration has no public API

    ensure_builtin_programs()
    for category, typ in BUILTINS:
        _declaration(category, typ)
    schemas = [global_config_schema(), *(spec.schema for spec in program_specs())]
    schemas.extend(entry.schema for entries in _REGISTRY.values() for entry in entries.values())
    seen = set()
    while schemas:
        schema = schemas.pop()
        if id(schema) in seen:
            continue
        seen.add(id(schema))
        schemas.extend(schema.variants)
        for field in schema.fields:
            assert re.fullmatch(r"[a-z][A-Za-z0-9]*", field.name), field.name
            schemas.extend(child for child in (field.nested, field.item_nested) if child is not None)
    assert len(seen) > 50


def test_programs_are_registered_under_their_exact_names():
    @program(
        program_spec(
            "TestCanonicalProgram", "Test a small registered callable.", fields=(integer("value", required=True),)
        )
    )
    def first(config, context):
        return 1

    @program(program_spec("TestCanonicalProgram", "Replacement.", fields=(integer("value", required=True),)))
    def second(config, context):
        return config["value"]

    assert get_program("TestCanonicalProgram")[1] is second
    assert run_program("TestCanonicalProgram", {"value": 3}, cast(Any, None)) == 3
    with pytest.raises(KeyError, match="Unknown program"):
        run_program("testcanonicalprogram", {"value": 3}, cast(Any, None))
    with pytest.raises(ValueError, match="unknown configuration"):
        resolve_program_config("TestCanonicalProgram", {"value": 3, "legacy": True})


def test_builtin_programs_are_discoverable():
    ensure_builtin_programs()
    assert {
        "LlrResiduals",
        "NormalPointsConvert",
        "LlrProcessing",
        "LlrObservationPrediction",
        "LlrObservationPredictionMerge",
        "LunarOrbitPropagation",
        "MassCatalogCreate",
        "StationCatalogCreate",
        "ReflectorCatalogCreate",
    } <= set(available_programs())


def test_artifact_validation_checks_input_types_and_required_catalogs(tmp_path):
    ensure_builtin_programs()
    with atomic_text_writer(tmp_path / "wrong.txt", "stationCatalog") as stream:
        stream.write("frame ITRF\nrecordCount 0\ndata\n")
    config = {
        "inputFilesNormalPoints": ["wrong.txt"],
        "outputFileObservationResults": "results.txt",
        "inputFileStationCatalog": "stations.txt",
        "inputFileReflectorCatalog": "reflectors.txt",
    }

    with pytest.raises(ValueError, match="expects 'normalPoint'"):
        validate_program_artifacts("LlrResiduals", config, RunContext(working_dir=tmp_path))
    with pytest.raises(ValueError, match="inputFileStationCatalog"):
        resolve_program_config(
            "LlrResiduals", {"inputFilesNormalPoints": ["normal.txt"], "outputFileObservationResults": "results.txt"}
        )


def test_validate_command_knows_outputs_produced_earlier_in_the_run(tmp_path, capsys):
    (tmp_path / "source.crd").write_text("external source placeholder\n", encoding="ascii")
    (tmp_path / "scenario.yml").write_text(
        """
programs:
  - program: NormalPointsConvert
    inputFilesNormalPoints: [source.crd]
    crdUncertaintyPolicy: binRmsAsSigma
    outputFileNormalPoints: normalPoints.txt.gz
    outputFileImportReport: importReport.txt.gz
  - program: StationCatalogCreate
    outputFileStationCatalog: stations.txt
    stationCoordinates: [{key: S, xyzM: [1.0, 2.0, 3.0]}]
  - program: ReflectorCatalogCreate
    outputFileReflectorCatalog: reflectors.txt
    reflectorCoordinates: [{key: R, xyzM: [4.0, 5.0, 6.0]}]
  - program: LlrResiduals
    inputFilesNormalPoints: [normalPoints.txt.gz]
    outputFileObservationResults: residuals.txt.gz
    inputFileStationCatalog: stations.txt
    inputFileReflectorCatalog: reflectors.txt
""",
        encoding="utf-8",
    )

    assert cli.main(["validate", str(tmp_path / "scenario.yml"), "--working-dir", str(tmp_path)]) == 0
    assert "valid: 4 program call(s)" in capsys.readouterr().out
