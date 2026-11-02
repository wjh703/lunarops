import json
from pathlib import Path
from uuid import uuid4

import pytest

from lunarops.config.class_configs import available, validate_global_class_configs
from lunarops.config.context import ClassCache, RunContext
from lunarops.config.loader import build_run_plan, parse_set_overrides, resolve_variables
from lunarops.config.registry import create, register_factory, resolve
from lunarops.config.schema import ConfigSchema, class_list, integer, sequence, string
from lunarops.programs.registry import ArtifactSlot, program_spec


def test_registry_schema_gets_a_canonical_type_and_defaults():
    category = f"contract_{uuid4().hex}"

    def factory(config, context):
        return config

    register_factory(
        category,
        "Model",
        factory,
        schema=ConfigSchema(
            fields=(
                integer("level", default=1),
            )
        ),
    )

    config = resolve(category, {"type": "Model"})
    assert config == {"type": "Model", "level": 1}
    assert available(category) == {category: ["Model"]}
    assert create(category, resolve(category, {"type": "Model"})) == config
    with pytest.raises(TypeError, match="explicit type: none"):
        resolve(category, None)

    context = RunContext()
    assert context.create_class(category, config) is context.create_class(category, dict(config))


def test_schema_preserves_exact_type_choices_and_copies_sequences():
    schema = ConfigSchema(
        fields=(
            string("mode", choices=("Fast", "Safe")),
            sequence("labels", item_kind="string"),
        ),
        type_name="Model",
    )
    source = {"type": "Model", "mode": "Fast", "labels": ["input"]}
    resolved = schema.resolve(source)

    assert resolved == {"type": "Model", "mode": "Fast", "labels": ["input"]}
    resolved["labels"].append("output")
    assert source == {"type": "Model", "mode": "Fast", "labels": ["input"]}
    assert source["labels"] == ["input"]


def test_schema_validator_receives_defaults_once():
    seen = []

    def validate(config, path):
        seen.append(dict(config))
        config["mode"] = "fast"
        return config

    schema = ConfigSchema(
        fields=(
            string("mode", default="safe"),
        ),
        validator=validate,
    )

    assert schema.resolve({}) == {"mode": "fast"}
    assert seen == [{"mode": "safe"}]


def test_nested_model_is_resolved_once_and_factory_receives_defaults():
    from lunarops.config.schema import class_config

    calls = []
    category = f"single_pass_{uuid4().hex}"

    def validate(config, path):
        calls.append(path)
        return config

    register_factory(
        category,
        "model",
        lambda config, context: config,
        schema=ConfigSchema(fields=(integer("value", default=7),), validator=validate),
    )
    config = ConfigSchema(fields=(class_config("model", category),)).resolve({"model": {"type": "model"}})
    context = RunContext()
    assert context.create_class(category, config["model"]) == {"type": "model", "value": 7}
    assert len(calls) == 1


def test_observation_defaults_apply_model_schema_before_assembly():

    context = RunContext(
        observation_model_configs={"troposphere": {"type": "mendesPavlis"}, "stationDisplacement": [{"type": "none"}]}
    )
    context.resolve_defaults()
    assert context.class_config("rangeBias", {}) is None
    assert context.observation_model_configs["stationDisplacement"] == [{"type": "none"}]
    with pytest.raises(ValueError, match="unknown configuration key"):
        RunContext(observation_model_configs={"troposphere": {"type": "none", "extra": 1}}).resolve_defaults()


def test_program_declarations_replace_previous_implementation():
    from lunarops.programs.registry import get_program, program

    name = f"replacement_{uuid4().hex}"

    @program(program_spec(name=name, summary="first"))
    def first(config, context):
        return 1

    @program(program_spec(name=name, summary="second"))
    def second(config, context):
        return 2

    assert get_program(name)[1] is second


def test_factory_without_schema_is_strict():
    category = f"strict_{uuid4().hex}"

    register_factory(category, "model", lambda config, context: config)
    with pytest.raises(ValueError, match="unknown configuration key"):
        resolve(category, {"type": "model", "legacy": True})


def test_global_scope_is_explicit():

    with pytest.raises(ValueError, match="unknown configuration key"):
        validate_global_class_configs({"parametrization": {"type": "reflectorPosition"}})


def test_shared_and_observation_model_sections_compile_separately():
    from lunarops.config.loader import build_run_plan

    plan = build_run_plan(
        {
            "shared": {"ephemerides": "calceph"},
            "observationModel": {"troposphere": {"type": "none"}},
            "programs": [],
        }
    )
    assert plan.shared == {"ephemerides": "calceph"}
    assert plan.observation_model == {"troposphere": {"type": "none"}}


def test_required_artifact_and_class_lists_are_strict():
    spec = program_spec(
        name="StrictProgram",
        summary="A strict test contract.",
        outputs=(ArtifactSlot("output", "TextFile"),),
    )
    with pytest.raises(ValueError, match="must not be null"):
        spec.schema.resolve({"output": None})

    schema = ConfigSchema(fields=(class_list("models", "test"),))
    with pytest.raises(TypeError, match="must be a sequence"):
        schema.resolve({"models": "single"})


def test_schema_honors_non_empty_class_lists():
    schema = ConfigSchema(fields=(class_list("models", "test", non_empty=True),))
    with pytest.raises(ValueError, match="must not be empty"):
        schema.resolve({"models": []})


def test_run_plan_resolves_shared_defaults_and_expands_conditions():
    plan = build_run_plan(
        {
            "variables": {"sites": ["A", "B"]},
            "shared": {"sites": "{sites}"},
            "programs": [
                {
                    "program": "P",
                    "loop": {"variable": "site", "values": "{sites}"},
                    "when": {"equals": ["{site}", "B"]},
                    "output": "{site}.txt",
                }
            ],
        }
    )

    assert plan.shared == {"sites": ["A", "B"]}
    assert plan.calls == (("P", {"output": "B.txt"}),)


def test_run_plan_leaves_program_fields_to_its_schema():
    plan = build_run_plan({"programs": [{"program": "P", "enabled": False}]})
    assert plan.calls == (("P", {"enabled": False}),)


def test_variable_cycles_and_cli_scalar_parsing_are_explicit():
    with pytest.raises(ValueError, match="cycle detected"):
        resolve_variables({"first": "{second}", "second": "{first}"})
    assert parse_set_overrides(["leading=001", "scientific=1e3"]) == {
        "leading": 1,
        "scientific": "1e3",
    }


def test_shared_run_context_cache_is_closed_by_its_owner():
    category = f"cache_{uuid4().hex}"

    class Resource:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    register_factory(category, "resource", lambda config, context: Resource())
    cache = ClassCache()
    worker_context = RunContext(class_cache=cache)
    config = resolve(category, {"type": "resource"})
    resource = worker_context.create_class(category, config)
    worker_context.close()
    assert resource.closed == 0
    assert cache

    cache.close()
    assert resource.closed == 1
    assert cache == {}

    transient_context = RunContext()
    transient = transient_context.create_class(category, config, cache=False)
    transient_context.close()
    assert transient.closed == 1


def test_registered_schema_field_names_use_lower_camel_case():
    import re

    from lunarops.classes.builtin_registry import BUILTINS
    from lunarops.config.class_configs import global_config_schema
    from lunarops.config.registry import _REGISTRY, _declaration
    from lunarops.programs.registry import ensure_builtin_programs, program_specs

    ensure_builtin_programs()
    for category, typ in BUILTINS:
        _declaration(category, typ)
    schemas = [global_config_schema(), *(spec.schema for spec in program_specs())]
    schemas.extend(entry.schema for category in _REGISTRY.values() for entry in category.values())
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


@pytest.mark.parametrize("config_path", sorted((*Path("configs").glob("*.yml"), *Path("configs").glob("*.yaml"))),
                         ids=lambda path: path.name)
def test_production_resolved_config_leaf_paths_match_269ea29(config_path):
    from lunarops.config.loader import load_config_file
    from lunarops.programs.registry import ensure_builtin_programs, resolve_program_config

    def leaves(value, prefix=""):
        if isinstance(value, dict) and value:
            return [path for key, child in value.items()
                    for path in leaves(child, f"{prefix}.{key}" if prefix else key)]
        if isinstance(value, list) and value:
            return [path for index, child in enumerate(value) for path in leaves(child, f"{prefix}[{index}]")]
        return [prefix]

    snapshot = json.loads(Path("tests/data/resolved_config_keys.json").read_text())
    assert snapshot["sourceCommit"] == "269ea29"
    ensure_builtin_programs()
    plan = build_run_plan(load_config_file(config_path))
    context = RunContext(shared_class_configs=plan.shared, observation_model_configs=plan.observation_model)
    context.resolve_defaults()
    actual = [{"program": name, "paths": sorted(leaves(resolve_program_config(name, body)))}
              for name, body in plan.calls]
    assert actual == snapshot["configs"][config_path.name]
