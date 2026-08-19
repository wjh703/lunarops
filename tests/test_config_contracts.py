import json
from uuid import uuid4

import pytest

from lunarops.config.context import RunContext
from lunarops.config.loader import build_run_plan, parse_set_overrides, resolve_variables, run_config_schema
from lunarops.config.registry import (
    available,
    class_json_schema,
    create,
    register_factory,
    registration_transaction,
    validate_class_config,
    validate_global_class_configs,
)
from lunarops.config.schema import ConfigSchema, class_list, integer, number, sequence, string
from lunarops.programs.registry import ArtifactSlot, ProgramSpec


def test_registry_schema_gets_a_canonical_type_and_defaults():
    category = f"contract_{uuid4().hex}"

    def factory(config, context):
        return config

    register_factory(
        category,
        "Model",
        factory,
        schema=ConfigSchema(fields=(integer("level", default=1, allow_none=False),)),
    )

    config = validate_class_config(category, {"type": "MODEL"})
    assert config == {"type": "Model", "level": 1}
    assert available(category) == ["Model"]
    assert create(category, "model") == config
    with pytest.raises(TypeError, match="explicit type: none"):
        validate_class_config(category, None)

    context = RunContext()
    assert context.create_class(category, "MODEL") is context.create_class(category, "model")


def test_schema_canonicalizes_type_choices_and_copies_sequences():
    schema = ConfigSchema(
        fields=(
            string("mode", choices=("Fast", "Safe"), allow_none=False),
            sequence("labels", item_kind="string"),
        ),
        type_name="Model",
    )
    source = {"type": "model", "mode": "fast", "labels": ["input"]}
    resolved = schema.resolve(source)

    assert resolved == {"type": "Model", "mode": "Fast", "labels": ["input"]}
    resolved["labels"].append("output")
    assert source == {"type": "model", "mode": "fast", "labels": ["input"]}
    assert source["labels"] == ["input"]


def test_schema_reapplies_defaults_after_custom_validation():
    schema = ConfigSchema(
        fields=(string("mode", default="safe", allow_none=False),),
        validator=lambda config, path: {},
    )

    assert schema.resolve({}) == {"mode": "safe"}


def test_program_registration_transaction_rolls_back_declarations():
    from lunarops.programs.registry import (
        ensure_builtin_programs,
        get_program,
        program,
        program_registration_transaction,
    )

    ensure_builtin_programs()
    name = f"TransientProgram_{uuid4().hex}"
    with pytest.raises(RuntimeError, match="abort"):
        with program_registration_transaction():

            @program(name, summary="transaction test")
            def transient(config, context):
                return None

            raise RuntimeError("abort")
    with pytest.raises(KeyError):
        get_program(name)


def test_factory_without_schema_is_strict_and_registration_batches_roll_back():
    category = f"strict_{uuid4().hex}"

    register_factory(category, "model", lambda config, context: config)
    with pytest.raises(ValueError, match="unknown configuration key"):
        create(category, {"type": "model", "legacy": True})

    with pytest.raises(RuntimeError, match="abort"):
        with registration_transaction():
            register_factory(category, "transient", lambda config, context: config)
            raise RuntimeError("abort")
    assert available(category) == ["model"]


def test_global_scope_is_explicit_and_recursive_class_schema_is_describable():
    from lunarops.classes.observation_factory import ensure_registered

    ensure_registered()
    with pytest.raises(ValueError, match="unknown configuration key"):
        validate_global_class_configs({"parametrization": {"type": "reflectorPosition"}})

    schema = class_json_schema("stationDisplacement")
    assert schema["anyOf"]
    registered_types = {
        choice.get("properties", {}).get("type", {}).get("anyOf", [{}])[0].get("const")
        for choice in schema["anyOf"][2:]
    }
    assert "iers2010SolidEarthTide" in registered_types
    assert "sum" not in registered_types


def test_configuration_catalog_is_gui_ready_and_json_serializable():
    from lunarops.config.catalog import configuration_catalog

    catalog = configuration_catalog()
    json.dumps(catalog)
    assert catalog["format"] == "lunarops-yaml"
    global_fields = catalog["sections"]["globals"]["configuration"]["fields"]
    assert {field["name"] for field in global_fields} >= {"ephemerides", "stationCatalog"}
    choices = catalog["sections"]["programs"]["choices"]
    assert any(choice["name"] == "LlrResiduals" for choice in choices)
    assert len(catalog["jsonSchema"]["properties"]["programs"]["items"]["anyOf"]) == 4
    station_displacement = next(field for field in global_fields if field["name"] == "stationDisplacement")
    assert station_displacement["type"] == "class_list"
    assert station_displacement["minItems"] == 1
    assert catalog["jsonSchema"]["properties"]["variables"]["type"] == "object"
    assert "enabled" not in {
        field["name"] for field in catalog["sections"]["programs"]["controls"]["fields"]
    }
    elevation = next(
        field for field in catalog["sections"]["programs"]["choices"]
        if field["name"] == "LlrResiduals"
    )["configuration"]["fields"]
    elevation = next(field for field in elevation if field["name"] == "minElevationDeg")
    assert elevation["ui"]["widget"] == "number"
    assert elevation["ui"]["unit"] == "deg"
    program_schemas = catalog["jsonSchema"]["properties"]["programs"]["items"]["anyOf"]
    residual_schema = next(schema for schema in program_schemas if "minElevationDeg" in schema.get("properties", {}))
    assert residual_schema["properties"]["minElevationDeg"]["x-lunarops-ui"]["unit"] == "deg"


def test_run_schema_and_program_schema_describe_variable_references():
    resolved = run_config_schema().resolve({})
    assert resolved == {"variables": {}, "globals": {}, "programs": []}

    from lunarops.programs.registry import ensure_builtin_programs, get_program

    ensure_builtin_programs()
    residual_properties = get_program("LlrResiduals").spec.json_schema()["properties"]
    elevation = residual_properties["minElevationDeg"]
    assert any(option.get("pattern") == r"^\{[A-Za-z_][A-Za-z0-9_]*\}$" for option in elevation["anyOf"])
    assert "combineInputs" not in residual_properties
    assert "combinedName" not in residual_properties


def test_required_artifact_and_class_lists_are_strict():
    spec = ProgramSpec(
        name="StrictProgram",
        summary="A strict test contract.",
        outputs=(ArtifactSlot("output", "TextFile"),),
    )
    with pytest.raises(ValueError, match="must not be null"):
        spec.schema.resolve({"output": None})

    schema = ConfigSchema(fields=(class_list("models", "test"),))
    with pytest.raises(TypeError, match="class configs"):
        schema.resolve({"models": "single"})


def test_schema_rejects_empty_ranges_and_honors_non_empty_class_lists():
    with pytest.raises(ValueError, match="minimum_exclusive needs a minimum"):
        number("value", minimum_exclusive=True)
    with pytest.raises(ValueError, match="empty numeric range"):
        number("value", minimum=1, maximum=1, maximum_exclusive=True)

    schema = ConfigSchema(fields=(class_list("models", "test", non_empty=True),))
    with pytest.raises(ValueError, match="must not be empty"):
        schema.resolve({"models": []})


def test_run_plan_resolves_globals_once_and_expands_conditions():
    plan = build_run_plan(
        {
            "variables": {"sites": ["A", "B"]},
            "globals": {"sites": "{sites}"},
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

    assert plan.globals == {"sites": ["A", "B"]}
    assert plan.calls == (("P", {"output": "B.txt"}),)


def test_program_entries_cannot_be_disabled_in_place():
    with pytest.raises(ValueError, match=r"programs\[0\]\.enabled has been removed"):
        build_run_plan({"programs": [{"program": "P", "enabled": False}]})


def test_variable_cycles_and_cli_scalar_parsing_are_explicit():
    with pytest.raises(ValueError, match="cycle detected"):
        resolve_variables({"first": "{second}", "second": "{first}"})
    assert parse_set_overrides(["leading=001", "scientific=1e3"]) == {
        "leading": "001",
        "scientific": 1000.0,
    }


def test_shared_run_context_cache_is_closed_by_its_owner():
    category = f"cache_{uuid4().hex}"

    class Resource:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    register_factory(category, "resource", lambda config, context: Resource())
    cache = {}
    worker_context = RunContext(class_cache=cache)
    resource = worker_context.create_class(category, "resource")
    worker_context.close()
    assert resource.closed == 0
    assert cache

    owner = RunContext(class_cache=cache, owns_class_cache=True)
    owner.close()
    assert resource.closed == 1
    assert cache == {}

    transient_context = RunContext()
    transient = transient_context.create_class(category, "resource", cache=False)
    transient_context.close()
    assert transient.closed == 1
