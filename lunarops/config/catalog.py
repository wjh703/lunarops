"""Machine-readable catalog for YAML editors and future GUI clients."""

from __future__ import annotations

from typing import Any

def _program_json_schema(spec, control_schema: dict[str, Any]) -> dict[str, Any]:
    from .schema import variable_reference_json_schema

    base = spec.json_schema()
    properties = {
        "program": {
            "anyOf": [
                {"const": spec.name},
                variable_reference_json_schema(),
            ]
        },
        **base.get("properties", {}),
        **control_schema.get("properties", {}),
    }
    return {
        "type": "object",
        "properties": properties,
        "required": ["program", *base.get("required", [])],
        "additionalProperties": False,
        "title": spec.name,
        "description": spec.summary,
    }


def configuration_catalog() -> dict[str, Any]:
    """Return the complete YAML contract as JSON-compatible metadata.

    Registration is lazy so importing :mod:`lunarops.config` stays cheap.  A
    GUI can use ``sections`` for form construction and ``jsonSchema`` for
    generic validation without importing implementation modules itself.
    """
    from lunarops.classes.observation_factory import ensure_registered
    from lunarops.programs.registry import ensure_builtin_programs, program_specs

    from .loader import condition_operators, program_control_schema, run_config_schema
    from .registry import global_config_schema

    ensure_builtin_programs()
    ensure_registered()

    global_schema = global_config_schema()
    control_schema = program_control_schema()
    specs = program_specs()
    program_choices = []
    for spec in specs:
        description = spec.describe()
        description["jsonSchema"] = spec.json_schema()
        program_choices.append(description)

    program_item_schema = {
        "anyOf": [
            _program_json_schema(spec, control_schema.json_schema())
            for spec in specs
        ]
    }
    json_schema = run_config_schema().json_schema()
    json_schema.update(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "LunarOps YAML run configuration",
        }
    )
    properties = json_schema["properties"]
    properties["globals"] = global_schema.json_schema()
    properties["programs"] = {"type": "array", "items": program_item_schema}
    return {
        "format": "lunarops-yaml",
        "version": 1,
        "jsonSchema": json_schema,
        "sections": {
            "variables": {
                "type": "mapping",
                "description": "Values substituted into globals and program entries.",
            },
            "globals": {
                "type": "mapping",
                "description": global_schema.description,
                "configuration": global_schema.describe(),
            },
            "programs": {
                "type": "sequence",
                "description": "Ordered program calls.",
                "controls": control_schema.describe(),
                "when": {
                    "type": "condition",
                    "operators": list(condition_operators()),
                },
                "choices": program_choices,
            },
        },
    }


__all__ = ["configuration_catalog"]
