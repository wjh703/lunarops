"""Machine-readable catalog for YAML editors and future GUI clients."""

from __future__ import annotations

from typing import Any


def _program_json_schema(spec, base: dict[str, Any], control_schema: dict[str, Any]) -> dict[str, Any]:
    from .schema import variable_reference_json_schema

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

    from .expressions import CONDITION_OPERATORS
    from .loader import PROGRAM_CONTROL_SCHEMA, RUN_CONFIG_SCHEMA
    from .registry import global_config_schema

    ensure_builtin_programs()
    ensure_registered()

    global_schema = global_config_schema()
    control_json_schema = PROGRAM_CONTROL_SCHEMA.json_schema()
    program_choices = []
    program_items = []
    for spec in program_specs():
        description = spec.describe()
        description["jsonSchema"] = spec.json_schema()
        program_choices.append(description)
        program_items.append(_program_json_schema(spec, description["jsonSchema"], control_json_schema))

    json_schema = RUN_CONFIG_SCHEMA.json_schema()
    json_schema.update(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "LunarOps YAML run configuration",
        }
    )
    properties = json_schema["properties"]
    properties["shared"] = global_schema.json_schema()
    properties["observationModel"] = global_schema.json_schema()
    properties["programs"] = {"type": "array", "items": {"anyOf": program_items}}
    return {
        "format": "lunarops-yaml",
        "version": 2,
        "jsonSchema": json_schema,
        "sections": {
            "variables": {
                "type": "mapping",
                "description": "Values substituted into shared, observationModel, and program entries.",
            },
            "shared": {
                "type": "mapping",
                "description": "Run-level resources shared by programs.",
                "configuration": global_schema.describe(),
            },
            "observationModel": {
                "type": "mapping",
                "description": "Default LLR observation models.",
                "configuration": global_schema.describe(),
            },
            "programs": {
                "type": "sequence",
                "description": "Ordered program calls.",
                "controls": PROGRAM_CONTROL_SCHEMA.describe(),
                "when": {
                    "type": "condition",
                    "operators": list(CONDITION_OPERATORS),
                },
                "choices": program_choices,
            },
        },
    }


__all__ = ["configuration_catalog"]
