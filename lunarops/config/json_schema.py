"""Export of the run-config schemas as a JSON Schema, for YAML editors.

The JSON Schema gives an editor completion, hover descriptions, enumeration choices and immediate errors. It
covers field names, types, bounds and required keys; cross-field rules stay with ``lunarops validate``. Any
non-string value may instead be a ``{variable}`` placeholder.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from lunarops.base.serialization import plain_data

from .class_configs import model_configs_schema
from .expressions import VARIABLE_NAME_PATTERN, VARIABLE_REFERENCE_PATTERN
from .loader import PROGRAM_CONTROL_SCHEMA
from .registry import categories, registered_class, type_names
from .schema import MISSING, ConfigSchema, FieldSpec

_PLACEHOLDER = {"type": "string", "pattern": VARIABLE_REFERENCE_PATTERN}
_BOUNDS = (("gt", "exclusiveMinimum"), ("ge", "minimum"), ("lt", "exclusiveMaximum"), ("le", "maximum"))


def _class_ref(category: str | None) -> dict[str, Any]:
    return {"$ref": f"#/definitions/class.{category}"}


def _value_schema(field: FieldSpec) -> dict[str, Any]:
    kind = field.kind
    if kind in {"string", "path", "time"}:
        schema: dict[str, Any] = {"type": "string"}
        if field.non_empty:
            schema["minLength"] = 1
    elif kind in {"boolean", "integer", "number"}:
        schema = {"type": kind}
        schema.update({key: getattr(field, name) for name, key in _BOUNDS if getattr(field, name) is not None})
    elif kind == "mapping":
        schema = _object_schema(field.nested) if field.nested is not None else {"type": "object"}
    elif kind in {"sequence", "class_list"}:
        schema = {"type": "array"}
        if kind == "class_list":
            schema["items"] = _class_ref(field.class_category)
        elif field.item_nested is not None:
            schema["items"] = _object_schema(field.item_nested)
        elif field.item_kind is not None:
            schema["items"] = _accepting_placeholder(
                _value_schema(FieldSpec(name="item", kind=field.item_kind, class_category=field.class_category)),
                field.item_kind,
            )
        if field.min_items is not None:
            schema["minItems"] = field.min_items
        if field.max_items is not None:
            schema["maxItems"] = field.max_items
        if field.unique:
            schema["uniqueItems"] = True
    elif kind == "class":
        schema = _class_ref(field.class_category)
    else:
        schema = {}
    if field.choices:
        schema["enum"] = list(field.choices)
    return schema


def _accepting_placeholder(schema: dict[str, Any], kind: str, *, has_choices: bool = False) -> dict[str, Any]:
    plain_string = kind in {"string", "path", "time"} and not has_choices
    return schema if kind == "any" or plain_string else {"anyOf": [schema, _PLACEHOLDER]}


def _field_schema(field: FieldSpec) -> dict[str, Any]:
    schema = _accepting_placeholder(_value_schema(field), field.kind, has_choices=bool(field.choices))
    if field.allow_none:
        schema = {"anyOf": [*schema.get("anyOf", [schema]), {"type": "null"}]}
    if field.description:
        schema = {**schema, "description": field.description}
    if field.default is not MISSING:
        try:
            default = plain_data(field.default)
            json.dumps(default, allow_nan=False)
        except TypeError, ValueError:
            return schema
        schema = {**schema, "default": default}
    return schema


def _object_schema(schema: ConfigSchema) -> dict[str, Any]:
    if schema.variants:
        return {"oneOf": [_object_schema(variant) for variant in schema.variants]}
    properties = {field.name: _field_schema(field) for field in schema.fields}
    required = [field.name for field in schema.fields if field.required]
    if schema.type_name is not None:
        properties = {"type": {"const": schema.type_name}, **properties}
        required.insert(0, "type")
    result: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        result["required"] = required
    if schema.description:
        result["description"] = schema.description
    return result


def _program_schema(spec: Any) -> dict[str, Any]:
    result = _object_schema(spec.schema)
    controls = {field.name: _field_schema(field) for field in PROGRAM_CONTROL_SCHEMA.fields}
    result["properties"] = {"program": {"const": spec.name}, **controls, **result["properties"]}
    result["required"] = ["program", *result.get("required", [])]
    return result


def run_config_json_schema(specs: Iterable[Any]) -> dict[str, Any]:
    """Return the JSON Schema (draft-07) of a run config.

    Parameters
    ----------
    specs : iterable of ProgramSpec
        Programs accepted in the ``programs:`` section.

    Returns
    -------
    dict
        JSON Schema document.
    """
    definitions = {
        f"class.{category}": {
            "oneOf": [_object_schema(registered_class(category, name).schema) for name in type_names(category)]
        }
        for category in categories()
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "LunarOps run configuration",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "extends": {
                "description": "File or list of files this file starts from, relative to this file.",
                "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}, "minItems": 1}],
            },
            "variables": {
                "description": "Named values used by {name} placeholders.",
                "type": "object",
                "propertyNames": {"pattern": f"^{VARIABLE_NAME_PATTERN}$"},
            },
            "models": _object_schema(model_configs_schema()),
            "programs": {
                "description": "Ordered program calls.",
                "type": "array",
                "items": {"oneOf": [_program_schema(spec) for spec in specs]},
            },
        },
        "definitions": definitions,
    }


def render_run_config_json_schema(specs: Iterable[Any]) -> str:
    """Return the JSON Schema of a run config as text.

    Parameters
    ----------
    specs : iterable of ProgramSpec
        Programs accepted in the ``programs:`` section.

    Returns
    -------
    str
        Indented JSON ending with a newline.
    """
    return json.dumps(run_config_json_schema(specs), indent=1, ensure_ascii=False) + "\n"


__all__ = ["render_run_config_json_schema", "run_config_json_schema"]
