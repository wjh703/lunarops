"""Serialize configuration contracts for documentation and GUI clients."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from .expressions import variable_reference_json_schema

if TYPE_CHECKING:
    from .schema import ConfigSchema, FieldSpec


_DEFAULT_WIDGETS = {
    "any": "expression",
    "string": "text",
    "boolean": "checkbox",
    "integer": "integer",
    "number": "number",
    "path": "path",
    "time": "datetime",
    "mapping": "object",
    "sequence": "list",
    "class": "class-selector",
    "class_list": "class-list",
}


def _widget(field: FieldSpec) -> str:
    if field.ui.widget:
        return field.ui.widget
    if field.choices:
        return "select"
    return _DEFAULT_WIDGETS[field.kind]


def describe_field(field: FieldSpec) -> dict[str, Any]:
    from .schema import MISSING

    result: dict[str, Any] = {
        "name": field.name,
        "type": field.kind,
        "required": field.required,
        "allowNone": field.allow_none,
        "allowVariableReference": field.allow_variable_reference,
        "ui": field.ui.describe(field_name=field.name, default_widget=_widget(field)),
    }
    optional_values = (
        ("description", field.description),
        ("choices", list(field.choices)),
        ("examples", deepcopy(list(field.examples))),
        ("itemType", field.item_kind),
        ("itemChoices", list(field.item_choices)),
        ("classCategory", field.class_category),
        ("minimum", field.minimum),
        ("maximum", field.maximum),
        ("minItems", field.min_items),
        ("maxItems", field.max_items),
    )
    for key, value in optional_values:
        if value not in (None, "", [], ()):
            result[key] = value
    if field.default is not MISSING:
        result["default"] = deepcopy(field.default)
    if field.class_category is not None:
        from .registry import available

        result["classTypes"] = available(field.class_category)
    if field.minimum is not None:
        result["exclusiveMinimum"] = field.minimum_exclusive
    if field.maximum is not None:
        result["exclusiveMaximum"] = field.maximum_exclusive
    if field.non_empty:
        result["nonEmpty"] = True
    if field.nested is not None:
        result["properties"] = describe_schema(field.nested)
    if field.item_nested is not None:
        result["itemProperties"] = describe_schema(field.item_nested)
    return result


def describe_schema(schema: ConfigSchema) -> dict[str, Any]:
    result: dict[str, Any] = {
        "description": schema.description,
        "allowUnknown": schema.allow_unknown,
        "fields": [describe_field(field) for field in schema.fields],
    }
    if schema.type_name is not None:
        result["typeName"] = schema.type_name
    return result


def _base_field_json_schema(field: FieldSpec, class_stack: frozenset[str]) -> dict[str, Any]:
    if field.kind == "any":
        result: dict[str, Any] = {}
    elif field.kind in {"string", "path"}:
        result = {"type": "string"}
        if field.non_empty:
            result["minLength"] = 1
    elif field.kind == "time":
        result = {
            "oneOf": [
                {"type": "string", "format": "date-time"},
                {"type": "string", "format": "date"},
            ]
        }
        if field.non_empty:
            for option in result["oneOf"]:
                option["minLength"] = 1
    elif field.kind == "boolean":
        result = {"type": "boolean"}
    elif field.kind == "integer":
        result = {"type": "integer"}
    elif field.kind == "number":
        result = {"type": "number"}
    elif field.kind == "mapping":
        result = {"type": "object"}
    elif field.kind == "sequence":
        result = {"type": "array"}
        if field.item_kind is not None:
            result["items"] = _item_json_schema(field, class_stack)
    elif field.kind in {"class", "class_list"}:
        class_schema = _class_json_schema(field, class_stack)
        result = {"type": "array", "items": class_schema} if field.kind == "class_list" else class_schema
    else:
        raise AssertionError(f"Unhandled schema kind {field.kind!r}.")
    return result


def field_json_schema(field: FieldSpec, *, class_stack: frozenset[str]) -> dict[str, Any]:
    from .schema import MISSING

    result = _base_field_json_schema(field, class_stack)
    if field.choices:
        result["enum"] = list(field.choices)
    if field.nested is not None:
        result = config_json_schema(field.nested, class_stack=class_stack)
    if field.item_nested is not None and field.kind == "sequence":
        result["items"] = config_json_schema(field.item_nested, class_stack=class_stack)
    for key, value in (
        ("exclusiveMinimum" if field.minimum_exclusive else "minimum", field.minimum),
        ("exclusiveMaximum" if field.maximum_exclusive else "maximum", field.maximum),
        ("minItems", field.min_items),
        ("maxItems", field.max_items),
    ):
        if value is not None:
            result[key] = value

    variants = [result]
    if field.allow_variable_reference and (field.kind not in {"any", "string", "path"} or field.choices):
        variants.append(variable_reference_json_schema())
    if field.allow_none:
        variants.append({"type": "null"})
    result = variants[0] if len(variants) == 1 else {"anyOf": variants}
    result["title"] = field.ui.label or field.name
    result["x-lunarops-ui"] = field.ui.describe(field_name=field.name, default_widget=_widget(field))
    if field.description:
        result["description"] = field.description
    if field.default is not MISSING:
        result["default"] = deepcopy(field.default)
    if field.examples:
        result["examples"] = deepcopy(list(field.examples))
    return result


def _item_json_schema(field: FieldSpec, class_stack: frozenset[str]) -> dict[str, Any]:
    from .schema import FieldSpec

    item = FieldSpec(
        name="item",
        kind=field.item_kind or "any",
        choices=field.item_choices,
        nested=field.item_nested,
        class_category=field.class_category,
        non_empty=field.non_empty and field.item_kind in {"string", "path", "time"},
        allow_none=False,
        allow_variable_reference=field.allow_variable_reference,
    )
    return field_json_schema(item, class_stack=class_stack)


def _class_json_schema(field: FieldSpec, class_stack: frozenset[str]) -> dict[str, Any]:
    if field.class_category is None:
        return {"oneOf": [{"type": "string"}, {"type": "object"}]}
    from .registry import class_json_schema

    return class_json_schema(field.class_category, _class_stack=class_stack)


def config_json_schema(schema: ConfigSchema, *, class_stack: frozenset[str]) -> dict[str, Any]:
    properties = {
        field.name: field_json_schema(field, class_stack=class_stack)
        for field in schema.fields
    }
    required = [field.name for field in schema.fields if field.required]
    if schema.type_name is not None:
        properties = {
            "type": {"anyOf": [{"const": schema.type_name}, variable_reference_json_schema()]},
            **properties,
        }
        required = ["type", *required]
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": schema.allow_unknown,
    }
    if required:
        result["required"] = required
    if schema.description:
        result["description"] = schema.description
    return result


__all__ = ["config_json_schema", "describe_field", "describe_schema", "field_json_schema"]
