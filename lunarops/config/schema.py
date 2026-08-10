"""Small declarative schemas for YAML program and class configuration.

The schema layer deliberately stays independent of any validation framework.
It is used for three related jobs: validating resolved configuration, applying
declared defaults, and exposing a machine-readable description to the CLI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
import math
from pathlib import Path
from typing import Any

from .expressions import (
    VARIABLE_NAME_PATTERN,
    VARIABLE_REFERENCE_PATTERN,
    variable_reference_json_schema,
)


MISSING = object()
SchemaValidator = Callable[[dict[str, Any], str], Mapping[str, Any] | None]


@dataclass(frozen=True, slots=True)
class UiHints:
    """Presentation hints consumed by form builders, never by validation."""

    label: str = ""
    group: str = ""
    widget: str = ""
    unit: str = ""
    placeholder: str = ""
    advanced: bool = False

    def __post_init__(self) -> None:
        for name in ("label", "group", "widget", "unit", "placeholder"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"UI hint {name} must be a string.")
        if not isinstance(self.advanced, bool):
            raise TypeError("UI hint advanced must be a boolean.")

    def describe(self, *, field_name: str, default_widget: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "label": self.label or field_name,
            "widget": self.widget or default_widget,
        }
        for key, value in (
            ("group", self.group),
            ("unit", self.unit),
            ("placeholder", self.placeholder),
        ):
            if value:
                result[key] = value
        if self.advanced:
            result["advanced"] = True
        return result


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Description and validation rules for one configuration field."""

    name: str
    kind: str = "any"
    required: bool = False
    default: Any = MISSING
    description: str = ""
    examples: tuple[Any, ...] = ()
    ui: UiHints = UiHints()
    choices: tuple[Any, ...] = ()
    item_kind: str | None = None
    item_choices: tuple[Any, ...] = ()
    nested: ConfigSchema | None = None
    item_nested: ConfigSchema | None = None
    class_category: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    minimum_exclusive: bool = False
    maximum_exclusive: bool = False
    min_items: int | None = None
    max_items: int | None = None
    non_empty: bool = False
    allow_none: bool = True
    allow_variable_reference: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Schema field names must not be empty.")
        if not isinstance(self.kind, str):
            raise TypeError(f"Schema field {self.name!r} kind must be a string.")
        for flag_name in (
            "required",
            "minimum_exclusive",
            "maximum_exclusive",
            "non_empty",
            "allow_none",
            "allow_variable_reference",
        ):
            if not isinstance(getattr(self, flag_name), bool):
                raise TypeError(f"Schema field {self.name!r} {flag_name} must be a boolean.")
        if not isinstance(self.description, str):
            raise TypeError(f"Schema field {self.name!r} description must be a string.")
        if not isinstance(self.ui, UiHints):
            raise TypeError(f"Schema field {self.name!r} ui must be a UiHints instance.")
        if self.kind not in {
            "any",
            "string",
            "boolean",
            "integer",
            "number",
            "path",
            "time",
            "mapping",
            "sequence",
            "class",
            "class_list",
        }:
            raise ValueError(f"Unknown schema field kind {self.kind!r} for {self.name!r}.")
        for bound_name in ("minimum", "maximum"):
            bound = getattr(self, bound_name)
            if bound is not None:
                if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(float(bound)):
                    raise TypeError(f"Schema field {self.name!r} {bound_name} must be a finite number.")
        for count_name in ("min_items", "max_items"):
            count = getattr(self, count_name)
            if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
                raise TypeError(f"Schema field {self.name!r} {count_name} must be an integer.")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"Schema field {self.name!r} has an invalid numeric range.")
        if self.minimum_exclusive and self.minimum is None:
            raise ValueError(f"Schema field {self.name!r} minimum_exclusive needs a minimum.")
        if self.maximum_exclusive and self.maximum is None:
            raise ValueError(f"Schema field {self.name!r} maximum_exclusive needs a maximum.")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum == self.maximum
            and (self.minimum_exclusive or self.maximum_exclusive)
        ):
            raise ValueError(f"Schema field {self.name!r} has an empty numeric range.")
        if self.min_items is not None and self.min_items < 0:
            raise ValueError(f"Schema field {self.name!r} has a negative min_items.")
        if self.max_items is not None and self.max_items < 0:
            raise ValueError(f"Schema field {self.name!r} has a negative max_items.")
        if self.min_items is not None and self.max_items is not None and self.min_items > self.max_items:
            raise ValueError(f"Schema field {self.name!r} has an invalid item range.")
        if self.required and self.default is not MISSING:
            raise ValueError(f"Schema field {self.name!r} cannot be both required and have a default.")
        if self.item_kind is not None and not isinstance(self.item_kind, str):
            raise TypeError(f"Schema field {self.name!r} item_kind must be a string or None.")
        if self.item_kind is not None and self.item_kind not in {
            "any",
            "string",
            "boolean",
            "integer",
            "number",
            "path",
            "time",
            "mapping",
            "class",
        }:
            raise ValueError(f"Unknown schema item kind {self.item_kind!r} for {self.name!r}.")
        if self.nested is not None and self.kind != "mapping":
            raise ValueError(f"Schema field {self.name!r} can only use nested with kind 'mapping'.")
        if self.item_nested is not None and self.kind != "sequence":
            raise ValueError(f"Schema field {self.name!r} can only use item_nested with kind 'sequence'.")
        if self.item_kind is not None and self.kind != "sequence":
            raise ValueError(f"Schema field {self.name!r} can only use item_kind with kind 'sequence'.")
        if self.item_choices and self.item_kind is None:
            raise ValueError(f"Schema field {self.name!r} item_choices needs item_kind.")
        if self.kind in {"class", "class_list"} and not self.class_category:
            raise ValueError(f"Schema class field {self.name!r} needs a class category.")
        if self.class_category and self.kind not in {"class", "class_list"} and self.item_kind != "class":
            raise ValueError(
                f"Schema field {self.name!r} class_category requires a class or class item kind."
            )
        if self.kind == "class_list" and (self.item_kind is not None or self.item_nested is not None):
            raise ValueError(f"Schema class-list field {self.name!r} cannot declare item validation.")
        if (self.minimum is not None or self.maximum is not None) and self.kind not in {"integer", "number"}:
            raise ValueError(f"Schema field {self.name!r} numeric bounds require an integer or number kind.")
        if (self.min_items is not None or self.max_items is not None) and self.kind not in {"sequence", "class_list"}:
            raise ValueError(f"Schema field {self.name!r} item bounds require a sequence kind.")
        if self.non_empty and self.kind not in {"string", "path", "time", "sequence", "class_list"}:
            raise ValueError(f"Schema field {self.name!r} non_empty is not valid for kind {self.kind!r}.")
        if self.nested is not None and not isinstance(self.nested, ConfigSchema):
            raise TypeError(f"Schema field {self.name!r} nested value must be a ConfigSchema.")
        if self.item_nested is not None and not isinstance(self.item_nested, ConfigSchema):
            raise TypeError(f"Schema field {self.name!r} item_nested value must be a ConfigSchema.")
        object.__setattr__(self, "choices", tuple(self.choices))
        object.__setattr__(self, "item_choices", tuple(self.item_choices))
        object.__setattr__(self, "examples", tuple(deepcopy(self.examples)))
        object.__setattr__(self, "name", self.name.strip())

    def validate(self, value: Any, path: str) -> Any:
        if value is None:
            if self.allow_none:
                return None
            raise TypeError(f"{path} must not be null.")

        if self.kind == "any":
            return deepcopy(self._validate_choices(value, path))

        if self.kind == "string":
            if not isinstance(value, str):
                raise TypeError(f"{path} must be a string.")
            if self.non_empty and not value.strip():
                raise ValueError(f"{path} must not be empty.")
            return self._validate_choices(value, path)

        if self.kind == "boolean":
            if not isinstance(value, bool):
                raise TypeError(f"{path} must be a boolean.")
            return self._validate_choices(value, path)

        if self.kind == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{path} must be an integer.")
            self._validate_number(value, path)
            return self._validate_choices(value, path)

        if self.kind == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{path} must be a number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{path} must be finite.")
            self._validate_number(value, path)
            return self._validate_choices(value, path)

        if self.kind == "path":
            if not isinstance(value, (str, Path)):
                raise TypeError(f"{path} must be a path string.")
            if self.non_empty and not str(value).strip():
                raise ValueError(f"{path} must not be empty.")
            return self._validate_choices(value, path)

        if self.kind == "time":
            if not isinstance(value, (str, date, datetime)):
                raise TypeError(f"{path} must be an ISO time/date value.")
            if self.non_empty and isinstance(value, str) and not value.strip():
                raise ValueError(f"{path} must not be empty.")
            return self._validate_choices(value, path)

        if self.kind == "mapping":
            if not isinstance(value, Mapping):
                raise TypeError(f"{path} must be a mapping.")
            if any(not isinstance(key, str) for key in value):
                raise TypeError(f"{path} keys must be strings.")
            if self.nested is not None:
                return self.nested.resolve(value, path=path)
            return deepcopy(dict(value))

        if self.kind == "sequence":
            return self._validate_sequence(value, path)

        if self.kind == "class":
            return self._validate_class(value, path)

        if self.kind == "class_list":
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{path} must be a list of class configs (components list).")
            values = list(value)
            if self.min_items is not None and len(values) < self.min_items:
                raise ValueError(f"{path} must contain at least {self.min_items} item(s).")
            if self.max_items is not None and len(values) > self.max_items:
                raise ValueError(f"{path} must contain at most {self.max_items} item(s).")
            if self.non_empty and not values:
                raise ValueError(f"{path} must not be empty.")
            return [self._validate_class(item, f"{path}[{index}]") for index, item in enumerate(values)]

        raise AssertionError(f"Unhandled schema kind {self.kind!r}.")

    def _validate_sequence(self, value: Any, path: str) -> list[Any]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{path} must be a sequence.")
        values = list(value)
        if self.min_items is not None and len(values) < self.min_items:
            raise ValueError(f"{path} must contain at least {self.min_items} item(s).")
        if self.max_items is not None and len(values) > self.max_items:
            raise ValueError(f"{path} must contain at most {self.max_items} item(s).")
        if self.non_empty and not values:
            raise ValueError(f"{path} must not be empty.")

        result = []
        for index, item in enumerate(values):
            item_path = f"{path}[{index}]"
            if self.item_kind is None and self.item_nested is None:
                result.append(deepcopy(item))
                continue
            item_spec = FieldSpec(
                name=str(index),
                kind=self.item_kind or "mapping",
                choices=self.item_choices,
                nested=self.item_nested,
                class_category=self.class_category,
                non_empty=self.non_empty and self.item_kind in {"string", "path", "time"},
                allow_none=False,
                allow_variable_reference=self.allow_variable_reference,
            )
            result.append(item_spec.validate(item, item_path))
        return result

    def _validate_class(self, value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, (str, Mapping)):
            raise TypeError(f"{path} must be a class name or class config mapping.")
        from .registry import normalize_class_config

        # The category is metadata for the schema and JSON description.  The
        # category-specific schema is applied by registry.create(), after the
        # built-in factories have been lazily registered.
        return normalize_class_config(value)

    def _validate_choices(self, value: Any, path: str) -> Any:
        if not self.choices:
            return value
        if isinstance(value, str):
            for choice in self.choices:
                if isinstance(choice, str) and value.casefold() == choice.casefold():
                    return choice
            if not any(isinstance(item, str) for item in self.choices):
                raise ValueError(f"{path} must use a non-string choice, got {value!r}.")
            if value.casefold() not in {item.casefold() for item in self.choices if isinstance(item, str)}:
                raise ValueError(f"{path} must be one of {list(self.choices)!r}, got {value!r}.")
            raise AssertionError("String choice validation did not return a canonical choice.")
        if value not in self.choices:
            raise ValueError(f"{path} must be one of {list(self.choices)!r}, got {value!r}.")
        return value

    def _validate_number(self, value: float, path: str) -> None:
        if self.minimum is not None:
            if self.minimum_exclusive and value <= self.minimum:
                raise ValueError(f"{path} must be greater than {self.minimum}.")
            if not self.minimum_exclusive and value < self.minimum:
                raise ValueError(f"{path} must be at least {self.minimum}.")
        if self.maximum is not None:
            if self.maximum_exclusive and value >= self.maximum:
                raise ValueError(f"{path} must be less than {self.maximum}.")
            if not self.maximum_exclusive and value > self.maximum:
                raise ValueError(f"{path} must be at most {self.maximum}.")

    def describe(self) -> dict[str, Any]:
        from .presentation import describe_field

        return describe_field(self)

    def json_schema(self, *, _class_stack: frozenset[str] = frozenset()) -> dict[str, Any]:
        from .presentation import field_json_schema

        return field_json_schema(self, class_stack=_class_stack)


@dataclass(frozen=True, slots=True)
class ConfigSchema:
    """A strict mapping schema with optional nested schemas."""

    fields: tuple[FieldSpec, ...] = ()
    description: str = ""
    allow_unknown: bool = False
    type_name: str | None = None
    validator: SchemaValidator | None = None

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        if any(not isinstance(field, FieldSpec) for field in fields):
            raise TypeError("ConfigSchema fields must be FieldSpec instances.")
        names = [field.name for field in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"Schema contains duplicate field names: {names!r}.")
        if self.type_name is not None and (not isinstance(self.type_name, str) or not self.type_name.strip()):
            raise ValueError("ConfigSchema type_name must be a non-empty string or None.")
        if not isinstance(self.description, str):
            raise TypeError("ConfigSchema description must be a string.")
        if not isinstance(self.allow_unknown, bool):
            raise TypeError("ConfigSchema allow_unknown must be a boolean.")
        if self.validator is not None and not callable(self.validator):
            raise TypeError("ConfigSchema validator must be callable or None.")
        object.__setattr__(self, "fields", fields)
        if self.type_name is not None:
            object.__setattr__(self, "type_name", self.type_name.strip())

    @property
    def field_map(self) -> dict[str, FieldSpec]:
        return {field.name: field for field in self.fields}

    @property
    def allowed_keys(self) -> frozenset[str]:
        keys = {field.name for field in self.fields}
        if self.type_name is not None:
            keys.add("type")
        return frozenset(keys)

    def resolve(self, value: Mapping[str, Any], *, path: str = "config") -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping.")
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} keys must be strings.")
        result = deepcopy(dict(value))

        self._check_structure(result, path)
        self._apply_fields(result, path)
        if self.validator is not None:
            validated = self.validator(result, path)
            if validated is not None:
                if not isinstance(validated, Mapping):
                    raise TypeError(f"{path} schema validator must return a mapping or None.")
                if any(not isinstance(key, str) for key in validated):
                    raise TypeError(f"{path} schema validator returned a non-string key.")
                result = deepcopy(dict(validated))
            self._check_structure(result, path)
            self._apply_fields(result, path)
        return result

    def _check_structure(self, result: dict[str, Any], path: str) -> None:
        if self.type_name is not None:
            raw_type = result.get("type")
            if not isinstance(raw_type, str):
                raise ValueError(f"{path}.type must be {self.type_name!r}.")
            if raw_type.casefold() != self.type_name.casefold():
                raise ValueError(f"{path}.type must be {self.type_name!r}, got {raw_type!r}.")
            result["type"] = self.type_name

        unknown = set(result) - self.allowed_keys
        if unknown and not self.allow_unknown:
            raise ValueError(f"{path} has unknown configuration key(s): {sorted(str(key) for key in unknown)}")

    def _apply_fields(self, result: dict[str, Any], path: str) -> None:
        for field in self.fields:
            field_path = f"{path}.{field.name}"
            if field.name not in result:
                if field.required:
                    raise ValueError(f"{path} is missing required key {field.name!r}.")
                if field.default is not MISSING:
                    result[field.name] = field.validate(deepcopy(field.default), field_path)
                continue
            raw = result[field.name]
            if raw is None:
                if not field.allow_none:
                    raise ValueError(f"{field_path} must not be null.")
                continue
            result[field.name] = field.validate(raw, field_path)

    def validate(self, value: Mapping[str, Any], *, path: str = "config") -> None:
        self.resolve(value, path=path)

    def resolve_classes(self, value: Mapping[str, Any], *, path: str = "config") -> dict[str, Any]:
        """Validate every registered class nested in an already-resolved mapping."""
        result = dict(value)
        for field in self.fields:
            if field.name not in result or result[field.name] is None:
                continue
            field_path = f"{path}.{field.name}"
            if field.kind == "class":
                result[field.name] = _validate_registered_class(
                    field.class_category,
                    result[field.name],
                    field_path,
                )
            elif field.kind == "class_list":
                result[field.name] = [
                    _validate_registered_class(field.class_category, item, f"{field_path}[{index}]")
                    for index, item in enumerate(result[field.name])
                ]
            elif field.kind == "sequence" and field.item_kind == "class":
                result[field.name] = [
                    _validate_registered_class(field.class_category, item, f"{field_path}[{index}]")
                    for index, item in enumerate(result[field.name])
                ]
            elif field.nested is not None:
                result[field.name] = field.nested.resolve_classes(result[field.name], path=field_path)
            elif field.item_nested is not None:
                result[field.name] = [
                    field.item_nested.resolve_classes(item, path=f"{field_path}[{index}]")
                    for index, item in enumerate(result[field.name])
                ]
        return result

    def describe(self) -> dict[str, Any]:
        from .presentation import describe_schema

        return describe_schema(self)

    def json_schema(self, *, _class_stack: frozenset[str] = frozenset()) -> dict[str, Any]:
        from .presentation import config_json_schema

        return config_json_schema(self, class_stack=_class_stack)


def _validate_registered_class(category: str | None, value: Any, path: str) -> dict[str, Any]:
    if not category:
        raise ValueError(f"{path} schema class field is missing its category.")
    from .registry import validate_class_config

    return validate_class_config(category, value, path=path)


def field(name: str, kind: str = "any", **kwargs: Any) -> FieldSpec:
    """Concise factory used by program and class schema declarations."""
    return FieldSpec(name=name, kind=kind, **kwargs)


def string(name: str, **kwargs: Any) -> FieldSpec:
    return field(name, "string", **kwargs)


def boolean(name: str, **kwargs: Any) -> FieldSpec:
    return field(name, "boolean", **kwargs)


def integer(name: str, **kwargs: Any) -> FieldSpec:
    return field(name, "integer", **kwargs)


def number(name: str, **kwargs: Any) -> FieldSpec:
    return field(name, "number", **kwargs)


def path(name: str, **kwargs: Any) -> FieldSpec:
    return field(name, "path", **kwargs)


def time(name: str, **kwargs: Any) -> FieldSpec:
    return field(name, "time", **kwargs)


def mapping(name: str, **kwargs: Any) -> FieldSpec:
    return field(name, "mapping", **kwargs)


def sequence(name: str, **kwargs: Any) -> FieldSpec:
    return field(name, "sequence", **kwargs)


def class_config(name: str, category: str, **kwargs: Any) -> FieldSpec:
    kwargs.setdefault("allow_none", False)
    return field(name, "class", class_category=category, **kwargs)


def class_list(name: str, category: str, **kwargs: Any) -> FieldSpec:
    kwargs.setdefault("allow_none", False)
    return field(name, "class_list", class_category=category, **kwargs)


__all__ = [
    "ConfigSchema",
    "FieldSpec",
    "MISSING",
    "SchemaValidator",
    "UiHints",
    "VARIABLE_NAME_PATTERN",
    "VARIABLE_REFERENCE_PATTERN",
    "boolean",
    "class_config",
    "class_list",
    "field",
    "integer",
    "mapping",
    "number",
    "path",
    "sequence",
    "string",
    "time",
    "variable_reference_json_schema",
]
