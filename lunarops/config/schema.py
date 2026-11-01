"""Schemas validate resolved configuration and fill declared defaults."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from lunarops.base.serialization import plain_data

from .expressions import (
    VARIABLE_NAME_PATTERN,
    VARIABLE_REFERENCE_PATTERN,
)

MISSING = object()
SchemaValidator = Callable[[dict[str, Any], str], Mapping[str, Any] | None]


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Description and validation rules for one configuration field."""

    name: str
    kind: str = "any"
    required: bool = False
    default: Any = MISSING
    description: str = ""
    choices: tuple[Any, ...] = ()
    item_kind: str | None = None
    nested: ConfigSchema | None = None
    item_nested: ConfigSchema | None = None
    class_category: str | None = None
    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None
    min_items: int | None = None
    max_items: int | None = None
    non_empty: bool = False
    allow_none: bool = False

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
            values = self._sequence_values(value, path)
            return [self._validate_class(item, f"{path}[{index}]") for index, item in enumerate(values)]

        raise AssertionError(f"Unhandled schema kind {self.kind!r}.")

    def _sequence_values(self, value: Any, path: str) -> list[Any]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{path} must be a sequence.")
        values = list(value)
        if self.min_items is not None and len(values) < self.min_items:
            raise ValueError(f"{path} must contain at least {self.min_items} item(s).")
        if self.max_items is not None and len(values) > self.max_items:
            raise ValueError(f"{path} must contain at most {self.max_items} item(s).")
        if self.non_empty and not values:
            raise ValueError(f"{path} must not be empty.")

        return values

    def _validate_sequence(self, value: Any, path: str) -> list[Any]:
        values = self._sequence_values(value, path)
        if self.item_kind is None and self.item_nested is None:
            return [deepcopy(item) for item in values]
        item_spec = FieldSpec(
            name="item",
            kind=self.item_kind or "mapping",
            nested=self.item_nested,
            class_category=self.class_category,
            non_empty=self.non_empty and self.item_kind in {"string", "path", "time"},
        )
        return [item_spec.validate(item, f"{path}[{index}]") for index, item in enumerate(values)]

    def _validate_class(self, value: Any, path: str) -> dict[str, Any]:
        return _validate_registered_class(self.class_category, value, path)

    def _validate_choices(self, value: Any, path: str) -> Any:
        if not self.choices:
            return value
        if value not in self.choices:
            raise ValueError(f"{path} must be one of {list(self.choices)!r}, got {value!r}.")
        return value

    def _validate_number(self, value: float, path: str) -> None:
        if self.gt is not None and value <= self.gt:
            raise ValueError(f"{path} must be greater than {self.gt}.")
        if self.ge is not None and value < self.ge:
            raise ValueError(f"{path} must be at least {self.ge}.")
        if self.lt is not None and value >= self.lt:
            raise ValueError(f"{path} must be less than {self.lt}.")
        if self.le is not None and value > self.le:
            raise ValueError(f"{path} must be at most {self.le}.")


@dataclass(frozen=True, slots=True)
class ConfigSchema:
    """A strict mapping schema with optional nested schemas."""

    fields: tuple[FieldSpec, ...] = ()
    description: str = ""
    type_name: str | None = None
    validator: SchemaValidator | None = None

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"Schema contains duplicate field names: {names!r}.")

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
        result = dict(value)

        self._check_structure(result, path)
        self._apply_fields(result, path)
        if self.validator is not None:
            validated = self.validator(result, path)
            if validated is not None:
                result = dict(validated)
        return plain_data(result)

    def _check_structure(self, result: dict[str, Any], path: str) -> None:
        if self.type_name is not None:
            raw_type = result.get("type")
            if not isinstance(raw_type, str):
                raise ValueError(f"{path}.type must be {self.type_name!r}.")
            if raw_type != self.type_name:
                raise ValueError(f"{path}.type must be {self.type_name!r}, got {raw_type!r}.")
            result["type"] = self.type_name

        unknown = set(result) - self.allowed_keys
        if unknown:
            raise ValueError(f"{path} has unknown configuration key(s): {sorted(str(key) for key in unknown)}")

    def _apply_fields(self, result: dict[str, Any], path: str) -> None:
        for field in self.fields:
            field_path = f"{path}.{field.name}"
            if field.name not in result:
                if field.required:
                    raise ValueError(f"{path} is missing required key {field.name!r}.")
                if field.default is not MISSING:
                    result[field.name] = field.validate(field.default, field_path)
                continue
            raw = result[field.name]
            if raw is None:
                if not field.allow_none:
                    raise ValueError(f"{field_path} must not be null.")
                continue
            result[field.name] = field.validate(raw, field_path)


def _validate_registered_class(category: str | None, value: Any, path: str) -> dict[str, Any]:
    if not category:
        raise ValueError(f"{path} schema class field is missing its category.")
    from .registry import resolve

    return resolve(category, value, path=path)


def field(name: str, kind: str = "any", **kwargs: Any) -> FieldSpec:
    return FieldSpec(name=name, kind=kind, **kwargs)


def string(name: str, **kwargs: Any) -> FieldSpec:
    kwargs.setdefault("non_empty", True)
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
    return field(name, "class", class_category=category, **kwargs)


def class_list(name: str, category: str, **kwargs: Any) -> FieldSpec:
    return field(name, "class_list", class_category=category, **kwargs)


__all__ = [
    "MISSING",
    "VARIABLE_NAME_PATTERN",
    "VARIABLE_REFERENCE_PATTERN",
    "ConfigSchema",
    "FieldSpec",
    "SchemaValidator",
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
]
