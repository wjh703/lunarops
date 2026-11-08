"""Declarative config schemas.

A ``ConfigSchema`` is a strict mapping schema: it rejects unknown keys, checks every declared field
(``FieldSpec``) for its kind and constraints and fills in declared defaults. The helper functions
(``string``, ``number``, ``path``, ``class_config``, ...) create field specs of one kind.
"""

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

#: Sentinel default of a field without a default value.
MISSING = object()
#: Callable ``validator(config, path)`` run after field validation; returns a replacement config or None.
SchemaValidator = Callable[[dict[str, Any], str], Mapping[str, Any] | None]


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Description and validation rules for one configuration field.

    Attributes
    ----------
    name : str
        Config key of the field.
    kind : str
        Value kind: ``any``, ``string``, ``boolean``, ``integer``, ``number``, ``path``, ``time``, ``mapping``,
        ``sequence``, ``class`` or ``class_list``.
    required : bool
        Whether the key must be present.
    default : object
        Default value filled in when the key is absent; ``MISSING`` for none.
    description : str
        Description shown in the generated reference and by ``describe-program``.
    choices : tuple
        Allowed values; empty for any value.
    item_kind : str or None
        Kind of each item of a ``sequence``; items are copied unchecked when None and ``item_nested`` is None.
    nested : ConfigSchema or None
        Schema of a ``mapping`` value.
    item_nested : ConfigSchema or None
        Schema of each mapping item of a ``sequence``.
    class_category : str or None
        Registry category of a ``class`` or ``class_list`` value.
    gt, ge, lt, le : float or None
        Strict and inclusive bounds of an ``integer`` or ``number`` value.
    unique : bool
        Whether the items of a ``sequence`` must be distinct.
    min_items, max_items : int or None
        Bounds on the number of items of a ``sequence`` or ``class_list``.
    non_empty : bool
        Whether a string, path, time string or sequence must not be empty.
    allow_none : bool
        Whether None is accepted.
    """

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
    unique: bool = False
    min_items: int | None = None
    max_items: int | None = None
    non_empty: bool = False
    allow_none: bool = False

    def validate(self, value: Any, path: str) -> Any:
        """Validate a value of this field.

        Parameters
        ----------
        value : object
            Value to validate.
        path : str
            Config path used in error messages.

        Returns
        -------
        object
            Validated value; mappings and class configs are resolved against their schemas.

        Raises
        ------
        TypeError
            If the value has the wrong type.
        ValueError
            If the value violates a constraint.
        """
        if value is None:
            if self.allow_none:
                return None
            raise ValueError(f"{path} must not be null.")

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
        if self.unique and len(set(values)) != len(values):
            raise ValueError(f"{path} items must be unique.")
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
    """Strict mapping schema with optional nested schemas.

    Attributes
    ----------
    fields : tuple of FieldSpec
        Declared fields; names must be distinct.
    description : str
        Description of the config shown in the generated reference.
    type_name : str or None
        Required value of the ``type`` entry for class configs; None for configs without ``type``.
    validator : SchemaValidator or None
        Cross-field check run after the fields are validated.
    variants : tuple of ConfigSchema
        Alternative schemas selected by the ``type`` entry; when given, ``fields`` is not used.

    Raises
    ------
    ValueError
        If two fields have the same name.
    """

    fields: tuple[FieldSpec, ...] = ()
    description: str = ""
    type_name: str | None = None
    validator: SchemaValidator | None = None
    variants: tuple[ConfigSchema, ...] = ()

    def __post_init__(self) -> None:
        """Check that the field names are distinct."""
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"Schema contains duplicate field names: {names!r}.")

    @property
    def field_map(self) -> dict[str, FieldSpec]:
        """Fields keyed by name."""
        return {field.name: field for field in self.fields}

    @property
    def allowed_keys(self) -> frozenset[str]:
        """Accepted config keys: the field names, plus ``type`` when ``type_name`` is set."""
        keys = {field.name for field in self.fields}
        if self.type_name is not None:
            keys.add("type")
        return frozenset(keys)

    def resolve(self, value: Mapping[str, Any], *, path: str = "config") -> dict[str, Any]:
        """Validate a config mapping and fill in defaults.

        With ``variants``, the variant whose ``type_name`` equals the ``type`` entry resolves the value.
        Otherwise the ``type`` entry and the keys are checked, every field is validated or defaulted, and
        the ``validator`` is applied.

        Parameters
        ----------
        value : Mapping
            Config mapping with string keys.
        path : str, optional
            Config path used in error messages.

        Returns
        -------
        dict
            Resolved config as plain data.

        Raises
        ------
        TypeError
            If the value is not a mapping with string keys, or a field has the wrong type.
        ValueError
            If the ``type`` entry is wrong, a key is unknown, a required key is missing, or a constraint is
            violated.
        """
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping.")
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} keys must be strings.")
        if self.variants:
            for variant in self.variants:
                if value.get("type") == variant.type_name:
                    return variant.resolve(value, path=path)
            raise ValueError(f"{path}.type must be one of {[variant.type_name for variant in self.variants]}.")
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
            result[field.name] = field.validate(raw, field_path)


def _validate_registered_class(category: str | None, value: Any, path: str) -> dict[str, Any]:
    if not category:
        raise ValueError(f"{path} schema class field is missing its category.")
    from .registry import resolve

    return resolve(category, value, path=path)


def field(name: str, kind: str = "any", **kwargs: Any) -> FieldSpec:
    """Return a field spec of the given kind; keyword arguments set the other ``FieldSpec`` attributes.

    Parameters
    ----------
    name : str
        Config key of the field.
    kind : str, optional
        Value kind (see ``FieldSpec.kind``). Default ``"any"``.
    **kwargs
        Other ``FieldSpec`` attributes.

    Returns
    -------
    FieldSpec
        The field spec.
    """
    return FieldSpec(name=name, kind=kind, **kwargs)


def string(name: str, **kwargs: Any) -> FieldSpec:
    """Return a ``string`` field spec; ``non_empty`` defaults to True."""
    kwargs.setdefault("non_empty", True)
    return field(name, "string", **kwargs)


def boolean(name: str, **kwargs: Any) -> FieldSpec:
    """Return a ``boolean`` field spec."""
    return field(name, "boolean", **kwargs)


def integer(name: str, **kwargs: Any) -> FieldSpec:
    """Return an ``integer`` field spec; booleans are rejected."""
    return field(name, "integer", **kwargs)


def number(name: str, **kwargs: Any) -> FieldSpec:
    """Return a ``number`` field spec for finite int or float values; booleans are rejected."""
    return field(name, "number", **kwargs)


def path(name: str, **kwargs: Any) -> FieldSpec:
    """Return a ``path`` field spec for string or ``Path`` values."""
    return field(name, "path", **kwargs)


def time(name: str, **kwargs: Any) -> FieldSpec:
    """Return a ``time`` field spec for ISO date or date-time strings, ``date`` or ``datetime`` values."""
    return field(name, "time", **kwargs)


def mapping(name: str, **kwargs: Any) -> FieldSpec:
    """Return a ``mapping`` field spec; ``nested`` gives the schema of the value."""
    return field(name, "mapping", **kwargs)


def sequence(name: str, **kwargs: Any) -> FieldSpec:
    """Return a ``sequence`` field spec; ``item_kind`` or ``item_nested`` describes the items."""
    return field(name, "sequence", **kwargs)


def class_config(name: str, category: str, **kwargs: Any) -> FieldSpec:
    """Return a field spec for one class config of a registry category."""
    return field(name, "class", class_category=category, **kwargs)


def class_list(name: str, category: str, **kwargs: Any) -> FieldSpec:
    """Return a field spec for a list of class configs of a registry category."""
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
