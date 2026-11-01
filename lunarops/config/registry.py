"""Class configs use an explicit, exactly spelled ``type`` name.

A list of class configs composes implementations. A factory without an
explicit schema accepts only ``{"type": "..."}``."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

from lunarops.classes.builtin_registry import BUILTINS

from .schema import ConfigSchema

Factory = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class RegisteredClass:
    type_name: str
    factory: Factory
    schema: ConfigSchema
    scope: str


_REGISTRY: dict[str, dict[str, RegisteredClass]] = {}


def register_factory(
    category: str,
    type_name: str,
    factory: Factory,
    *,
    schema: ConfigSchema | None = None,
    scope: str = "program",
) -> None:
    """Declare a factory; later declarations replace earlier ones."""
    canonical_type_name = type_name
    if scope not in {"shared", "program"}:
        raise ValueError("scope must be 'shared' or 'program'.")
    if schema is None:
        schema = ConfigSchema(type_name=canonical_type_name)
    elif schema.type_name is None:
        schema = dataclass_replace(schema, type_name=canonical_type_name)
    elif schema.type_name != canonical_type_name:
        raise ValueError(
            f"Schema type name {schema.type_name!r} does not match registered type {canonical_type_name!r}."
        )
    _REGISTRY.setdefault(category, {})[canonical_type_name] = RegisteredClass(
        type_name=canonical_type_name,
        factory=factory,
        schema=schema,
        scope=scope,
    )


for _category in ("troposphere", "relativity", "stationDisplacement", "reflectorDisplacement"):
    register_factory(_category, "none", lambda cfg, ctx: None, schema=ConfigSchema(type_name="none"), scope="shared")


def _available_type_names(category: str) -> list[str]:
    return sorted(
        set(_REGISTRY.get(category, {})) | {typ for cat, typ in BUILTINS if cat == category},
        key=str.casefold,
    )


def _global_categories() -> list[str]:
    return sorted(
        {cat for cat, _ in BUILTINS if cat != "parametrization"}
        | {
            category
            for category, implementations in _REGISTRY.items()
            if any(entry.scope == "shared" for entry in implementations.values())
        }
    )


def _declaration(category: str, type_name: str) -> RegisteredClass:
    registered = _REGISTRY.get(category, {}).get(type_name)
    if registered is None and (category, type_name) in BUILTINS:
        module, name = BUILTINS[category, type_name].split(":")
        cls = getattr(importlib.import_module(module), name)
        register_factory(
            category,
            type_name,
            cls.from_config,
            schema=cls.SCHEMA,
            scope="program" if category == "parametrization" else "shared",
        )
        registered = _REGISTRY[category][type_name]
    if registered is None:
        raise KeyError(
            f"No implementation {type_name!r} registered for category {category!r}. "
            f"Available: {_available_type_names(category)}"
        )
    return registered


def resolve(category: str, config, *, path: str | None = None) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("Class configs must be a mapping with explicit type; use an explicit type: none when needed.")
    if "type" not in config:
        raise ValueError("Class config mapping requires a 'type' key.")
    registered = _declaration(category, config["type"])
    return registered.schema.resolve(config, path=path or f"{category}/{registered.type_name}")


def create(category: str, config: dict[str, Any], context=None, **dependencies):
    return _declaration(category, config["type"]).factory(config, context, **dependencies)


__all__ = ["create", "register_factory", "resolve"]
