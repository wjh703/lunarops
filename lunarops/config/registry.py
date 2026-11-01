"""Class configs use an explicit, exactly spelled ``type`` name.

A list of class configs composes implementations. A factory without an
explicit schema accepts only ``{"type": "..."}``."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

from lunarops.classes.builtin_registry import BUILTINS

from .schema import ConfigSchema, class_config, class_list

Factory = Callable[[dict, "object"], Any]


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
    category = _normalize_category(category)
    canonical_type_name = _normalize_type_name(type_name)
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


def _normalize_category(category: str) -> str:
    if not isinstance(category, str) or not category or category != category.strip():
        raise ValueError("Class categories must be non-empty strings.")
    return category


def _normalize_type_name(type_name: str) -> str:
    if not isinstance(type_name, str) or not type_name or type_name != type_name.strip():
        raise ValueError("Class type names must be non-empty strings.")
    return type_name


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


def get_class_declaration(category: str, type_name: str) -> RegisteredClass:
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


def resolve_class_config(
    category: str,
    config,
    *,
    path: str | None = None,
) -> tuple[dict[str, Any], RegisteredClass]:
    cfg = normalize_class_config(config)
    registered = get_class_declaration(category, cfg["type"])
    config_path = path or f"{category}/{registered.type_name}"
    resolved = registered.schema.resolve(cfg, path=config_path)
    return resolved, registered


def normalize_class_config(config) -> dict:
    if config is None:
        raise TypeError("Class configs must be a mapping with explicit type; use an explicit type: none when needed.")
    if isinstance(config, Mapping):
        if any(not isinstance(key, str) for key in config):
            raise TypeError("Class config keys must be strings.")
        if "type" not in config:
            raise ValueError(f"Class config mapping requires a 'type' key: {config!r}")
        result = deepcopy(dict(config))
        result["type"] = _normalize_type_name(result["type"])
        return result
    raise TypeError(f"Unsupported class config: {config!r}")


def instantiate(category: str, config: dict[str, Any], context=None):
    return get_class_declaration(category, config["type"]).factory(config, context)


def validate_class_config(category: str, config, *, path: str | None = None) -> dict:
    resolved, _ = resolve_class_config(category, config, path=path)
    return resolved


def available(category: str | None = None):
    if category is None:
        return {cat: _available_type_names(cat) for cat in sorted(set(_REGISTRY) | {cat for cat, _ in BUILTINS})}
    category = _normalize_category(category)
    return _available_type_names(category)


def global_config_schema() -> ConfigSchema:
    categories = _global_categories()
    return ConfigSchema(
        fields=tuple(
            (
                class_list(
                    category,
                    category,
                    min_items=1,
                    description=f"Shared {category} model configurations applied additively.",
                )
                if category == "stationDisplacement"
                else class_config(category, category, description=f"Shared {category} model configuration.")
            )
            for category in categories
        ),
        description="Run-level shared model and catalog configuration.",
    )


def validate_global_class_configs(configs: Mapping[str, Any], *, path: str = "globals") -> dict[str, Any]:
    schema = global_config_schema()
    return schema.resolve(configs, path=path)


__all__ = [
    "RegisteredClass",
    "available",
    "global_config_schema",
    "instantiate",
    "normalize_class_config",
    "register_factory",
    "resolve_class_config",
    "validate_class_config",
    "validate_global_class_configs",
]
