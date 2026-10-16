"""Class registry design informed by GROOPS.

In GROOPS every configurable concept (ephemerides, troposphere, tides,
parametrization, ...) is an abstract *class category*; concrete
implementations register themselves under a ``type`` name and are
instantiated from the config file.  This module provides that mechanism.

Usage
-----
Registering an implementation::

    @register("troposphere", "mendesPavlis")
    class Iers2010MendesPavlisTroposphere: ...

or, when the class lives in an unmodified physics module::

    register_factory("troposphere", "mendesPavlis",
                     lambda cfg, ctx: Iers2010MendesPavlisTroposphere())

Instantiating from config::

    model = instantiate("troposphere", validate_class_config("troposphere", {"type": "mendesPavlis"}), context)

Config conventions
------------------
* A class config is a mapping ``{"type": "mendesPavlis", ...options...}``
  with an explicit, exactly spelled type.
* A *list* of class configs is used by schema fields that compose several
  implementations.
* Every registered type has a strict schema.  A factory without an explicit
  schema therefore accepts only ``{"type": "..."}``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

from .schema import ConfigSchema, class_config, class_list, variable_reference_json_schema

Factory = Callable[[dict, "object"], Any]


@dataclass(frozen=True, slots=True)
class RegisteredClass:
    """One complete class declaration used by validation, creation, and UI."""

    type_name: str
    factory: Factory
    schema: ConfigSchema
    scope: str


_REGISTRY: dict[str, dict[str, RegisteredClass]] = {}


class UnknownClassError(KeyError):
    pass


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
    if scope not in {"shared", "program", "transient"}:
        raise ValueError("scope must be 'shared', 'program', or 'transient'.")
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
        (registered.type_name for registered in _REGISTRY.get(category, {}).values()),
        key=str.casefold,
    )


def _global_categories() -> list[str]:
    return sorted(
        category
        for category, implementations in _REGISTRY.items()
        if implementations and any(registered.scope == "shared" for registered in implementations.values())
    )


def get_class_declaration(category: str, type_name: str) -> RegisteredClass:
    """Look up an exact declaration without resolving or filling configuration."""
    registered = _REGISTRY.get(category, {}).get(type_name)
    if registered is None:
        raise UnknownClassError(
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
    """Resolve one class config and return its canonical declaration and factory."""
    cfg = normalize_class_config(config)
    registered = get_class_declaration(category, cfg["type"])
    config_path = path or f"{category}/{registered.type_name}"
    resolved = registered.schema.resolve(cfg, path=config_path)
    return resolved, registered


def register(
    category: str,
    type_name: str,
    *,
    schema: ConfigSchema | None = None,
    scope: str = "program",
):
    """Decorator form.  The class must accept ``**options`` in ``__init__`` or
    provide ``from_config(cls, config, context)``."""

    def _wrap(cls):
        def _factory(config: dict, context) -> Any:
            if hasattr(cls, "from_config"):
                return cls.from_config(config, context)
            options = {k: v for k, v in config.items() if k != "type"}
            return cls(**options)

        register_factory(
            category,
            type_name,
            _factory,
            schema=schema,
            scope=scope,
        )
        cls._registry_type = type_name
        return cls

    return _wrap


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
    """Construct a model from a declaration already resolved by its schema."""
    return _REGISTRY[category][config["type"]].factory(config, context)


def validate_class_config(category: str, config, *, path: str | None = None) -> dict:
    """Normalize and validate one class config without instantiating it."""
    resolved, _ = resolve_class_config(category, config, path=path)
    return resolved


def available(category: str | None = None):
    if category is None:
        return {cat: _available_type_names(cat) for cat in sorted(_REGISTRY)}
    category = _normalize_category(category)
    return _available_type_names(category)


def class_json_schema(category: str, *, _class_stack: frozenset[str] = frozenset()) -> dict:
    """Return a JSON-schema union for all registered types in a category."""
    if category in _class_stack:
        return {"type": "object", "properties": {"type": {"type": "string"}}, "required": ["type"]}
    class_stack = _class_stack | {category}
    choices = [
        registered.schema.json_schema(_class_stack=class_stack)
        for registered in sorted(_REGISTRY.get(category, {}).values(), key=lambda item: item.type_name.casefold())
    ]
    if not choices:
        return {"type": "object"}
    return {
        "anyOf": [
            variable_reference_json_schema(),
            *choices,
        ]
    }


def global_config_schema() -> ConfigSchema:
    """Build the single schema shared by global validation and GUI metadata."""
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
    """Validate the run-level class map without constructing heavyweight objects."""
    schema = global_config_schema()
    return schema.resolve(configs, path=path)


__all__ = [
    "RegisteredClass",
    "UnknownClassError",
    "available",
    "class_json_schema",
    "global_config_schema",
    "instantiate",
    "normalize_class_config",
    "register",
    "register_factory",
    "resolve_class_config",
    "validate_class_config",
    "validate_global_class_configs",
]
