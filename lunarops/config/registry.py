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

    model = create("troposphere", {"type": "mendesPavlis"}, context)

Config conventions
------------------
* A class config is either a plain string ``"mendesPavlis"`` (no options) or a
  mapping ``{"type": "mendesPavlis", ...options...}``.
* A *list* of class configs is allowed for categories whose base class
  supports composition (e.g. stationDisplacement); ``create_list`` returns the
  instantiated list.
* Every registered type has a strict schema.  A factory without an explicit
  schema therefore accepts only ``{"type": "..."}``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace as dataclass_replace
from threading import RLock
from typing import Any, Callable

from .schema import ConfigSchema, class_config, path, variable_reference_json_schema

Factory = Callable[[dict, "object"], Any]


@dataclass(frozen=True, slots=True)
class RegisteredClass:
    """One complete class declaration used by validation, creation, and UI."""

    type_name: str
    factory: Factory
    schema: ConfigSchema
    global_scope: bool


_REGISTRY: dict[str, dict[str, RegisteredClass]] = {}
_REGISTRY_LOCK = RLock()


class UnknownClassError(KeyError):
    pass


class DuplicateClassRegistrationError(ValueError):
    """Raised when a factory would replace an existing type implicitly."""


def register_factory(
    category: str,
    type_name: str,
    factory: Factory,
    *,
    replace: bool = False,
    schema: ConfigSchema | None = None,
    global_scope: bool = False,
) -> None:
    """Register one config factory.

    Replacing an existing ``(category, type)`` is opt-in.  This prevents a
    plugin or an import-order change from silently changing a configured
    physical model.
    """
    if not callable(factory):
        raise TypeError("Class factories must be callable.")
    category = _normalize_category(category)
    canonical_type_name = _normalize_type_name(type_name)
    normalized_type_name = canonical_type_name.casefold()
    if not isinstance(global_scope, bool):
        raise TypeError("global_scope must be a boolean.")
    if schema is None:
        schema = ConfigSchema(type_name=canonical_type_name)
    elif not isinstance(schema, ConfigSchema):
        raise TypeError("Class schemas must be ConfigSchema instances.")
    if schema.type_name is None:
        schema = dataclass_replace(schema, type_name=canonical_type_name)
    elif schema.type_name.casefold() != normalized_type_name:
        raise ValueError(
            f"Schema type name {schema.type_name!r} does not match registered type {canonical_type_name!r}."
        )
    else:
        schema = dataclass_replace(schema, type_name=canonical_type_name)
    with _REGISTRY_LOCK:
        category_factories = _REGISTRY.get(category)
        if category_factories is not None and normalized_type_name in category_factories and not replace:
            raise DuplicateClassRegistrationError(
                f"Implementation {canonical_type_name!r} is already registered for category {category!r}. "
                "Pass replace=True to replace it explicitly."
            )
        if category_factories is None:
            category_factories = {}
            _REGISTRY[category] = category_factories
        elif category_factories and any(
            registered.global_scope != global_scope for registered in category_factories.values()
        ):
            raise ValueError(
                f"Class category {category!r} must use one consistent global_scope value for all implementations."
            )
        category_factories[normalized_type_name] = RegisteredClass(
            type_name=canonical_type_name,
            factory=factory,
            schema=schema,
            global_scope=global_scope,
        )


def _normalize_category(category: str) -> str:
    if not isinstance(category, str) or not category.strip():
        raise ValueError("Class categories must be non-empty strings.")
    return category.strip()


def _normalize_type_name(type_name: str) -> str:
    if not isinstance(type_name, str) or not type_name.strip():
        raise ValueError("Class type names must be non-empty strings.")
    return type_name.strip()


@contextmanager
def registration_transaction() -> Iterator[None]:
    """Restore the registry if a built-in registration batch fails."""
    with _REGISTRY_LOCK:
        snapshot = {category: factories.copy() for category, factories in _REGISTRY.items()}
        try:
            yield
        except Exception:
            _REGISTRY.clear()
            _REGISTRY.update(snapshot)
            raise


def _available_type_names(category: str) -> list[str]:
    return sorted(
        (registered.type_name for registered in _REGISTRY.get(category, {}).values()),
        key=str.casefold,
    )


def _global_categories() -> list[str]:
    return sorted(
        category
        for category, implementations in _REGISTRY.items()
        if implementations and next(iter(implementations.values())).global_scope
    )


def resolve_class_config(
    category: str,
    config,
    *,
    path: str | None = None,
) -> tuple[dict[str, Any], RegisteredClass]:
    """Resolve one class config and return its canonical declaration and factory."""
    category = _normalize_category(category)
    cfg = normalize_class_config(config)
    type_name = str(cfg["type"]).casefold()
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(category, {}).get(type_name)
        if registered is None:
            raise UnknownClassError(
                f"No implementation {cfg['type']!r} registered for category {category!r}. "
                f"Available: {_available_type_names(category)}"
            )
    cfg["type"] = registered.type_name
    config_path = path or f"{category}/{registered.type_name}"
    resolved = registered.schema.resolve(cfg, path=config_path)
    return registered.schema.resolve_classes(resolved, path=config_path), registered


def register(
    category: str,
    type_name: str,
    *,
    replace: bool = False,
    schema: ConfigSchema | None = None,
    global_scope: bool = False,
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
            replace=replace,
            schema=schema,
            global_scope=global_scope,
        )
        cls._registry_category = _normalize_category(category)
        cls._registry_type = _normalize_type_name(type_name)
        return cls

    return _wrap


def normalize_class_config(config) -> dict:
    if config is None:
        raise TypeError("Class configs must be a type name or mapping; use an explicit type: none when needed.")
    if isinstance(config, str):
        return {"type": _normalize_type_name(config)}
    if isinstance(config, Mapping):
        if any(not isinstance(key, str) for key in config):
            raise TypeError("Class config keys must be strings.")
        if "type" not in config:
            raise ValueError(f"Class config mapping requires a 'type' key: {config!r}")
        result = deepcopy(dict(config))
        result["type"] = _normalize_type_name(result["type"])
        return result
    raise TypeError(f"Unsupported class config: {config!r}")


def create(category: str, config, context=None):
    """Instantiate one implementation of *category* from *config*."""
    cfg, registered = resolve_class_config(category, config)
    return registered.factory(cfg, context)


def validate_class_config(category: str, config, *, path: str | None = None) -> dict:
    """Normalize and validate one class config without instantiating it."""
    resolved, _ = resolve_class_config(category, config, path=path)
    return resolved


def create_list(category: str, configs, context=None) -> list[Any]:
    if configs is None:
        return []
    if isinstance(configs, (str, bytes, bytearray)) or not isinstance(configs, Sequence):
        raise TypeError(f"Class list for category {category!r} must be a sequence of class configs.")
    return [create(category, cfg, context) for cfg in configs]


def available(category: str | None = None):
    with _REGISTRY_LOCK:
        if category is None:
            return {
                cat: _available_type_names(cat)
                for cat in sorted(_REGISTRY)
            }
        category = _normalize_category(category)
        return _available_type_names(category)


def class_json_schema(category: str, *, _class_stack: frozenset[str] = frozenset()) -> dict:
    """Return a JSON-schema union for all registered types in a category."""
    category = _normalize_category(category)
    if category in _class_stack:
        return {
            "anyOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {"type": {"type": "string"}},
                    "required": ["type"],
                    "additionalProperties": True,
                },
            ]
        }
    with _REGISTRY_LOCK:
        implementations = dict(_REGISTRY.get(category, {}))
        types = sorted((registered.type_name for registered in implementations.values()), key=str.casefold)
    class_stack = _class_stack | {category}
    choices = []
    for registered in sorted(implementations.values(), key=lambda item: item.type_name.casefold()):
        choices.append(registered.schema.json_schema(_class_stack=class_stack))
    if not choices:
        return {"anyOf": [{"type": "string"}, {"type": "object"}]}
    return {
        "anyOf": [
            {"type": "string", "enum": types},
            variable_reference_json_schema(),
            *choices,
        ]
    }


def class_descriptions(category: str | None = None) -> dict:
    """Return registered class schemas in a CLI-friendly form."""
    with _REGISTRY_LOCK:
        selected = sorted(_REGISTRY) if category is None else [_normalize_category(category)]
        snapshots = {
            cat: {
                registered.type_name: registered.schema
                for registered in sorted(
                    _REGISTRY.get(cat, {}).values(),
                    key=lambda item: item.type_name.casefold(),
                )
            }
            for cat in selected
        }
    return {
        cat: {type_name: schema.describe() for type_name, schema in schemas.items()}
        for cat, schemas in snapshots.items()
    }


_GLOBAL_SCALAR_FIELDS = (
    path("stationCatalog", non_empty=True, description="Station catalog path or 'builtin'."),
    path("reflectorCatalog", non_empty=True, description="Reflector catalog path or 'builtin'."),
)


def global_config_schema() -> ConfigSchema:
    """Build the single schema shared by global validation and GUI metadata."""
    with _REGISTRY_LOCK:
        categories = _global_categories()
    return ConfigSchema(
        fields=tuple(
            class_config(category, category, description=f"Shared {category} model configuration.")
            for category in categories
        )
        + _GLOBAL_SCALAR_FIELDS,
        description="Run-level shared model and catalog configuration.",
    )


def validate_global_class_configs(configs: Mapping[str, Any], *, path: str = "globals") -> dict[str, Any]:
    """Validate the run-level class map without constructing heavyweight objects."""
    schema = global_config_schema()
    resolved = schema.resolve(configs, path=path)
    return schema.resolve_classes(resolved, path=path)


__all__ = [
    "DuplicateClassRegistrationError",
    "RegisteredClass",
    "UnknownClassError",
    "available",
    "class_descriptions",
    "class_json_schema",
    "create",
    "create_list",
    "global_config_schema",
    "normalize_class_config",
    "register",
    "register_factory",
    "registration_transaction",
    "resolve_class_config",
    "validate_class_config",
    "validate_global_class_configs",
]
