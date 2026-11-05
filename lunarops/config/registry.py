"""Registry of model class implementations, by category and type name.

A class config is a mapping whose ``type`` entry names an implementation exactly as registered, for
example ``{"type": "calceph", ...}`` in the ``ephemerides`` category; a list of class configs composes
several implementations. Built-in classes are registered on first use from
``lunarops.classes.builtin_registry``. A factory registered without a schema accepts only
``{"type": "..."}``. The categories ``troposphere``, ``relativity``, ``stationDisplacement`` and
``reflectorDisplacement`` each have a ``none`` type that disables the model.
"""

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
    """Registered implementation of a model class.

    Attributes
    ----------
    type_name : str
        Type name used in the ``type`` entry of a class config.
    factory : callable
        Factory called as ``factory(config, context, **dependencies)``.
    schema : ConfigSchema
        Schema of the class config.
    scope : str
        ``"shared"`` for classes configured at run level, ``"program"`` for classes configured in a
        program call.
    """

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
    """Register a factory for a type name in a category; a later registration replaces an earlier one.

    Parameters
    ----------
    category : str
        Class category, such as ``troposphere``.
    type_name : str
        Type name of the implementation.
    factory : callable
        Factory called as ``factory(config, context, **dependencies)`` with the resolved config.
    schema : ConfigSchema, optional
        Schema of the class config; its ``type_name`` is set to ``type_name`` when unset. When None, the
        config accepts only the ``type`` entry.
    scope : {"program", "shared"}, optional
        Whether the class is configured in a program call or at run level. Default ``"program"``.

    Raises
    ------
    ValueError
        If ``scope`` is invalid or the schema has a different type name.
    """
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
    """Validate a class config against the schema of its type and fill in defaults.

    Parameters
    ----------
    category : str
        Class category.
    config : Mapping
        Class config with a ``type`` entry.
    path : str, optional
        Config path used in error messages; ``category/type`` when None.

    Returns
    -------
    dict
        Resolved class config.

    Raises
    ------
    TypeError
        If ``config`` is not a mapping.
    ValueError
        If ``config`` has no ``type`` entry or does not match the schema.
    KeyError
        If the type is not registered in the category.
    """
    if not isinstance(config, Mapping):
        raise TypeError("Class configs must be a mapping with explicit type; use an explicit type: none when needed.")
    if "type" not in config:
        raise ValueError("Class config mapping requires a 'type' key.")
    registered = _declaration(category, config["type"])
    return registered.schema.resolve(config, path=path or f"{category}/{registered.type_name}")


def create(category: str, config: dict[str, Any], context=None, **dependencies):
    """Create a model class instance from a resolved class config.

    Parameters
    ----------
    category : str
        Class category.
    config : dict
        Resolved class config with a ``type`` entry.
    context : RunContext, optional
        Run context passed to the factory.
    **dependencies
        Extra keyword arguments passed to the factory.

    Returns
    -------
    object
        Instance returned by the factory; None for the ``none`` types.

    Raises
    ------
    KeyError
        If the type is not registered in the category.
    """
    return _declaration(category, config["type"]).factory(config, context, **dependencies)


__all__ = ["create", "register_factory", "resolve"]
