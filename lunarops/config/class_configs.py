"""Listing of registered model classes and the schema of the run-level ``shared:`` section."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lunarops.classes.builtin_registry import BUILTINS

from .registry import _REGISTRY, _available_type_names, _global_categories
from .schema import ConfigSchema, class_config, class_list


def available(category: str | None = None) -> dict[str, list[str]]:
    """Return the registered model class types, by category.

    Parameters
    ----------
    category : str, optional
        Category to list, such as ``troposphere``; all categories when None.

    Returns
    -------
    dict of str to list of str
        Type names keyed by category.
    """
    categories = sorted(set(_REGISTRY) | {cat for cat, _ in BUILTINS}) if category is None else [category]
    return {cat: _available_type_names(cat) for cat in categories}


def global_config_schema() -> ConfigSchema:
    """Return the schema of the ``shared:`` section.

    The section has one class config per shared category. ``stationDisplacement`` is a non-empty list of
    displacement models whose effects add up, and ``rangeBias`` is optional.

    Returns
    -------
    ConfigSchema
        Schema of the shared model and catalog configuration.
    """
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
                else class_config(
                    category,
                    category,
                    default=None,
                    allow_none=True,
                    description=f"Shared {category} model configuration.",
                )
                if category == "rangeBias"
                else class_config(category, category, description=f"Shared {category} model configuration.")
            )
            for category in categories
        ),
        description="Run-level shared model and catalog configuration.",
    )


def validate_global_class_configs(configs: Mapping[str, Any], *, path: str = "globals") -> dict[str, Any]:
    """Validate the ``shared:`` section and fill in defaults.

    Parameters
    ----------
    configs : Mapping
        Shared class configs keyed by category.
    path : str, optional
        Config path used in error messages.

    Returns
    -------
    dict
        Resolved shared class configs.
    """
    schema = global_config_schema()
    return schema.resolve(configs, path=path)
