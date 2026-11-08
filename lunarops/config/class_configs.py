"""Listing of registered model classes and the schema of the run-level ``models:`` section."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .registry import categories, model_categories, type_names
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
    return {name: type_names(name) for name in (categories() if category is None else [category])}


def model_configs_schema() -> ConfigSchema:
    """Return the schema of the ``models:`` section.

    The section has one class config per model category. ``stationDisplacement`` is a non-empty list of
    displacement models whose effects add up, and ``rangeBias`` is optional.

    Returns
    -------
    ConfigSchema
        Schema of the run-level model configuration.
    """
    fields = []
    for category in model_categories():
        if category == "stationDisplacement":
            fields.append(
                class_list(category, category, min_items=1, description=f"{category} models, applied additively.")
            )
        elif category == "rangeBias":
            fields.append(
                class_config(category, category, default=None, allow_none=True, description=f"{category} model.")
            )
        else:
            fields.append(class_config(category, category, description=f"{category} model."))
    return ConfigSchema(fields=tuple(fields), description="Run-level models, used by every program of the run.")


def validate_model_configs(configs: Mapping[str, Any], *, path: str = "models") -> dict[str, Any]:
    """Validate the ``models:`` section and fill in defaults.

    Parameters
    ----------
    configs : Mapping
        Model class configs keyed by category.
    path : str, optional
        Config path used in error messages.

    Returns
    -------
    dict
        Resolved model class configs.
    """
    return model_configs_schema().resolve(configs, path=path)
