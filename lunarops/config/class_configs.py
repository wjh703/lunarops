from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lunarops.classes.builtin_registry import BUILTINS

from .registry import _REGISTRY, _available_type_names, _global_categories
from .schema import ConfigSchema, class_config, class_list


def available(category: str | None = None):
    if category is None:
        return {cat: _available_type_names(cat) for cat in sorted(set(_REGISTRY) | {cat for cat, _ in BUILTINS})}
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
