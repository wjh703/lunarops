"""Configuration helpers for assembling an LLR observation model.

This module owns the declarative model boundary.  Runtime construction remains
in :mod:`lunarops.classes.observation_factory`.
"""

from __future__ import annotations

from collections.abc import Mapping

from lunarops.config.schema import ConfigSchema, class_config, class_list

OBSERVATION_MODEL_CATEGORIES = (
    "ephemerides",
    "earthRotation",
    "troposphere",
    "relativity",
    "stationDisplacement",
    "reflectorDisplacement",
    "rangeBias",
)


def resolve_model_configs(context, program_config: Mapping[str, object]) -> dict[str, object]:
    """Merge program overrides with run defaults."""
    merged: dict[str, object] = {}
    for category in OBSERVATION_MODEL_CATEGORIES:
        value = context.class_config(category, dict(program_config))
        if value is None:
            continue
        merged[category] = value
    return merged


def observation_model_schema() -> ConfigSchema:
    return ConfigSchema(
        fields=tuple(
            class_list(category, category, min_items=1)
            if category == "stationDisplacement"
            else class_config(category, category)
            for category in OBSERVATION_MODEL_CATEGORIES
        )
    )


__all__ = ["OBSERVATION_MODEL_CATEGORIES", "resolve_model_configs"]
