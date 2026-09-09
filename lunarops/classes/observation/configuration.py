"""Configuration helpers for assembling an LLR observation model.

This module owns the declarative model boundary.  Runtime construction remains
in :mod:`lunarops.classes.observation_factory`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lunarops.config.registry import validate_class_config


OBSERVATION_MODEL_CATEGORIES = (
    "ephemerides",
    "earthRotation",
    "troposphere",
    "relativity",
    "stationDisplacement",
    "reflectorDisplacement",
    "rangeBias",
    "lunarDynamics",
)


def resolve_model_configs(context, program_config: Mapping[str, object]) -> dict[str, object]:
    """Merge program overrides with run defaults and validate each model."""
    if not isinstance(program_config, Mapping):
        raise TypeError("program_config must be a mapping.")
    merged: dict[str, object] = {}
    for category in OBSERVATION_MODEL_CATEGORIES:
        value = context.class_config(category, dict(program_config))
        if value is None:
            continue
        if category == "stationDisplacement":
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
                raise TypeError("observationModel.stationDisplacement must be a non-empty class list.")
            merged[category] = [
                validate_class_config(category, item, path=f"observationModel.{category}[{index}]")
                for index, item in enumerate(value)
            ]
        else:
            merged[category] = validate_class_config(category, value, path=f"observationModel.{category}")
    return merged


__all__ = ["OBSERVATION_MODEL_CATEGORIES", "resolve_model_configs"]
