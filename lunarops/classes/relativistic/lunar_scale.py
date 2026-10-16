"""Relativistic scale metadata for lunar-surface coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    LunarRelativisticScaleConvention,
    l_b_minus_l_l_for_convention,
    normalize_lunar_relativistic_scale_convention,
)


@dataclass(frozen=True, slots=True)
class LunarRelativisticScale:
    convention: LunarRelativisticScaleConvention
    l_b_minus_l_l: float

    @classmethod
    def from_convention(cls, convention):
        normalized = normalize_lunar_relativistic_scale_convention(convention)
        return cls(normalized, l_b_minus_l_l_for_convention(normalized))


__all__ = ["LunarRelativisticScale"]
