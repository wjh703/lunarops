"""Validated gravitating-body parameters."""

from __future__ import annotations

from dataclasses import dataclass

from lunarops.classes.ephemerides.body_ids import body_name


@dataclass(frozen=True, slots=True)
class GravitatingBody:
    body_id: str
    gravitational_parameter_m3_s2: float

    def __post_init__(self):
        key = body_name(self.body_id)
        object.__setattr__(self, "body_id", key)
