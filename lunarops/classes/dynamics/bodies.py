"""Validated gravitating-body parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.classes.ephemerides.body_ids import body_name


@dataclass(frozen=True, slots=True)
class GravitatingBody:
    body_id: str
    gravitational_parameter_m3_s2: float

    def __post_init__(self):
        key = body_name(self.body_id)
        if not key or not np.isfinite(self.gravitational_parameter_m3_s2) or self.gravitational_parameter_m3_s2 <= 0:
            raise ValueError("A body requires a non-empty id and positive finite gravitational parameter.")
        object.__setattr__(self, "body_id", key)
