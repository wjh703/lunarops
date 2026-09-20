"""Body identifiers and provider contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from lunarops.classes.ephemerides import BodyState
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.time import Epoch


class BodyRole(StrEnum):
    INTEGRATED = "integrated"
    PRESCRIBED = "prescribed"


@dataclass(frozen=True, slots=True)
class DynamicalBody:
    body_id: str
    mu_m3_s2: float
    role: BodyRole = BodyRole.PRESCRIBED

    def __post_init__(self):
        key = body_name(self.body_id)
        if not key or not np.isfinite(self.mu_m3_s2) or self.mu_m3_s2 <= 0:
            raise ValueError("A body requires a non-empty id and positive finite mu_m3_s2.")
        object.__setattr__(self, "body_id", key)
        object.__setattr__(self, "role", BodyRole(self.role))


class ExternalStateProvider(Protocol):
    def body_state_bcrs(self, body_name: str, epoch_tdb: Epoch) -> BodyState: ...
    def body_states_bcrs(self, body_names: Sequence[str], epoch_tdb: Epoch) -> dict[str, BodyState]: ...
    def body_state_vectors_bcrs(self, body_names: Sequence[str], epoch_tdb: Epoch) -> np.ndarray: ...
