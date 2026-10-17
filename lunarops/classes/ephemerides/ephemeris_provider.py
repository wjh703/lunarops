"""Body-state provider contract and immutable state values."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from lunarops.base.array_validation import finite_array
from lunarops.classes.time import Epoch


@dataclass(frozen=True, slots=True, eq=False)
class BodyState:
    """BCRS position and velocity of one body relative to the SSB."""

    position_m: np.ndarray
    velocity_mps: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", finite_array(
            self.position_m, size=3, name="position_m", copy=True, readonly=True
        ))
        object.__setattr__(self, "velocity_mps", finite_array(
            self.velocity_mps, size=3, name="velocity_mps", copy=True, readonly=True
        ))


@runtime_checkable
class Ephemeris(Protocol):
    """Provide SSB-relative BCRS states and accelerations at TDB epochs."""

    @property
    def source_path(self) -> Path | None: ...

    def body_state_bcrs(self, body: str, epoch_tdb: Epoch) -> BodyState: ...

    def body_position_bcrs(self, body: str, epoch_tdb: Epoch) -> np.ndarray: ...

    def body_state_matrix_bcrs(self, bodies: Sequence[str], epoch_tdb: Epoch) -> np.ndarray:
        """Return requested states as a ``(body, state)`` SI array."""
        ...

    def body_state_matrices_bcrs(self, bodies: Sequence[str], epochs: Sequence[Epoch]) -> np.ndarray:
        """Return requested states as an ``(epoch, body, state)`` SI array."""
        ...

    def body_accelerations_bcrs(self, body: str, epochs: Sequence[Epoch]) -> np.ndarray:
        """Return one body's second TDB derivative as an ``(epoch, component)`` SI array."""
        ...

    def close(self) -> None: ...


__all__ = ["BodyState", "Ephemeris"]
