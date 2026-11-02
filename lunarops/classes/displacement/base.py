"""Core displacement interfaces and immutable evaluation inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lunarops.classes.time import Epoch


@dataclass(frozen=True, slots=True, eq=False)
class StationDisplacementInput:
    reference_position_itrf_m: np.ndarray
    epoch_utc: Epoch
    station_id: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ReflectorDisplacementInput:
    reference_position_lcrs_m: np.ndarray
    epoch_tdb: Epoch


class StationDisplacement(Protocol):
    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray: ...


class CompositeStationDisplacement:
    def __init__(self, components: Sequence[StationDisplacement]) -> None:
        self.components = tuple(components)

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        total = np.zeros(3, dtype=float)
        for component in self.components:
            total += component.displacement_itrf_m(data)
        return total


__all__ = [
    "CompositeStationDisplacement",
    "ReflectorDisplacementInput",
    "StationDisplacement",
    "StationDisplacementInput",
]
