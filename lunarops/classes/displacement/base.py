"""Core displacement interfaces and immutable evaluation inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lunarops.base.array_validation import finite_array
from lunarops.classes.time import Epoch, TimeScale


@dataclass(frozen=True, slots=True, eq=False)
class StationDisplacementInput:
    reference_position_itrf_m: np.ndarray
    epoch_utc: Epoch
    station_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_position_itrf_m",
            finite_array(
                self.reference_position_itrf_m,
                name="reference_position_itrf_m",
            size=3, copy=True, readonly=True),
        )
        self.epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")


@dataclass(frozen=True, slots=True, eq=False)
class ReflectorDisplacementInput:
    reference_position_lcrs_m: np.ndarray
    epoch_tdb: Epoch

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_position_lcrs_m",
            finite_array(
                self.reference_position_lcrs_m,
                name="reference_position_lcrs_m",
            size=3, copy=True, readonly=True),
        )
        self.epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")


class StationDisplacement(Protocol):
    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray: ...


class ReflectorDisplacement(Protocol):
    def displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray: ...


class ZeroStationDisplacement:
    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        return np.zeros(3, dtype=float)


class ZeroReflectorDisplacement:
    def displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        return np.zeros(3, dtype=float)


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
    "ReflectorDisplacement",
    "ReflectorDisplacementInput",
    "StationDisplacement",
    "StationDisplacementInput",
    "ZeroReflectorDisplacement",
    "ZeroStationDisplacement",
]
