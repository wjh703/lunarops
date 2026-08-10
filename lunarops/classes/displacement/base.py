"""Core displacement interfaces and immutable evaluation inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from lunarops.base.array_validation import readonly_vector3
from lunarops.classes.time import Epoch, TimeScale
from lunarops.base.station_identity import canonical_station_id


def _epoch(value: Epoch, *, scale: TimeScale, name: str) -> Epoch:
    if not isinstance(value, Epoch):
        raise TypeError(f"{name} must be an Epoch.")
    return value.require_scale(scale, name=name)


@dataclass(frozen=True, slots=True, eq=False)
class StationDisplacementInput:
    reference_position_itrf_m: np.ndarray
    epoch_utc: Epoch
    station_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_position_itrf_m",
            readonly_vector3(
                self.reference_position_itrf_m,
                name="reference_position_itrf_m",
            ),
        )
        object.__setattr__(
            self,
            "epoch_utc",
            _epoch(self.epoch_utc, scale=TimeScale.UTC, name="epoch_utc"),
        )
        if self.station_id is not None:
            station_id = canonical_station_id(self.station_id)
            if not station_id:
                raise ValueError("station_id must not be empty when supplied.")
            object.__setattr__(self, "station_id", station_id)


@dataclass(frozen=True, slots=True, eq=False)
class ReflectorDisplacementInput:
    reference_position_lcrs_m: np.ndarray
    epoch_tdb: Epoch

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_position_lcrs_m",
            readonly_vector3(
                self.reference_position_lcrs_m,
                name="reference_position_lcrs_m",
            ),
        )
        object.__setattr__(
            self,
            "epoch_tdb",
            _epoch(self.epoch_tdb, scale=TimeScale.TDB, name="epoch_tdb"),
        )


@runtime_checkable
class StationDisplacement(Protocol):
    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray: ...


@runtime_checkable
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
        normalized = tuple(components)
        if not normalized:
            raise ValueError("CompositeStationDisplacement requires at least one component.")
        for index, component in enumerate(normalized):
            if component is None:
                raise TypeError(
                    f"CompositeStationDisplacement components cannot contain None; component {index} is invalid."
                )
            if not callable(getattr(component, "displacement_itrf_m", None)):
                raise TypeError(
                    "CompositeStationDisplacement components must implement "
                    f"displacement_itrf_m(data); component {index} is {type(component)!r}."
                )
        self.components = normalized

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        total = np.zeros(3, dtype=float)
        for component in self.components:
            value = np.asarray(component.displacement_itrf_m(data), dtype=float)
            if value.size != 3:
                raise ValueError(f"{type(component).__name__}.displacement_itrf_m() must return three values.")
            value = value.reshape(3)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{type(component).__name__}.displacement_itrf_m() returned non-finite values.")
            total += value
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
