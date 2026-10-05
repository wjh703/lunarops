"""Explicit physical and integration state objects for Moon dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.base.array_validation import finite_array
from lunarops.classes.ephemerides import BodyState
from lunarops.classes.time import require_tdb_epoch
from lunarops.classes.time import Epoch


@dataclass(frozen=True, slots=True)
class MoonRelativeState:
    """Six-variable state integrated with a prescribed Earth ephemeris.

    The state is the Moon position and velocity relative to the ephemeris
    Earth, in BCRS axes and SI units.
    """

    epoch_tdb: Epoch
    position_m: np.ndarray
    velocity_mps: np.ndarray

    def __post_init__(self) -> None:
        require_tdb_epoch(self.epoch_tdb)
        object.__setattr__(
            self,
            "position_m",
            finite_array(self.position_m, size=3, name="position_m", copy=True, readonly=True),
        )
        object.__setattr__(
            self,
            "velocity_mps",
            finite_array(self.velocity_mps, size=3, name="velocity_mps", copy=True, readonly=True),
        )

    def as_relative_vector(self) -> np.ndarray:
        return np.concatenate((self.position_m, self.velocity_mps))

    @classmethod
    def from_relative_vector(cls, epoch_tdb, vector):
        y = finite_array(vector, shape=(6,), name="relative integration state")
        return cls(epoch_tdb, y[:3], y[3:6])

    @classmethod
    def from_barycentric_state(cls, physical: BarycentricEarthMoonState):
        return cls(
            physical.epoch_tdb,
            physical.moon.position_m - physical.earth.position_m,
            physical.moon.velocity_mps - physical.earth.velocity_mps,
        )

    def to_barycentric_state(self, earth: BodyState) -> BarycentricEarthMoonState:
        if not isinstance(earth, BodyState):
            raise TypeError("earth must be a BodyState")
        moon = BodyState(
            earth.position_m + self.position_m,
            earth.velocity_mps + self.velocity_mps,
        )
        return BarycentricEarthMoonState(self.epoch_tdb, earth, moon)


@dataclass(frozen=True, slots=True)
class BarycentricEarthMoonState:
    epoch_tdb: Epoch
    earth: BodyState
    moon: BodyState

    def __post_init__(self):
        require_tdb_epoch(self.epoch_tdb)
        if not isinstance(self.earth, BodyState) or not isinstance(self.moon, BodyState):
            raise TypeError("earth and moon must be BodyState objects")

    @property
    def relative_state(self) -> MoonRelativeState:
        return MoonRelativeState(
            self.epoch_tdb,
            self.moon.position_m - self.earth.position_m,
            self.moon.velocity_mps - self.earth.velocity_mps,
        )
