"""Core ephemeris interfaces and immutable query/result objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import numpy as np

from lunarops.base.array_validation import finite_array
from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.relativistic import LunarRelativisticScaleConvention


def require_tdb_epoch(epoch: Epoch, *, name: str = "epoch") -> Epoch:
    if not isinstance(epoch, Epoch):
        raise TypeError(f"{name} must be an Epoch.")
    return epoch.require_scale(TimeScale.TDB, name=name)


class LongitudeLibrationCorrectionType(StrEnum):
    NONE = "none"
    INPOP21A = "inpop21a"


@dataclass(frozen=True, slots=True, eq=False)
class BodyState:
    """BCRS position and velocity of one body relative to the SSB."""

    position_m: np.ndarray
    velocity_mps: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_m",
            finite_array(
                self.position_m,
                size=3,
                name="position_m",
                copy=True,
                readonly=True,
            ),
        )
        object.__setattr__(
            self,
            "velocity_mps",
            finite_array(
                self.velocity_mps,
                size=3,
                name="velocity_mps",
                copy=True,
                readonly=True,
            ),
        )


class Ephemeris(ABC):
    """Abstract ephemeris used by the LLR physical models."""

    @property
    @abstractmethod
    def source_file_path(self) -> Path | None: ...

    @abstractmethod
    def body_state_bcrs(self, body_name: str, epoch_tdb: Epoch) -> BodyState:
        """Return a body's SSB-relative BCRS state at a TDB epoch."""

    def body_position_bcrs(self, body_name: str, epoch_tdb: Epoch) -> np.ndarray:
        return np.array(
            self.body_state_bcrs(body_name, epoch_tdb).position_m,
            copy=True,
        )

    @abstractmethod
    def pa2lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        """Return the passive rotation from lunar PA axes to LCRS axes at TDB."""

    def geocentric_tdb_minus_tt_s(self, epoch_tdb: Epoch) -> float | None:
        """Return geocentric TDB-TT in seconds at a TDB epoch."""
        require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        return None

    @property
    def longitude_libration_correction_type(
        self,
    ) -> LongitudeLibrationCorrectionType:
        return LongitudeLibrationCorrectionType.NONE

    def longitude_libration_correction_rad(self, epoch_tdb: Epoch) -> float:
        require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        return 0.0

    @property
    def l_b_minus_l_l(self) -> float:
        return 0.0

    @property
    def lunar_relativistic_scale_convention(
        self,
    ) -> LunarRelativisticScaleConvention:
        return LunarRelativisticScaleConvention.ALREADY_SCALED

    def close(self) -> None:
        """Release resources; the default implementation owns none."""
        return

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "BodyState",
    "Ephemeris",
    "LongitudeLibrationCorrectionType",
    "require_tdb_epoch",
]
