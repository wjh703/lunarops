"""Core ephemeris interfaces and immutable query/result objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import numpy as np

from lunarops.base.array_validation import finite_array
from lunarops.classes.relativistic import LunarRelativisticScaleConvention
from lunarops.classes.time import Epoch, TimeScale

from .body_ids import body_name


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

    def body_states_bcrs(self, body_names: Sequence[str], epoch_tdb: Epoch) -> dict[str, BodyState]:
        """Return several body states at one epoch.

        Providers may override this to use a native batch/prefetch facility.
        The default preserves compatibility for simple providers.
        """
        names = tuple(body_name(name) for name in body_names)
        return {name: self.body_state_bcrs(name, epoch_tdb) for name in names}

    def body_state_vectors_bcrs(self, body_names: Sequence[str], epoch_tdb: Epoch) -> np.ndarray:
        """Return BCRS states as one contiguous ``(N,6)`` SI array."""
        names = tuple(body_name(name) for name in body_names)
        states = self.body_states_bcrs(names, epoch_tdb)
        return np.ascontiguousarray(
            [np.concatenate((states[name].position_m, states[name].velocity_mps)) for name in names],
            dtype=float,
        )

    def body_acceleration_bcrs(self, body_name: str, epoch_tdb: Epoch) -> np.ndarray:
        """Return the trajectory's second TDB derivative in m/s^2, relative to SSB.

        This is not the auxiliary Newtonian acceleration used inside EIH.
        Providers without derivative support fail explicitly.
        """
        require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        raise NotImplementedError("This ephemeris does not provide acceleration.")

    @abstractmethod
    def pa2lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        """Return the passive rotation from lunar PA axes to LCRS axes at TDB."""

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
