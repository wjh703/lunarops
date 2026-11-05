"""Core displacement interfaces and immutable evaluation inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lunarops.classes.time import Epoch


@dataclass(frozen=True, slots=True, eq=False)
class StationDisplacementInput:
    """Evaluation input of a station displacement model.

    Attributes
    ----------
    reference_position_itrf_m : ndarray, shape (3,)
        Undisplaced geocentric station position (X, Y, Z) in the ITRF, in metres.
    epoch_utc : Epoch
        Evaluation epoch on the UTC scale.
    station_id : str or None
        Station identifier, used by models with per-station coefficients.
    """

    reference_position_itrf_m: np.ndarray
    epoch_utc: Epoch
    station_id: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ReflectorDisplacementInput:
    """Evaluation input of a lunar-reflector displacement model.

    Attributes
    ----------
    reference_position_lcrs_m : ndarray, shape (3,)
        Undisplaced selenocentric reflector position on LCRS (ICRF-aligned) axes, in metres.
    epoch_tdb : Epoch
        Evaluation epoch on the TDB scale.
    """

    reference_position_lcrs_m: np.ndarray
    epoch_tdb: Epoch


class StationDisplacement(Protocol):
    """Interface of a station displacement model."""

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        """Return the station displacement (X, Y, Z) in the ITRF, shape (3,), in metres."""
        ...


class CompositeStationDisplacement:
    """Sum of several station displacement models.

    Parameters
    ----------
    components : sequence of StationDisplacement
        Models whose displacements are added.
    """

    def __init__(self, components: Sequence[StationDisplacement]) -> None:
        self.components = tuple(components)

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        """Return the summed displacement (X, Y, Z) in the ITRF, shape (3,), in metres."""
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
