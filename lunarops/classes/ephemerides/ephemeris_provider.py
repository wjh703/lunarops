"""Body-state provider contracts and immutable state values."""

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
    """Provide one body's SSB-relative BCRS state at a TDB epoch."""

    @property
    def source_path(self) -> Path | None: ...

    def body_state_bcrs(self, body: str, epoch_tdb: Epoch) -> BodyState: ...

    def body_position_bcrs(self, body: str, epoch_tdb: Epoch) -> np.ndarray: ...

    def close(self) -> None: ...


@runtime_checkable
class BcrsAccelerationProvider(Protocol):
    """Provide the trajectory's second TDB derivative in BCRS."""

    def body_acceleration_bcrs(self, body: str, epoch_tdb: Epoch) -> np.ndarray: ...


def body_state_matrix(ephemeris: Ephemeris, bodies: Sequence[str], epoch_tdb: Epoch) -> np.ndarray:
    """Stack requested states as a ``(body_count, 6)`` SI array."""
    batch_provider = getattr(ephemeris, "body_state_matrix_bcrs", None)
    if callable(batch_provider):
        result = np.asarray(batch_provider(tuple(bodies), epoch_tdb), dtype=float)
        if result.shape != (len(bodies), 6) or not np.all(np.isfinite(result)):
            raise ValueError("Ephemeris returned an invalid body state matrix")
        return result
    return np.asarray(
        [
            np.concatenate((state.position_m, state.velocity_mps))
            for state in (ephemeris.body_state_bcrs(body, epoch_tdb) for body in bodies)
        ],
        dtype=float,
    ).reshape(len(bodies), 6)


def body_state_matrices(ephemeris: Ephemeris, bodies: Sequence[str], epochs: Sequence[Epoch]) -> np.ndarray:
    """Return requested BCRS states as ``(epoch, body, state)`` SI values."""
    epoch_array = tuple(epochs)
    names = tuple(bodies)
    batch_provider = getattr(ephemeris, "body_state_matrices_bcrs", None)
    if callable(batch_provider):
        result = np.asarray(batch_provider(names, epoch_array), dtype=float)
    else:
        result = np.asarray(
            [body_state_matrix(ephemeris, names, epoch) for epoch in epoch_array],
            dtype=float,
        )
        if not epoch_array:
            result = np.empty((0, len(names), 6), dtype=float)
    if result.shape != (len(epoch_array), len(names), 6) or not np.all(np.isfinite(result)):
        raise ValueError("Ephemeris returned an invalid epoch body state matrix")
    return result


def body_accelerations(ephemeris: BcrsAccelerationProvider, body: str, epochs: Sequence[Epoch]) -> np.ndarray:
    """Return one body's BCRS accelerations as ``(epoch, component)`` SI values."""
    epoch_array = tuple(epochs)
    batch_provider = getattr(ephemeris, "body_accelerations_bcrs", None)
    if callable(batch_provider):
        result = np.asarray(batch_provider(body, epoch_array), dtype=float)
    else:
        result = np.asarray([ephemeris.body_acceleration_bcrs(body, epoch) for epoch in epoch_array], dtype=float)
    if result.shape != (len(epoch_array), 3) or not np.all(np.isfinite(result)):
        raise ValueError("Ephemeris returned an invalid epoch acceleration matrix")
    return result


__all__ = [
    "BcrsAccelerationProvider",
    "BodyState",
    "Ephemeris",
    "body_accelerations",
    "body_state_matrices",
    "body_state_matrix",
]
