"""Ephemeris interface and body-state values."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from lunarops.classes.time import Epoch


@dataclass(frozen=True, slots=True, eq=False)
class BodyState:
    """BCRS position and velocity of one body relative to the solar-system barycentre.

    Attributes
    ----------
    position_m : ndarray, shape (3,)
        Barycentric position on ICRF axes, in metres, in the ephemeris's TDB-compatible coordinates.
    velocity_mps : ndarray, shape (3,)
        Barycentric velocity on ICRF axes, in metres per TDB second.
    """

    position_m: np.ndarray
    velocity_mps: np.ndarray


class Ephemeris(Protocol):
    """Interface of an ephemeris giving barycentric BCRS body states at TDB epochs.

    Positions are in metres, velocities in m/s and accelerations in m/s^2, relative to the solar-system
    barycentre on ICRF axes. All epochs must be on the TDB scale.
    """

    @property
    def source_path(self) -> Path | None:
        """Path of the ephemeris data, or ``None`` if it has no file source."""
        ...

    def body_state_bcrs(self, body: str, epoch_tdb: Epoch) -> BodyState:
        """Return the barycentric BCRS state of ``body`` at a TDB epoch."""
        ...

    def body_position_bcrs(self, body: str, epoch_tdb: Epoch) -> np.ndarray:
        """Return the barycentric BCRS position of ``body`` at a TDB epoch, shape (3,), in metres."""
        ...

    def body_state_matrix_bcrs(self, bodies: Sequence[str], epoch_tdb: Epoch) -> np.ndarray:
        """Return the barycentric states of several bodies at one TDB epoch.

        Parameters
        ----------
        bodies : sequence of str
            Body identifiers (see ``canonical_body_id``).
        epoch_tdb : Epoch
            Epoch on the TDB scale.

        Returns
        -------
        ndarray, shape (n_bodies, 6)
            Rows (x, y, z, vx, vy, vz) on ICRF axes, in m and m/s, in the order of ``bodies``.
        """
        ...

    def body_state_matrices_bcrs(self, bodies: Sequence[str], epochs: Sequence[Epoch]) -> np.ndarray:
        """Return the barycentric states of several bodies at several TDB epochs.

        Parameters
        ----------
        bodies : sequence of str
            Body identifiers (see ``canonical_body_id``).
        epochs : sequence of Epoch
            Epochs on the TDB scale.

        Returns
        -------
        ndarray, shape (n_epochs, n_bodies, 6)
            States (x, y, z, vx, vy, vz) on ICRF axes, in m and m/s.
        """
        ...

    def body_accelerations_bcrs(self, body: str, epochs: Sequence[Epoch]) -> np.ndarray:
        """Return the barycentric acceleration of one body at several TDB epochs.

        Parameters
        ----------
        body : str
            Body identifier (see ``canonical_body_id``).
        epochs : sequence of Epoch
            Epochs on the TDB scale.

        Returns
        -------
        ndarray, shape (n_epochs, 3)
            Second TDB derivative of the barycentric position on ICRF axes, in m/s^2.
        """
        ...

    def close(self) -> None:
        """Release the resources held by the ephemeris."""
        ...


__all__ = ["BodyState", "Ephemeris"]
