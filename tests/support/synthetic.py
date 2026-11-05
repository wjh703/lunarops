"""Small analytic stand-ins for kernels and data files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from lunarops.classes.ephemerides import BodyState, Ephemeris
from lunarops.classes.time import Epoch, TimeScale

J2000_TDB = Epoch(2451545.0, 0.0, TimeScale.TDB)


class LinearEphemeris(Ephemeris):
    """Bodies in uniform motion: ``position + velocity * (t - J2000)`` in BCRS metres and m/s."""

    def __init__(self, bodies: Mapping[str, tuple[Sequence[float], Sequence[float]]], *, source: str = "linear.eph"):
        self._bodies = {name: (np.asarray(p, dtype=float), np.asarray(v, dtype=float)) for name, (p, v) in bodies.items()}
        self._source = Path(source)
        self.closed = False

    @property
    def source_path(self) -> Path:
        return self._source

    def _state(self, body: str, epoch_tdb: Epoch) -> np.ndarray:
        epoch_tdb.require_scale(TimeScale.TDB)
        position, velocity = self._bodies[body.strip().upper()]
        return np.r_[position + velocity * J2000_TDB.seconds_until(epoch_tdb), velocity]

    def body_state_bcrs(self, body: str, epoch_tdb: Epoch) -> BodyState:
        state = self._state(body, epoch_tdb)
        return BodyState(state[:3], state[3:])

    def body_position_bcrs(self, body: str, epoch_tdb: Epoch) -> np.ndarray:
        return self._state(body, epoch_tdb)[:3]

    def body_state_matrix_bcrs(self, bodies: Sequence[str], epoch_tdb: Epoch) -> np.ndarray:
        return np.asarray([self._state(body, epoch_tdb) for body in bodies]).reshape(-1, 6)

    def body_state_matrices_bcrs(self, bodies: Sequence[str], epochs: Sequence[Epoch]) -> np.ndarray:
        return np.asarray([self.body_state_matrix_bcrs(bodies, epoch) for epoch in epochs])

    def body_accelerations_bcrs(self, body: str, epochs: Sequence[Epoch]) -> np.ndarray:
        return np.zeros((len(epochs), 3))

    def close(self) -> None:
        self.closed = True
