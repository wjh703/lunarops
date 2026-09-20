"""Explicit physical and integration state objects for Earth--Moon dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.base.array_validation import finite_array
from lunarops.classes.ephemerides import BodyState, require_tdb_epoch
from lunarops.classes.time import Epoch


@dataclass(frozen=True, slots=True)
class CartesianState:
    position_m: np.ndarray
    velocity_mps: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self, "position_m", finite_array(self.position_m, size=3, name="position_m", copy=True, readonly=True)
        )
        object.__setattr__(
            self, "velocity_mps", finite_array(self.velocity_mps, size=3, name="velocity_mps", copy=True, readonly=True)
        )

    def as_body_state(self) -> BodyState:
        return BodyState(self.position_m, self.velocity_mps)


@dataclass(frozen=True, slots=True)
class EarthMoonIntegrationState:
    """The 12 variables integrated by the translational propagator.

    Ordering is ``[r_EMB, v_EMB, r_M/E, v_M/E]`` in BCRS SI units.
    """

    epoch_tdb: Epoch
    emb: CartesianState
    moon_relative: CartesianState

    def __post_init__(self):
        require_tdb_epoch(self.epoch_tdb)

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            (self.emb.position_m, self.emb.velocity_mps, self.moon_relative.position_m, self.moon_relative.velocity_mps)
        )

    @classmethod
    def from_vector(cls, epoch_tdb, vector):
        y = finite_array(vector, shape=(12,), name="integration state")
        return cls(epoch_tdb, CartesianState(y[:3], y[3:6]), CartesianState(y[6:9], y[9:12]))

    def to_barycentric(self, earth_mu_m3_s2: float, moon_mu_m3_s2: float):
        if not np.all(np.isfinite((earth_mu_m3_s2, moon_mu_m3_s2))) or earth_mu_m3_s2 <= 0 or moon_mu_m3_s2 <= 0:
            raise ValueError("Earth and Moon gravitational parameters must be positive and finite.")
        total = float(earth_mu_m3_s2) + float(moon_mu_m3_s2)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Earth and Moon gravitational parameters must have a positive finite sum.")
        q_e, q_m = earth_mu_m3_s2 / total, moon_mu_m3_s2 / total
        r, v = self.moon_relative.position_m, self.moon_relative.velocity_mps
        er = self.emb.position_m - q_m * r
        ev = self.emb.velocity_mps - q_m * v
        mr = self.emb.position_m + q_e * r
        mv = self.emb.velocity_mps + q_e * v
        return EarthMoonBarycentricState(self.epoch_tdb, BodyState(er, ev), BodyState(mr, mv))


@dataclass(frozen=True, slots=True)
class EarthMoonBarycentricState:
    epoch_tdb: Epoch
    earth: BodyState
    moon: BodyState

    def __post_init__(self):
        require_tdb_epoch(self.epoch_tdb)
        if not isinstance(self.earth, BodyState) or not isinstance(self.moon, BodyState):
            raise TypeError("earth and moon must be BodyState objects")

    @property
    def relative(self) -> BodyState:
        return BodyState(self.moon.position_m - self.earth.position_m, self.moon.velocity_mps - self.earth.velocity_mps)

    def to_integration(self, earth_mu_m3_s2: float, moon_mu_m3_s2: float):
        if not np.all(np.isfinite((earth_mu_m3_s2, moon_mu_m3_s2))) or earth_mu_m3_s2 <= 0 or moon_mu_m3_s2 <= 0:
            raise ValueError("Earth and Moon gravitational parameters must be positive and finite.")
        total = float(earth_mu_m3_s2) + float(moon_mu_m3_s2)
        q_e, q_m = earth_mu_m3_s2 / total, moon_mu_m3_s2 / total
        emb = CartesianState(
            q_e * self.earth.position_m + q_m * self.moon.position_m,
            q_e * self.earth.velocity_mps + q_m * self.moon.velocity_mps,
        )
        rel = CartesianState(
            self.moon.position_m - self.earth.position_m, self.moon.velocity_mps - self.earth.velocity_mps
        )
        return EarthMoonIntegrationState(self.epoch_tdb, emb, rel)
