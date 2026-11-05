"""Relativistic BCRS/GCRS/LCRS spatial transformations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.constants import C2
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.ephemerides.ephemeris_provider import BodyState, Ephemeris
from lunarops.classes.frames.coordinate_scaling import (
    EARTH_EXTERNAL_POTENTIAL_BODIES,
    GM_BY_BODY,
    L_B_MINUS_L_G,
    MOON_EXTERNAL_POTENTIAL_BODIES,
    LunarRelativisticScale,
)
from lunarops.classes.time import Epoch, require_tdb_epoch


class RelativisticFrameTransform:
    def __init__(self, ephemeris: Ephemeris, lunar_scale: LunarRelativisticScale) -> None:
        self.ephemeris = ephemeris
        self.lunar_scale = lunar_scale
        self._states: dict[tuple, np.ndarray] = {}
        self._potentials: dict[tuple, float] = {}

    def begin_observation(self) -> None:
        self._states.clear()
        self._potentials.clear()

    def body_state_matrix_bcrs(self, bodies, epoch: Epoch) -> np.ndarray:
        names = tuple(body_name(body) for body in bodies)
        missing = tuple(name for name in names if (name, epoch.jd1, epoch.jd2) not in self._states)
        if missing:
            values = self.ephemeris.body_state_matrix_bcrs(missing, epoch)
            for name, value in zip(missing, values):
                self._states[name, epoch.jd1, epoch.jd2] = value
        return np.asarray([self._states[name, epoch.jd1, epoch.jd2] for name in names]).reshape(len(names), 6)

    def body_state(self, body: str, epoch: Epoch) -> BodyState:
        state = self.body_state_matrix_bcrs((body,), epoch)[0]
        return BodyState(state[:3], state[3:])

    def external_gravitational_potential_m2_s2(
        self,
        center_body_name: str,
        epoch_tdb: Epoch,
        perturbing_body_names: Iterable[str],
    ) -> float:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        center_name = body_name(center_body_name)
        names = tuple(dict.fromkeys(body_name(name) for name in perturbing_body_names))
        if center_name in names:
            raise ValueError("center_body_name must not also appear in perturbing_body_names.")
        key = (center_name, names, epoch.jd1, epoch.jd2)
        if key in self._potentials:
            return self._potentials[key]
        positions = self.body_state_matrix_bcrs((center_name, *names), epoch)[:, :3]
        center_position = positions[0]
        total = 0.0
        for name, position in zip(names, positions[1:], strict=True):
            try:
                gm = GM_BY_BODY[name]
            except KeyError:
                raise KeyError(f"No gravitational parameter configured for body {name!r}.") from None
            displacement = position - center_position
            distance = float(np.linalg.norm(displacement))
            if distance <= 0.0:
                raise RuntimeError(f"Ephemeris returned coincident positions for {center_name!r} and {name!r}.")
            total += gm / distance
        self._potentials[key] = float(total)
        return self._potentials[key]

    def _to_bcrs(self, position, epoch, center, perturbing, scale_difference):
        epoch = require_tdb_epoch(epoch, name="epoch_tdb")
        state = self.body_state(center, epoch)
        position = np.asarray(position, dtype=float)
        potential = self.external_gravitational_potential_m2_s2(center, epoch, perturbing)
        scale = 1.0 - scale_difference - potential / C2
        tdb_position = scale * position - 0.5 * (np.dot(state.velocity_mps, position) / C2) * state.velocity_mps
        return state.position_m + tdb_position

    def _from_bcrs(self, position, epoch, center, perturbing, scale_difference, *, vector=False):
        epoch = require_tdb_epoch(epoch, name="epoch_tdb")
        state = self.body_state(center, epoch)
        relative = np.asarray(position, dtype=float) if vector else np.asarray(position, dtype=float) - state.position_m
        potential = self.external_gravitational_potential_m2_s2(center, epoch, perturbing)
        scale = 1.0 + scale_difference + potential / C2
        return scale * relative + 0.5 * (np.dot(state.velocity_mps, relative) / C2) * state.velocity_mps

    def gcrs2bcrs(self, position_gcrs_m, epoch_tdb):
        return self._to_bcrs(position_gcrs_m, epoch_tdb, "EARTH", EARTH_EXTERNAL_POTENTIAL_BODIES, L_B_MINUS_L_G)

    def bcrs2gcrs(self, position_bcrs_m, epoch_tdb):
        return self._from_bcrs(position_bcrs_m, epoch_tdb, "EARTH", EARTH_EXTERNAL_POTENTIAL_BODIES, L_B_MINUS_L_G)

    def bcrs_vector2gcrs(self, vector_bcrs_m, epoch_tdb):
        return self._from_bcrs(vector_bcrs_m, epoch_tdb, "EARTH", EARTH_EXTERNAL_POTENTIAL_BODIES, L_B_MINUS_L_G, vector=True)

    def lcrs2bcrs(self, position_lcrs_m, epoch_tdb):
        return self._to_bcrs(position_lcrs_m, epoch_tdb, "MOON", MOON_EXTERNAL_POTENTIAL_BODIES, self.lunar_scale.l_b_minus_l_l)

    def bcrs2lcrs(self, position_bcrs_m, epoch_tdb):
        return self._from_bcrs(position_bcrs_m, epoch_tdb, "MOON", MOON_EXTERNAL_POTENTIAL_BODIES, self.lunar_scale.l_b_minus_l_l)

    def bcrs_vector2lcrs(self, vector_bcrs_m, epoch_tdb):
        return self._from_bcrs(vector_bcrs_m, epoch_tdb, "MOON", MOON_EXTERNAL_POTENTIAL_BODIES, self.lunar_scale.l_b_minus_l_l, vector=True)

    def lcrs2gcrs(self, position_lcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.bcrs2gcrs(self.lcrs2bcrs(position_lcrs_m, epoch_tdb), epoch_tdb)

    def gcrs2lcrs(self, position_gcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.bcrs2lcrs(self.gcrs2bcrs(position_gcrs_m, epoch_tdb), epoch_tdb)


__all__ = ["RelativisticFrameTransform"]
