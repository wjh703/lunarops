"""Relativistic BCRS/GCRS/LCRS spatial transformations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.constants import C2
from lunarops.classes.ephemerides.ephemeris_provider import Ephemeris
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.frames.coordinate_scaling import (
    EARTH_EXTERNAL_POTENTIAL_BODIES,
    GM_BY_BODY,
    L_B_MINUS_L_G,
    MOON_EXTERNAL_POTENTIAL_BODIES,
)
from lunarops.classes.frames.coordinate_scaling import LunarRelativisticScale
from lunarops.classes.time import Epoch, require_tdb_epoch


class RelativisticFrameTransform:
    def __init__(self, ephemeris: Ephemeris, lunar_scale: LunarRelativisticScale) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        self.ephemeris = ephemeris
        self.lunar_scale = lunar_scale

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
        positions = self.ephemeris.body_state_matrix_bcrs((center_name, *names), epoch)[:, :3]
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
        return float(total)

    def _to_bcrs(self, position, epoch, center, perturbing, scale_difference):
        epoch = require_tdb_epoch(epoch, name="epoch_tdb")
        state = self.ephemeris.body_state_bcrs(center, epoch)
        position = np.asarray(position, dtype=float)
        potential = self.external_gravitational_potential_m2_s2(center, epoch, perturbing)
        scale = 1.0 - scale_difference - potential / C2
        tdb_position = scale * position - 0.5 * (np.dot(state.velocity_mps, position) / C2) * state.velocity_mps
        return state.position_m + tdb_position

    def _from_bcrs(self, position, epoch, center, perturbing, scale_difference, *, vector=False):
        epoch = require_tdb_epoch(epoch, name="epoch_tdb")
        state = self.ephemeris.body_state_bcrs(center, epoch)
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
