"""Relativistic BCRS/GCRS/LCRS spatial transformations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from lunarops.base.array_validation import vector3
from lunarops.base.constants import C2
from lunarops.classes.time import Epoch
from lunarops.classes.ephemerides import Ephemeris, require_tdb_epoch
from lunarops.classes.relativistic.constants import (
    EARTH_EXTERNAL_POTENTIAL_BODIES,
    GM_BY_BODY,
    L_B_MINUS_L_G,
    MOON_EXTERNAL_POTENTIAL_BODIES,
)


class RelativisticFrameTransform:
    def __init__(self, ephemeris: Ephemeris) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        self.ephemeris = ephemeris

    @staticmethod
    def _normalize_body_name(value: str, *, parameter_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{parameter_name} must contain body-name strings.")
        name = value.strip().upper()
        if not name:
            raise ValueError(f"{parameter_name} must not contain empty body names.")
        return name

    def external_gravitational_potential_m2_s2(
        self,
        center_body_name: str,
        epoch_tdb: Epoch,
        perturbing_body_names: Iterable[str],
    ) -> float:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        center_name = self._normalize_body_name(
            center_body_name,
            parameter_name="center_body_name",
        )
        raw_names = tuple(perturbing_body_names)
        names = tuple(
            dict.fromkeys(self._normalize_body_name(name, parameter_name="perturbing_body_names") for name in raw_names)
        )
        if center_name in names:
            raise ValueError("center_body_name must not also appear in perturbing_body_names.")
        center_position = self.ephemeris.body_position_bcrs(center_name, epoch)
        total = 0.0
        for body_name in names:
            try:
                gm = GM_BY_BODY[body_name]
            except KeyError:
                raise KeyError(f"No gravitational parameter configured for body {body_name!r}.") from None
            displacement = self.ephemeris.body_position_bcrs(body_name, epoch) - center_position
            distance = float(np.linalg.norm(displacement))
            if distance <= 0.0:
                raise RuntimeError(f"Ephemeris returned coincident positions for {center_name!r} and {body_name!r}.")
            total += gm / distance
        return float(total)

    def gcrs2bcrs(self, position_gcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        earth = self.ephemeris.body_state_bcrs("EARTH", epoch)
        position = vector3(position_gcrs_m, name="position_gcrs_m")
        potential = self.external_gravitational_potential_m2_s2(
            "EARTH",
            epoch,
            EARTH_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 - L_B_MINUS_L_G - potential / C2
        tdb_position = scale * position - 0.5 * (np.dot(earth.velocity_mps, position) / C2) * earth.velocity_mps
        return earth.position_m + tdb_position

    def bcrs2gcrs(self, position_bcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        earth = self.ephemeris.body_state_bcrs("EARTH", epoch)
        relative = vector3(position_bcrs_m, name="position_bcrs_m") - earth.position_m
        potential = self.external_gravitational_potential_m2_s2(
            "EARTH",
            epoch,
            EARTH_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 + L_B_MINUS_L_G + potential / C2
        return scale * relative + 0.5 * (np.dot(earth.velocity_mps, relative) / C2) * earth.velocity_mps

    def bcrs_vector2gcrs(self, vector_bcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        """Transform a BCRS displacement vector using the epoch's Earth frame.

        Unlike :meth:`bcrs2gcrs`, this operation does not subtract the Earth's
        barycentric position.  It is needed when the two endpoints of a light
        path are evaluated at different event epochs.
        """
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        earth = self.ephemeris.body_state_bcrs("EARTH", epoch)
        vector = vector3(vector_bcrs_m, name="vector_bcrs_m")
        potential = self.external_gravitational_potential_m2_s2(
            "EARTH",
            epoch,
            EARTH_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 + L_B_MINUS_L_G + potential / C2
        return scale * vector + 0.5 * (np.dot(earth.velocity_mps, vector) / C2) * earth.velocity_mps

    def lcrs2bcrs(self, position_lcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        moon = self.ephemeris.body_state_bcrs("MOON", epoch)
        position = vector3(position_lcrs_m, name="position_lcrs_m")
        potential = self.external_gravitational_potential_m2_s2(
            "MOON",
            epoch,
            MOON_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 - self.ephemeris.l_b_minus_l_l - potential / C2
        tdb_position = scale * position - 0.5 * (np.dot(moon.velocity_mps, position) / C2) * moon.velocity_mps
        return moon.position_m + tdb_position

    def bcrs2lcrs(self, position_bcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        moon = self.ephemeris.body_state_bcrs("MOON", epoch)
        relative = vector3(position_bcrs_m, name="position_bcrs_m") - moon.position_m
        potential = self.external_gravitational_potential_m2_s2(
            "MOON",
            epoch,
            MOON_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 + self.ephemeris.l_b_minus_l_l + potential / C2
        return scale * relative + 0.5 * (np.dot(moon.velocity_mps, relative) / C2) * moon.velocity_mps

    def bcrs_vector2lcrs(self, vector_bcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        """Transform a BCRS displacement vector into the Moon-centered frame."""
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        moon = self.ephemeris.body_state_bcrs("MOON", epoch)
        vector = vector3(vector_bcrs_m, name="vector_bcrs_m")
        potential = self.external_gravitational_potential_m2_s2(
            "MOON",
            epoch,
            MOON_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 + self.ephemeris.l_b_minus_l_l + potential / C2
        return scale * vector + 0.5 * (np.dot(moon.velocity_mps, vector) / C2) * moon.velocity_mps

    def lcrs2gcrs(self, position_lcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.bcrs2gcrs(self.lcrs2bcrs(position_lcrs_m, epoch_tdb), epoch_tdb)

    def gcrs2lcrs(self, position_gcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        return self.bcrs2lcrs(self.gcrs2bcrs(position_gcrs_m, epoch_tdb), epoch_tdb)


__all__ = ["RelativisticFrameTransform"]
