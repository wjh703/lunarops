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
    """First post-Newtonian spatial transformations between BCRS and the GCRS or LCRS.

    A local position ``x`` (GCRS or LCRS, relative to the geocentre or selenocentre) and its barycentric
    position ``X`` are related by

        X = X_c + (1 - L - U/c^2) x - (V . x) V / (2 c^2),

    where ``X_c`` and ``V`` are the barycentric position and velocity of the centre body (Earth or Moon),
    ``U`` is the Newtonian potential of the external bodies at the centre and ``L`` is the constant scale
    difference: L_B - L_G for the GCRS, and ``lunar_scale.l_b_minus_l_l`` for the LCRS. The inverse uses the
    first-order inverse coefficients. BCRS coordinates are TDB-compatible, GCRS coordinates TT-compatible.
    All positions are in metres on ICRF axes, and all epochs are on TDB. Body states and potentials are
    cached until ``begin_observation`` is called.

    Parameters
    ----------
    ephemeris : Ephemeris
        Source of barycentric body states.
    lunar_scale : LunarRelativisticScale
        BCRS<->LCRS coordinate scaling.

    References
    ----------
    .. [1] Petit, G., & Luzum, B. (Eds.) (2010). IERS Conventions (2010), IERS Technical Note 36,
       Eqs. 11.18-11.19.
    """

    def __init__(self, ephemeris: Ephemeris, lunar_scale: LunarRelativisticScale) -> None:
        self.ephemeris = ephemeris
        self.lunar_scale = lunar_scale
        self._states: dict[tuple, np.ndarray] = {}
        self._potentials: dict[tuple, float] = {}

    def begin_observation(self) -> None:
        """Clear the cached body states and external potentials."""
        self._states.clear()
        self._potentials.clear()

    def body_state_matrix_bcrs(self, bodies, epoch: Epoch) -> np.ndarray:
        """Return cached barycentric states of several bodies, shape (n_bodies, 6), in m and m/s."""
        names = tuple(body_name(body) for body in bodies)
        missing = tuple(name for name in names if (name, epoch.jd1, epoch.jd2) not in self._states)
        if missing:
            values = self.ephemeris.body_state_matrix_bcrs(missing, epoch)
            for name, value in zip(missing, values):
                self._states[name, epoch.jd1, epoch.jd2] = value
        return np.asarray([self._states[name, epoch.jd1, epoch.jd2] for name in names]).reshape(len(names), 6)

    def body_state(self, body: str, epoch: Epoch) -> BodyState:
        """Return the cached barycentric BCRS state of one body."""
        state = self.body_state_matrix_bcrs((body,), epoch)[0]
        return BodyState(state[:3], state[3:])

    def external_gravitational_potential_m2_s2(
        self,
        center_body_name: str,
        epoch_tdb: Epoch,
        perturbing_body_names: Iterable[str],
    ) -> float:
        """Return the Newtonian point-mass potential of external bodies at a centre body.

        Parameters
        ----------
        center_body_name : str
            Body at which the potential is evaluated.
        epoch_tdb : Epoch
            Epoch on the TDB scale.
        perturbing_body_names : iterable of str
            Bodies contributing GM / distance; each needs an entry in ``GM_BY_BODY``.

        Returns
        -------
        float
            Potential sum(GM_i / r_i), positive, in m^2/s^2.

        Raises
        ------
        ValueError
            If the centre body is among the perturbing bodies or the epoch is not on TDB.
        KeyError
            If a perturbing body has no gravitational parameter.
        RuntimeError
            If a perturbing body coincides with the centre body.
        """
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
        """Return the barycentric BCRS position, shape (3,), of a geocentric GCRS position, in metres."""
        return self._to_bcrs(position_gcrs_m, epoch_tdb, "EARTH", EARTH_EXTERNAL_POTENTIAL_BODIES, L_B_MINUS_L_G)

    def bcrs2gcrs(self, position_bcrs_m, epoch_tdb):
        """Return the geocentric GCRS position, shape (3,), of a barycentric BCRS position, in metres."""
        return self._from_bcrs(position_bcrs_m, epoch_tdb, "EARTH", EARTH_EXTERNAL_POTENTIAL_BODIES, L_B_MINUS_L_G)

    def bcrs_vector2gcrs(self, vector_bcrs_m, epoch_tdb):
        """Return a BCRS difference vector, shape (3,), in metres, scaled to GCRS without the origin shift."""
        return self._from_bcrs(vector_bcrs_m, epoch_tdb, "EARTH", EARTH_EXTERNAL_POTENTIAL_BODIES, L_B_MINUS_L_G, vector=True)

    def lcrs2bcrs(self, position_lcrs_m, epoch_tdb):
        """Return the barycentric BCRS position, shape (3,), of a selenocentric LCRS position, in metres."""
        return self._to_bcrs(position_lcrs_m, epoch_tdb, "MOON", MOON_EXTERNAL_POTENTIAL_BODIES, self.lunar_scale.l_b_minus_l_l)

    def bcrs2lcrs(self, position_bcrs_m, epoch_tdb):
        """Return the selenocentric LCRS position, shape (3,), of a barycentric BCRS position, in metres."""
        return self._from_bcrs(position_bcrs_m, epoch_tdb, "MOON", MOON_EXTERNAL_POTENTIAL_BODIES, self.lunar_scale.l_b_minus_l_l)

    def bcrs_vector2lcrs(self, vector_bcrs_m, epoch_tdb):
        """Return a BCRS difference vector, shape (3,), in metres, scaled to LCRS without the origin shift."""
        return self._from_bcrs(vector_bcrs_m, epoch_tdb, "MOON", MOON_EXTERNAL_POTENTIAL_BODIES, self.lunar_scale.l_b_minus_l_l, vector=True)

    def lcrs2gcrs(self, position_lcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        """Return the GCRS position, shape (3,), of an LCRS position through the BCRS, in metres."""
        return self.bcrs2gcrs(self.lcrs2bcrs(position_lcrs_m, epoch_tdb), epoch_tdb)

    def gcrs2lcrs(self, position_gcrs_m: ArrayLike, epoch_tdb: Epoch) -> np.ndarray:
        """Return the LCRS position, shape (3,), of a GCRS position through the BCRS, in metres."""
        return self.bcrs2lcrs(self.gcrs2bcrs(position_gcrs_m, epoch_tdb), epoch_tdb)


__all__ = ["RelativisticFrameTransform"]
