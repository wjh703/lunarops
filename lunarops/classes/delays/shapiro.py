"""Gravitational (Shapiro) path delay of a light signal in the BCRS."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np

from lunarops.base.constants import C2
from lunarops.classes.frames.coordinate_scaling import GM_BY_BODY
from lunarops.classes.time import Epoch
from lunarops.config.schema import ConfigSchema

# IERS Conventions 2010 S11.2 (Eq. 11.17) one-way path delay used for LLR.
# Pavlov, Williams & Suvorkin (2016) S4 explicitly include Sun, Earth, Moon,
# Jupiter and Saturn as point masses contributing to the gravitational delay.
_DEFAULT_LLR_SHAPIRO_BODIES = (
    "SUN",
    "EARTH",
    "MOON",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
)


class Iers2010ShapiroDelay:
    """IERS 2010 gravitational (Shapiro) path delay of a one-way light-time leg.

    For each body B among the Sun, Earth, Moon and the Jupiter and Saturn barycentres, the delay in
    length units is

        2 GM_B / c^2 ln[(r1 + r2 + rho) / (r1 + r2 - rho)],

    where r1 and r2 are the distances of the transmitter and receiver from B and rho is the
    transmitter-receiver distance (PPN gamma = 1). The body positions are taken at a single TDB epoch.

    Parameters
    ----------
    body_states : callable
        Function of a tuple of body names and a TDB ``Epoch`` returning their BCRS states, shape
        (n_bodies, 6), in m and m/s.

    References
    ----------
    .. [1] Petit, G., & Luzum, B. (Eds.) (2010). IERS Conventions (2010), IERS Technical Note 36,
       Sec. 11.2, Eq. 11.17.
    .. [2] Pavlov, D. A., Williams, J. G., & Suvorkin, V. V. (2016). Determining parameters of Moon's
       orbital and rotational motion from LLR observations using GRAIL and IERS-recommended models.
       Celestial Mechanics and Dynamical Astronomy, 126, 61-88, Sec. 4.
    """

    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(fields=(), type_name="iersShapiro")

    @classmethod
    def from_config(cls, cfg, ctx, *, frames):
        """Create the model using the cached BCRS body states of ``frames.relativistic_transform``."""
        return cls(body_states=frames.relativistic_transform.body_state_matrix_bcrs)

    def __init__(self, body_states: Callable[[tuple[str, ...], Epoch], np.ndarray]) -> None:
        self.body_states = body_states

    def path_delay_m(
        self,
        transmitter_bcrs_m: np.ndarray,
        receiver_bcrs_m: np.ndarray,
        epoch_tdb: Epoch,
    ) -> float:
        """Return the gravitational path delay of one light-time leg.

        Parameters
        ----------
        transmitter_bcrs_m : ndarray, shape (3,)
            BCRS position of the transmitter at emission, in metres.
        receiver_bcrs_m : ndarray, shape (3,)
            BCRS position of the receiver at reception, in metres.
        epoch_tdb : Epoch
            TDB epoch at which the deflecting-body positions are evaluated.

        Returns
        -------
        float
            Path delay in metres (positive; divide by c for seconds). Bodies for which the logarithm
            is undefined are skipped.
        """
        x1 = transmitter_bcrs_m
        x2 = receiver_bcrs_m
        rho = float(np.linalg.norm(x2 - x1))

        total = 0.0
        positions = self.body_states(_DEFAULT_LLR_SHAPIRO_BODIES, epoch_tdb)[:, :3]
        for body, xb in zip(_DEFAULT_LLR_SHAPIRO_BODIES, positions, strict=True):
            r1 = float(np.linalg.norm(x1 - xb))
            r2 = float(np.linalg.norm(x2 - xb))
            denom = r1 + r2 - rho
            numer = r1 + r2 + rho
            if denom <= 0.0 or numer <= 0.0:
                continue
            total += 2.0 * GM_BY_BODY[body] / C2 * np.log(numer / denom)
        return float(total)
