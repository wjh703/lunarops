"""Night sessions: the normal points of one key (variance component or station) observed in one night.

Normal points of one station night share calibration and atmospheric errors. Each session ``n`` (the normal points
of one variance component without a gap longer than the session gap) gets a random offset ``b_n ~ N(0, tau_k^2)``,
with ``tau_k`` estimated per variance component. This is the compound-symmetric covariance
``C_n = D_n + tau^2 1 1^T`` of a session; the offsets are eliminated from the normal equations in closed form
(:mod:`lunarops.estimation.lsq.random_effects`). ``tau^2`` is estimated with the white variance components
(:func:`lunarops.estimation.stochastic.vce.session_variance_step`). The station nights of the Earth-orientation
corrections are built the same way, keyed by station.
"""

from __future__ import annotations

import numpy as np

from lunarops.estimation.lsq.blocks import RowBlocks

#: Lower bound of an estimated session sigma, in metres; keeps ``1 / tau^2`` finite.
MINIMUM_SESSION_SIGMA_M = 1.0e-5
#: Session gap used for the clustering diagnostic when the session model is not enabled, in hours.
DEFAULT_SESSION_GAP_HOURS = 4.0


def night_sessions(keys: np.ndarray, epoch_jd: np.ndarray, maximum_gap_hours: float) -> RowBlocks:
    """Split the rows into night sessions: same key, no gap longer than ``maximum_gap_hours``.

    Parameters
    ----------
    keys : ndarray of object, shape (n,)
        Variance component or station per row.
    epoch_jd : ndarray, shape (n,)
        Transmit epoch per row, as a Julian date.
    maximum_gap_hours : float
        Largest gap between consecutive normal points of one session, in hours.
    """
    return RowBlocks.build(keys, np.asarray(epoch_jd, dtype=float) * 24.0, maximum_gap_hours)


__all__ = ["DEFAULT_SESSION_GAP_HOURS", "MINIMUM_SESSION_SIGMA_M", "night_sessions"]
