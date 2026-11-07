"""Consistent robust variance-component estimation with night-session correlation.

This module holds the pieces of the ``consistent`` estimation scheme of :class:`LlrAdjustmentSolver`:

Consistency factor
    With IGG-III or direct-rejection weights ``w(z)`` of the standardized residual ``z``, the robust variance
    estimate ``sum w v^2 / sum r`` of a Gaussian sample is biased low by ``kappa = E[w(z) z^2 | w(z) > 0]``
    (0.786 for IGG-III with k0 = 1.5, k1 = 6). Dividing by ``kappa`` gives the consistent M-scale (Huber's
    proposal 2) of the component.

Residual variance under robust weights
    A residual is standardized by its actual variance under the weights of its solution,
    ``Var(v_i) = s^2 sigma_i^2 (1 - f_i t_i) (1 + (1 - f_i) t_i)`` with ``t_i = q_i / (s sigma_i)^2`` and ``q_i`` the
    variance of the fitted value (``a_i N^-1 a_i^T``, plus the session term). This is the redundancy ``1 - t_i`` of a
    full-weight observation and the prediction variance ``1 + t_i`` of a rejected one, and the standardized residual
    is Baarda's w-statistic, which does not depend on the observation's own weight factor, so a weight cannot feed
    back on itself. The legacy standardization uses the redundancy of the weighted system, which tends to one as an
    observation is down-weighted: a rejected observation then looks better than it is, and a full-weight observation
    whose session partners are rejected (its residual absorbed by the session offset) looks worse. The same factor
    is the denominator of the consistent scale, ``s^2 = sum f v^2 / sigma^2 / (kappa sum rho)`` over the active
    observations, so observations absorbed by their session offset carry neither residual nor redundancy.

Night sessions
    Normal points of one station night share calibration and atmospheric errors. Each session ``n`` (the
    normal points of one variance component without a gap longer than the session gap) gets a random offset
    ``b_n ~ N(0, tau_k^2)``, with ``tau_k`` estimated per variance component. This is the compound-symmetric
    covariance ``C_n = D_n + tau^2 1 1^T`` of a session; the offsets are eliminated from the normal equations
    in closed form (:mod:`lunarops.estimation.random_effects`). ``tau^2`` is estimated with the white variance
    components (:func:`lunarops.estimation.stochastic_model.session_variance_step`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import erf, sqrt

import numpy as np

from lunarops.estimation.robust_weights import DIRECT_REJECTION_MODEL, robust_factors

#: Lower bound of an estimated session sigma, in metres; keeps ``1 / tau^2`` finite.
MINIMUM_SESSION_SIGMA_M = 1.0e-5
#: Session gap used for the clustering diagnostic when the session model is not enabled, in hours.
DEFAULT_SESSION_GAP_HOURS = 4.0


@lru_cache(maxsize=32)
def consistency_factor(model: str, k0: float, k1: float | None, weight_power: int = 1) -> float:
    """Return ``E[w(z)^k z^2 | w(z) > 0]`` for a standard normal ``z``, with ``k = weight_power``.

    ``k = 1`` is the consistency factor ``kappa`` of the white variance estimate. ``k = 2`` is that of the session
    variance estimate: the noise part of a session offset estimated with weights ``f p_i`` has the expectation
    ``E[w^2 z^2] sum p_i``, where the weights taken as inverse variances give ``sum f p_i``.

    Parameters
    ----------
    model : str
        ``igg3`` or ``directRejection``.
    k0, k1 : float
        Thresholds of the weight function.
    weight_power : int, default 1
        Power ``k`` of the weight.

    Returns
    -------
    float
        The consistency factor.
    """
    upper = float(k0 if model == DIRECT_REJECTION_MODEL else k1)
    z = np.linspace(0.0, upper, 400_001)
    density = np.exp(-0.5 * z * z) / sqrt(2.0 * np.pi)
    numerator = 2.0 * np.trapezoid(robust_factors(model, k0, k1, z) ** weight_power * z * z * density, z)
    return float(numerator / erf(upper / sqrt(2.0)))


def gaussian_tail_fractions(model: str, k0: float, k1: float | None) -> tuple[float, float]:
    """Return the fractions of a Gaussian sample that are down-weighted and rejected by the weight function."""
    beyond_k0 = 1.0 - erf(k0 / sqrt(2.0))
    if model == DIRECT_REJECTION_MODEL:
        return 0.0, beyond_k0
    beyond_k1 = 1.0 - erf(float(k1) / sqrt(2.0))
    return beyond_k0 - beyond_k1, beyond_k1


@dataclass(frozen=True, eq=False, slots=True)
class Sessions:
    """Night sessions of the rows of one linearization.

    Attributes
    ----------
    index : ndarray of int, shape (n,)
        Session of each row.
    component : ndarray of object, shape (s,)
        Variance component of each session.
    order : ndarray of int, shape (n,)
        Rows sorted by session.
    starts : ndarray of int, shape (s,)
        First position of each session in ``order``.
    """

    index: np.ndarray
    component: np.ndarray
    order: np.ndarray
    starts: np.ndarray

    @property
    def count(self) -> int:
        """Number of sessions."""
        return len(self.component)

    @classmethod
    def build(cls, component_ids: np.ndarray, epoch_jd: np.ndarray, maximum_gap_hours: float) -> Sessions:
        """Split the rows into sessions: same component, no gap longer than ``maximum_gap_hours``.

        Parameters
        ----------
        component_ids : ndarray of object, shape (n,)
            Variance component per row.
        epoch_jd : ndarray, shape (n,)
            Transmit epoch per row, as a Julian date.
        maximum_gap_hours : float
            Largest gap between consecutive normal points of one session, in hours.
        """
        n = len(component_ids)
        if n == 0:
            empty = np.empty(0, dtype=int)
            return cls(empty, np.empty(0, dtype=object), empty, empty)
        _, component_code = np.unique(component_ids.astype(str), return_inverse=True)
        order = np.lexsort((epoch_jd, component_code))
        sorted_code = component_code[order]
        sorted_epoch = epoch_jd[order]
        new = np.ones(n, dtype=bool)
        new[1:] = (sorted_code[1:] != sorted_code[:-1]) | (np.diff(sorted_epoch) * 24.0 > maximum_gap_hours)
        sorted_session = np.cumsum(new) - 1
        index = np.empty(n, dtype=int)
        index[order] = sorted_session
        starts = np.flatnonzero(new)
        component = np.asarray(component_ids, dtype=object)[order[starts]]
        return cls(index, component, order, starts)

    def sums(self, values: np.ndarray) -> np.ndarray:
        """Sum ``values`` (rows first) over the rows of each session."""
        return np.add.reduceat(values[self.order], self.starts, axis=0)


def clustering_diagnostics(
    sessions: Sessions,
    component_ids: np.ndarray,
    weight_factors: np.ndarray,
    components: list[str],
    expected: tuple[float, float],
) -> dict[str, dict[str, object]]:
    """Return per-component outlier statistics that separate random outliers from systematic errors.

    For every component this reports the observed fractions of down-weighted (``0 < f < 1``) and rejected
    (``f = 0``) observations next to their Gaussian expectation, and the dispersion index of the number of
    affected (``f < 1``) observations per session: ``sum (c_n - m_n p)^2 / (m_n p (1 - p)) / (S - 1)`` with
    ``m_n`` observations of session ``n``, ``c_n`` of them affected, and the component's affected fraction ``p``.
    Independent outliers give an index near one; outliers clustered in nights, the signature of a systematic
    (model or calibration) error rather than of random gross errors, give an index well above one.
    """
    expected_down, expected_rejected = expected
    affected = (weight_factors < 1.0).astype(float)
    counts = np.bincount(sessions.index, minlength=sessions.count).astype(float)
    hits = np.bincount(sessions.index, affected, minlength=sessions.count)
    result: dict[str, dict[str, object]] = {}
    for component in components:
        rows = component_ids == component
        session_rows = sessions.component == component
        n = int(np.count_nonzero(rows))
        record: dict[str, object] = {
            "expectedDownweightedFraction": expected_down,
            "expectedRejectedFraction": expected_rejected,
            "downweightedFraction": None,
            "rejectedFraction": None,
            "sessionClusteringIndex": None,
        }
        if n:
            factors = weight_factors[rows]
            record["downweightedFraction"] = float(np.mean((factors > 0.0) & (factors < 1.0)))
            record["rejectedFraction"] = float(np.mean(factors == 0.0))
            m, c = counts[session_rows], hits[session_rows]
            p = float(np.sum(c) / np.sum(m))
            if 0.0 < p < 1.0 and len(m) > 1:
                record["sessionClusteringIndex"] = float(
                    np.sum((c - m * p) ** 2 / (m * p * (1.0 - p))) / (len(m) - 1)
                )
        result[component] = record
    return result


__all__ = [
    "DEFAULT_SESSION_GAP_HOURS",
    "MINIMUM_SESSION_SIGMA_M",
    "Sessions",
    "clustering_diagnostics",
    "consistency_factor",
    "gaussian_tail_fractions",
]
