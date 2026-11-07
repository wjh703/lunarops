"""Diagnostics of the stochastic model: whether outliers are random or clustered in nights."""

from __future__ import annotations

import numpy as np

from lunarops.estimation.lsq.blocks import RowBlocks


def clustering_diagnostics(
    sessions: RowBlocks,
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
        session_rows = sessions.key == component
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
                record["sessionClusteringIndex"] = float(np.sum((c - m * p) ** 2 / (m * p * (1.0 - p))) / (len(m) - 1))
        result[component] = record
    return result


__all__ = ["clustering_diagnostics"]
