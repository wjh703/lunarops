"""Robust observation weights: the weight functions, their Gaussian statistics and the residual standardization.

Weight functions
    IGG-III and direct rejection map a standardized residual ``z`` to a weight factor ``w(z)`` in [0, 1]
    (:func:`robust_factors`).

Consistency factor
    With weights ``w(z)``, the robust variance estimate ``sum w v^2 / sum r`` of a Gaussian sample is biased low by
    ``kappa = E[w(z) z^2 | w(z) > 0]`` (0.786 for IGG-III with k0 = 1.5, k1 = 6). Dividing by ``kappa`` gives the
    consistent M-scale (Huber's proposal 2) of the component (:func:`consistency_factor`).

Residual variance under robust weights
    A residual is standardized by its actual variance under the weights of its solution,
    ``Var(v_i) = s^2 sigma_i^2 (1 - f_i t_i) (1 + (1 - f_i) t_i)`` with ``t_i = q_i / (s sigma_i)^2`` and ``q_i`` the
    variance of the fitted value (``a_i N^-1 a_i^T``, plus the random-effect term). This is the redundancy ``1 - t_i``
    of a full-weight observation and the prediction variance ``1 + t_i`` of a rejected one, and the standardized
    residual is Baarda's w-statistic, which does not depend on the observation's own weight factor, so a weight
    cannot feed back on itself (:func:`residual_variance_factors`). The redundancy of the weighted system is not
    suitable: it tends to one as an observation is down-weighted, so a rejected observation would look better than
    it is, and a full-weight observation whose session partners are rejected (its residual absorbed by the session
    offset) would look worse. The same factor is the denominator of the consistent scale,
    ``s^2 = sum f v^2 / sigma^2 / (kappa sum rho)`` over the active observations, so observations absorbed by their
    session offset carry neither residual nor redundancy.
"""

from __future__ import annotations

from functools import lru_cache
from math import erf, sqrt
from typing import cast

import numpy as np

IGG3_MODEL = "igg3"
DIRECT_REJECTION_MODEL = "directRejection"
ROBUST_WEIGHT_MODELS = frozenset((IGG3_MODEL, DIRECT_REJECTION_MODEL))
#: Observations whose residual-variance factor is at or below this value get a standardized residual of 0.
MINIMUM_ROBUST_REDUNDANCY = 0.1
REDUNDANCY_EPSILON = 1.0e-12


def robust_factors(model: str, k0: float, k1: float | None, standardized: np.ndarray) -> np.ndarray:
    """Map standardized residuals to robust weight factors in [0, 1].

    With ``u = |standardized|``, ``directRejection`` gives 1 for ``u <= k0`` and 0 otherwise. ``igg3`` is the
    IGG-III model for iteratively reweighted least squares with thresholds ``k0 < k1``: 1 for ``u <= k0``,
    ``(k0 / u) * ((k1 - u) / (k1 - k0))**2`` for ``k0 < u <= k1`` and 0 for ``u > k1``.

    Parameters
    ----------
    model : str
        ``directRejection`` or ``igg3``; any other value is treated as ``igg3``.
    k0 : float
        Full-weight threshold.
    k1 : float or None
        Rejection threshold of ``igg3``; not used by ``directRejection``.
    standardized : ndarray, shape (n,)
        Standardized residuals.

    Returns
    -------
    ndarray, shape (n,)
        Weight factors.
    """
    if model not in ROBUST_WEIGHT_MODELS:
        raise ValueError(f"Unknown robust weight model {model!r}; expected one of {sorted(ROBUST_WEIGHT_MODELS)}.")
    magnitude = np.abs(standardized)
    if model == DIRECT_REJECTION_MODEL:
        return np.where(magnitude <= k0, 1.0, 0.0)
    k1 = cast(float, k1)
    result = np.zeros_like(magnitude)
    full = magnitude <= k0
    middle = (magnitude > k0) & (magnitude <= k1)
    result[full] = 1.0
    result[middle] = k0 / magnitude[middle] * ((k1 - magnitude[middle]) / (k1 - k0)) ** 2
    return result


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


def residual_variance_factors(ratios: np.ndarray, weight_factors: np.ndarray) -> np.ndarray:
    """Return ``(1 - f t) (1 + (1 - f) t)``: residual variance over observation variance under weight factors ``f``.

    ``t`` is the fitted-value ratio of the solution that used the weight factors ``f``. With the observation's own
    weight ``f p`` in the normal equations, its residual is ``v = e / (1 + f t_e)`` with the prediction residual
    ``e`` (variance ``1 + t_e``) of the system without it, and ``t = t_e / (1 + f t_e)``; hence the factor. It is the
    redundancy ``1 - t`` at full weight and the prediction variance ``1 + t`` at zero weight, and ``v / sqrt`` of it
    is Baarda's w-statistic of the observation, which does not depend on its own weight (the other observations'
    weights taken as their inverse variances).
    """
    return np.maximum((1.0 - weight_factors * ratios) * (1.0 + (1.0 - weight_factors) * ratios), 0.0)


def standardized_residuals(
    residuals: np.ndarray, standardization: np.ndarray, sigmas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized residuals ``v / (sigma sqrt(rho))`` and residual sigmas ``sigma sqrt(rho)`` in metres.

    ``rho`` is the residual-variance factor (``standardization``): the redundancy number, or under robust weights
    :func:`residual_variance_factors`. Observations with ``rho`` at or below ``MINIMUM_ROBUST_REDUNDANCY`` get a
    standardized residual of 0.
    """
    residual_sigmas = sigmas * np.sqrt(np.maximum(standardization, REDUNDANCY_EPSILON))
    standardized = np.zeros_like(residuals)
    eligible = standardization > MINIMUM_ROBUST_REDUNDANCY
    standardized[eligible] = residuals[eligible] / residual_sigmas[eligible]
    return standardized, residual_sigmas


__all__ = [
    "DIRECT_REJECTION_MODEL",
    "IGG3_MODEL",
    "MINIMUM_ROBUST_REDUNDANCY",
    "ROBUST_WEIGHT_MODELS",
    "consistency_factor",
    "gaussian_tail_fractions",
    "residual_variance_factors",
    "robust_factors",
    "standardized_residuals",
]
