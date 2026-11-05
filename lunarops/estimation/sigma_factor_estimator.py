"""Variance-component estimation of observation sigma factors.

For each component ``k`` the sigma factor is ``s_k = sqrt(sum_i f_i (v_i / sigma_i)**2 / sum_i r_i)`` over its
active observations, with robust weight factor ``f_i``, post-fit one-way residual ``v_i`` and a-priori one-way
sigma ``sigma_i`` in metres, and redundancy number ``r_i``. Active observations are those with ``f_i > 0``.
The adjustment weights observation ``i`` with ``f_i / (s_k sigma_i)**2``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from lunarops.estimation.variance_component_groups import VarianceComponentDefinition

# Algorithm invariant, not a dataset-tuning parameter: a component needs a total
# redundancy above this value before its sigma factor is updated.
MINIMUM_COMPONENT_REDUNDANCY = 3.0


@dataclass(frozen=True)
class SigmaFactorEstimate:
    """Result of one sigma-factor estimation.

    Attributes
    ----------
    sigma_factors : dict[str, float]
        Proposed sigma factor per component.
    diagnostics : dict[str, dict[str, object]]
        Per component: active count, effective redundancy, weighted square sum, sigma factor before and after,
        variance ratio and ``updateStatus``.
    """

    sigma_factors: dict[str, float]
    diagnostics: dict[str, dict[str, object]]


@dataclass(frozen=True)
class SigmaFactorEstimator:
    """Variance-component sigma-factor estimator using frozen residuals and redundancies.

    Attributes
    ----------
    components : tuple[VarianceComponentDefinition, ...]
        Components to estimate.
    """

    components: tuple[VarianceComponentDefinition, ...]

    def estimate(
        self,
        *,
        apriori_sigmas: np.ndarray,
        residuals: np.ndarray,
        redundancies: np.ndarray,
        component_ids: np.ndarray,
        weight_factors: np.ndarray,
        sigma_factors: Mapping[str, float],
    ) -> SigmaFactorEstimate:
        """Estimate the sigma factor of every component.

        A component keeps its current factor when it has no observations (``NO_OBSERVATIONS``), its total
        redundancy is at most ``MINIMUM_COMPONENT_REDUNDANCY`` (``INSUFFICIENT_REDUNDANCY``) or the estimated
        variance is zero (``ZERO_VARIANCE_TARGET``); otherwise it is updated (``UPDATED``).

        Parameters
        ----------
        apriori_sigmas : ndarray, shape (n,)
            A-priori one-way sigmas, in metres.
        residuals : ndarray, shape (n,)
            Post-fit one-way residuals, in metres.
        redundancies : ndarray, shape (n,)
            Redundancy numbers in [0, 1].
        component_ids : ndarray of object, shape (n,)
            Component identifier per observation.
        weight_factors : ndarray, shape (n,)
            Robust weight factors; observations with factor 0 are inactive.
        sigma_factors : Mapping[str, float]
            Current sigma factor per component.

        Returns
        -------
        SigmaFactorEstimate
            Proposed sigma factors and diagnostics.

        Raises
        ------
        RuntimeError
            If an estimated variance is negative or not finite.
        """
        updates: dict[str, float] = {}
        diagnostics: dict[str, dict[str, object]] = {}
        active = weight_factors > 0.0
        for component in self.components:
            current = float(sigma_factors[component.id])
            rows = component_ids == component.id
            mask = active & rows
            redundancy = float(np.sum(redundancies[mask]))
            weighted_square_sum = float(
                np.sum(weight_factors[mask] * (residuals[mask] / apriori_sigmas[mask]) ** 2)
            )
            if not np.any(rows):
                proposed = current
                variance_ratio = 1.0
                status = "NO_OBSERVATIONS"
            elif redundancy <= MINIMUM_COMPONENT_REDUNDANCY:
                proposed = current
                variance_ratio = 1.0
                status = "INSUFFICIENT_REDUNDANCY"
            else:
                proposed_variance = weighted_square_sum / redundancy
                if not np.isfinite(proposed_variance) or proposed_variance < 0.0:
                    raise RuntimeError(f"Invalid sigma-factor estimate for {component.id!r}.")
                if proposed_variance == 0.0:
                    proposed = current
                    variance_ratio = 1.0
                    status = "ZERO_VARIANCE_TARGET"
                else:
                    proposed = float(np.sqrt(proposed_variance))
                    variance_ratio = proposed_variance / current**2
                    status = "UPDATED"
            updates[component.id] = proposed
            diagnostics[component.id] = {
                "activeCount": int(np.count_nonzero(mask)),
                "effectiveRedundancy": redundancy,
                "weightedSquareSum": weighted_square_sum,
                "sigmaFactorBefore": current,
                "sigmaFactorAfter": proposed,
                "varianceRatio": float(variance_ratio),
                "updateStatus": status,
            }
        return SigmaFactorEstimate(updates, diagnostics)


__all__ = [
    "MINIMUM_COMPONENT_REDUNDANCY",
    "SigmaFactorEstimate",
    "SigmaFactorEstimator",
]
