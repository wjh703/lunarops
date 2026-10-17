"""Variance-component sigma-factor adjustment informed by GROOPS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from lunarops.estimation.variance_component_groups import VarianceComponentDefinition

# These are algorithm invariants in the corresponding GROOPS workflow, not
# dataset-tuning parameters.
MINIMUM_COMPONENT_REDUNDANCY = 3.0


@dataclass(frozen=True)
class SigmaFactorEstimate:
    sigma_factors: dict[str, float]
    diagnostics: dict[str, dict[str, object]]


@dataclass(frozen=True)
class SigmaFactorEstimator:
    """Adjust component sigma factors from frozen residuals and redundancies."""

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
