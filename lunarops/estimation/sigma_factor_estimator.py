"""GROOPS-style variance-component sigma-factor adjustment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

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
    active_weight_threshold: float = 1.0e-12

    def __post_init__(self) -> None:
        if isinstance(self.components, (str, bytes)) or not isinstance(self.components, Sequence):
            raise TypeError("Sigma-factor components must be a sequence.")
        components = tuple(self.components)
        if not components or not all(isinstance(item, VarianceComponentDefinition) for item in components):
            raise ValueError("At least one valid sigma-factor component is required.")
        if len({item.id for item in components}) != len(components):
            raise ValueError("Sigma-factor component IDs must be unique.")
        threshold = float(self.active_weight_threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("Active weight threshold must be finite and in (0, 1).")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "active_weight_threshold", threshold)

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
        apriori_sigmas = np.asarray(apriori_sigmas, dtype=float).reshape(-1)
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        redundancies = np.asarray(redundancies, dtype=float).reshape(-1)
        component_ids = np.asarray(component_ids, dtype=object).reshape(-1)
        weight_factors = np.asarray(weight_factors, dtype=float).reshape(-1)
        size = residuals.size
        if any(array.size != size for array in (apriori_sigmas, redundancies, component_ids, weight_factors)):
            raise ValueError("Sigma-factor row arrays must have equal length.")
        if not np.all(np.isfinite(apriori_sigmas)) or np.any(apriori_sigmas <= 0.0):
            raise ValueError("A-priori sigmas must be positive and finite.")
        if not np.all(np.isfinite(residuals)):
            raise ValueError("Residuals must be finite.")
        if not np.all(np.isfinite(redundancies)) or np.any((redundancies < 0.0) | (redundancies > 1.0)):
            raise ValueError("Observation redundancies must be finite and in [0, 1].")
        if not np.all(np.isfinite(weight_factors)) or np.any((weight_factors < 0.0) | (weight_factors > 1.0)):
            raise ValueError("Weight factors must be finite and in [0, 1].")

        configured = {item.id for item in self.components}
        if set(component_ids) - configured:
            raise ValueError("Sigma-factor rows contain unknown component IDs.")
        if set(sigma_factors) != configured:
            raise ValueError("Sigma factors must match configured components.")

        updates: dict[str, float] = {}
        diagnostics: dict[str, dict[str, object]] = {}
        active = weight_factors > self.active_weight_threshold
        for component in self.components:
            current = sigma_factors[component.id]
            if isinstance(current, bool) or not isinstance(current, Real):
                raise TypeError(f"Sigma factor for {component.id!r} must be real.")
            current = float(current)
            if not np.isfinite(current) or current <= 0.0:
                raise ValueError(f"Sigma factor for {component.id!r} must be positive and finite.")
            mask = active & (component_ids == component.id)
            redundancy = float(np.sum(redundancies[mask]))
            weighted_square_sum = float(
                np.sum(weight_factors[mask] * (residuals[mask] / apriori_sigmas[mask]) ** 2)
            )
            if redundancy <= MINIMUM_COMPONENT_REDUNDANCY:
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
                "active_count": int(np.count_nonzero(mask)),
                "effective_redundancy": redundancy,
                "weighted_square_sum": weighted_square_sum,
                "sigma_factor_before": current,
                "sigma_factor_after": proposed,
                "variance_ratio": float(variance_ratio),
                "update_status": status,
            }
        return SigmaFactorEstimate(updates, diagnostics)


__all__ = [
    "MINIMUM_COMPONENT_REDUNDANCY",
    "SigmaFactorEstimate",
    "SigmaFactorEstimator",
]
