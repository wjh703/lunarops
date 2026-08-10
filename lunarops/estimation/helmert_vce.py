"""Helmert variance-component estimation for LLR observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, cast

import numpy as np

from lunarops.estimation.variance_component_groups import VarianceComponentDefinition
from lunarops.estimation.normal_equation_solver import normal_matrix_rank
from lunarops.estimation.normal_equations import NormalEquations


@dataclass(frozen=True)
class VarianceComponentEstimate:
    scales: dict[str, float]
    diagnostics: dict[str, dict[str, object]]


@dataclass(frozen=True)
class HelmertVceEstimator:
    """Helmert trace VCE using exact component effective redundancies."""

    components: tuple[VarianceComponentDefinition, ...]
    minimum_nonzero_factor: float = 1.0e-12
    minimum_effective_redundancy: float = 20.0
    minimum_variance_ratio_per_iteration: float = 0.25
    maximum_variance_ratio_per_iteration: float = 4.0

    def __post_init__(self) -> None:
        if isinstance(self.components, (str, bytes)) or not isinstance(self.components, Sequence):
            raise TypeError("Helmert VCE components must be a sequence of VarianceComponentDefinition instances.")
        components = tuple(self.components)
        if not components:
            raise ValueError("Helmert VCE requires at least one component.")
        if not all(isinstance(component, VarianceComponentDefinition) for component in components):
            raise TypeError("Helmert VCE components must be VarianceComponentDefinition instances.")
        component_ids = [component.id for component in components]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("Helmert VCE component IDs must be unique.")
        raw_values = (
            self.minimum_nonzero_factor,
            self.minimum_effective_redundancy,
            self.minimum_variance_ratio_per_iteration,
            self.maximum_variance_ratio_per_iteration,
        )
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_values):
            raise TypeError("Helmert VCE thresholds must be real numbers.")
        minimum_nonzero_factor = float(self.minimum_nonzero_factor)
        minimum_effective_redundancy = float(self.minimum_effective_redundancy)
        minimum_ratio = float(self.minimum_variance_ratio_per_iteration)
        maximum_ratio = float(self.maximum_variance_ratio_per_iteration)
        values = (minimum_nonzero_factor, minimum_effective_redundancy, minimum_ratio, maximum_ratio)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Helmert VCE thresholds must be finite.")
        if not 0.0 < minimum_nonzero_factor < 1.0:
            raise ValueError("Helmert VCE active-factor threshold must be in (0, 1).")
        if minimum_effective_redundancy < 0.0:
            raise ValueError("Helmert VCE minimum effective redundancy must be non-negative.")
        if not (0.0 < minimum_ratio <= maximum_ratio):
            raise ValueError("Helmert VCE variance-ratio limits are invalid.")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "minimum_nonzero_factor", minimum_nonzero_factor)
        object.__setattr__(self, "minimum_effective_redundancy", minimum_effective_redundancy)
        object.__setattr__(self, "minimum_variance_ratio_per_iteration", minimum_ratio)
        object.__setattr__(self, "maximum_variance_ratio_per_iteration", maximum_ratio)

    def _finalize(
        self,
        *,
        covariance: np.ndarray,
        component_normal_matrices: Mapping[str, np.ndarray],
        counts: Mapping[str, int],
        numerators: Mapping[str, float],
        scales: Mapping[str, float],
        normals: NormalEquations,
        active_count: int,
    ) -> VarianceComponentEstimate:
        updates: dict[str, float] = {}
        diagnostics: dict[str, dict[str, object]] = {}
        for component in self.components:
            component_id = component.id
            consumed = float(np.trace(covariance @ component_normal_matrices[component_id]))
            count = counts[component_id]
            redundancy = float(count - consumed)
            current_variance = float(scales[component_id]) ** 2
            if not np.isfinite(current_variance) or current_variance <= 0.0:
                raise ValueError(f"Helmert VCE variance for {component_id!r} must be positive and finite.")
            if redundancy <= 0.0:
                raw_variance = current_variance
                raw_ratio = limited_ratio = 1.0
                next_variance = current_variance
                status = "ZERO_EFFECTIVE_REDUNDANCY"
            elif redundancy < self.minimum_effective_redundancy:
                raw_variance = current_variance
                raw_ratio = limited_ratio = 1.0
                next_variance = current_variance
                status = "INSUFFICIENT_REDUNDANCY"
            else:
                raw_variance = numerators[component_id] / redundancy
                if not np.isfinite(raw_variance) or raw_variance < 0.0:
                    raise RuntimeError(
                        f"Invalid Helmert VCE estimate for component {component_id!r}: {raw_variance!r}."
                    )
                raw_ratio = raw_variance / current_variance
                if raw_ratio == 0.0:
                    # A zero target would collapse the observation variance
                    # and make the next weighted solve ill-conditioned. It is
                    # a degenerate estimate, not a meaningful scale update.
                    limited_ratio = 1.0
                    next_variance = current_variance
                    status = "ZERO_VARIANCE_TARGET"
                else:
                    limited_ratio = float(
                        np.clip(
                            raw_ratio,
                            self.minimum_variance_ratio_per_iteration,
                            self.maximum_variance_ratio_per_iteration,
                        )
                    )
                    next_variance = current_variance * limited_ratio
                    status = "UPDATED"
            proposed_scale = float(np.sqrt(next_variance))
            updates[component_id] = proposed_scale
            diagnostics[component_id] = {
                "active_count": float(count),
                "consumed_dof": consumed,
                "effective_redundancy": redundancy,
                "current_variance": current_variance,
                "estimated_variance": float(raw_variance),
                "estimated_variance_ratio": float(raw_ratio),
                "bounded_variance_ratio": float(limited_ratio),
                # A zero residual variance is a degenerate target. It is
                # reported separately and leaves the current scale unchanged.
                "target_scale_log_change": float(0.0 if raw_ratio == 0.0 else abs(np.log(raw_ratio))),
                "proposed_variance": float(next_variance),
                "proposed_scale": proposed_scale,
                "update_status": status,
            }
        expected = float(active_count - normal_matrix_rank(normals))
        actual = sum(cast(float, item["effective_redundancy"]) for item in diagnostics.values())
        if not np.isclose(actual, expected, rtol=1.0e-10, atol=1.0e-8):
            raise RuntimeError(f"Helmert redundancy check failed: {actual:.12g} != {expected:.12g}.")
        return VarianceComponentEstimate(updates, diagnostics)

    def estimate(
        self,
        *,
        design,
        sigmas,
        residuals,
        component_ids,
        factors,
        scales,
        normals,
        covariance,
    ) -> VarianceComponentEstimate:
        design = np.asarray(design, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        factors = np.asarray(factors, dtype=float).reshape(-1)
        component_ids = np.asarray(component_ids, dtype=object).reshape(-1)
        if design.ndim != 2 or design.shape[0] == 0:
            raise ValueError("Helmert VCE design must be a non-empty two-dimensional matrix.")
        observation_count, parameter_count = design.shape
        if any(values.size != observation_count for values in (sigmas, residuals, factors, component_ids)):
            raise ValueError("Helmert VCE row arrays must match the design row count.")
        if not np.all(np.isfinite(design)) or not np.all(np.isfinite(residuals)):
            raise ValueError("Helmert VCE design and residuals must be finite.")
        if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            raise ValueError("Helmert VCE sigmas must be positive and finite.")
        if not np.all(np.isfinite(factors)) or np.any((factors < 0.0) | (factors > 1.0)):
            raise ValueError("Helmert VCE factors must be finite and in [0, 1].")
        if not all(isinstance(component_id, str) for component_id in component_ids):
            raise TypeError("Helmert VCE component IDs must be strings.")
        if not isinstance(normals, NormalEquations):
            raise TypeError("Helmert VCE normals must be a NormalEquations instance.")
        if normals.N.shape != (parameter_count, parameter_count) or not np.all(np.isfinite(normals.N)):
            raise ValueError("Helmert VCE normal matrix does not match the design column count.")
        active = factors > self.minimum_nonzero_factor
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (parameter_count, parameter_count) or not np.all(np.isfinite(covariance)):
            raise ValueError("Helmert VCE covariance does not match the design columns or is non-finite.")
        component_id_set = {component.id for component in self.components}
        unknown_ids = set(component_ids) - component_id_set
        if unknown_ids:
            raise ValueError(f"Helmert VCE received unknown component IDs: {sorted(unknown_ids)!r}.")
        if not all(isinstance(component_id, str) for component_id in scales):
            raise TypeError("Helmert VCE scale keys must be strings.")
        missing_scales = component_id_set - set(scales)
        extra_scales = set(scales) - component_id_set
        if missing_scales or extra_scales:
            raise ValueError(
                "Helmert VCE scales must match configured components: "
                f"missing={sorted(missing_scales)!r}, extra={sorted(extra_scales)!r}."
            )
        for component_id in component_id_set:
            raw_scale = scales[component_id]
            if isinstance(raw_scale, bool) or not isinstance(raw_scale, Real):
                raise TypeError(f"Helmert VCE scale for {component_id!r} must be a real number.")
            scale = float(raw_scale)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Helmert VCE scale for {component_id!r} must be positive and finite.")
        counts: dict[str, int] = {}
        numerators: dict[str, float] = {}
        component_normal_matrices: dict[str, np.ndarray] = {}
        for component in self.components:
            mask = active & (component_ids == component.id)
            A = design[mask]
            factor = factors[mask]
            sigma = sigmas[mask]
            weight = factor / (float(scales[component.id]) ** 2 * sigma**2)
            if not np.all(np.isfinite(weight)):
                raise ValueError(f"Helmert VCE weights for {component.id!r} are non-finite.")
            component_normal_matrices[component.id] = A.T @ (weight[:, None] * A)
            if not np.all(np.isfinite(component_normal_matrices[component.id])):
                raise ValueError(f"Helmert VCE component normals for {component.id!r} are non-finite.")
            counts[component.id] = int(np.count_nonzero(mask))
            numerators[component.id] = float(np.sum(factor * residuals[mask] ** 2 / sigma**2))
            if not np.isfinite(numerators[component.id]):
                raise ValueError(f"Helmert VCE residual numerator for {component.id!r} is non-finite.")

        return self._finalize(
            covariance=covariance,
            component_normal_matrices=component_normal_matrices,
            counts=counts,
            numerators=numerators,
            scales=scales,
            normals=normals,
            active_count=int(np.count_nonzero(active)),
        )


__all__ = ["HelmertVceEstimator", "VarianceComponentEstimate"]
