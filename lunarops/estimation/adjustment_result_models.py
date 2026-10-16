"""Typed result models emitted by the nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.estimation.normal_equations import NormalEquations

ObsKey = Hashable


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class LlrAdjustmentObservationDomain:
    """Permanent observation domain shared by all estimate processing steps."""

    gross_rejected: dict[ObsKey, float]
    retained_keys: set[ObsKey]
    assignments: dict[ObsKey, str]
    accuracy_records: dict[ObsKey, dict[str, object]]
    accuracy_groups: dict[str, dict[str, object]]
    observation_signatures: dict[ObsKey, tuple[str, str, object, float | None]]


@dataclass(frozen=True, eq=False, slots=True)
class LlrAdjustmentIteration:
    iteration: int
    adjustment_iteration: int
    sigma_weight_iteration: int
    elapsed_seconds: float
    maximum_sigma_factor_change: float
    maximum_weight_factor_change: float
    active_observation_count: int
    rejected_observation_count: int
    total_frozen_redundancy: float
    expected_total_redundancy: float
    normal_matrix_condition: float | None
    candidate_wrms_m: float | None
    maximum_candidate_parameter_update_m: float
    candidate_update_by_block_m: dict[str, float]
    sigma_factors: dict[str, float]
    weight_factor_summary: dict[str, object]
    variance_components: dict[str, dict[str, object]]


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class LlrAdjustmentEstimateResult:
    """State passed to the next estimate without computing final report products."""

    converged: bool
    termination_reason: str
    equation_evaluations: list[dict[str, object]]
    state: dict[str, object]
    sigma_factors: dict[str, float]
    weight_factors: dict[ObsKey, float]
    sigma_weight_iterations: list[LlrAdjustmentIteration]
    adjustment_iterations: list[dict[str, object]]
    summary: dict[str, object]
    observation_domain: LlrAdjustmentObservationDomain


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class LlrAdjustmentResult:
    converged: bool
    termination_reason: str
    settings: dict[str, object]
    equation_evaluations: list[dict[str, object]]
    state: dict[str, object]
    gross_rejected: dict[ObsKey, float]
    accuracy_screening: dict[str, object]
    sigma_factors: dict[str, float]
    weight_factors: dict[ObsKey, float]
    sigma_weight_iterations: list[LlrAdjustmentIteration]
    adjustment_iterations: list[dict[str, object]]
    summary: dict[str, object]
    parameters: list[dict[str, object]]
    global_residuals: dict[str, object]
    variance_components: list[dict[str, object]]
    observations: list[dict[str, object]]
    normals: NormalEquations
    remaining_correction: np.ndarray
    cofactor: np.ndarray
    sigma0_post: float | None

    @property
    def parameter_names(self) -> list[ParameterName]:
        return self.normals.parameter_names

    def parameter_vector(self):
        """Serialize the final linearization state, without applying its remaining correction."""
        from lunarops.estimation.parameter_products import ParameterVector
        from lunarops.estimation.uncertainty_conventions import PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER

        cofactor_sigma = np.sqrt(np.maximum(np.diag(self.cofactor), 0.0))
        one_sigma = cofactor_sigma if self.sigma0_post is None else self.sigma0_post * cofactor_sigma
        return ParameterVector(
            tuple(self.parameter_names),
            self.normals.x0,
            tuple(self.normals.parameter_units),
            PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER * one_sigma,
            PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER,
        )

    def covariance_matrix(self):
        from lunarops.estimation.parameter_products import CovarianceMatrix

        matrix = self.cofactor if self.sigma0_post is None else self.sigma0_post**2 * self.cofactor
        return CovarianceMatrix(
            tuple(self.parameter_names),
            matrix,
            tuple(self.normals.parameter_units),
            "cofactor" if self.sigma0_post is None else "posteriorCovariance",
        )

    def publication_quality(self) -> dict[str, object]:
        reasons = []
        if not self.converged:
            reasons.append(self.termination_reason)
        if self.normals.obs_count <= len(self.parameter_names) or self.sigma0_post is None:
            reasons.append("insufficient posterior redundancy")
        if not np.all(np.isfinite(self.cofactor)):
            reasons.append("invalid covariance")
        if any(not bool(row.get("light_time_converged", True)) for row in self.observations):
            reasons.append("light-time nonconvergence")
        if any(item.get("update_status") == "INSUFFICIENT_REDUNDANCY" for item in self.variance_components):
            reasons.append("insufficient variance-component redundancy")
        if any(bool(row.get("troposphere_clamped", False)) for row in self.observations):
            reasons.append("clamped low-elevation observations")
        return {
            "formalProductEligible": not reasons,
            "status": "passed" if not reasons else "diagnostic",
            "reasons": reasons,
        }

    def require_formal_product(self) -> None:
        quality = self.publication_quality()
        if not quality["formalProductEligible"]:
            raise ValueError(f"Adjustment is diagnostic only: {quality['reasons']}")

    def to_dict(self) -> dict[str, object]:
        from lunarops.estimation.report_serialization import serialize_adjustment_result

        return serialize_adjustment_result(self)



__all__ = [
    "LlrAdjustmentEstimateResult",
    "LlrAdjustmentIteration",
    "LlrAdjustmentObservationDomain",
    "LlrAdjustmentResult",
]
