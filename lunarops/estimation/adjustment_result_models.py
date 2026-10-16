"""Typed result models emitted by the nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from lunarops.base.parameter_name import ParameterName, names_to_strings
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


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class LlrAdjustmentEstimateResult:
    """State passed to the next estimate without computing final report products."""

    converged: bool
    termination_reason: str
    equation_evaluations: list[dict[str, object]]
    state: dict[str, object]
    sigma_factors: dict[str, float]
    weight_factors: dict[ObsKey, float]
    sigma_weight_iterations: list[dict[str, object]]
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
    sigma_weight_iterations: list[dict[str, object]]
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
        from lunarops.estimation.parameter_products import PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER, ParameterVector

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
        if any(item.get("updateStatus") == "INSUFFICIENT_REDUNDANCY" for item in self.variance_components):
            reasons.append("insufficient variance-component redundancy")
        if any(bool(row.get("troposphereClamped", False)) for row in self.observations):
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
        """Return the processing-report fields of this result."""
        return {
            "converged": self.converged,
            "publicationQuality": self.publication_quality(),
            "terminationReason": self.termination_reason,
            "settings": self.settings,
            "equationEvaluations": self.equation_evaluations,
            "parameterNames": names_to_strings(self.parameter_names),
            "state": self.state,
            "observationScreening": {
                "residual": {
                    "rejectedCount": len(self.gross_rejected),
                    "rejectedObservations": self.gross_rejected,
                },
                "reportedSigma": self.accuracy_screening,
            },
            "sigmaFactors": self.sigma_factors,
            "weightFactors": self.weight_factors,
            "sigmaWeightIterations": self.sigma_weight_iterations,
            "adjustmentIterations": self.adjustment_iterations,
            "summary": self.summary,
            "parameters": self.parameters,
            "globalResiduals": self.global_residuals,
            "varianceComponents": self.variance_components,
            "observations": self.observations,
        }


__all__ = [
    "LlrAdjustmentEstimateResult",
    "LlrAdjustmentObservationDomain",
    "LlrAdjustmentResult",
]
