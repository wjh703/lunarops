"""Typed result models emitted by the nonlinear LLR adjustment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Hashable

import numpy as np

from lunarops.base.parameter_name import ParameterName, names_to_strings
from lunarops.estimation.normal_equations import NormalEquations

ObsKey = Hashable


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
class LlrAdjustmentResult:
    converged: bool
    termination_reason: str
    settings: dict[str, object]
    equation_evaluations: list[dict[str, object]]
    parameter_names: list[ParameterName]
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

    def to_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "termination_reason": self.termination_reason,
            "settings": self.settings,
            "equation_evaluations": self.equation_evaluations,
            "parameter_names": names_to_strings(self.parameter_names),
            "state": self.state,
            "gross_rejected_observations": {str(key): value for key, value in self.gross_rejected.items()},
            "accuracy_screening": self.accuracy_screening,
            "sigma_factors": self.sigma_factors,
            "weight_factors": {str(key): value for key, value in self.weight_factors.items()},
            "sigma_weight_iterations": [asdict(item) for item in self.sigma_weight_iterations],
            "adjustment_iterations": self.adjustment_iterations,
            "summary": self.summary,
            "parameters": self.parameters,
            "global_residuals": self.global_residuals,
            "variance_components": self.variance_components,
            "observations": self.observations,
        }


__all__ = ["LlrAdjustmentIteration", "LlrAdjustmentResult"]
