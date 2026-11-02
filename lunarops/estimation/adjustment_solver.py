"""Strict nonlinear LLR adjustment informed by GROOPS processing patterns."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.adjustment_plan import LlrAdjustmentSettings
from lunarops.estimation.adjustment_reporting import build_report, weight_factor_summary
from lunarops.estimation.adjustment_result_models import (
    AdjustmentState,
    LlrAdjustmentObservationDomain,
    LlrAdjustmentResult,
    ObsKey,
    _LinearizedSolution,
)
from lunarops.estimation.linearization import DenseLinearization
from lunarops.estimation.normal_equation_solver import (
    solve_normal_equations,
)
from lunarops.estimation.parameter_convergence import parameter_convergence
from lunarops.estimation.robust_weights import robust_factors
from lunarops.estimation.sigma_factor_estimator import SigmaFactorEstimator

SIGMA_WEIGHT_ITERATION_COUNT = 10
MINIMUM_ROBUST_REDUNDANCY = 0.1
REDUNDANCY_EPSILON = 1.0e-12




class LlrAdjustmentSolver:
    def __init__(
        self,
        *,
        equation_source: Callable[[int], EquationBatch],
        parametrization: ParametrizationList,
        settings: LlrAdjustmentSettings,
        observation_domain: LlrAdjustmentObservationDomain,
        model_state=None,
        initial_sigma_factors: Mapping[str, float] | None = None,
        initial_weight_factors: Mapping[ObsKey, float] | None = None,
        iteration_callback: Callable[[dict[str, object]], None] | None = None,
        utc_offset_hours: float = 0.0,
    ) -> None:
        self.equation_source = equation_source
        self.parametrization = parametrization
        self.settings = settings
        self.adjustment = settings.adjustment
        self.accuracy_screening = settings.accuracy_screening
        self.robust = settings.robust_weights
        self.variance_components = settings.variance_components
        self.model_state = model_state
        self.initial_sigma_factors = dict(initial_sigma_factors or {})
        self.initial_weight_factors = dict(initial_weight_factors or {})
        self.observation_domain = observation_domain
        self.iteration_callback = iteration_callback
        self.utc_offset_hours = float(utc_offset_hours)
        self.sigma_factor_estimator = SigmaFactorEstimator(self.variance_components.components)

        self._equation_iteration = 0
        self._names: list[ParameterName] = []
        self._equation_evaluations: list[dict[str, object]] = []
        self._linearization: DenseLinearization
        self._performance_seconds = {"cacheBuild": 0.0, "normalSolve": 0.0, "redundancy": 0.0, "adjustSigma0": 0.0}

    def _equations(self, purpose: str) -> EquationBatch:
        self._equation_iteration += 1
        equations = self.equation_source(self._equation_iteration)
        identities = equations.observation_id.tolist()
        if len(set(identities)) != len(identities):
            raise ValueError("Observation identities must be unique.")
        light_time_valid = equations[equations.converged]
        retained = light_time_valid[np.isin(light_time_valid.observation_id, np.fromiter(self.observation_domain.retained_keys, dtype=object))]
        signatures = zip(retained.station.tolist(), retained.reflector.tolist(), retained.wavelength_nm.tolist())
        for index, (station, reflector, wavelength) in enumerate(signatures):
            identity = retained.observation_id[index]
            signature = (station, reflector, retained.epoch(index), wavelength)
            expected = self.observation_domain.observation_signatures.get(identity)
            if expected is not None and signature != expected:
                raise ValueError(f"Observation {identity!r} changed immutable metadata.")
        self._equation_evaluations.append(
            {
                "evaluation": self._equation_iteration,
                "purpose": purpose,
                "sourceObservationCount": len(equations),
                "lightTimeConvergedCount": len(light_time_valid),
                "lightTimeNonconvergedCount": len(equations) - len(light_time_valid),
                "fixedDomainReturnedCount": len(retained),
                "convergedButOutsideFixedDomainCount": len(light_time_valid) - len(retained),
            }
        )
        return retained

    def _prepare_linearization(self, equations: EquationBatch) -> None:
        if not equations:
            raise ValueError("Adjustment has no usable observations.")
        started = perf_counter()
        self._linearization = DenseLinearization.build(equations, self.parametrization, self._names)
        self._performance_seconds["cacheBuild"] += perf_counter() - started

    def _observation_weights(
        self,
        dense: DenseLinearization,
        sigma_factors: Mapping[str, float],
        weight_factors: np.ndarray,
    ) -> np.ndarray:
        sigma = np.asarray([sigma_factors[self.observation_domain.assignments[key]] for key in dense.identities])
        return weight_factors / (sigma * dense.apriori_sigmas) ** 2

    def _solve_linearized(
        self,
        equations: EquationBatch,
        sigma_factors: Mapping[str, float],
        weight_factors: np.ndarray,
    ) -> _LinearizedSolution:
        dense = self._linearization
        started = perf_counter()
        weights = self._observation_weights(dense, sigma_factors, weight_factors)
        active = weight_factors > 0.0
        if np.count_nonzero(active) < len(self._names):
            raise RuntimeError("Too few active observations for the parameter set.")
        weights = np.where(active, weights, 0.0)
        x0 = self.parametrization.x0()
        normals = dense.normal_equations(weights, active=active, x0=x0)
        solved = solve_normal_equations(normals)
        delta = normals.correction_at_x0(solved.values)
        residual_vector = dense.reduced_observations - dense.design @ delta
        degrees_of_freedom = int(np.count_nonzero(active)) - len(self._names)
        sigma0_post = (
            None
            if degrees_of_freedom <= 0
            else float(np.sqrt(np.dot(weights, residual_vector**2) / degrees_of_freedom))
        )
        weight_sum = float(np.sum(weights))
        wrms = None if weight_sum <= 0.0 else float(np.sqrt(np.dot(weights, residual_vector**2) / weight_sum))
        self._performance_seconds["normalSolve"] += perf_counter() - started
        return _LinearizedSolution(
            equations=equations,
            normals=normals,
            delta=delta,
            wrms_m=wrms,
            covariance=solved.covariance,
            sigma0_post=sigma0_post,
            residual_vector=residual_vector,
            observation_weights=weights,
            rank=solved.rank,
        )

    def _redundancies(self, solution: _LinearizedSolution) -> np.ndarray:
        dense = self._linearization
        started = perf_counter()
        projected = dense.design @ solution.covariance
        leverage = solution.observation_weights * np.einsum("ij,ij->i", projected, dense.design)
        if not np.all(np.isfinite(leverage)) or np.any(leverage < -1.0e-8) or np.any(leverage > 1.0 + 1.0e-8):
            raise RuntimeError("Observation leverage is outside its valid range.")
        redundancies = np.clip(1.0 - leverage, 0.0, 1.0)
        self._performance_seconds["redundancy"] += perf_counter() - started
        return redundancies

    def _standardized_residuals(
        self,
        residuals: np.ndarray,
        redundancies: np.ndarray,
        sigma_factors: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        dense = self._linearization
        sigma0 = np.asarray(
            [
                sigma_factors[self.observation_domain.assignments[key]] * dense.apriori_sigmas[index]
                for index, key in enumerate(dense.identities)
            ],
            dtype=float,
        )
        residual_sigmas = sigma0 * np.sqrt(np.maximum(redundancies, REDUNDANCY_EPSILON))
        standardized = np.zeros_like(residuals)
        eligible = redundancies > MINIMUM_ROBUST_REDUNDANCY
        standardized[eligible] = residuals[eligible] / residual_sigmas[eligible]
        return standardized, residual_sigmas

    def _adjust_sigma0(
        self,
        residuals: np.ndarray,
        redundancies: np.ndarray,
        sigma_factors: Mapping[str, float],
        weight_factors: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
        dense = self._linearization
        started = perf_counter()
        estimate = self.sigma_factor_estimator.estimate(
            apriori_sigmas=dense.apriori_sigmas,
            residuals=residuals,
            redundancies=redundancies,
            component_ids=np.asarray([self.observation_domain.assignments[key] for key in dense.identities], dtype=object),
            weight_factors=weight_factors,
            sigma_factors=sigma_factors,
        )
        self._performance_seconds["adjustSigma0"] += perf_counter() - started
        return estimate.sigma_factors, estimate.diagnostics

    def run(self, *, finalize: bool = True) -> LlrAdjustmentResult:
        return build_report(self.solve(finalize=finalize), self.settings)

    def solve(self, *, finalize: bool = True) -> AdjustmentState:
        initial_equations = self._equations("initialization")
        if not initial_equations:
            raise ValueError("Adjustment has no light-time-converged observations.")
        self.parametrization.setup(initial_equations)
        self._names = self.parametrization.names
        initial_reference_values = self.parametrization.x0().copy()
        sigma_factors = {item.id: 1.0 for item in self.variance_components.components}
        sigma_factors.update({key: float(value) for key, value in self.initial_sigma_factors.items()})
        initial_sigma_factors = dict(sigma_factors)
        # A retained observation can temporarily disappear when its light-time
        # solution does not converge. Keep its weight so it can safely return in
        # a later linearization or processing step.
        weight_factors: dict[ObsKey, float] = {
            identity: float(self.initial_weight_factors.get(identity, 1.0)) for identity in self.observation_domain.retained_keys
        }
        warm_sigma_count = len(self.initial_sigma_factors)
        warm_weight_count = sum(key in self.initial_weight_factors for key in self.observation_domain.retained_keys)

        current_equations = initial_equations
        iterations: list[dict[str, object]] = []
        adjustment_iterations: list[dict[str, object]] = []
        converged = False
        termination_reason = "MAX_ITERATION_COUNT_REACHED"
        global_inner = 0
        diagnostics: dict[str, dict[str, object]] = {}

        for outer in range(1, self.adjustment.maxIterationCount + 1):
            self._prepare_linearization(current_equations)
            sigma_factors_used = dict(sigma_factors)
            weight_factors_used = dict(weight_factors)
            weight_vector = np.asarray([weight_factors[key] for key in self._linearization.identities])
            base_solution = self._solve_linearized(current_equations, sigma_factors_used, weight_vector)
            candidate_by_block = self.parametrization.update_norms(base_solution.delta)
            frozen_residuals = base_solution.residual_vector
            frozen_redundancies = (
                self._redundancies(base_solution)
                if self.adjustment.computeResiduals
                else np.zeros(len(current_equations), dtype=float)
            )
            keys = current_equations.observation_id.tolist()

            summary_used = weight_factor_summary(current_equations, weight_factors_used)
            sigma_weight_iteration_count = 0
            if self.adjustment.computeResiduals:
                if self.adjustment.estimateVarianceFactors and self.adjustment.estimateRobustWeights:
                    sigma_weight_iteration_count = SIGMA_WEIGHT_ITERATION_COUNT
                elif self.adjustment.estimateVarianceFactors or self.adjustment.estimateRobustWeights:
                    sigma_weight_iteration_count = 1
            for inner in range(1, sigma_weight_iteration_count + 1):
                started = perf_counter()
                previous_sigma = dict(sigma_factors)
                previous_weights = weight_vector.copy()
                if self.adjustment.estimateVarianceFactors:
                    sigma_factors, diagnostics = self._adjust_sigma0(
                        frozen_residuals,
                        frozen_redundancies,
                        sigma_factors,
                        weight_vector,
                    )
                if self.adjustment.estimateRobustWeights:
                    standardized, _ = self._standardized_residuals(
                        frozen_residuals,
                        frozen_redundancies,
                        sigma_factors,
                    )
                    weight_vector = robust_factors(self.robust.model, self.robust.k0, self.robust.k1, standardized)
                weight_factors.update(zip(keys, map(float, weight_vector), strict=True))
                max_sigma_change = max(
                    (abs(sigma_factors[key] - previous_sigma[key]) for key in sigma_factors),
                    default=0.0,
                )
                max_weight_change = np.max(np.abs(weight_vector - previous_weights), initial=0.0)
                global_inner += 1
                iteration_components = {
                    item.id: dict(diagnostics[item.id])
                    for item in self.variance_components.components
                    if item.id in diagnostics
                }
                record: dict[str, object] = {
                    "iteration": global_inner,
                    "adjustmentIteration": outer,
                    "sigmaWeightIteration": inner,
                    "elapsedSeconds": float(perf_counter() - started),
                    "maximumSigmaFactorChange": float(max_sigma_change),
                    "maximumWeightFactorChange": float(max_weight_change),
                    "activeObservationCount": int(np.count_nonzero(weight_vector > 0.0)),
                    "rejectedObservationCount": int(np.count_nonzero(weight_vector == 0.0)),
                    "totalFrozenRedundancy": float(np.sum(frozen_redundancies)),
                    "expectedTotalRedundancy": float(len(keys) - base_solution.rank),
                    "normalMatrixCondition": base_solution.condition,
                    "candidateWrmsM": base_solution.wrms_m,
                    "maximumCandidateParameterUpdateM": max(
                        candidate_by_block.values(),
                        default=0.0,
                    ),
                    "candidateUpdateByBlockM": candidate_by_block,
                    "sigmaFactors": dict(sigma_factors),
                    "weightFactorSummary": weight_factor_summary(
                        current_equations,
                        weight_factors,
                    ),
                    "varianceComponents": iteration_components,
                }
                iterations.append(record)
                if self.iteration_callback is not None:
                    self.iteration_callback(record)

            parameter_converged, tolerances, normalized_updates = parameter_convergence(
                candidate_by_block,
                self.adjustment.convergenceThreshold,
                self.adjustment.convergenceThresholdByParametrizations or {},
            )
            applied_updates = self.parametrization.update(base_solution.delta)
            adjustment_iterations.append(
                {
                    "iteration": outer,
                    "sigmaWeightIterations": sigma_weight_iteration_count,
                    "maximumParameterUpdateM": max(candidate_by_block.values(), default=0.0),
                    "parameterUpdateWithinThreshold": parameter_converged,
                    "convergenceThresholdByParametrizationM": tolerances,
                    "normalizedParameterUpdateByBlock": normalized_updates,
                    "parameterConverged": parameter_converged,
                    "appliedUpdateByBlockM": applied_updates,
                    "wrmsM": base_solution.wrms_m,
                    "equationCount": len(current_equations),
                    "candidateUpdateByBlockM": candidate_by_block,
                    "candidateParameterCorrectionsM": {
                        str(name): float(value) for name, value in zip(self._names, base_solution.delta)
                    },
                    "sigmaFactorsUsedInSolve": sigma_factors_used,
                    "sigmaFactorsForNextIteration": dict(sigma_factors),
                    "weightFactorSummaryUsedInSolve": summary_used,
                    "weightFactorSummaryForNextIteration": (
                        record["weightFactorSummary"] if sigma_weight_iteration_count else summary_used
                    ),
                    "normalMatrixRank": base_solution.rank,
                    "normalMatrixCondition": base_solution.condition,
                    "stateAfterUpdate": self.parametrization.state(),
                }
            )
            final_solution = base_solution
            if parameter_converged:
                converged = True
                termination_reason = "CONVERGED"
                break
            if outer < self.adjustment.maxIterationCount:
                current_equations = self._equations("adjustment-iteration")

        last_solution = final_solution
        final_equations = current_equations
        standardized = np.empty(0)
        residual_sigmas = np.empty(0)
        final_proposed_weight_factors: dict[ObsKey, float] = {}
        if finalize:
            final_equations = self._equations("final-state-report")
            self._prepare_linearization(final_equations)
            weight_vector = np.asarray([weight_factors[key] for key in self._linearization.identities])
            final_solution = self._solve_linearized(final_equations, sigma_factors, weight_vector)
            final_redundancies = self._redundancies(final_solution)
            standardized, residual_sigmas = self._standardized_residuals(
                final_solution.residual_vector,
                final_redundancies,
                sigma_factors,
            )
            keys = list(self._linearization.identities)
            final_proposed_weight_factors = dict(zip(keys, map(float, robust_factors(self.robust.model, self.robust.k0, self.robust.k1, standardized)), strict=True)) if self.adjustment.estimateRobustWeights else dict(weight_factors)
            diagnostics = {}
            if self.adjustment.estimateVarianceFactors:
                _, diagnostics = self._adjust_sigma0(
                    final_solution.residual_vector,
                    final_redundancies,
                    sigma_factors,
                    weight_vector,
                )
        return AdjustmentState(
            solution=final_solution, last_solution=last_solution, equations=final_equations,
            converged=converged, termination_reason=termination_reason, iterations=iterations,
            adjustment_iterations=adjustment_iterations, sigma_factors=sigma_factors, weight_factors=weight_factors,
            initial_sigma_factors=initial_sigma_factors, initial_reference_values=initial_reference_values,
            warm_sigma_count=warm_sigma_count, warm_weight_count=warm_weight_count,
            equation_evaluations=self._equation_evaluations, performance_seconds=self._performance_seconds,
            observation_domain=self.observation_domain, parametrization=self.parametrization,
            utc_offset_hours=self.utc_offset_hours, standardized=standardized, residual_sigmas=residual_sigmas,
            proposed_weight_factors=final_proposed_weight_factors, diagnostics=diagnostics, finalize=finalize,
        )



__all__ = ["LlrAdjustmentSolver"]
