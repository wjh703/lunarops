"""Strict nonlinear LLR adjustment informed by GROOPS processing patterns."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.classes.time import validate_utc_offset_hours
from lunarops.estimation.adjustment_reporting import (
    observation_records,
    parameter_records,
    report_state,
    residual_summary,
    variance_component_records,
    weight_factor_summary,
)
from lunarops.estimation.adjustment_result_models import (
    LlrAdjustmentEstimateResult,
    LlrAdjustmentObservationDomain,
    LlrAdjustmentResult,
)
from lunarops.estimation.adjustment_settings import LlrAdjustmentSettings
from lunarops.estimation.linearization import DenseLinearization
from lunarops.estimation.normal_equation_solver import (
    normal_matrix_condition,
    normal_matrix_rank,
    solve_normal_equations,
)
from lunarops.estimation.normal_equations import NormalEquations
from lunarops.estimation.parameter_convergence import parameter_convergence
from lunarops.estimation.parameter_products import PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER
from lunarops.estimation.robust_weights import robust_factors
from lunarops.estimation.sigma_factor_estimator import SigmaFactorEstimator

ObsKey = Hashable
SIGMA_WEIGHT_ITERATION_COUNT = 10
MINIMUM_ROBUST_REDUNDANCY = 0.1
REDUNDANCY_EPSILON = 1.0e-12


@dataclass(eq=False, repr=False, slots=True)
class _LinearizedSolution:
    equations: list[ObservationEquation]
    residuals: dict[ObsKey, float]
    normals: NormalEquations
    delta: np.ndarray
    wrms_m: float | None
    covariance: np.ndarray
    sigma0_post: float | None
    residual_vector: np.ndarray
    observation_weights: np.ndarray


class LlrAdjustmentSolver:
    def __init__(
        self,
        *,
        equation_source: Callable[[int], list[ObservationEquation]],
        parametrization: ParametrizationList,
        settings: LlrAdjustmentSettings,
        observation_domain: LlrAdjustmentObservationDomain,
        model_state=None,
        initial_sigma_factors: Mapping[str, float] | None = None,
        initial_weight_factors: Mapping[ObsKey, float] | None = None,
        iteration_callback: Callable[[dict[str, object]], None] | None = None,
        utc_offset_hours: object = 0.0,
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
        self.utc_offset_hours = validate_utc_offset_hours(utc_offset_hours)
        self.sigma_factor_estimator = SigmaFactorEstimator(self.variance_components.components)

        self._equation_iteration = 0
        self._gross_rejected: dict[ObsKey, float] = {}
        self._assignments: dict[ObsKey, str] = {}
        self._retained_keys: set[ObsKey] = set()
        self._names: list[ParameterName] = []
        self._equation_evaluations: list[dict[str, object]] = []
        self._accuracy_records: dict[ObsKey, dict[str, object]] = {}
        self._accuracy_groups: dict[str, dict[str, object]] = {}
        self._observation_signatures: dict[ObsKey, tuple[str, str, object, float | None]] = {}
        self._linearization: DenseLinearization
        self._performance_seconds = {"cacheBuild": 0.0, "normalSolve": 0.0, "redundancy": 0.0, "adjustSigma0": 0.0}

    def _domain(self) -> LlrAdjustmentObservationDomain:
        return LlrAdjustmentObservationDomain(
            gross_rejected=dict(self._gross_rejected),
            retained_keys=set(self._retained_keys),
            assignments=dict(self._assignments),
            accuracy_records={key: dict(value) for key, value in self._accuracy_records.items()},
            accuracy_groups={key: dict(value) for key, value in self._accuracy_groups.items()},
            observation_signatures=dict(self._observation_signatures),
        )

    def _equations(self, purpose: str) -> list[ObservationEquation]:
        self._equation_iteration += 1
        equations = list(self.equation_source(self._equation_iteration))
        identities = [item.observation_id for item in equations]
        if len(set(identities)) != len(identities):
            raise ValueError("Observation identities must be unique.")
        light_time_valid = [item for item in equations if item.light_time_converged]
        retained = [item for item in light_time_valid if item.observation_id in self._retained_keys]
        for equation in retained:
            signature = (
                equation.station_key,
                equation.reflector_key,
                equation.transmit_epoch_utc,
                equation.wavelength_nm,
            )
            expected = self._observation_signatures.get(equation.observation_id)
            if expected is not None and signature != expected:
                raise ValueError(f"Observation {equation.observation_id!r} changed immutable metadata.")
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

    def _prepare_linearization(self, equations: Sequence[ObservationEquation]) -> None:
        if not equations:
            raise ValueError("Adjustment has no usable observations.")
        started = perf_counter()
        self._linearization = DenseLinearization.build(equations, self.parametrization, self._names)
        self._performance_seconds["cacheBuild"] += perf_counter() - started

    def _observation_weights(
        self,
        dense: DenseLinearization,
        sigma_factors: Mapping[str, float],
        weight_factors: Mapping[ObsKey, float],
    ) -> np.ndarray:
        weights = []
        for index, identity in enumerate(dense.identities):
            factor = float(weight_factors[identity])
            sigma_factor = float(sigma_factors[self._assignments[identity]])
            weights.append(factor / (sigma_factor * dense.apriori_sigmas[index]) ** 2)
        return np.asarray(weights, dtype=float)

    def _solve_linearized(
        self,
        equations: Sequence[ObservationEquation],
        sigma_factors: Mapping[str, float],
        weight_factors: Mapping[ObsKey, float],
    ) -> _LinearizedSolution:
        dense = self._linearization
        started = perf_counter()
        weights = self._observation_weights(dense, sigma_factors, weight_factors)
        active = np.asarray(
            [weight_factors[key] > 0.0 for key in dense.identities],
            dtype=bool,
        )
        if np.count_nonzero(active) < len(self._names):
            raise RuntimeError("Too few active observations for the parameter set.")
        weights = np.where(active, weights, 0.0)
        x0 = self.parametrization.reference_values_for(self._names)
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
            equations=list(equations),
            residuals={key: float(value) for key, value in zip(dense.identities, residual_vector)},
            normals=normals,
            delta=delta,
            wrms_m=wrms,
            covariance=solved.covariance,
            sigma0_post=sigma0_post,
            residual_vector=residual_vector,
            observation_weights=weights,
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
    ) -> tuple[dict[ObsKey, float], dict[ObsKey, float]]:
        dense = self._linearization
        sigma0 = np.asarray(
            [
                sigma_factors[self._assignments[key]] * dense.apriori_sigmas[index]
                for index, key in enumerate(dense.identities)
            ],
            dtype=float,
        )
        residual_sigmas = sigma0 * np.sqrt(np.maximum(redundancies, REDUNDANCY_EPSILON))
        standardized = np.zeros_like(residuals)
        eligible = redundancies > MINIMUM_ROBUST_REDUNDANCY
        standardized[eligible] = residuals[eligible] / residual_sigmas[eligible]
        return (
            {key: float(value) for key, value in zip(dense.identities, standardized)},
            {key: float(value) for key, value in zip(dense.identities, residual_sigmas)},
        )

    def _adjust_sigma0(
        self,
        residuals: np.ndarray,
        redundancies: np.ndarray,
        sigma_factors: Mapping[str, float],
        weight_factors: Mapping[ObsKey, float],
    ) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
        dense = self._linearization
        started = perf_counter()
        estimate = self.sigma_factor_estimator.estimate(
            apriori_sigmas=dense.apriori_sigmas,
            residuals=residuals,
            redundancies=redundancies,
            component_ids=np.asarray([self._assignments[key] for key in dense.identities], dtype=object),
            weight_factors=np.asarray([weight_factors[key] for key in dense.identities], dtype=float),
            sigma_factors=sigma_factors,
        )
        self._performance_seconds["adjustSigma0"] += perf_counter() - started
        return estimate.sigma_factors, estimate.diagnostics

    def _robust_weight_factors(
        self,
        standardized: Mapping[ObsKey, float],
        keys: Sequence[ObsKey],
    ) -> dict[ObsKey, float]:
        values = np.asarray([standardized[key] for key in keys], dtype=float)
        factors = robust_factors(self.robust.model, self.robust.k0, self.robust.k1, values)
        return {key: float(value) for key, value in zip(keys, factors)}

    def run(self, *, finalize: bool = True) -> LlrAdjustmentResult | LlrAdjustmentEstimateResult:
        domain = self.observation_domain
        self._gross_rejected = dict(domain.gross_rejected)
        self._retained_keys = set(domain.retained_keys)
        self._assignments = dict(domain.assignments)
        self._accuracy_records = {key: dict(value) for key, value in domain.accuracy_records.items()}
        self._accuracy_groups = {key: dict(value) for key, value in domain.accuracy_groups.items()}
        self._observation_signatures = dict(domain.observation_signatures)
        initial_equations = self._equations("initialization")
        if not initial_equations:
            raise ValueError("Adjustment has no light-time-converged observations.")
        self.parametrization.setup(initial_equations, self.model_state)
        self._names = self.parametrization.parameter_names()
        initial_reference_values = self.parametrization.reference_values_for(self._names).copy()
        sigma_factors = {item.id: 1.0 for item in self.variance_components.components}
        sigma_factors.update({key: float(value) for key, value in self.initial_sigma_factors.items()})
        initial_sigma_factors = dict(sigma_factors)
        # A retained observation can temporarily disappear when its light-time
        # solution does not converge. Keep its weight so it can safely return in
        # a later linearization or processing step.
        weight_factors: dict[ObsKey, float] = {
            identity: float(self.initial_weight_factors.get(identity, 1.0)) for identity in self._retained_keys
        }
        warm_sigma_count = len(self.initial_sigma_factors)
        warm_weight_count = sum(key in self.initial_weight_factors for key in self._retained_keys)

        current_equations = initial_equations
        iterations: list[dict[str, object]] = []
        adjustment_iterations: list[dict[str, object]] = []
        converged = False
        termination_reason = "MAX_ITERATION_COUNT_REACHED"
        global_inner = 0
        diagnostics: dict[str, dict[str, object]] = {}

        for outer in range(1, self.adjustment.max_iteration_count + 1):
            self._prepare_linearization(current_equations)
            sigma_factors_used = dict(sigma_factors)
            weight_factors_used = dict(weight_factors)
            base_solution = self._solve_linearized(current_equations, sigma_factors_used, weight_factors_used)
            frozen_residuals = base_solution.residual_vector
            frozen_redundancies = (
                self._redundancies(base_solution)
                if self.adjustment.compute_residuals
                else np.zeros(len(current_equations), dtype=float)
            )
            keys = [item.observation_id for item in current_equations]

            sigma_weight_iteration_count = 0
            if self.adjustment.compute_residuals:
                if self.adjustment.adjust_sigma0 and self.adjustment.compute_weights:
                    sigma_weight_iteration_count = SIGMA_WEIGHT_ITERATION_COUNT
                elif self.adjustment.adjust_sigma0 or self.adjustment.compute_weights:
                    sigma_weight_iteration_count = 1
            for inner in range(1, sigma_weight_iteration_count + 1):
                started = perf_counter()
                previous_sigma = dict(sigma_factors)
                previous_weights = dict(weight_factors)
                if self.adjustment.adjust_sigma0:
                    sigma_factors, diagnostics = self._adjust_sigma0(
                        frozen_residuals,
                        frozen_redundancies,
                        sigma_factors,
                        weight_factors,
                    )
                if self.adjustment.compute_weights:
                    standardized, _ = self._standardized_residuals(
                        frozen_residuals,
                        frozen_redundancies,
                        sigma_factors,
                    )
                    weight_factors.update(self._robust_weight_factors(standardized, keys))
                max_sigma_change = max(
                    (abs(sigma_factors[key] - previous_sigma[key]) for key in sigma_factors),
                    default=0.0,
                )
                max_weight_change = max(
                    (abs(weight_factors[key] - previous_weights[key]) for key in keys),
                    default=0.0,
                )
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
                    "activeObservationCount": sum(weight_factors[key] > 0.0 for key in keys),
                    "rejectedObservationCount": sum(weight_factors[key] == 0.0 for key in keys),
                    "totalFrozenRedundancy": float(np.sum(frozen_redundancies)),
                    "expectedTotalRedundancy": float(len(keys) - normal_matrix_rank(base_solution.normals)),
                    "normalMatrixCondition": normal_matrix_condition(base_solution.normals),
                    "candidateWrmsM": base_solution.wrms_m,
                    "maximumCandidateParameterUpdateM": max(
                        self.parametrization.update_norms(base_solution.delta).values(),
                        default=0.0,
                    ),
                    "candidateUpdateByBlockM": self.parametrization.update_norms(base_solution.delta),
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

            candidate_by_block = self.parametrization.update_norms(base_solution.delta)
            parameter_converged, tolerances, normalized_updates = parameter_convergence(
                candidate_by_block,
                self.adjustment.convergence_threshold_m,
                self.adjustment.convergence_threshold_by_parametrization_m or {},
            )
            applied_updates = self.parametrization.apply_update(base_solution.delta)
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
                    "weightFactorSummaryUsedInSolve": weight_factor_summary(
                        current_equations,
                        weight_factors_used,
                    ),
                    "weightFactorSummaryForNextIteration": weight_factor_summary(
                        current_equations,
                        weight_factors,
                    ),
                    "normalMatrixRank": normal_matrix_rank(base_solution.normals),
                    "normalMatrixCondition": normal_matrix_condition(base_solution.normals),
                    "stateAfterUpdate": report_state(self.parametrization.state()),
                }
            )
            final_solution = base_solution
            if parameter_converged:
                converged = True
                termination_reason = "CONVERGED"
                break
            if outer < self.adjustment.max_iteration_count:
                current_equations = self._equations("adjustment-iteration")

        accuracy_rejected = {
            str(key): value for key, value in self._accuracy_records.items() if value["status"] == "REJECTED"
        }
        summary = {
            "converged": converged,
            "terminationReason": termination_reason,
            "sourceObservationCount": self._equation_evaluations[0]["sourceObservationCount"],
            "initialLightTimeConvergedCount": self._equation_evaluations[0]["lightTimeConvergedCount"],
            "grossRejectedCount": len(self._gross_rejected),
            "accuracyRejectedCount": len(accuracy_rejected),
            "retainedObservationCount": len(self._retained_keys),
            "lastSolvedEquationCount": len(final_solution.equations),
            "equationEvaluationCount": len(self._equation_evaluations),
            "adjustmentIterationCount": len(adjustment_iterations),
            "sigmaWeightIterationCount": len(iterations),
            "lastNormalMatrixRank": normal_matrix_rank(final_solution.normals),
            "lastNormalMatrixCondition": normal_matrix_condition(final_solution.normals),
            "performanceSeconds": dict(self._performance_seconds),
        }
        if not finalize:
            return LlrAdjustmentEstimateResult(
                converged=converged,
                termination_reason=termination_reason,
                equation_evaluations=list(self._equation_evaluations),
                state=report_state(self.parametrization.state()),
                sigma_factors=dict(sigma_factors),
                weight_factors=dict(weight_factors),
                sigma_weight_iterations=iterations,
                adjustment_iterations=adjustment_iterations,
                summary=summary,
                observation_domain=self._domain(),
            )

        final_equations = self._equations("final-state-report")
        self._prepare_linearization(final_equations)
        final_solution = self._solve_linearized(final_equations, sigma_factors, weight_factors)
        final_redundancies = self._redundancies(final_solution)
        standardized, residual_sigmas = self._standardized_residuals(
            final_solution.residual_vector,
            final_redundancies,
            sigma_factors,
        )
        final_proposed_weight_factors = (
            self._robust_weight_factors(
                standardized,
                [item.observation_id for item in final_equations],
            )
            if self.adjustment.compute_weights
            else dict(weight_factors)
        )
        diagnostics = {}
        if self.adjustment.adjust_sigma0:
            _, diagnostics = self._adjust_sigma0(
                final_solution.residual_vector,
                final_redundancies,
                sigma_factors,
                weight_factors,
            )
        current_state_residuals = {
            item.observation_id: float(self.parametrization.reduced_observation(item)) for item in final_equations
        }
        final_parameter_records = parameter_records(
            final_solution.normals,
            final_solution.delta,
            self._names,
            final_solution.covariance,
            final_solution.sigma0_post,
            initial_reference_values,
        )
        global_residuals = residual_summary(
            final_equations,
            final_solution.residuals,
            standardized,
            final_solution.observation_weights,
            weight_factors,
        )
        summary.update(
            {
                "finalEquationCount": len(final_equations),
                "equationEvaluationCount": len(self._equation_evaluations),
                "performanceSeconds": dict(self._performance_seconds),
                "observationCount": int(final_solution.normals.obs_count),
                "parameterCount": len(self._names),
                "rank": normal_matrix_rank(final_solution.normals),
                "conditionNumber": normal_matrix_condition(final_solution.normals),
                "sigma0Post": final_solution.sigma0_post,
                "parameterUncertaintySigmaMultiplier": PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER,
            }
        )
        component_records = variance_component_records(
            final_equations,
            final_solution.residuals,
            standardized,
            sigma_factors,
            initial_sigma_factors,
            weight_factors,
            final_proposed_weight_factors,
            assignments=self._assignments,
            components=self.variance_components.components,
            diagnostics=diagnostics,
            accuracy_screening_groups=self._accuracy_groups,
            utc_offset_hours=self.utc_offset_hours,
        )
        return LlrAdjustmentResult(
            converged=converged,
            termination_reason=termination_reason,
            settings={
                **self.settings.to_report_settings(),
                "warmStartedSigmaFactorCount": warm_sigma_count,
                "warmStartedWeightFactorCount": warm_weight_count,
            },
            equation_evaluations=list(self._equation_evaluations),
            state=report_state(self.parametrization.state()),
            gross_rejected=dict(self._gross_rejected),
            accuracy_screening={
                "action": "reject",
                "minimumOneWayM": self.accuracy_screening.minimum_one_way_m,
                "minimumFractionOfGroupMedian": self.accuracy_screening.minimum_fraction_of_group_median,
                "rejectedCount": len(accuracy_rejected),
                "rejectedObservations": accuracy_rejected,
                "groups": dict(self._accuracy_groups),
            },
            sigma_factors=dict(sigma_factors),
            weight_factors=dict(weight_factors),
            sigma_weight_iterations=iterations,
            adjustment_iterations=adjustment_iterations,
            summary=summary,
            parameters=final_parameter_records,
            global_residuals=global_residuals,
            variance_components=component_records,
            observations=observation_records(
                final_equations,
                current_state_residuals,
                final_solution.residuals,
                residual_sigmas,
                standardized,
                sigma_factors,
                weight_factors,
                final_proposed_weight_factors,
                assignments=self._assignments,
                parametrization=self.parametrization,
                components=self.variance_components.components,
                accuracy_screening_records=self._accuracy_records,
                utc_offset_hours=self.utc_offset_hours,
            ),
            normals=final_solution.normals,
            remaining_correction=final_solution.delta,
            cofactor=final_solution.covariance,
            sigma0_post=final_solution.sigma0_post,
        )


__all__ = ["LlrAdjustmentSolver"]
