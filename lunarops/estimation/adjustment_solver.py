"""Strict GROOPS-style nonlinear LLR adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from time import perf_counter
from typing import Callable, Hashable, Mapping, Optional, Sequence

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.adjustment_preprocessing import (
    prefit_gross_rejections,
    reject_implausible_apriori_accuracies,
)
from lunarops.estimation.adjustment_reporting import (
    observation_records,
    parameter_records,
    residual_summary,
    weight_factor_summary,
    variance_component_records,
)
from lunarops.estimation.adjustment_result_models import (
    LlrAdjustmentIteration,
    LlrAdjustmentResult,
    LlrAdjustmentStageResult,
)
from lunarops.estimation.adjustment_settings import LlrAdjustmentSettings
from lunarops.estimation.linearization import DenseLinearization
from lunarops.estimation.normal_equation_solver import (
    normal_matrix_condition,
    normal_matrix_rank,
    solve_normal_equations,
)
from lunarops.estimation.normal_equations import NormalEquations
from lunarops.estimation.parameter_convergence import ParameterConvergencePolicy
from lunarops.estimation.robust_weights import create_robust_weight_model
from lunarops.estimation.sigma_factor_estimator import SigmaFactorEstimator
from lunarops.estimation.variance_component_groups import assign_variance_components

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
    wrms_m: Optional[float]
    covariance: np.ndarray
    sigma0_post: Optional[float]
    residual_vector: np.ndarray
    observation_weights: np.ndarray


class LlrAdjustmentSolver:
    def __init__(
        self,
        *,
        equation_source: Callable[[int], list[ObservationEquation]],
        parametrization: ParametrizationList,
        settings: LlrAdjustmentSettings,
        model_state=None,
        initial_sigma_factors: Optional[Mapping[str, float]] = None,
        initial_weight_factors: Optional[Mapping[ObsKey, float]] = None,
        iteration_callback: Optional[Callable[[LlrAdjustmentIteration], None]] = None,
    ) -> None:
        if not callable(equation_source):
            raise TypeError("equation_source must be callable.")
        if not isinstance(parametrization, ParametrizationList):
            raise TypeError("parametrization must be a ParametrizationList.")
        if not isinstance(settings, LlrAdjustmentSettings):
            raise TypeError("settings must be LlrAdjustmentSettings.")
        if initial_sigma_factors is not None and not isinstance(initial_sigma_factors, Mapping):
            raise TypeError("initial_sigma_factors must be a mapping or null.")
        if initial_weight_factors is not None and not isinstance(initial_weight_factors, Mapping):
            raise TypeError("initial_weight_factors must be a mapping or null.")
        if iteration_callback is not None and not callable(iteration_callback):
            raise TypeError("iteration_callback must be callable or null.")

        self.equation_source = equation_source
        self.parametrization = parametrization
        self.settings = settings
        self.adjustment = settings.adjustment
        self.accuracy_screening = settings.accuracy_screening
        self.initialization = settings.initialization
        self.robust = settings.robust_weights
        self.variance_components = settings.variance_components
        self.model_state = model_state
        self.initial_sigma_factors = dict(initial_sigma_factors or {})
        self.initial_weight_factors = dict(initial_weight_factors or {})
        self.iteration_callback = iteration_callback
        self.convergence_policy = ParameterConvergencePolicy(
            default_tolerance_m=self.adjustment.convergence_threshold_m,
            tolerance_by_block_m=self.adjustment.convergence_threshold_by_block_m or {},
        )
        self.robust_weight_model = create_robust_weight_model(
            model=self.robust.model,
            k0=self.robust.k0,
            k1=self.robust.k1,
            active_threshold=self.robust.active_weight_threshold,
        )
        self.sigma_factor_estimator = SigmaFactorEstimator(
            self.variance_components.components,
            active_weight_threshold=self.robust.active_weight_threshold,
        )
        component_ids = {item.id for item in self.variance_components.components}
        if set(self.initial_sigma_factors) - component_ids:
            raise ValueError("Warm-start sigma factors contain unknown components.")
        for component_id, raw in self.initial_sigma_factors.items():
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise TypeError(f"Warm-start sigma factor for {component_id!r} must be real.")
            if not np.isfinite(float(raw)) or float(raw) <= 0.0:
                raise ValueError(f"Warm-start sigma factor for {component_id!r} must be positive and finite.")

        self._equation_iteration = 0
        self._gross_rejected: dict[ObsKey, float] = {}
        self._assignments: dict[ObsKey, str] = {}
        self._retained_keys: Optional[set[ObsKey]] = None
        self._names: list[ParameterName] = []
        self._equation_evaluations: list[dict[str, object]] = []
        self._accuracy_records: dict[ObsKey, dict[str, object]] = {}
        self._accuracy_groups: dict[str, dict[str, object]] = {}
        self._observation_signatures: dict[ObsKey, tuple[str, str, object, float | None]] = {}
        self._linearization: Optional[DenseLinearization] = None
        self._performance_seconds = {"cache_build": 0.0, "normal_solve": 0.0, "redundancy": 0.0, "adjust_sigma0": 0.0}

    def _equations(self, purpose: str) -> list[ObservationEquation]:
        self._equation_iteration += 1
        try:
            equations = list(self.equation_source(self._equation_iteration))
        except TypeError as exc:
            raise TypeError("equation_source must return observation equations.") from exc
        if not all(isinstance(item, ObservationEquation) for item in equations):
            raise TypeError("equation_source must return ObservationEquation objects.")
        identities = [item.observation_id for item in equations]
        if len(set(identities)) != len(identities):
            raise ValueError("Observation identities must be unique.")
        light_time_valid = [item for item in equations if item.light_time_converged]
        retained = (
            light_time_valid
            if self._retained_keys is None
            else [item for item in light_time_valid if item.observation_id in self._retained_keys]
        )
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
                "source_observation_count": len(equations),
                "light_time_converged_count": len(light_time_valid),
                "light_time_nonconverged_count": len(equations) - len(light_time_valid),
                "fixed_domain_returned_count": len(retained),
                "converged_but_outside_fixed_domain_count": len(light_time_valid) - len(retained),
            }
        )
        return retained

    def _prepare_linearization(self, equations: Sequence[ObservationEquation]) -> None:
        if not equations:
            raise ValueError("Adjustment has no usable observations.")
        started = perf_counter()
        self._linearization = DenseLinearization.build(equations, self.parametrization, self._names)
        self._performance_seconds["cache_build"] += perf_counter() - started

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
            if not np.isfinite(factor) or not 0.0 <= factor <= 1.0:
                raise ValueError("Weight factors must be finite and in [0, 1].")
            if not np.isfinite(sigma_factor) or sigma_factor <= 0.0:
                raise ValueError("Sigma factors must be positive and finite.")
            weights.append(factor / (sigma_factor * dense.apriori_sigmas[index]) ** 2)
        result = np.asarray(weights, dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError("Observation weights must be finite.")
        return result

    def _solve_linearized(
        self,
        equations: Sequence[ObservationEquation],
        sigma_factors: Mapping[str, float],
        weight_factors: Mapping[ObsKey, float],
    ) -> _LinearizedSolution:
        dense = self._linearization
        if dense is None or tuple(item.observation_id for item in equations) != dense.identities:
            raise RuntimeError("Prepared linearization does not match equations.")
        started = perf_counter()
        weights = self._observation_weights(dense, sigma_factors, weight_factors)
        active = np.asarray(
            [weight_factors[key] > self.robust.active_weight_threshold for key in dense.identities],
            dtype=bool,
        )
        if np.count_nonzero(active) < len(self._names):
            raise RuntimeError("Too few active observations for the parameter set.")
        weights = np.where(active, weights, 0.0)
        normals = dense.normal_equations(weights, active=active)
        solved = solve_normal_equations(normals)
        delta = np.asarray(solved.delta, dtype=float)
        residual_vector = dense.reduced_observations - dense.design @ delta
        weight_sum = float(np.sum(weights))
        wrms = None if weight_sum <= 0.0 else float(np.sqrt(np.dot(weights, residual_vector**2) / weight_sum))
        self._performance_seconds["normal_solve"] += perf_counter() - started
        return _LinearizedSolution(
            equations=list(equations),
            residuals={key: float(value) for key, value in zip(dense.identities, residual_vector)},
            normals=normals,
            delta=delta,
            wrms_m=wrms,
            covariance=solved.covariance,
            sigma0_post=solved.sigma0_post,
            residual_vector=residual_vector,
            observation_weights=weights,
        )

    def _redundancies(self, solution: _LinearizedSolution) -> np.ndarray:
        dense = self._linearization
        if dense is None:
            raise RuntimeError("Linearization has not been prepared.")
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
        if dense is None:
            raise RuntimeError("Linearization has not been prepared.")
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
        if dense is None:
            raise RuntimeError("Linearization has not been prepared.")
        started = perf_counter()
        estimate = self.sigma_factor_estimator.estimate(
            apriori_sigmas=dense.apriori_sigmas,
            residuals=residuals,
            redundancies=redundancies,
            component_ids=np.asarray([self._assignments[key] for key in dense.identities], dtype=object),
            weight_factors=np.asarray([weight_factors[key] for key in dense.identities], dtype=float),
            sigma_factors=sigma_factors,
        )
        self._performance_seconds["adjust_sigma0"] += perf_counter() - started
        return estimate.sigma_factors, estimate.diagnostics

    def run(self, *, finalize: bool = True) -> LlrAdjustmentResult | LlrAdjustmentStageResult:
        if not isinstance(finalize, bool):
            raise TypeError("finalize must be a boolean.")
        initial_equations = self._equations("initialization")
        if not initial_equations:
            raise ValueError("Adjustment has no light-time-converged observations.")
        self.parametrization.setup(initial_equations, self.model_state)
        self._names = self.parametrization.parameter_names()
        self._gross_rejected = prefit_gross_rejections(
            initial_equations,
            self.parametrization,
            threshold_m=self.adjustment.prefit_gross_threshold_m,
            threshold_by_station_m=self.adjustment.prefit_gross_threshold_by_station_m,
        )
        gross_retained = [item for item in initial_equations if item.observation_id not in self._gross_rejected]
        initial_assignments = assign_variance_components(gross_retained, self.variance_components.components)
        active_initial, self._accuracy_records, self._accuracy_groups = reject_implausible_apriori_accuracies(
            gross_retained,
            initial_assignments,
            minimum_one_way_m=self.accuracy_screening.minimum_one_way_m,
            minimum_group_median_fraction=self.accuracy_screening.minimum_fraction_of_group_median,
        )
        if not active_initial:
            raise ValueError("Adjustment has no observations after permanent prefit rejection.")
        self._retained_keys = {item.observation_id for item in active_initial}
        self._assignments = {key: value for key, value in initial_assignments.items() if key in self._retained_keys}
        self._observation_signatures = {
            item.observation_id: (item.station_key, item.reflector_key, item.transmit_epoch_utc, item.wavelength_nm)
            for item in active_initial
        }
        self.parametrization.setup(active_initial, self.model_state)
        self._names = self.parametrization.parameter_names()
        bias_delta = self.parametrization.initial_update(
            active_initial,
            weight_cap=self.initialization.bias_weight_cap,
            maximum_iterations=self.initialization.bias_maximum_iterations,
        )
        self.parametrization.apply_update(bias_delta)

        sigma_factors = {item.id: 1.0 for item in self.variance_components.components}
        sigma_factors.update({key: float(value) for key, value in self.initial_sigma_factors.items()})
        initial_sigma_factors = dict(sigma_factors)
        weight_factors: dict[ObsKey, float] = {}
        for equation in active_initial:
            raw = self.initial_weight_factors.get(equation.observation_id, 1.0)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, Real)
                or not np.isfinite(float(raw))
                or not 0.0 <= float(raw) <= 1.0
            ):
                raise ValueError(
                    f"Warm-start weight factor for {equation.observation_id!r} must be finite and in [0, 1]."
                )
            weight_factors[equation.observation_id] = float(raw)
        warm_sigma_count = len(self.initial_sigma_factors)
        warm_weight_count = sum(key in self.initial_weight_factors for key in self._retained_keys)

        current_equations = list(active_initial)
        iterations: list[LlrAdjustmentIteration] = []
        adjustment_iterations: list[dict[str, object]] = []
        converged = False
        termination_reason = "MAX_ITERATION_COUNT_REACHED"
        consecutive_converged = 0
        global_inner = 0
        final_solution: Optional[_LinearizedSolution] = None
        diagnostics: dict[str, dict[str, object]] = {}

        for outer in range(1, self.adjustment.max_iteration_count + 1):
            self._prepare_linearization(current_equations)
            sigma_factors_used = dict(sigma_factors)
            weight_factors_used = dict(weight_factors)
            base_solution = self._solve_linearized(current_equations, sigma_factors_used, weight_factors_used)
            frozen_residuals = np.array(base_solution.residual_vector, copy=True)
            frozen_redundancies = self._redundancies(base_solution)
            keys = [item.observation_id for item in current_equations]

            for inner in range(1, SIGMA_WEIGHT_ITERATION_COUNT + 1):
                started = perf_counter()
                previous_sigma = dict(sigma_factors)
                previous_weights = dict(weight_factors)
                sigma_factors, diagnostics = self._adjust_sigma0(
                    frozen_residuals,
                    frozen_redundancies,
                    sigma_factors,
                    weight_factors,
                )
                standardized, _ = self._standardized_residuals(
                    frozen_residuals,
                    frozen_redundancies,
                    sigma_factors,
                )
                weight_factors.update(self.robust_weight_model.target_factors(standardized, keys))
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
                    item.id: dict(diagnostics[item.id]) for item in self.variance_components.components
                }
                record = LlrAdjustmentIteration(
                    iteration=global_inner,
                    adjustment_iteration=outer,
                    sigma_weight_iteration=inner,
                    elapsed_seconds=float(perf_counter() - started),
                    maximum_sigma_factor_change=float(max_sigma_change),
                    maximum_weight_factor_change=float(max_weight_change),
                    active_observation_count=sum(
                        weight_factors[key] > self.robust.active_weight_threshold for key in keys
                    ),
                    rejected_observation_count=sum(
                        weight_factors[key] <= self.robust.active_weight_threshold for key in keys
                    ),
                    total_frozen_redundancy=float(np.sum(frozen_redundancies)),
                    expected_total_redundancy=float(len(keys) - normal_matrix_rank(base_solution.normals)),
                    normal_matrix_condition=normal_matrix_condition(base_solution.normals),
                    candidate_wrms_m=base_solution.wrms_m,
                    maximum_candidate_parameter_update_m=max(
                        self.parametrization.update_norms(base_solution.delta).values(),
                        default=0.0,
                    ),
                    candidate_update_by_block_m=self.parametrization.update_norms(base_solution.delta),
                    sigma_factors=dict(sigma_factors),
                    weight_factor_summary=weight_factor_summary(
                        current_equations,
                        weight_factors,
                        active_threshold=self.robust.active_weight_threshold,
                    ),
                    variance_components=iteration_components,
                )
                iterations.append(record)
                if self.iteration_callback is not None:
                    self.iteration_callback(record)

            candidate_by_block = self.parametrization.update_norms(base_solution.delta)
            convergence = self.convergence_policy.evaluate(candidate_by_block)
            consecutive_converged = consecutive_converged + 1 if convergence.converged else 0
            parameter_converged = consecutive_converged >= self.adjustment.required_consecutive_converged_iterations
            applied_delta = self.adjustment.parameter_update_factor * base_solution.delta
            applied_updates = self.parametrization.apply_update(applied_delta)
            adjustment_iterations.append(
                {
                    "iteration": outer,
                    "sigma_weight_iterations": SIGMA_WEIGHT_ITERATION_COUNT,
                    "maximum_parameter_update_m": max(candidate_by_block.values(), default=0.0),
                    "parameter_update_within_threshold": convergence.converged,
                    "convergence_threshold_by_block_m": convergence.tolerances_m,
                    "normalized_parameter_update_by_block": convergence.normalized_updates,
                    "consecutive_converged_iterations": consecutive_converged,
                    "parameter_converged": parameter_converged,
                    "parameter_update_factor": self.adjustment.parameter_update_factor,
                    "applied_update_by_block_m": applied_updates,
                    "wrms_m": base_solution.wrms_m,
                    "equation_count": len(current_equations),
                    "candidate_update_by_block_m": candidate_by_block,
                    "candidate_parameter_corrections_m": {
                        str(name): float(value) for name, value in zip(self._names, base_solution.delta)
                    },
                    "sigma_factors_used_in_solve": sigma_factors_used,
                    "sigma_factors_for_next_iteration": dict(sigma_factors),
                    "weight_factor_summary_used_in_solve": weight_factor_summary(
                        current_equations,
                        weight_factors_used,
                        active_threshold=self.robust.active_weight_threshold,
                    ),
                    "weight_factor_summary_for_next_iteration": weight_factor_summary(
                        current_equations,
                        weight_factors,
                        active_threshold=self.robust.active_weight_threshold,
                    ),
                    "normal_matrix_rank": normal_matrix_rank(base_solution.normals),
                    "normal_matrix_condition": normal_matrix_condition(base_solution.normals),
                    "state_after_update": self.parametrization.state(),
                }
            )
            final_solution = base_solution
            if parameter_converged:
                converged = True
                termination_reason = "CONVERGED"
                break
            if outer < self.adjustment.max_iteration_count:
                current_equations = self._equations("adjustment-iteration")

        if final_solution is None:
            raise RuntimeError("Adjustment produced no solution.")

        accuracy_rejected = {
            str(key): value for key, value in self._accuracy_records.items() if value["status"] == "REJECTED"
        }
        stage_summary = {
            "converged": converged,
            "termination_reason": termination_reason,
            "source_observation_count": self._equation_evaluations[0]["source_observation_count"],
            "initial_light_time_converged_count": self._equation_evaluations[0]["light_time_converged_count"],
            "gross_rejected_count": len(self._gross_rejected),
            "accuracy_rejected_count": len(accuracy_rejected),
            "retained_observation_count": len(self._retained_keys),
            "last_solved_equation_count": len(final_solution.equations),
            "equation_evaluation_count": len(self._equation_evaluations),
            "adjustment_iteration_count": len(adjustment_iterations),
            "sigma_weight_iteration_count": len(iterations),
            "consecutive_converged_iterations": consecutive_converged,
            "last_normal_matrix_rank": normal_matrix_rank(final_solution.normals),
            "last_normal_matrix_condition": normal_matrix_condition(final_solution.normals),
            "performance_seconds": dict(self._performance_seconds),
        }
        if not finalize:
            return LlrAdjustmentStageResult(
                converged=converged,
                termination_reason=termination_reason,
                equation_evaluations=list(self._equation_evaluations),
                state=self.parametrization.state(),
                sigma_factors=dict(sigma_factors),
                weight_factors=dict(weight_factors),
                sigma_weight_iterations=iterations,
                adjustment_iterations=adjustment_iterations,
                summary=stage_summary,
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
        final_proposed_weight_factors = self.robust_weight_model.target_factors(
            standardized,
            [item.observation_id for item in final_equations],
        )
        _, diagnostics = self._adjust_sigma0(
            final_solution.residual_vector,
            final_redundancies,
            sigma_factors,
            weight_factors,
        )
        current_state_residuals = {
            item.observation_id: float(self.parametrization.reduced_observation(item)) for item in final_equations
        }
        final_parameter_records, normal_summary = parameter_records(
            final_solution.normals,
            final_solution.delta,
            self._names,
            final_solution.covariance,
            final_solution.sigma0_post,
        )
        global_residuals = residual_summary(
            final_equations,
            final_solution.residuals,
            standardized,
            final_solution.observation_weights,
            weight_factors,
            active_threshold=self.robust.active_weight_threshold,
        )
        summary = {
            **stage_summary,
            "final_equation_count": len(final_equations),
            "equation_evaluation_count": len(self._equation_evaluations),
            "performance_seconds": dict(self._performance_seconds),
            **normal_summary,
        }
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
            active_threshold=self.robust.active_weight_threshold,
        )
        return LlrAdjustmentResult(
            converged=converged,
            termination_reason=termination_reason,
            settings={
                **self.settings.to_report_settings(),
                "warm_started_sigma_factor_count": warm_sigma_count,
                "warm_started_weight_factor_count": warm_weight_count,
            },
            equation_evaluations=list(self._equation_evaluations),
            parameter_names=list(self._names),
            state=self.parametrization.state(),
            gross_rejected=dict(self._gross_rejected),
            accuracy_screening={
                "action": "reject",
                "minimum_one_way_m": self.accuracy_screening.minimum_one_way_m,
                "minimum_fraction_of_group_median": self.accuracy_screening.minimum_fraction_of_group_median,
                "rejected_count": len(accuracy_rejected),
                "rejected_observations": accuracy_rejected,
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
                active_threshold=self.robust.active_weight_threshold,
            ),
            normals=final_solution.normals,
            remaining_correction=final_solution.delta,
            cofactor=final_solution.covariance,
            sigma0_post=final_solution.sigma0_post,
        )


__all__ = ["LlrAdjustmentSolver"]
