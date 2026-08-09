"""LLR nonlinear adjustment with robust weights and variance-component estimation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Hashable, Mapping, Optional, Sequence, cast

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.convergence import ParameterConvergencePolicy
from lunarops.estimation.adjustment_preprocessing import (
    floor_prefit_uncertainties,
    initialize_mad_scales,
    prefit_gross_rejections,
)
from lunarops.estimation.adjustment_options import LlrAdjustmentOptions
from lunarops.estimation.adjustment_results import (
    LlrAdjustmentIteration,
    LlrAdjustmentResult,
    observation_records,
    parameter_records,
    residual_summary,
    robust_factor_summary,
    variance_component_records,
)
from lunarops.estimation.linearized_least_squares import (
    DenseLinearization,
    normal_matrix_condition,
    solve_normal_equations,
)
from lunarops.estimation.normal_equations import NormalEquations
from lunarops.estimation.robust_weights import (
    create_robust_weight_model,
    maximum_robust_factor_change,
)
from lunarops.estimation.helmert_vce import HelmertVceEstimator
from lunarops.estimation.variance_components import assign_variance_components

ObsKey = Hashable


@dataclass(eq=False, repr=False, slots=True)
class _InnerSolution:
    equations: list[ObservationEquation]
    residuals: dict[ObsKey, float]
    normals: NormalEquations
    delta: np.ndarray
    wrms_m: Optional[float]
    covariance: np.ndarray
    sigma0_post: Optional[float]
    residual_vector: np.ndarray
    weights: np.ndarray


class LlrAdjustmentSolver:
    def __init__(
        self,
        *,
        equation_source: Callable[[int], list[ObservationEquation]],
        parametrization: ParametrizationList,
        options: LlrAdjustmentOptions,
        model_state=None,
        initial_scales: Optional[Mapping[str, float]] = None,
        initial_factors: Optional[Mapping[ObsKey, float]] = None,
        iteration_callback: Optional[Callable[[LlrAdjustmentIteration], None]] = None,
    ) -> None:
        self.equation_source = equation_source
        self.parametrization = parametrization
        self.options = options
        self.model_state = model_state
        self.initial_scales = dict(initial_scales or {})
        self.initial_factors = dict(initial_factors or {})
        self.iteration_callback = iteration_callback
        self.convergence_policy = ParameterConvergencePolicy(
            default_tolerance_m=self.options.update_tolerance_m,
            tolerance_by_block_m=self.options.update_tolerance_by_block_m or {},
        )
        self.robust_weight_model = create_robust_weight_model(
            model=self.options.robust_model,
            k0=self.options.k0,
            k1=self.options.k1,
            active_threshold=self.options.minimum_nonzero_robust_factor,
            convergence_floor=self.options.minimum_robust_factor_for_convergence,
            change_quantile=self.options.robust_factor_change_quantile,
        )
        self.vce_estimator = HelmertVceEstimator(
            tuple(self.options.components),
            minimum_nonzero_factor=self.options.minimum_nonzero_robust_factor,
            minimum_effective_redundancy=self.options.minimum_effective_redundancy,
            minimum_variance_ratio_per_iteration=self.options.minimum_variance_ratio_per_iteration,
            maximum_variance_ratio_per_iteration=self.options.maximum_variance_ratio_per_iteration,
        )
        self._equation_iteration = 0
        self._gross_rejected: dict[ObsKey, float] = {}
        self._assignments: dict[ObsKey, str] = {}
        self._retained_keys: Optional[set[ObsKey]] = None
        self._names: list[ParameterName] = []
        self._equation_evaluations: list[dict[str, object]] = []
        self._uncertainty_qc_records: dict[ObsKey, dict[str, object]] = {}
        self._uncertainty_qc_groups: dict[str, dict[str, object]] = {}
        self._linearization: Optional[DenseLinearization] = None
        self._performance_seconds = {
            "cache_build": 0.0,
            "normal_solve": 0.0,
            "leverage": 0.0,
            "vce": 0.0,
        }

    def _equations(self, purpose: str) -> list[ObservationEquation]:
        self._equation_iteration += 1
        equations = self.equation_source(self._equation_iteration)
        identities = [eq.observation_id for eq in equations]
        if len(set(identities)) != len(identities):
            duplicate = next(key for index, key in enumerate(identities) if key in identities[:index])
            raise ValueError(f"Observation identity {duplicate!r} is not unique.")

        active = [eq for eq in equations if eq.light_time_converged]
        retained = (
            active if self._retained_keys is None else [eq for eq in active if eq.observation_id in self._retained_keys]
        )
        self._equation_evaluations.append(
            {
                "linearization_iteration": self._equation_iteration,
                "purpose": purpose,
                "source_observation_count": len(equations),
                "light_time_converged_count": len(active),
                "light_time_nonconverged_count": len(equations) - len(active),
                "fixed_domain_returned_count": len(retained),
                "converged_but_outside_fixed_domain_count": len(active) - len(retained),
            }
        )
        if self._uncertainty_qc_records:
            retained = [
                replace(
                    equation,
                    sigma_one_way_m=cast(
                        float,
                        self._uncertainty_qc_records[equation.observation_id]["effective_sigma_m"],
                    ),
                )
                for equation in retained
            ]
        return retained

    def _prepare_linearization(self, equations: Sequence[ObservationEquation]) -> None:
        started = perf_counter()
        self._linearization = DenseLinearization.build(equations, self.parametrization, self._names)
        self._performance_seconds["cache_build"] += perf_counter() - started

    def _dense_weights(
        self,
        linearization: DenseLinearization,
        scales: Mapping[str, float],
        factors: Mapping[ObsKey, float],
    ) -> np.ndarray:
        factor_values = np.asarray(
            [float(factors[key]) for key in linearization.identities],
            dtype=float,
        )
        scale_values = np.asarray(
            [float(scales[self._assignments[key]]) for key in linearization.identities],
            dtype=float,
        )
        return factor_values / (scale_values**2 * linearization.sigmas**2)

    def _solve_linearized(
        self,
        equations: Sequence[ObservationEquation],
        scales: Mapping[str, float],
        factors: Mapping[ObsKey, float],
    ) -> _InnerSolution:
        equations = list(equations)
        dense = self._linearization
        if dense is None:
            raise RuntimeError("Linearization has not been prepared.")
        started = perf_counter()
        if tuple(eq.observation_id for eq in equations) != dense.identities:
            raise RuntimeError("Linearization does not match equations.")
        weights = self._dense_weights(dense, scales, factors)
        active = (
            np.asarray([float(factors[key]) for key in dense.identities], dtype=float)
            > self.options.minimum_nonzero_robust_factor
        )
        if np.count_nonzero(active) < len(self._names):
            raise RuntimeError("Too few non-zero-factor observations for the current parameter set.")
        normals = dense.normal_equations(weights, active=active)
        solved = solve_normal_equations(normals)
        delta = np.asarray(solved.delta, dtype=float)
        residual_vector = dense.reduced_observations - dense.design @ delta
        identities = dense.identities
        if solved.covariance is None:
            raise RuntimeError("Normal-equation covariance is unavailable.")

        sum_weight = float(np.sum(weights))
        wrms = None if sum_weight <= 0.0 else float(np.sqrt(np.dot(weights, residual_vector**2) / sum_weight))
        self._performance_seconds["normal_solve"] += perf_counter() - started
        return _InnerSolution(
            equations=equations,
            residuals={key: float(value) for key, value in zip(identities, residual_vector)},
            normals=normals,
            delta=delta,
            wrms_m=wrms,
            covariance=solved.covariance,
            sigma0_post=solved.sigma0_post,
            residual_vector=residual_vector,
            weights=weights,
        )

    def _standardized_residuals(
        self,
        solution: _InnerSolution,
        scales: Mapping[str, float],
        residuals: Optional[Mapping[ObsKey, float]] = None,
    ) -> tuple[dict[ObsKey, float], dict[ObsKey, float]]:
        started = perf_counter()
        dense = self._linearization
        if dense is None:
            raise RuntimeError("Linearization has not been prepared.")
        identities = dense.identities
        residual_values = (
            solution.residual_vector
            if residuals is None
            else np.asarray([residuals[key] for key in identities], dtype=float)
        )
        scale_values = np.asarray(
            [float(scales[self._assignments[key]]) for key in identities],
            dtype=float,
        )
        base_variances = scale_values**2 * dense.sigmas**2
        base_weights_array = 1.0 / base_variances
        base_normals = dense.normal_equations(base_weights_array)
        covariance = solve_normal_equations(base_normals).covariance
        if covariance is None:
            raise RuntimeError("Base normal equation covariance is unavailable.")
        projected = dense.design @ covariance
        leverage = base_weights_array * np.einsum("ij,ij->i", projected, dense.design)

        if not np.all(np.isfinite(leverage)):
            raise RuntimeError("Observation leverage contains non-finite values.")
        one_minus_leverage = np.maximum(
            1.0 - leverage,
            self.options.minimum_one_minus_leverage,
        )
        residual_sigma_values = np.sqrt(base_variances * one_minus_leverage)
        if not np.all(np.isfinite(residual_sigma_values)) or np.any(residual_sigma_values <= 0.0):
            raise RuntimeError("Residual standard deviations must be positive and finite.")
        standardized_values = residual_values / residual_sigma_values
        self._performance_seconds["leverage"] += perf_counter() - started
        return (
            {key: float(value) for key, value in zip(identities, standardized_values)},
            {key: float(value) for key, value in zip(identities, residual_sigma_values)},
        )

    def _update_scales(
        self,
        solution: _InnerSolution,
        scales: Mapping[str, float],
        factors: Mapping[ObsKey, float],
    ) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
        started = perf_counter()
        dense = self._linearization
        if dense is None:
            raise RuntimeError("Linearization has not been prepared.")
        factor_values = np.asarray([float(factors[key]) for key in dense.identities], dtype=float)
        component_ids = np.asarray([self._assignments[key] for key in dense.identities], dtype=object)
        estimate = self.vce_estimator.estimate(
            design=dense.design,
            sigmas=dense.sigmas,
            residuals=solution.residual_vector,
            component_ids=component_ids,
            factors=factor_values,
            scales=scales,
            normals=solution.normals,
            covariance=solution.covariance,
        )
        self._performance_seconds["vce"] += perf_counter() - started
        return estimate.scales, estimate.diagnostics

    def run(self) -> LlrAdjustmentResult:
        initial_equations = self._equations("initialization")
        self.parametrization.setup(initial_equations, self.model_state)
        self._names = self.parametrization.parameter_names()
        self._gross_rejected = prefit_gross_rejections(
            initial_equations,
            self.parametrization,
            threshold_m=self.options.prefit_gross_threshold_m,
            threshold_by_station_m=self.options.prefit_gross_threshold_by_station_m,
        )
        active_initial = [eq for eq in initial_equations if eq.observation_id not in self._gross_rejected]
        initial_assignments = assign_variance_components(active_initial, self.options.components)
        (
            active_initial,
            self._uncertainty_qc_records,
            self._uncertainty_qc_groups,
        ) = floor_prefit_uncertainties(
            active_initial,
            initial_assignments,
            minimum_sigma_m=self.options.uncertainty_floor_minimum_m,
            minimum_group_median_fraction=(self.options.uncertainty_floor_group_median_fraction),
        )
        self._retained_keys = {eq.observation_id for eq in active_initial}
        self.parametrization.setup(active_initial, self.model_state)
        self._names = self.parametrization.parameter_names()
        self._assignments = dict(initial_assignments)

        bias_delta = self.parametrization.initial_update(
            active_initial,
            weight_cap=self.options.bias_weight_cap,
            maximum_iterations=self.options.bias_maximum_iterations,
        )
        self.parametrization.apply_update(bias_delta)
        scales = initialize_mad_scales(
            active_initial,
            self.parametrization,
            self._assignments,
            self.options.components,
            minimum_count=self.options.minimum_mad_count,
            minimum_scale=self.options.minimum_initial_scale,
        )
        warm_scale_count = 0
        for component in self.options.components:
            value = self.initial_scales.get(component.id)
            if value is not None and np.isfinite(value) and value > 0.0:
                scales[component.id] = float(value)
                warm_scale_count += 1
        initial_scales = dict(scales)
        factors = {}
        warm_factor_count = 0
        for equation in active_initial:
            value = self.initial_factors.get(equation.observation_id)
            if value is not None and np.isfinite(value) and 0.0 <= value <= 1.0:
                factors[equation.observation_id] = float(value)
                warm_factor_count += 1
            else:
                factors[equation.observation_id] = 1.0
        target_factors = dict(factors)
        current_equations = list(active_initial)
        self._prepare_linearization(current_equations)
        iterations: list[LlrAdjustmentIteration] = []
        linearizations: list[dict[str, object]] = []
        diagnostics: dict[str, dict[str, object]] = {}
        converged = False
        termination_reason = "MAXIMUM_LINEARIZATIONS_REACHED"
        final_solution: Optional[_InnerSolution] = None
        global_stochastic_iteration = 0
        consecutive_converged_linearizations = 0

        for linearization in range(1, self.options.maximum_linearizations + 1):
            stochastic_converged = False
            stochastic_iterations_used = 0

            for stochastic in range(1, self.options.maximum_stochastic_iterations + 1):
                stochastic_started = perf_counter()
                keys = [eq.observation_id for eq in current_equations]
                base_solution = self._solve_linearized(
                    current_equations,
                    scales,
                    factors,
                )
                standardized, _ = self._standardized_residuals(
                    base_solution,
                    scales,
                )
                robust_update = self.robust_weight_model.update(
                    standardized,
                    factors,
                    target_factors,
                    keys,
                )
                next_target_factors = dict(target_factors)
                next_target_factors.update(robust_update.target_factors)
                next_factors = dict(factors)
                next_factors.update(robust_update.applied_factors)
                factor_target_change = robust_update.target_change_quantile
                active_set_change = robust_update.active_set_change_fraction
                robust_solution = self._solve_linearized(
                    current_equations,
                    scales,
                    next_factors,
                )
                next_scales, diagnostics = self._update_scales(
                    robust_solution,
                    scales,
                    next_factors,
                )
                variance_ratio_change = max(
                    abs((next_scales[component.id] ** 2) / (scales[component.id] ** 2) - 1.0)
                    for component in self.options.components
                )
                scale_log_target_change = max(
                    cast(float, diagnostics[component.id]["target_scale_log_change"])
                    for component in self.options.components
                )
                factor_change = maximum_robust_factor_change(
                    factors,
                    next_factors,
                    keys,
                    significance_floor=(self.options.minimum_robust_factor_for_convergence),
                )
                current_factor_values = [next_factors[eq.observation_id] for eq in current_equations]
                candidate_update_by_block = self.parametrization.update_norms(robust_solution.delta)
                stochastic_converged = (
                    scale_log_target_change <= self.options.scale_log_tolerance
                    and factor_target_change <= self.options.robust_factor_change_tolerance
                    and active_set_change <= self.options.active_set_change_tolerance
                )
                iteration_components = {
                    component.id: {
                        **diagnostics[component.id],
                        "scale_before": float(scales[component.id]),
                        "scale_after": float(next_scales[component.id]),
                    }
                    for component in self.options.components
                }
                global_stochastic_iteration += 1
                stochastic_iterations_used = stochastic
                iterations.append(
                    LlrAdjustmentIteration(
                        iteration=global_stochastic_iteration,
                        linearization_iteration=linearization,
                        stochastic_iteration=stochastic,
                        elapsed_seconds=float(perf_counter() - stochastic_started),
                        maximum_variance_ratio_change=float(variance_ratio_change),
                        maximum_robust_factor_change=float(factor_change),
                        maximum_scale_log_target_change=float(scale_log_target_change),
                        robust_factor_target_change_quantile=float(factor_target_change),
                        active_set_change_fraction=float(active_set_change),
                        stochastic_converged=bool(stochastic_converged),
                        target_rejected_observation_count=sum(
                            next_target_factors[key] <= self.options.minimum_nonzero_robust_factor for key in keys
                        ),
                        active_observation_count=sum(
                            value > self.options.minimum_nonzero_robust_factor for value in current_factor_values
                        ),
                        rejected_observation_count=sum(
                            value <= self.options.minimum_nonzero_robust_factor for value in current_factor_values
                        ),
                        total_effective_redundancy=float(
                            sum(cast(float, item["effective_redundancy"]) for item in diagnostics.values())
                        ),
                        expected_total_redundancy=float(
                            sum(cast(float, item["active_count"]) for item in diagnostics.values())
                            - np.linalg.matrix_rank(robust_solution.normals.N)
                        ),
                        normal_matrix_condition=normal_matrix_condition(robust_solution.normals),
                        candidate_wrms_m=robust_solution.wrms_m,
                        maximum_candidate_parameter_update_m=max(
                            candidate_update_by_block.values(),
                            default=0.0,
                        ),
                        candidate_update_by_block_m=candidate_update_by_block,
                        scales={key: float(value) for key, value in next_scales.items()},
                        robust_factor_summary=robust_factor_summary(
                            current_equations,
                            next_factors,
                            active_threshold=self.options.minimum_nonzero_robust_factor,
                        ),
                        variance_components=iteration_components,
                    )
                )
                if self.iteration_callback is not None:
                    self.iteration_callback(iterations[-1])

                scales = next_scales
                factors = next_factors
                target_factors = next_target_factors
                final_solution = robust_solution
                if stochastic_converged:
                    break

            if final_solution is None:
                raise RuntimeError("Stochastic model produced no linearized solution.")

            final_solution = self._solve_linearized(
                current_equations,
                scales,
                factors,
            )
            candidate_update_by_block = self.parametrization.update_norms(final_solution.delta)
            maximum_update = max(
                candidate_update_by_block.values(),
                default=0.0,
            )
            convergence_evaluation = self.convergence_policy.evaluate(candidate_update_by_block)
            update_within_tolerance = convergence_evaluation.converged
            if stochastic_converged and update_within_tolerance:
                consecutive_converged_linearizations += 1
            else:
                consecutive_converged_linearizations = 0
            parameter_converged = (
                consecutive_converged_linearizations >= self.options.required_consecutive_converged_linearizations
            )

            applied_delta = self.options.parameter_update_factor * final_solution.delta
            applied_updates = self.parametrization.apply_update(applied_delta)
            linearizations.append(
                {
                    "iteration": linearization,
                    "stochastic_iterations": stochastic_iterations_used,
                    "stochastic_converged": stochastic_converged,
                    "stochastic_iteration_limit_reached": (not stochastic_converged),
                    "maximum_parameter_update_m": float(maximum_update),
                    "parameter_update_within_tolerance": bool(update_within_tolerance),
                    "parameter_update_tolerance_by_block_m": convergence_evaluation.tolerances_m,
                    "normalized_parameter_update_by_block": convergence_evaluation.normalized_updates,
                    "consecutive_converged_linearizations": (consecutive_converged_linearizations),
                    "parameter_converged": bool(parameter_converged),
                    "parameter_update_factor": (self.options.parameter_update_factor),
                    "applied_update_by_block_m": applied_updates,
                    "wrms_m": final_solution.wrms_m,
                    "equation_count": len(current_equations),
                    "candidate_update_by_block_m": candidate_update_by_block,
                    "candidate_parameter_corrections_m": {
                        str(name): float(value)
                        for name, value in zip(
                            self._names,
                            final_solution.delta,
                        )
                    },
                    "scales": {key: float(value) for key, value in scales.items()},
                    "robust_factor_summary": robust_factor_summary(
                        current_equations,
                        factors,
                        active_threshold=self.options.minimum_nonzero_robust_factor,
                    ),
                    "normal_matrix_rank": int(np.linalg.matrix_rank(final_solution.normals.N)),
                    "normal_matrix_condition": normal_matrix_condition(final_solution.normals),
                    "state_after_update": self.parametrization.state(),
                }
            )

            if parameter_converged:
                converged = True
                termination_reason = "CONVERGED"
                break
            if linearization < self.options.maximum_linearizations:
                current_equations = self._equations("linearization")
                self._prepare_linearization(current_equations)

        if final_solution is None:
            raise RuntimeError("Adjustment produced no final linearized solution.")

        # The last candidate update has already been absorbed into the model.
        # Re-evaluate once so state, residuals, normals, and remaining
        # correction all refer to the same final model state.
        final_equations = self._equations("final-state-report")
        self._prepare_linearization(final_equations)
        final_solution = self._solve_linearized(
            final_equations,
            scales,
            factors,
        )
        current_state_residuals = {
            equation.observation_id: float(self.parametrization.reduced_observation(equation))
            for equation in final_equations
        }
        standardized, residual_sigmas = self._standardized_residuals(
            final_solution,
            scales,
            residuals=current_state_residuals,
        )
        final_state_proposed_factors = self.robust_weight_model.target_factors(
            standardized,
            [equation.observation_id for equation in final_equations],
        )
        _, diagnostics = self._update_scales(
            final_solution,
            scales,
            factors,
        )
        final_parameter_records, normal_summary = parameter_records(
            final_solution.normals,
            final_solution.delta,
            self._names,
            final_solution.covariance,
            final_solution.sigma0_post,
        )
        global_residuals = residual_summary(
            final_solution.equations,
            current_state_residuals,
            standardized,
            final_solution.weights,
            factors,
            active_threshold=self.options.minimum_nonzero_robust_factor,
        )
        settings = {
            **self.options.settings(),
            "warm_started_scale_count": warm_scale_count,
            "warm_started_factor_count": warm_factor_count,
        }
        first_evaluation = self._equation_evaluations[0]
        summary = {
            "converged": converged,
            "termination_reason": termination_reason,
            "source_observation_count": first_evaluation["source_observation_count"],
            "initial_light_time_converged_count": first_evaluation["light_time_converged_count"],
            "initial_light_time_nonconverged_count": first_evaluation["light_time_nonconverged_count"],
            "gross_rejected_count": len(self._gross_rejected),
            "uncertainty_sigma_floored_count": sum(
                item["status"] == "FLOORED" for item in self._uncertainty_qc_records.values()
            ),
            "retained_uncertainty_sigma_floored_count": sum(
                self._uncertainty_qc_records[key]["status"] == "FLOORED" for key in (self._retained_keys or ())
            ),
            "retained_observation_count": len(self._retained_keys or ()),
            "final_equation_count": len(final_solution.equations),
            "equation_evaluation_count": len(self._equation_evaluations),
            "linearization_count": len(linearizations),
            "stochastic_iteration_count": len(iterations),
            "performance_seconds": {key: float(value) for key, value in self._performance_seconds.items()},
            "consecutive_converged_linearizations": (consecutive_converged_linearizations),
            **normal_summary,
        }
        component_records = variance_component_records(
            final_solution.equations,
            current_state_residuals,
            standardized,
            scales,
            initial_scales,
            factors,
            final_state_proposed_factors,
            assignments=self._assignments,
            components=self.options.components,
            diagnostics=diagnostics,
            uncertainty_qc_groups=self._uncertainty_qc_groups,
            active_threshold=self.options.minimum_nonzero_robust_factor,
        )

        return LlrAdjustmentResult(
            converged=converged,
            termination_reason=termination_reason,
            settings=settings,
            equation_evaluations=list(self._equation_evaluations),
            parameter_names=list(self._names),
            state=self.parametrization.state(),
            gross_rejected=dict(self._gross_rejected),
            uncertainty_quality_control={
                "action": "floor",
                "minimum_sigma_m": self.options.uncertainty_floor_minimum_m,
                "minimum_group_median_fraction": (self.options.uncertainty_floor_group_median_fraction),
                "floored_count": summary["uncertainty_sigma_floored_count"],
                "retained_floored_count": summary["retained_uncertainty_sigma_floored_count"],
                "groups": dict(self._uncertainty_qc_groups),
            },
            scales=dict(scales),
            robust_factors=dict(factors),
            iterations=iterations,
            linearizations=linearizations,
            summary=summary,
            parameters=final_parameter_records,
            global_residuals=global_residuals,
            variance_components=component_records,
            observations=observation_records(
                final_solution.equations,
                current_state_residuals,
                final_solution.residuals,
                residual_sigmas,
                standardized,
                scales,
                factors,
                final_state_proposed_factors,
                assignments=self._assignments,
                parametrization=self.parametrization,
                components=self.options.components,
                uncertainty_qc_records=self._uncertainty_qc_records,
                active_threshold=self.options.minimum_nonzero_robust_factor,
            ),
            normals=final_solution.normals,
        )


__all__ = ["LlrAdjustmentSolver"]
