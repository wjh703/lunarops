"""Nonlinear weighted least-squares adjustment of LLR observations.

Each adjustment iteration re-evaluates the observation model at the current parameter values, linearizes the
reduced one-way O-C residuals, and solves the normal equations ``N x = W`` around the linearization point
``x0`` with strict Cholesky factorization. Observation weights are ``f / sigma**2`` with weight factor ``f`` and
one-way sigma ``sigma`` in metres from the stochastic model (:mod:`lunarops.estimation.stochastic_model`):
``sigma = s sigma_rep`` (``scaled``) or ``sigma^2 = a^2 + (s sigma_rep)^2`` (``floorAndScaled``) with component sigma
factor ``s``, floor ``a`` and reported sigma ``sigma_rep``.

Two schemes estimate the variance parameters and the robust (IGG-III or direct-rejection) weight factors ``f``:

``legacy``
    One solve per linearization; ``s`` and ``f`` are then re-estimated ten times alternately from its frozen
    residuals and redundancies, and the new values enter the next linearization.
``consistent``
    At every linearization, solve and reweight until the weights are stable (see
    :mod:`lunarops.estimation.robust_vce`): the variances are consistent M-scales (divided by the Gaussian
    consistency factor of the weight function), residuals are standardized by their variance under the current
    weights (redundancy at full weight, prediction variance at zero weight), and optional night-session offsets
    are eliminated with their sigma estimated per component.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import cast

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
from lunarops.estimation.normal_equations import NormalEquations
from lunarops.estimation.parameter_convergence import parameter_convergence
from lunarops.estimation.robust_vce import (
    DEFAULT_SESSION_GAP_HOURS,
    SessionElimination,
    Sessions,
    clustering_diagnostics,
    consistency_factor,
    gaussian_tail_fractions,
)
from lunarops.estimation.robust_weights import robust_factors
from lunarops.estimation.stochastic_model import (
    ObservationVariances,
    VarianceComponentEstimator,
    session_variance_step,
)

SIGMA_WEIGHT_ITERATION_COUNT = 10
MINIMUM_ROBUST_REDUNDANCY = 0.1
REDUNDANCY_EPSILON = 1.0e-12
#: ``consistent`` scheme: limit of solve/reweight passes per linearization.
CONSISTENT_PASS_LIMIT = 100
#: ``consistent`` scheme: limit of scale/weight fixed-point updates on the residuals of one pass.
SCALE_WEIGHT_UPDATE_LIMIT = 50
#: ``consistent`` scheme: a pass is converged when no weight factor changes by more than this.
WEIGHT_FACTOR_TOLERANCE = 1.0e-4
#: ``consistent`` scheme: ... and no observation sigma or session sigma changes by more than this fraction.
SIGMA_RELATIVE_TOLERANCE = 1.0e-4




class LlrAdjustmentSolver:
    """Iterative nonlinear LLR adjustment over a fixed observation domain.

    Parameters
    ----------
    equation_source : Callable[[int], EquationBatch]
        Evaluates the observation model at the current parameter state; called with a running evaluation count.
    parametrization : ParametrizationList
        Estimated parameter blocks; updated in place after every iteration.
    settings : LlrAdjustmentSettings
        Iteration control, robust weight model and variance components.
    observation_domain : LlrAdjustmentObservationDomain
        Retained observations and their variance-component assignments.
    initial_sigma_factors : Mapping[str, float], optional
        Starting sigma factor per component; the others start at 1.
    initial_floor_sigmas : Mapping[str, float], optional
        Starting floor sigma per component, in metres; the others start at 0. Ignored by the ``scaled`` model.
    initial_weight_factors : Mapping[ObsKey, float], optional
        Starting weight factor per observation; the others start at 1.
    iteration_callback : Callable[[dict[str, object]], None], optional
        Called with the record of every sigma/weight iteration.
    utc_offset_hours : float, default 0.0
        Offset of local time from UTC, in hours, for reports.
    """

    def __init__(
        self,
        *,
        equation_source: Callable[[int], EquationBatch],
        parametrization: ParametrizationList,
        settings: LlrAdjustmentSettings,
        observation_domain: LlrAdjustmentObservationDomain,
        initial_sigma_factors: Mapping[str, float] | None = None,
        initial_floor_sigmas: Mapping[str, float] | None = None,
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
        self.initial_sigma_factors = dict(initial_sigma_factors or {})
        self.initial_floor_sigmas = dict(initial_floor_sigmas or {})
        self.initial_weight_factors = dict(initial_weight_factors or {})
        self.observation_domain = observation_domain
        self.iteration_callback = iteration_callback
        self.utc_offset_hours = float(utc_offset_hours)
        self.variance_model = self.adjustment.varianceModel
        self.variance_estimator = VarianceComponentEstimator(self.variance_components.components, self.variance_model)

        self._equation_iteration = 0
        self._names: list[ParameterName] = []
        self._equation_evaluations: list[dict[str, object]] = []
        self._linearization: DenseLinearization
        self._performance_seconds = {"cacheBuild": 0.0, "normalSolve": 0.0, "redundancy": 0.0, "adjustSigma0": 0.0}
        self._consistent = self.adjustment.varianceEstimation == "consistent"
        self._consistency = (
            consistency_factor(self.robust.model, self.robust.k0, self.robust.k1)
            if self._consistent and self.adjustment.estimateRobustWeights
            else 1.0
        )
        # Night sessions of the current linearization: for the session model, and for the clustering diagnostic.
        self._sessions: Sessions | None = None
        self._session_sigma: dict[str, float] | None = (
            {item.id: float(self.adjustment.sessionInitialSigmaM) for item in self.variance_components.components}
            if self.adjustment.sessionCorrelation
            else None
        )
        self._session_diagnostics: dict[str, dict[str, object]] = {}

    def _equations(self, purpose: str) -> EquationBatch:
        """Evaluate the equations and return the converged rows inside the domain, checking their metadata."""
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
        gap = self.adjustment.sessionMaximumGapHours if self.adjustment.sessionCorrelation else DEFAULT_SESSION_GAP_HOURS
        self._sessions = Sessions.build(
            self._component_ids(), equations.epoch_jd1 + equations.epoch_jd2, float(gap)
        )
        self._performance_seconds["cacheBuild"] += perf_counter() - started

    def _component_ids(self) -> np.ndarray:
        return np.asarray(
            [self.observation_domain.assignments[key] for key in self._linearization.identities], dtype=object
        )

    def _session_sigma_rows(self) -> np.ndarray:
        """Session sigma of every session of the current linearization, in metres."""
        sessions, sigma = cast(Sessions, self._sessions), cast(dict[str, float], self._session_sigma)
        return np.asarray([sigma[component] for component in sessions.component], dtype=float)

    def _normal_equations(
        self, weights: np.ndarray, active: np.ndarray, x0: np.ndarray
    ) -> tuple[NormalEquations, SessionElimination | None]:
        """Return the normal equations, with the session offsets eliminated when the session model is on."""
        dense = self._linearization
        if self._session_sigma is None:
            return dense.normal_equations(weights, active=active, x0=x0), None
        session_weights = np.where(active, weights, 0.0)
        elimination = SessionElimination.build(
            cast(Sessions, self._sessions), dense.design, dense.reduced_observations, session_weights,
            self._session_sigma_rows(),
        )
        plain = dense.normal_equations(weights, active=active, x0=x0)
        rhs = plain.W - plain.N @ x0
        normal_matrix, reduced_rhs, lpl = elimination.reduce(plain.N, rhs, plain.lPl)
        normals = NormalEquations.from_linearized_statistics(
            parameter_names=plain.parameter_names, N=normal_matrix, correction_rhs=reduced_rhs,
            correction_lPl=lpl, obs_count=plain.obs_count, x0=x0, meta={}, parameter_units=plain.parameter_units,
        )
        return normals, elimination

    def _observation_sigmas(self, variances: ObservationVariances) -> np.ndarray:
        """One-way sigma of every row of the current linearization under ``variances``, in metres."""
        return variances.sigmas(self._component_ids(), self._linearization.apriori_sigmas)

    def _observation_weights(self, variances: ObservationVariances, weight_factors: np.ndarray) -> np.ndarray:
        return weight_factors / self._observation_sigmas(variances) ** 2

    def _solve_linearized(
        self,
        equations: EquationBatch,
        variances: ObservationVariances,
        weight_factors: np.ndarray,
    ) -> _LinearizedSolution:
        dense = self._linearization
        started = perf_counter()
        weights = self._observation_weights(variances, weight_factors)
        active = weight_factors > 0.0
        if np.count_nonzero(active) < len(self._names):
            raise RuntimeError("Too few active observations for the parameter set.")
        weights = np.where(active, weights, 0.0)
        x0 = self.parametrization.x0()
        normals, elimination = self._normal_equations(weights, active, x0)
        solved = solve_normal_equations(normals)
        delta = normals.correction_at_x0(solved.values)
        residual_vector = dense.reduced_observations - dense.design @ delta
        offsets = None
        quadratic = 0.0
        if elimination is not None:
            offsets = elimination.offsets(delta)
            residual_vector = residual_vector - offsets[cast(Sessions, self._sessions).index]
            quadratic = float(np.sum(elimination.precision * offsets**2))
        degrees_of_freedom = int(np.count_nonzero(active)) - len(self._names)
        sigma0_post = (
            None
            if degrees_of_freedom <= 0
            else float(np.sqrt((np.dot(weights, residual_vector**2) + quadratic) / degrees_of_freedom))
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
            elimination=elimination,
            session_offsets=offsets,
        )

    def _redundancies(self, solution: _LinearizedSolution) -> np.ndarray:
        """Return the redundancy numbers ``r_i = 1 - p_i a_i N^-1 a_i^T``, clipped to [0, 1].

        With session offsets, ``a_i`` is extended by the row's session column (see ``SessionElimination``).
        """
        dense = self._linearization
        started = perf_counter()
        if solution.elimination is None:
            projected = dense.design @ solution.covariance
            leverage = solution.observation_weights * np.einsum("ij,ij->i", projected, dense.design)
        else:
            leverage = cast(SessionElimination, solution.elimination).leverage(
                cast(Sessions, self._sessions), dense.design, solution.observation_weights, solution.covariance
            )
        if not np.all(np.isfinite(leverage)) or np.any(leverage < -1.0e-8) or np.any(leverage > 1.0 + 1.0e-8):
            raise RuntimeError("Observation leverage is outside its valid range.")
        redundancies = np.clip(1.0 - leverage, 0.0, 1.0)
        self._performance_seconds["redundancy"] += perf_counter() - started
        return redundancies

    def _fitted_value_ratios(self, solution: _LinearizedSolution, variances: ObservationVariances) -> np.ndarray:
        """Return ``t_i = q_i / sigma_i^2``: variance of the fitted value over the observation variance.

        ``q_i = a_i N^-1 a_i^T`` (plus the session term with night sessions) for every row, rejected ones included;
        a full-weight row's leverage is ``t_i``.
        """
        dense = self._linearization
        started = perf_counter()
        if solution.elimination is None:
            fitted = np.einsum("ij,ij->i", dense.design @ solution.covariance, dense.design)
        else:
            fitted = cast(SessionElimination, solution.elimination).prediction_variance(
                cast(Sessions, self._sessions), dense.design, solution.covariance
            )
        unit_weights = self._observation_weights(variances, np.ones(len(fitted)))
        self._performance_seconds["redundancy"] += perf_counter() - started
        return np.maximum(fitted * unit_weights, 0.0)

    def _standardized_residuals(
        self,
        residuals: np.ndarray,
        redundancies: np.ndarray,
        variances: ObservationVariances,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return standardized residuals ``v / (sigma sqrt(r))`` and residual sigmas in metres.

        Observations with redundancy at or below ``MINIMUM_ROBUST_REDUNDANCY`` get a standardized residual of 0.
        """
        sigma0 = self._observation_sigmas(variances)
        residual_sigmas = sigma0 * np.sqrt(np.maximum(redundancies, REDUNDANCY_EPSILON))
        standardized = np.zeros_like(residuals)
        eligible = redundancies > MINIMUM_ROBUST_REDUNDANCY
        standardized[eligible] = residuals[eligible] / residual_sigmas[eligible]
        return standardized, residual_sigmas

    def _estimate_variances(
        self,
        residuals: np.ndarray,
        standardization: np.ndarray,
        variances: ObservationVariances,
        weight_factors: np.ndarray,
        consistency: float = 1.0,
    ) -> tuple[ObservationVariances, dict[str, dict[str, object]]]:
        """One variance-component update from residuals and their residual-variance factors."""
        started = perf_counter()
        estimate = self.variance_estimator.estimate(
            apriori_sigmas=self._linearization.apriori_sigmas,
            residuals=residuals,
            standardization=standardization,
            component_ids=self._component_ids(),
            weight_factors=weight_factors,
            variances=variances,
            consistency=consistency,
        )
        self._performance_seconds["adjustSigma0"] += perf_counter() - started
        return estimate.variances, estimate.diagnostics

    def _sigma_change(self, before: ObservationVariances, after: ObservationVariances) -> float:
        """Largest relative change of an observation sigma of the current linearization."""
        return before.relative_change(after, self._component_ids(), self._linearization.apriori_sigmas)

    def _floor_record(self, variances: ObservationVariances, key: str = "floorSigmasM") -> dict[str, object]:
        """Return the floor sigmas under ``key`` for a record with ``floorAndScaled``, else nothing."""
        return {key: dict(variances.floor_sigmas)} if self.variance_model == "floorAndScaled" else {}

    def _model_error_diagnostics(
        self, diagnostics: Mapping[str, Mapping[str, object]], weight_vector: np.ndarray
    ) -> dict[str, dict[str, object]]:
        """Add the outlier-clustering statistics, the session sigmas and the consistency factor per component."""
        components = [item.id for item in self.variance_components.components]
        expected = (
            gaussian_tail_fractions(self.robust.model, self.robust.k0, self.robust.k1)
            if self.adjustment.estimateRobustWeights
            else (0.0, 0.0)
        )
        clustering = clustering_diagnostics(
            cast(Sessions, self._sessions), self._component_ids(), weight_vector, components, expected
        )
        merged: dict[str, dict[str, object]] = {}
        for component in components:
            item = dict(diagnostics.get(component, {}))
            item.update(clustering[component])
            item["varianceEstimation"] = self.adjustment.varianceEstimation
            item["varianceModel"] = self.variance_model
            item["consistencyFactor"] = self._consistency
            item.update(self._session_diagnostics.get(component, {}))
            merged[component] = item
        return merged

    def _consistent_passes(
        self,
        outer: int,
        global_inner: int,
        equations: EquationBatch,
        variances: ObservationVariances,
        weight_vector: np.ndarray,
    ) -> tuple[
        _LinearizedSolution, ObservationVariances, np.ndarray, dict[str, dict[str, object]], list[dict[str, object]]
    ]:
        """Solve and reweight at the current linearization until the stochastic model is stable.

        Every pass solves with the current variance parameters, weight factors and session sigmas, then updates them
        from its residuals. With robust weights, the variance parameters (consistent M-scales) and the weight
        factors are iterated to their joint fixed point on the pass's residuals, standardized by
        ``residual_variance_factors`` of the pass's fitted-value ratios and of the weight factors of the pass's solve
        (the residuals' own variance; the standardized residuals are then w-statistics and change only with the
        variance parameters); without, the variance parameters get one update with the solution's redundancies. The
        session sigmas get one profile Fisher-scoring step. Passes stop when no
        weight factor changes by more than ``WEIGHT_FACTOR_TOLERANCE`` and no observation or session sigma by more
        than ``SIGMA_RELATIVE_TOLERANCE``, or after ``CONSISTENT_PASS_LIMIT`` passes. The returned solution is that
        of the last pass.
        """
        robust = self.adjustment.estimateRobustWeights
        variance = self.adjustment.estimateVarianceFactors
        diagnostics: dict[str, dict[str, object]] = {}
        records: list[dict[str, object]] = []
        for index in range(1, CONSISTENT_PASS_LIMIT + 1):
            started = perf_counter()
            solution = self._solve_linearized(equations, variances, weight_vector)
            redundancies = self._redundancies(solution)
            residuals = solution.residual_vector
            previous_variances, previous_weights = variances, weight_vector.copy()
            previous_session = dict(self._session_sigma or {})
            standardization = redundancies
            if robust:
                standardization = residual_variance_factors(self._fitted_value_ratios(solution, variances), weight_vector)
                for _ in range(SCALE_WEIGHT_UPDATE_LIMIT):
                    variances_before, weights_before = variances, weight_vector
                    if variance:
                        variances, diagnostics = self._estimate_variances(
                            residuals, standardization, variances, weight_vector, self._consistency
                        )
                    standardized, _ = self._standardized_residuals(residuals, standardization, variances)
                    weight_vector = robust_factors(self.robust.model, self.robust.k0, self.robust.k1, standardized)
                    if (
                        self._sigma_change(variances_before, variances) <= 0.1 * SIGMA_RELATIVE_TOLERANCE
                        and np.max(np.abs(weight_vector - weights_before), initial=0.0) <= 0.1 * WEIGHT_FACTOR_TOLERANCE
                    ):
                        break
            else:
                variances, diagnostics = self._estimate_variances(residuals, redundancies, variances, weight_vector)
            offset_redundancy = np.empty(0)
            if self._session_sigma is not None and variance:
                offset_redundancy = cast(SessionElimination, solution.elimination).offset_redundancy(solution.covariance)
                self._session_sigma, self._session_diagnostics = session_variance_step(
                    model=self.variance_model,
                    sessions=cast(Sessions, self._sessions),
                    component_ids=self._component_ids(),
                    apriori_sigmas=self._linearization.apriori_sigmas,
                    residuals=residuals,
                    standardization=standardization,
                    weight_factors=weight_vector,
                    offsets=cast(np.ndarray, solution.session_offsets),
                    offset_redundancy=offset_redundancy,
                    variances=variances,
                    session_sigma=self._session_sigma,
                    consistency=self._consistency,
                )
            sigma_change = self._sigma_change(previous_variances, variances)
            session_change = _relative_change(previous_session, self._session_sigma or {})
            weight_change = float(np.max(np.abs(weight_vector - previous_weights), initial=0.0))
            candidate_by_block = self.parametrization.update_norms(solution.delta)
            sigma_factors, previous_sigma = variances.sigma_factors, previous_variances.sigma_factors
            records.append(
                {
                    "iteration": global_inner + index,
                    "adjustmentIteration": outer,
                    "sigmaWeightIteration": index,
                    "elapsedSeconds": float(perf_counter() - started),
                    "maximumSigmaFactorChange": max(
                        (abs(sigma_factors[key] - previous_sigma[key]) for key in sigma_factors), default=0.0
                    ),
                    "maximumObservationSigmaRelativeChange": sigma_change,
                    "maximumSessionSigmaRelativeChange": session_change,
                    "maximumWeightFactorChange": weight_change,
                    "activeObservationCount": int(np.count_nonzero(weight_vector > 0.0)),
                    "rejectedObservationCount": int(np.count_nonzero(weight_vector == 0.0)),
                    "totalFrozenRedundancy": float(np.sum(redundancies) + np.sum(offset_redundancy)),
                    "expectedTotalRedundancy": float(len(equations) - solution.rank),
                    "normalMatrixCondition": solution.condition,
                    "candidateWrmsM": solution.wrms_m,
                    "maximumCandidateParameterUpdateM": max(candidate_by_block.values(), default=0.0),
                    "candidateUpdateByBlockM": candidate_by_block,
                    "sigmaFactors": dict(sigma_factors),
                    **self._floor_record(variances),
                    "sessionSigmaM": dict(self._session_sigma) if self._session_sigma is not None else None,
                    "varianceComponents": {
                        item.id: dict(diagnostics[item.id])
                        for item in self.variance_components.components
                        if item.id in diagnostics
                    },
                }
            )
            if (
                weight_change <= WEIGHT_FACTOR_TOLERANCE
                and sigma_change <= SIGMA_RELATIVE_TOLERANCE
                and session_change <= SIGMA_RELATIVE_TOLERANCE
            ):
                break
        return solution, variances, weight_vector, diagnostics, records

    def run(self, *, finalize: bool = True) -> LlrAdjustmentResult:
        """Run ``solve`` and build the adjustment report from its state."""
        return build_report(self.solve(finalize=finalize), self.settings)

    def solve(self, *, finalize: bool = True) -> AdjustmentState:
        """Iterate the nonlinear adjustment until convergence or ``maxIterationCount`` iterations.

        In each adjustment iteration the equations are linearized and solved with the current sigma and weight
        factors. With ``computeResiduals``, the sigma factors and/or robust weights are then re-estimated from
        the frozen residuals and redundancies: 10 alternating passes when both are estimated, otherwise one.
        The solved correction is applied to the parametrization, and the estimate has converged when every
        block's update norm is within its threshold.

        Parameters
        ----------
        finalize : bool, default True
            Re-evaluate and solve once more at the final parameter state, to obtain final residuals, residual
            sigmas, standardized residuals, proposed weight factors and variance-component diagnostics. That
            solution's correction is reported but not applied.

        Returns
        -------
        AdjustmentState
            Final state of the estimate.

        Raises
        ------
        ValueError
            If no usable observation is available or an observation's station, reflector, transmit epoch or
            wavelength changes between evaluations.
        RuntimeError
            If fewer active observations than parameters remain or the leverage is outside [0, 1].
        NormalEquationSingularError
            If the normal matrix is singular or not positive definite.
        """
        initial_equations = self._equations("initialization")
        if not initial_equations:
            raise ValueError("Adjustment has no light-time-converged observations.")
        self.parametrization.setup(initial_equations)
        self._names = self.parametrization.names
        initial_reference_values = self.parametrization.x0().copy()
        variances = ObservationVariances.initial(
            (item.id for item in self.variance_components.components),
            self.initial_sigma_factors,
            self.initial_floor_sigmas,
        )
        if self.variance_model == "scaled":
            variances = variances.without_floors()
        initial_variances = variances
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
            variances_used = variances
            weight_factors_used = dict(weight_factors)
            weight_vector = np.asarray([weight_factors[key] for key in self._linearization.identities])
            keys = current_equations.observation_id.tolist()
            summary_used = weight_factor_summary(current_equations, weight_factors_used)
            reweighting = self.adjustment.computeResiduals and (
                self.adjustment.estimateVarianceFactors or self.adjustment.estimateRobustWeights
            )
            records: list[dict[str, object]] = []
            if self._consistent and reweighting:
                base_solution, variances, weight_vector, diagnostics, records = self._consistent_passes(
                    outer, global_inner, current_equations, variances, weight_vector
                )
                weight_factors.update(zip(keys, map(float, weight_vector), strict=True))
                for record in records:
                    record["weightFactorSummary"] = weight_factor_summary(current_equations, weight_factors)
                    iterations.append(record)
                    if self.iteration_callback is not None:
                        self.iteration_callback(record)
                global_inner += len(records)
            else:
                base_solution = self._solve_linearized(current_equations, variances_used, weight_vector)
            candidate_by_block = self.parametrization.update_norms(base_solution.delta)
            frozen_residuals = base_solution.residual_vector
            frozen_redundancies = (
                self._redundancies(base_solution)
                if self.adjustment.computeResiduals and not (self._consistent and reweighting)
                else np.zeros(len(current_equations), dtype=float)
            )

            legacy_count = 0
            if self.adjustment.computeResiduals and not self._consistent:
                if self.adjustment.estimateVarianceFactors and self.adjustment.estimateRobustWeights:
                    legacy_count = SIGMA_WEIGHT_ITERATION_COUNT
                elif self.adjustment.estimateVarianceFactors or self.adjustment.estimateRobustWeights:
                    legacy_count = 1
            sigma_weight_iteration_count = len(records) if self._consistent and reweighting else legacy_count
            for inner in range(1, legacy_count + 1):
                started = perf_counter()
                previous_sigma = dict(variances.sigma_factors)
                previous_weights = weight_vector.copy()
                if self.adjustment.estimateVarianceFactors:
                    variances, diagnostics = self._estimate_variances(
                        frozen_residuals,
                        frozen_redundancies,
                        variances,
                        weight_vector,
                    )
                if self.adjustment.estimateRobustWeights:
                    standardized, _ = self._standardized_residuals(
                        frozen_residuals,
                        frozen_redundancies,
                        variances,
                    )
                    weight_vector = robust_factors(self.robust.model, self.robust.k0, self.robust.k1, standardized)
                weight_factors.update(zip(keys, map(float, weight_vector), strict=True))
                max_sigma_change = max(
                    (abs(value - previous_sigma[key]) for key, value in variances.sigma_factors.items()),
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
                    "sigmaFactors": dict(variances.sigma_factors),
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
                    "sigmaFactorsUsedInSolve": dict(variances_used.sigma_factors),
                    "sigmaFactorsForNextIteration": dict(variances.sigma_factors),
                    **self._floor_record(variances_used, "floorSigmasUsedInSolveM"),
                    **self._floor_record(variances, "floorSigmasForNextIterationM"),
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
            final_solution = self._solve_linearized(final_equations, variances, weight_vector)
            final_redundancies = self._redundancies(final_solution)
            robust_consistent = self._consistent and self.adjustment.estimateRobustWeights
            standardization = (
                residual_variance_factors(self._fitted_value_ratios(final_solution, variances), weight_vector)
                if robust_consistent
                else final_redundancies
            )
            standardized, residual_sigmas = self._standardized_residuals(
                final_solution.residual_vector,
                standardization,
                variances,
            )
            keys = list(self._linearization.identities)
            final_proposed_weight_factors = dict(zip(keys, map(float, robust_factors(self.robust.model, self.robust.k0, self.robust.k1, standardized)), strict=True)) if self.adjustment.estimateRobustWeights else dict(weight_factors)
            diagnostics = {}
            if self.adjustment.estimateVarianceFactors:
                _, diagnostics = self._estimate_variances(
                    final_solution.residual_vector,
                    standardization,
                    variances,
                    weight_vector,
                    self._consistency,
                )
            diagnostics = self._model_error_diagnostics(diagnostics, weight_vector)
        return AdjustmentState(
            solution=final_solution, last_solution=last_solution, equations=final_equations,
            converged=converged, termination_reason=termination_reason, iterations=iterations,
            adjustment_iterations=adjustment_iterations, sigma_factors=dict(variances.sigma_factors),
            floor_sigmas=dict(variances.floor_sigmas), weight_factors=weight_factors,
            initial_sigma_factors=dict(initial_variances.sigma_factors),
            initial_floor_sigmas=dict(initial_variances.floor_sigmas), initial_reference_values=initial_reference_values,
            warm_sigma_count=warm_sigma_count, warm_weight_count=warm_weight_count,
            equation_evaluations=self._equation_evaluations, performance_seconds=self._performance_seconds,
            observation_domain=self.observation_domain, parametrization=self.parametrization,
            utc_offset_hours=self.utc_offset_hours, standardized=standardized, residual_sigmas=residual_sigmas,
            proposed_weight_factors=final_proposed_weight_factors, diagnostics=diagnostics, finalize=finalize,
        )



def residual_variance_factors(ratios: np.ndarray, weight_factors: np.ndarray) -> np.ndarray:
    """Return ``(1 - f t) (1 + (1 - f) t)``: residual variance over observation variance under weight factors ``f``.

    ``t`` is the fitted-value ratio of the solution that used the weight factors ``f``. With the observation's own
    weight ``f p`` in the normal equations, its residual is ``v = e / (1 + f t_e)`` with the prediction residual
    ``e`` (variance ``1 + t_e``) of the system without it, and ``t = t_e / (1 + f t_e)``; hence the factor. It is the
    redundancy ``1 - t`` at full weight and the prediction variance ``1 + t`` at zero weight, and ``v / sqrt`` of it
    is Baarda's w-statistic of the observation, which does not depend on its own weight (the other observations'
    weights taken as their inverse variances).
    """
    return np.maximum((1.0 - weight_factors * ratios) * (1.0 + (1.0 - weight_factors) * ratios), 0.0)


def _relative_change(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    """Largest relative change between two mappings of positive values."""
    return max((abs(after[key] - value) / value for key, value in before.items() if value > 0.0), default=0.0)


__all__ = ["LlrAdjustmentSolver"]
