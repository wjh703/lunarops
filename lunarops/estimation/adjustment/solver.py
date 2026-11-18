"""Nonlinear weighted least-squares adjustment of LLR observations.

Each adjustment iteration re-evaluates the observation model at the current parameter values, linearizes the
reduced one-way O-C residuals, and solves the normal equations ``N x = W`` around the linearization point
``x0`` with strict Cholesky factorization. Observation weights are ``f / sigma**2`` with weight factor ``f`` and
one-way sigma ``sigma`` in metres from the stochastic model (:mod:`lunarops.estimation.stochastic.vce`):
``sigma = s sigma_rep`` (``scaled``) or ``sigma^2 = a^2 + (s sigma_rep)^2`` (``floorAndScaled``) with component sigma
factor ``s``, floor ``a`` and reported sigma ``sigma_rep``.

The variance parameters and the robust (IGG-III or direct-rejection) weight factors ``f`` are estimated at every
linearization by solve/reweight passes until they are stable (see :mod:`lunarops.estimation.stochastic.sessions`): the
variances are consistent M-scales (divided by the Gaussian consistency factor of the weight function), residuals are
standardized by their variance under the current weights (redundancy at full weight, prediction variance at zero
weight), and optional night-session offsets are eliminated with their sigma estimated per component.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import cast

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.adjustment.convergence import parameter_convergence
from lunarops.estimation.adjustment.plan import LlrAdjustmentSettings
from lunarops.estimation.adjustment.reporting import (
    build_report,
    component_diagnostics,
    floor_record,
    iteration_record,
    weight_factor_summary,
)
from lunarops.estimation.adjustment.results import (
    AdjustmentState,
    LinearizedSolution,
    LlrAdjustmentObservationDomain,
    LlrAdjustmentResult,
)
from lunarops.estimation.constraints import station_position_columns
from lunarops.estimation.earth_orientation import EarthOrientationUncertainty, StationNightCorrections
from lunarops.estimation.lsq.blocks import RowBlocks
from lunarops.estimation.lsq.linearization import DenseLinearization
from lunarops.estimation.lsq.normal_equations import NormalEquations
from lunarops.estimation.lsq.pseudo_observations import PseudoObservations
from lunarops.estimation.lsq.random_effects import RandomEffectElimination, RandomEffectLayout, fitted_value_variance
from lunarops.estimation.lsq.solve import solve_normal_equations
from lunarops.estimation.stochastic.components import ObsKey, observation_signatures, require_unique_identities
from lunarops.estimation.stochastic.robust import residual_variance_factors, standardized_residuals
from lunarops.estimation.stochastic.sessions import DEFAULT_SESSION_GAP_HOURS, MINIMUM_SESSION_SIGMA_M, night_sessions
from lunarops.estimation.stochastic.variances import ObservationVariances
from lunarops.estimation.stochastic.vce import VarianceComponentEstimator, session_variance_step

#: Limit of solve/reweight passes per linearization. Large components with many down-weighted observations converge
#: linearly and slowly (the EPM2021 joint estimate with night sessions needs about 125 passes).
PASS_LIMIT = 200
#: ``floorAndScaled``: fraction of each proposed ``(s^2, a^2)`` update taken per pass (under-relaxation; the fixed
#: point is unchanged). The two-parameter update with robust weights can cycle in a component whose floor is small
#: next to its scaled reported sigmas.
FLOOR_MODEL_RELAXATION = 0.5
#: An observation whose rejection status changed this often within the passes of a linearization stays rejected.
REJECTION_STATUS_CHANGE_LIMIT = 2
#: A pass is converged when no weight factor changes by more than this.
WEIGHT_FACTOR_TOLERANCE = 1.0e-4
#: ... and no observation sigma or session sigma changes by more than this fraction.
SIGMA_RELATIVE_TOLERANCE = 1.0e-4


@dataclass(frozen=True, eq=False)
class _Linearization:
    """One linearization point: the equations with their design, night sessions and random effects.

    Attributes
    ----------
    equations : EquationBatch
        Equations evaluated at the linearization point.
    dense : DenseLinearization
        Design matrix, reduced observations and a-priori sigmas, in row order.
    component_ids : ndarray of object, shape (n,)
        Variance component of every row.
    sessions : RowBlocks
        Night sessions per variance component: for the session model, and for the clustering diagnostic.
    layout : RandomEffectLayout or None
        Random effects (session offsets and/or Earth-orientation corrections); None when none are modelled.
    eop : StationNightCorrections or None
        Earth-orientation corrections per station night; None without ``earthOrientationCorrection``.
    """

    equations: EquationBatch
    dense: DenseLinearization
    component_ids: np.ndarray
    sessions: RowBlocks
    layout: RandomEffectLayout | None
    eop: StationNightCorrections | None


@dataclass(frozen=True, eq=False)
class _StochasticModelUpdate:
    """Solution of one linearization and the stochastic model re-estimated with it.

    Attributes
    ----------
    solution : LinearizedSolution
        Solution whose correction is applied (that of the last pass).
    variances : ObservationVariances
        Variance parameters for the next linearization.
    weight_vector : ndarray, shape (n,)
        Robust weight factors for the next linearization, in row order.
    diagnostics : dict[str, dict[str, object]]
        Variance-component diagnostics of the last update; empty without variance estimation.
    records : list[dict[str, object]]
        One record per sigma/weight iteration.
    converged : bool
        Whether the stochastic model became stable; ``False`` when the passes reached ``PASS_LIMIT``.
    session_diagnostics : dict[str, dict[str, object]]
        Session-sigma diagnostics of the last update; empty without session-sigma estimation.
    """

    solution: LinearizedSolution
    variances: ObservationVariances
    weight_vector: np.ndarray
    diagnostics: dict[str, dict[str, object]]
    records: list[dict[str, object]]
    converged: bool = True
    session_diagnostics: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class _FinalEvaluation:
    """Solution reported as the final state of an estimate, with its residual statistics.

    Attributes
    ----------
    solution : LinearizedSolution
        Solution at the final parameter state (with ``finalize``) or that of the last linearization.
    equations : EquationBatch
        Equations of ``solution``.
    standardized : ndarray, shape (n,)
        Standardized residuals; empty without ``finalize``.
    residual_sigmas : ndarray, shape (n,)
        Residual sigmas, in metres; empty without ``finalize``.
    proposed_weight_factors : dict[ObsKey, float]
        Weight factors proposed from the final residuals; empty without ``finalize``.
    diagnostics : dict[str, dict[str, object]]
        Variance-component diagnostics.
    constraints : list[dict[str, object]]
        Records of the constraint pseudo-observations; empty without ``finalize``.
    earth_orientation : list[dict[str, object]]
        Earth-orientation correction records; empty without ``finalize`` or ``earthOrientationCorrection``.
    """

    solution: LinearizedSolution
    equations: EquationBatch
    standardized: np.ndarray
    residual_sigmas: np.ndarray
    proposed_weight_factors: dict[ObsKey, float]
    diagnostics: dict[str, dict[str, object]]
    constraints: list[dict[str, object]]
    earth_orientation: list[dict[str, object]]


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
    initial_session_sigmas : Mapping[str, float], optional
        Starting session sigma per component, in metres; the others start at ``sessionInitialSigmaM``. Ignored
        without ``sessionCorrelation``.
    initial_weight_factors : Mapping[ObsKey, float], optional
        Starting weight factor per observation; the others start at 1.
    iteration_callback : Callable[[dict[str, object]], None], optional
        Called with the record of every sigma/weight iteration.
    utc_offset_hours : float, default 0.0
        Offset of local time from UTC, in hours, for reports.
    earth_orientation_uncertainty : EarthOrientationUncertainty, optional
        Prior sigmas of the Earth-orientation corrections; required with ``earthOrientationCorrection``.
    station_positions : Mapping[str, ArrayLike], optional
        ITRS station positions in metres, the lever arms of the Earth-orientation partials; required with
        ``earthOrientationCorrection``.

    Raises
    ------
    ValueError
        If ``earthOrientationCorrection`` is set without the uncertainties or the station positions.
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
        initial_session_sigmas: Mapping[str, float] | None = None,
        initial_weight_factors: Mapping[ObsKey, float] | None = None,
        iteration_callback: Callable[[dict[str, object]], None] | None = None,
        utc_offset_hours: float = 0.0,
        earth_orientation_uncertainty: EarthOrientationUncertainty | None = None,
        station_positions: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        if settings.adjustment.earthOrientationCorrection and (
            earth_orientation_uncertainty is None or station_positions is None
        ):
            raise ValueError(
                "earthOrientationCorrection needs the Earth-orientation uncertainties and station positions."
            )
        self.earth_orientation_uncertainty = earth_orientation_uncertainty
        self.station_positions = {key: np.asarray(value, float) for key, value in (station_positions or {}).items()}
        self.equation_source = equation_source
        self.parametrization = parametrization
        self.settings = settings
        self.adjustment = settings.adjustment
        self.robust = settings.robust_weights
        self.variance_components = settings.variance_components
        self.initial_sigma_factors = dict(initial_sigma_factors or {})
        self.initial_floor_sigmas = dict(initial_floor_sigmas or {})
        self.initial_session_sigmas = dict(initial_session_sigmas or {})
        self.initial_weight_factors = dict(initial_weight_factors or {})
        self.observation_domain = observation_domain
        self.iteration_callback = iteration_callback
        self.utc_offset_hours = float(utc_offset_hours)
        self.variance_model = self.adjustment.varianceModel
        self._component_names = [item.id for item in self.variance_components.components]
        self.variance_estimator = VarianceComponentEstimator(self.variance_components.components, self.variance_model)

        self._equation_iteration = 0
        self._names: list[ParameterName] = []
        self._equation_evaluations: list[dict[str, object]] = []
        self._performance_seconds = {"cacheBuild": 0.0, "normalSolve": 0.0, "redundancy": 0.0, "adjustSigma0": 0.0}
        robust = self.adjustment.estimateRobustWeights
        self._consistency = self.robust.consistency() if robust else 1.0
        # E[w^2 z^2 | w > 0]: consistency factor of the session-variance equation under robust weights.
        self._session_consistency = self.robust.consistency(2) if robust else 1.0
        # Pseudo-observations of the plan's constraints over this estimate's parameters, set after setup.
        self._pseudo = PseudoObservations.empty(0)

    def _equations(self, purpose: str) -> EquationBatch:
        """Evaluate the equations and return the converged rows inside the domain, checking their metadata."""
        self._equation_iteration += 1
        equations = self.equation_source(self._equation_iteration)
        require_unique_identities(equations)
        light_time_valid = equations[equations.converged]
        retained = light_time_valid[
            np.isin(light_time_valid.observation_id, np.fromiter(self.observation_domain.retained_keys, dtype=object))
        ]
        for identity, signature in observation_signatures(retained).items():
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

    def _linearize(self, equations: EquationBatch) -> _Linearization:
        """Linearize the equations at the current parameter state, with their night sessions and random effects."""
        if not equations:
            raise ValueError("Adjustment has no usable observations.")
        started = perf_counter()
        dense = DenseLinearization.build(equations, self.parametrization, self._names)
        component_ids = np.asarray([self.observation_domain.assignments[key] for key in dense.identities], dtype=object)
        random_effects = self.adjustment.sessionCorrelation or self.adjustment.earthOrientationCorrection
        gap = float(self.adjustment.sessionMaximumGapHours if random_effects else DEFAULT_SESSION_GAP_HOURS)
        epochs = equations.epoch_jd1 + equations.epoch_jd2
        sessions = night_sessions(component_ids, epochs, gap)
        layout: RandomEffectLayout | None = None
        eop: StationNightCorrections | None = None
        if self.adjustment.earthOrientationCorrection:
            eop = StationNightCorrections.build(
                equations,
                night_sessions(np.asarray(equations.station, dtype=object), epochs, gap),
                self.station_positions,
                self._earth_orientation_uncertainty(),
                self.adjustment.earthOrientationSigmaScale,
                sessions if self.adjustment.sessionCorrelation else None,
            )
            layout = eop.layout
        elif self.adjustment.sessionCorrelation:
            layout = RandomEffectLayout.offsets(sessions)
        self._performance_seconds["cacheBuild"] += perf_counter() - started
        return _Linearization(equations, dense, component_ids, sessions, layout, eop)

    def _earth_orientation_uncertainty(self) -> EarthOrientationUncertainty:
        if self.earth_orientation_uncertainty is None:
            raise RuntimeError("earthOrientationCorrection needs the Earth-orientation uncertainties.")
        return self.earth_orientation_uncertainty

    @staticmethod
    def _effect_precision(
        linearization: _Linearization, layout: RandomEffectLayout, variances: ObservationVariances
    ) -> np.ndarray:
        """Prior precision of every random effect of ``layout``, shape ``(B, q)``; 1 for padding.

        The Earth-orientation corrections occupy the ``partial_slots`` of the layout, the session offsets its
        offset slots.
        """
        precision = np.ones((layout.blocks.count, layout.size))
        if linearization.eop is not None:
            precision[:, list(layout.partial_slots)] = linearization.eop.prior_precision
        if len(layout.offset_block):
            precision[layout.offset_block, layout.offset_slot] = (
                1.0 / np.maximum(_session_sigma_rows(linearization, variances), MINIMUM_SESSION_SIGMA_M) ** 2
            )
        return precision

    def _normal_equations(
        self,
        linearization: _Linearization,
        weights: np.ndarray,
        active: np.ndarray,
        x0: np.ndarray,
        variances: ObservationVariances,
    ) -> tuple[NormalEquations, RandomEffectElimination | None]:
        """Return the normal equations, with the random effects eliminated when they are modelled.

        The random effects are the session offsets and the Earth-orientation corrections. The pseudo-observations
        of the constraints are added and counted in ``obs_count``, so ``obs_count - m`` are the degrees of freedom
        of the solution. Each eliminated random effect adds one unknown and one prior pseudo-observation and so
        leaves them unchanged.
        """
        dense, layout = linearization.dense, linearization.layout
        plain = dense.normal_equations(weights, active=active, x0=x0)
        if layout is None and not len(self._pseudo):
            return plain, None
        normal_matrix, rhs, lpl = plain.N, plain.correction_rhs, plain.lPl
        elimination = None
        if layout is not None:
            elimination = RandomEffectElimination.build(
                layout,
                dense.design,
                dense.reduced_observations,
                np.where(active, weights, 0.0),
                self._effect_precision(linearization, layout, variances),
            )
            normal_matrix, rhs, lpl = elimination.reduce(normal_matrix, rhs, lpl)
        normal_matrix, reduced_rhs, lpl = self._pseudo.reduce(normal_matrix, rhs, lpl, x0)
        normals = NormalEquations.from_linearized_statistics(
            parameter_names=plain.parameter_names,
            N=normal_matrix,
            correction_rhs=reduced_rhs,
            correction_lPl=lpl,
            obs_count=plain.obs_count + len(self._pseudo),
            x0=x0,
            meta={},
            parameter_units=plain.parameter_units,
        )
        return normals, elimination

    def _solve_linearized(
        self,
        linearization: _Linearization,
        variances: ObservationVariances,
        weight_factors: np.ndarray,
    ) -> LinearizedSolution:
        dense = linearization.dense
        started = perf_counter()
        weights = _observation_weights(linearization, variances, weight_factors)
        active = weight_factors > 0.0
        if np.count_nonzero(active) < len(self._names):
            raise RuntimeError("Too few active observations for the parameter set.")
        weights = np.where(active, weights, 0.0)
        x0 = self.parametrization.x0()
        normals, elimination = self._normal_equations(linearization, weights, active, x0, variances)
        solved = solve_normal_equations(normals)
        delta = solved.delta
        residual_vector = dense.reduced_observations - dense.design @ delta
        offsets = effects = None
        quadratic = 0.0
        if elimination is not None:
            effects = elimination.effects(delta)
            residual_vector = residual_vector - elimination.effect_rows(effects)
            quadratic = elimination.prior_quadratic(effects)
            offsets = elimination.offset_values(effects) if self.adjustment.sessionCorrelation else None
        quadratic += float(np.sum(self._pseudo.weights * self._pseudo.residuals(solved.values) ** 2))
        degrees_of_freedom = normals.obs_count - len(self._names)
        sigma0_post = (
            None
            if degrees_of_freedom <= 0
            else float(np.sqrt((np.dot(weights, residual_vector**2) + quadratic) / degrees_of_freedom))
        )
        weight_sum = float(np.sum(weights))
        wrms = None if weight_sum <= 0.0 else float(np.sqrt(np.dot(weights, residual_vector**2) / weight_sum))
        self._performance_seconds["normalSolve"] += perf_counter() - started
        return LinearizedSolution(
            equations=linearization.equations,
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
            random_effects=effects,
        )

    def _redundancies(self, linearization: _Linearization, solution: LinearizedSolution) -> np.ndarray:
        """Return the redundancy numbers ``r_i = 1 - p_i a_i N^-1 a_i^T``, clipped to [0, 1].

        With random effects, ``a_i`` is extended by the row's effect design (see ``RandomEffectElimination``).
        """
        started = perf_counter()
        leverage = solution.observation_weights * fitted_value_variance(
            linearization.dense.design, solution.covariance, solution.elimination
        )
        if not np.all(np.isfinite(leverage)) or np.any(leverage < -1.0e-8) or np.any(leverage > 1.0 + 1.0e-8):
            raise RuntimeError("Observation leverage is outside its valid range.")
        redundancies = np.clip(1.0 - leverage, 0.0, 1.0)
        self._performance_seconds["redundancy"] += perf_counter() - started
        return redundancies

    def _fitted_value_ratios(
        self, linearization: _Linearization, solution: LinearizedSolution, variances: ObservationVariances
    ) -> np.ndarray:
        """Return ``t_i = q_i / sigma_i^2``: variance of the fitted value over the observation variance.

        ``q_i = a_i N^-1 a_i^T`` (plus the session term with night sessions) for every row, rejected ones included;
        a full-weight row's leverage is ``t_i``.
        """
        started = perf_counter()
        fitted = fitted_value_variance(linearization.dense.design, solution.covariance, solution.elimination)
        unit_weights = _observation_weights(linearization, variances, np.ones(len(fitted)))
        self._performance_seconds["redundancy"] += perf_counter() - started
        return np.maximum(fitted * unit_weights, 0.0)

    def _standardization(
        self,
        linearization: _Linearization,
        solution: LinearizedSolution,
        variances: ObservationVariances,
        weight_factors: np.ndarray,
        redundancies: np.ndarray,
    ) -> np.ndarray:
        """Residual-variance factors that standardize the residuals of ``solution``.

        Under robust weights ``residual_variance_factors`` of the fitted-value ratios and the weight factors used in
        the solve, otherwise the redundancy numbers.
        """
        if self.adjustment.estimateRobustWeights:
            return residual_variance_factors(
                self._fitted_value_ratios(linearization, solution, variances), weight_factors
            )
        return redundancies

    def _estimate_variances(
        self,
        linearization: _Linearization,
        residuals: np.ndarray,
        standardization: np.ndarray,
        variances: ObservationVariances,
        weight_factors: np.ndarray,
        consistency: float = 1.0,
    ) -> tuple[ObservationVariances, dict[str, dict[str, object]]]:
        """One variance-component update from residuals and their residual-variance factors."""
        started = perf_counter()
        estimate = self.variance_estimator.estimate(
            apriori_sigmas=linearization.dense.apriori_sigmas,
            residuals=residuals,
            standardization=standardization,
            component_ids=linearization.component_ids,
            weight_factors=weight_factors,
            variances=variances,
            consistency=consistency,
        )
        self._performance_seconds["adjustSigma0"] += perf_counter() - started
        return estimate.variances, estimate.diagnostics

    def _solve_reweight_passes(
        self,
        linearization: _Linearization,
        outer: int,
        global_inner: int,
        variances: ObservationVariances,
        weight_vector: np.ndarray,
    ) -> _StochasticModelUpdate:
        """Solve and reweight at the current linearization until the stochastic model is stable.

        Every pass solves with the current variance parameters, weight factors and session sigmas, then updates each
        of them once from its residuals: the variance parameters (consistent M-scales under robust weights), then the
        weight factors from the residuals standardized by ``residual_variance_factors`` of the pass's fitted-value
        ratios and weight factors (the residuals' own variance, so that the standardized residuals are w-statistics;
        the redundancy numbers without robust weights), then the session sigmas by one profile Fisher-scoring step.
        Passes stop when no weight factor changes by more than ``WEIGHT_FACTOR_TOLERANCE`` and no observation or
        session sigma by more than ``SIGMA_RELATIVE_TOLERANCE`` (``converged``), or after ``PASS_LIMIT`` passes. The
        returned solution is that of the last pass.

        One update per pass: iterating variances and weights to their joint fixed point on the residuals of one pass
        overshoots, since the next solve moves the residuals, and on LLR data the passes then settle into a 2-cycle.

        Rejection locking: the variance estimating equations count an active observation (``f > 0``) with its full
        expected contribution (redundancy, and ``kappa_2 p`` in the noise of its session offset) however small its
        weight, and a rejected one not at all. They jump when an observation near the rejection threshold changes
        status, and in a component of short sessions the jump can feed back into a limit cycle of rejection, variance
        parameters and session sigma. An observation whose rejection status has changed
        ``REJECTION_STATUS_CHANGE_LIMIT`` times within the passes stays rejected for the remaining passes of the
        linearization, as in iterative outlier elimination (data snooping), where a rejected observation is not
        readmitted. Observations that settle are not affected.

        Under-relaxation: with ``floorAndScaled`` each pass moves ``(s^2, a^2)`` only ``FLOOR_MODEL_RELAXATION`` of
        the way to the LS-VCE proposal, which damps the cycle of floor and robust weights in components whose floor is
        small next to their scaled reported sigmas; the fixed point is the same.
        """
        robust = self.adjustment.estimateRobustWeights
        variance = self.adjustment.estimateVarianceFactors
        equations = linearization.equations
        diagnostics: dict[str, dict[str, object]] = {}
        session_diagnostics: dict[str, dict[str, object]] = {}
        records: list[dict[str, object]] = []
        converged = False
        status_changes = np.zeros(len(weight_vector), dtype=int)
        locked = np.zeros(len(weight_vector), dtype=bool)
        for index in range(1, PASS_LIMIT + 1):
            started = perf_counter()
            solution = self._solve_linearized(linearization, variances, weight_vector)
            redundancies = self._redundancies(linearization, solution)
            residuals = solution.residual_vector
            previous_variances, previous_weights = variances, weight_vector.copy()
            standardization = self._standardization(linearization, solution, variances, weight_vector, redundancies)
            if variance:
                proposed, diagnostics = self._estimate_variances(
                    linearization, residuals, standardization, variances, weight_vector, self._consistency
                )
                variances = (
                    variances.toward(proposed, FLOOR_MODEL_RELAXATION)
                    if self.variance_model == "floorAndScaled"
                    else proposed
                )
            if robust:
                standardized, _ = standardized_residuals(
                    residuals, standardization, _observation_sigmas(linearization, variances)
                )
                weight_vector = self.robust.factors(standardized)
                status_changes += (weight_vector == 0.0) != (previous_weights == 0.0)
                locked |= status_changes >= REJECTION_STATUS_CHANGE_LIMIT
                weight_vector = np.where(locked, 0.0, weight_vector)
            offset_redundancy = np.empty(0)
            if self.adjustment.sessionCorrelation and variance:
                elimination, offsets = _session_offsets(solution)
                offset_redundancy = elimination.offset_values(elimination.effect_redundancy(solution.covariance))
                # Excess of the offsets' noise variance under robust weights over that with the weights taken as
                # inverse variances: rows contribute kappa_2 p (active) instead of f p.
                precision = np.where(weight_vector > 0.0, 1.0 / _observation_sigmas(linearization, variances) ** 2, 0.0)
                noise_excess = elimination.offset_values(
                    elimination.noise_variance((self._session_consistency - weight_vector) * precision)
                )
                variances, session_diagnostics = session_variance_step(
                    model=self.variance_model,
                    sessions=linearization.sessions,
                    component_ids=linearization.component_ids,
                    apriori_sigmas=linearization.dense.apriori_sigmas,
                    residuals=residuals,
                    standardization=standardization,
                    weight_factors=weight_vector,
                    offsets=offsets,
                    offset_redundancy=offset_redundancy,
                    variances=variances,
                    consistency=self._consistency,
                    offset_noise_excess=noise_excess,
                )
            record = iteration_record(
                outer=outer,
                iteration=global_inner + index,
                index=index,
                started=started,
                equations=equations,
                solution=solution,
                candidate_by_block=self.parametrization.update_norms(solution.delta),
                before=previous_variances,
                after=variances,
                sigma_change=previous_variances.relative_change(
                    variances, linearization.component_ids, linearization.dense.apriori_sigmas
                ),
                previous_weights=previous_weights,
                weights=weight_vector,
                redundancy=float(np.sum(redundancies) + np.sum(offset_redundancy)),
                diagnostics=diagnostics,
                components=self._component_names,
                variance_model=self.variance_model,
            )
            record["lockedRejectionCount"] = int(np.count_nonzero(locked))
            records.append(record)
            if _is_stable(record):
                converged = True
                break
        return _StochasticModelUpdate(
            solution, variances, weight_vector, diagnostics, records, converged, session_diagnostics
        )

    def run(self, *, finalize: bool = True) -> LlrAdjustmentResult:
        """Run ``solve`` and build the adjustment report from its state."""
        return build_report(self.solve(finalize=finalize), self.settings)

    def solve(self, *, finalize: bool = True) -> AdjustmentState:
        """Iterate the nonlinear adjustment until convergence or ``maxIterationCount`` iterations.

        In each adjustment iteration the equations are linearized and solved with the current variance parameters
        and weight factors. With ``computeResiduals``, the variance parameters and/or robust weights are
        re-estimated by solve/reweight passes to their fixed point (see ``_solve_reweight_passes``). The solved
        correction is applied to the parametrization, and the estimate has converged when every block's update norm
        is within its threshold.

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
        self._pseudo = self.settings.constraints.pseudo_observations(
            self._names, self.parametrization.apriori(), station_position_columns(self.parametrization)
        )
        initial_reference_values = self.parametrization.x0().copy()
        variances, weight_factors = self._initial_stochastic_state()
        initial_variances = variances
        warm_sigma_count = len(self.initial_sigma_factors)
        warm_weight_count = sum(key in self.initial_weight_factors for key in self.observation_domain.retained_keys)

        current_equations = initial_equations
        iterations: list[dict[str, object]] = []
        adjustment_iterations: list[dict[str, object]] = []
        converged = False
        termination_reason = "MAX_ITERATION_COUNT_REACHED"
        global_inner = 0
        reweighting = self.adjustment.computeResiduals and (
            self.adjustment.estimateVarianceFactors or self.adjustment.estimateRobustWeights
        )

        for outer in range(1, self.adjustment.maxIterationCount + 1):
            linearization = self._linearize(current_equations)
            variances_used = variances
            weight_vector = np.asarray([weight_factors[key] for key in linearization.dense.identities])
            summary_used = weight_factor_summary(current_equations, weight_factors)
            if not reweighting:
                update = _StochasticModelUpdate(
                    self._solve_linearized(linearization, variances, weight_vector),
                    variances,
                    weight_vector,
                    {},
                    [],
                )
            else:
                update = self._solve_reweight_passes(linearization, outer, global_inner, variances, weight_vector)
            variances, weight_vector = update.variances, update.weight_vector
            keys = current_equations.observation_id.tolist()
            weight_factors.update(zip(keys, map(float, weight_vector), strict=True))
            for record in update.records:
                iterations.append(record)
                if self.iteration_callback is not None:
                    self.iteration_callback(record)
            global_inner += len(update.records)
            candidate_by_block = self.parametrization.update_norms(update.solution.delta)

            parameter_converged, tolerances, normalized_updates = parameter_convergence(
                candidate_by_block,
                self.adjustment.convergenceThreshold,
                self.adjustment.convergenceThresholdByParametrizations or {},
            )
            applied_updates = self.parametrization.update(update.solution.delta)
            adjustment_iterations.append(
                self._adjustment_iteration_record(
                    outer,
                    update,
                    equations=current_equations,
                    variances_used=variances_used,
                    summary_used=summary_used,
                    candidate_by_block=candidate_by_block,
                    parameter_converged=parameter_converged,
                    tolerances=tolerances,
                    normalized_updates=normalized_updates,
                    applied_updates=applied_updates,
                )
            )
            if parameter_converged:
                converged = True
                termination_reason = "CONVERGED"
                break
            if outer < self.adjustment.maxIterationCount:
                current_equations = self._equations("adjustment-iteration")

        # EstimateControl requires maxIterationCount >= 1, so the loop has produced an update.
        last_solution = update.solution
        if finalize:
            final = self._finalize(variances, weight_factors, update.session_diagnostics)
        else:
            final = _FinalEvaluation(
                solution=last_solution,
                equations=current_equations,
                standardized=np.empty(0),
                residual_sigmas=np.empty(0),
                proposed_weight_factors={},
                diagnostics=update.diagnostics,
                constraints=[],
                earth_orientation=[],
            )
        return AdjustmentState(
            solution=final.solution,
            last_solution=last_solution,
            equations=final.equations,
            converged=converged,
            termination_reason=termination_reason,
            iterations=iterations,
            adjustment_iterations=adjustment_iterations,
            sigma_factors=dict(variances.sigma_factors),
            floor_sigmas=dict(variances.floor_sigmas),
            session_sigmas=dict(variances.session_sigmas),
            weight_factors=weight_factors,
            initial_sigma_factors=dict(initial_variances.sigma_factors),
            initial_floor_sigmas=dict(initial_variances.floor_sigmas),
            initial_reference_values=initial_reference_values,
            warm_sigma_count=warm_sigma_count,
            warm_weight_count=warm_weight_count,
            equation_evaluations=self._equation_evaluations,
            performance_seconds=self._performance_seconds,
            observation_domain=self.observation_domain,
            parametrization=self.parametrization,
            utc_offset_hours=self.utc_offset_hours,
            standardized=final.standardized,
            residual_sigmas=final.residual_sigmas,
            proposed_weight_factors=final.proposed_weight_factors,
            diagnostics=final.diagnostics,
            finalize=finalize,
            constraints=final.constraints,
            earth_orientation=final.earth_orientation,
        )

    def _initial_stochastic_state(self) -> tuple[ObservationVariances, dict[ObsKey, float]]:
        """Return the starting variance parameters and the starting weight factor of every retained observation.

        A retained observation can temporarily disappear when its light-time solution does not converge; its
        weight factor is kept so that it can return in a later linearization or processing step.
        """
        variances = ObservationVariances.initial(
            (item.id for item in self.variance_components.components),
            self.initial_sigma_factors,
            self.initial_floor_sigmas,
            self.adjustment.sessionInitialSigmaM if self.adjustment.sessionCorrelation else None,
            self.initial_session_sigmas,
        )
        if self.variance_model == "scaled":
            variances = variances.without_floors()
        weight_factors = {
            identity: float(self.initial_weight_factors.get(identity, 1.0))
            for identity in self.observation_domain.retained_keys
        }
        return variances, weight_factors

    def _adjustment_iteration_record(
        self,
        outer: int,
        update: _StochasticModelUpdate,
        *,
        equations: EquationBatch,
        variances_used: ObservationVariances,
        summary_used: dict[str, object],
        candidate_by_block: dict[str, float],
        parameter_converged: bool,
        tolerances: dict[str, float],
        normalized_updates: dict[str, float],
        applied_updates: dict[str, float],
    ) -> dict[str, object]:
        """Report record of one adjustment iteration; called after its correction has been applied."""
        solution, variances = update.solution, update.variances
        return {
            "iteration": outer,
            "sigmaWeightIterations": len(update.records),
            "stochasticModelConverged": update.converged,
            "maximumParameterUpdateM": max(candidate_by_block.values(), default=0.0),
            "parameterUpdateWithinThreshold": parameter_converged,
            "convergenceThresholdByParametrizationM": tolerances,
            "normalizedParameterUpdateByBlock": normalized_updates,
            "parameterConverged": parameter_converged,
            "appliedUpdateByBlockM": applied_updates,
            "wrmsM": solution.wrms_m,
            "equationCount": len(equations),
            "candidateUpdateByBlockM": candidate_by_block,
            "candidateParameterCorrectionsM": {
                str(name): float(value) for name, value in zip(self._names, solution.delta)
            },
            "sigmaFactorsUsedInSolve": dict(variances_used.sigma_factors),
            "sigmaFactorsForNextIteration": dict(variances.sigma_factors),
            **floor_record(variances_used, self.variance_model, "floorSigmasUsedInSolveM"),
            **floor_record(variances, self.variance_model, "floorSigmasForNextIterationM"),
            "weightFactorSummaryUsedInSolve": summary_used,
            "weightFactorSummaryForNextIteration": (
                update.records[-1]["weightFactorSummary"] if update.records else summary_used
            ),
            "normalMatrixRank": solution.rank,
            "normalMatrixCondition": solution.condition,
            "stateAfterUpdate": self.parametrization.state(),
        }

    def _finalize(
        self,
        variances: ObservationVariances,
        weight_factors: dict[ObsKey, float],
        session_diagnostics: dict[str, dict[str, object]],
    ) -> _FinalEvaluation:
        """Re-evaluate and solve once more at the final parameter state, without applying the correction.

        Parameters
        ----------
        variances : ObservationVariances
            Final variance parameters.
        weight_factors : dict[ObsKey, float]
            Final weight factor of every retained observation.
        session_diagnostics : dict[str, dict[str, object]]
            Session-sigma diagnostics of the last stochastic-model update.
        """
        equations = self._equations("final-state-report")
        linearization = self._linearize(equations)
        weight_vector = np.asarray([weight_factors[key] for key in linearization.dense.identities])
        solution = self._solve_linearized(linearization, variances, weight_vector)
        redundancies = self._redundancies(linearization, solution)
        standardization = self._standardization(linearization, solution, variances, weight_vector, redundancies)
        standardized, residual_sigmas = standardized_residuals(
            solution.residual_vector, standardization, _observation_sigmas(linearization, variances)
        )
        proposed_weight_factors = dict(weight_factors)
        if self.adjustment.estimateRobustWeights:
            proposed = self.robust.factors(standardized)
            proposed_weight_factors = dict(zip(linearization.dense.identities, map(float, proposed), strict=True))
        diagnostics: dict[str, dict[str, object]] = {}
        if self.adjustment.estimateVarianceFactors:
            _, diagnostics = self._estimate_variances(
                linearization,
                solution.residual_vector,
                standardization,
                variances,
                weight_vector,
                self._consistency,
            )
        diagnostics = component_diagnostics(
            diagnostics,
            sessions=linearization.sessions,
            component_ids=linearization.component_ids,
            weight_factors=weight_vector,
            components=self._component_names,
            expected_tail_fractions=(
                self.robust.tail_fractions() if self.adjustment.estimateRobustWeights else (0.0, 0.0)
            ),
            variance_model=self.variance_model,
            consistency=self._consistency,
            session_consistency=self._session_consistency if self.adjustment.sessionCorrelation else None,
            session_diagnostics=session_diagnostics,
        )
        constraints = self._pseudo.records(
            solution.normals.x0 + solution.delta, solution.covariance, solution.sigma0_post
        )
        return _FinalEvaluation(
            solution=solution,
            equations=equations,
            standardized=standardized,
            residual_sigmas=residual_sigmas,
            proposed_weight_factors=proposed_weight_factors,
            diagnostics=diagnostics,
            constraints=constraints,
            earth_orientation=_earth_orientation_records(linearization, solution),
        )


def _observation_sigmas(linearization: _Linearization, variances: ObservationVariances) -> np.ndarray:
    """One-way sigma of every row of a linearization under ``variances``, in metres."""
    return variances.sigmas(linearization.component_ids, linearization.dense.apriori_sigmas)


def _observation_weights(
    linearization: _Linearization, variances: ObservationVariances, weight_factors: np.ndarray
) -> np.ndarray:
    return weight_factors / _observation_sigmas(linearization, variances) ** 2


def _session_sigma_rows(linearization: _Linearization, variances: ObservationVariances) -> np.ndarray:
    """Session sigma of every night session of a linearization, in metres."""
    sigma = variances.session_sigmas
    return np.asarray([sigma[component] for component in linearization.sessions.key], dtype=float)


def _session_offsets(solution: LinearizedSolution) -> tuple[RandomEffectElimination, np.ndarray]:
    """Return the random-effect elimination and the session offsets of a solution with night sessions."""
    if solution.elimination is None or solution.session_offsets is None:
        raise RuntimeError("The solution has no session offsets although sessionCorrelation is set.")
    return solution.elimination, solution.session_offsets


def _earth_orientation_records(linearization: _Linearization, solution: LinearizedSolution) -> list[dict[str, object]]:
    """Earth-orientation correction records of a solution; empty without ``earthOrientationCorrection``."""
    if linearization.eop is None:
        return []
    if solution.elimination is None or solution.random_effects is None:
        raise RuntimeError("The solution has no random effects although earthOrientationCorrection is set.")
    return linearization.eop.records(
        solution.equations, solution.elimination, solution.random_effects, solution.covariance
    )


def _is_stable(record: Mapping[str, object]) -> bool:
    """Whether a pass changed no weight factor, observation sigma or session sigma beyond the tolerances."""
    return (
        cast(float, record["maximumWeightFactorChange"]) <= WEIGHT_FACTOR_TOLERANCE
        and cast(float, record["maximumObservationSigmaRelativeChange"]) <= SIGMA_RELATIVE_TOLERANCE
        and cast(float, record["maximumSessionSigmaRelativeChange"]) <= SIGMA_RELATIVE_TOLERANCE
    )


__all__ = ["LlrAdjustmentSolver"]
