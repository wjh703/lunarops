"""Data models of the LLR adjustment: observation domain, solver state and result products."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import numpy as np

from lunarops.base.parameter_name import ParameterName, names_to_strings
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.normal_equations import NormalEquations

ObsKey = Hashable


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class LlrAdjustmentObservationDomain:
    """Fixed observation domain shared by all estimate processing steps.

    Attributes
    ----------
    gross_rejected : dict[ObsKey, float]
        Reduced pre-fit one-way residual, in metres, per observation rejected as a gross error.
    retained_keys : set[ObsKey]
        Identities of the observations used by the estimates.
    assignments : dict[ObsKey, str]
        Variance-component identifier per retained observation.
    accuracy_records : dict[ObsKey, dict[str, object]]
        A-priori sigma screening record per screened observation.
    accuracy_groups : dict[str, dict[str, object]]
        A-priori sigma screening summary per variance component.
    observation_signatures : dict[ObsKey, tuple[str, str, object, float | None]]
        Station key, reflector key, transmit epoch (UTC) and wavelength (nm) per retained observation; a later
        evaluation that changes them is an error.
    """

    gross_rejected: dict[ObsKey, float]
    retained_keys: set[ObsKey]
    assignments: dict[ObsKey, str]
    accuracy_records: dict[ObsKey, dict[str, object]]
    accuracy_groups: dict[str, dict[str, object]]
    observation_signatures: dict[ObsKey, tuple[str, str, object, float | None]]


@dataclass(eq=False, repr=False, slots=True)
class _LinearizedSolution:
    """Weighted least-squares solution at one linearization point.

    ``delta`` is the parameter correction from ``x0`` and ``residual_vector`` the linearized post-fit one-way
    residuals in metres (after removing the session offsets, with night sessions); ``covariance`` is the
    cofactor matrix ``N^-1`` (with the session offsets eliminated).
    """

    equations: EquationBatch
    normals: NormalEquations
    delta: np.ndarray
    wrms_m: float | None
    covariance: np.ndarray
    sigma0_post: float | None
    residual_vector: np.ndarray
    observation_weights: np.ndarray
    rank: int
    #: Random-effect elimination of a solution with session offsets or Earth-orientation corrections, else ``None``.
    elimination: object | None = None
    #: Estimated session offsets, in metres, with night sessions, else ``None``.
    session_offsets: np.ndarray | None = None
    #: Estimated random effects per block, shape ``(B, q)``, with random effects, else ``None``.
    random_effects: np.ndarray | None = None

    @property
    def condition(self) -> float:
        return self.normals.condition


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class AdjustmentState:
    """Complete state of a finished nonlinear estimate, the input of ``build_report``.

    Attributes
    ----------
    solution : _LinearizedSolution
        Final solution: the re-linearized final-state solution when ``finalize`` is true, otherwise the last
        iteration's solution.
    last_solution : _LinearizedSolution
        Solution of the last adjustment iteration.
    equations : EquationBatch
        Equations of ``solution``.
    converged : bool
        Whether all block updates were within their thresholds.
    termination_reason : str
        ``CONVERGED`` or ``MAX_ITERATION_COUNT_REACHED``.
    iterations : list[dict[str, object]]
        Records of the sigma/weight iterations.
    adjustment_iterations : list[dict[str, object]]
        Records of the adjustment (linearization) iterations.
    sigma_factors : dict[str, float]
        Sigma factor per variance component.
    floor_sigmas : dict[str, float]
        Floor sigma per variance component, in metres (zero with the ``scaled`` model).
    weight_factors : dict[ObsKey, float]
        Robust weight factor in [0, 1] per retained observation.
    initial_sigma_factors : dict[str, float]
        Sigma factors at the start of the estimate.
    initial_floor_sigmas : dict[str, float]
        Floor sigmas at the start of the estimate, in metres.
    initial_reference_values : ndarray
        Parameter values at the start of the estimate.
    warm_sigma_count : int
        Number of sigma factors taken from a previous estimate.
    warm_weight_count : int
        Number of weight factors taken from a previous estimate.
    equation_evaluations : list[dict[str, object]]
        Record of each observation-model evaluation.
    performance_seconds : dict[str, float]
        Wall-clock time per solver phase, in seconds.
    observation_domain : LlrAdjustmentObservationDomain
        Fixed observation domain.
    parametrization : ParametrizationList
        Estimated parametrization, holding the final values.
    utc_offset_hours : float
        Offset of local time from UTC, in hours, for reports.
    standardized : ndarray
        Final standardized residuals; empty when not finalized.
    residual_sigmas : ndarray
        Final residual sigmas, in metres; empty when not finalized.
    proposed_weight_factors : dict[ObsKey, float]
        Weight factors proposed from the final residuals (not applied); empty when not finalized.
    diagnostics : dict[str, dict[str, object]]
        Variance-component diagnostics.
    finalize : bool
        Whether a final-state solution was computed.
    constraints : list[dict[str, object]]
        Per pseudo-observation of the constraints: label, value at the final solution (``c x - t``), a-priori
        and a-posteriori sigma; empty without constraints or when not finalized.
    earth_orientation : list[dict[str, object]]
        Per station night with Earth-orientation corrections: station, first epoch, observation count, the
        corrections (ms, mas) with posterior and prior sigmas; empty without them or when not finalized.
    """

    solution: _LinearizedSolution
    last_solution: _LinearizedSolution
    equations: EquationBatch
    converged: bool
    termination_reason: str
    iterations: list[dict[str, object]]
    adjustment_iterations: list[dict[str, object]]
    sigma_factors: dict[str, float]
    floor_sigmas: dict[str, float]
    weight_factors: dict[ObsKey, float]
    initial_sigma_factors: dict[str, float]
    initial_floor_sigmas: dict[str, float]
    initial_reference_values: np.ndarray
    warm_sigma_count: int
    warm_weight_count: int
    equation_evaluations: list[dict[str, object]]
    performance_seconds: dict[str, float]
    observation_domain: LlrAdjustmentObservationDomain
    parametrization: ParametrizationList
    utc_offset_hours: float
    standardized: np.ndarray
    residual_sigmas: np.ndarray
    proposed_weight_factors: dict[ObsKey, float]
    diagnostics: dict[str, dict[str, object]]
    finalize: bool
    constraints: list[dict[str, object]] = field(default_factory=list)
    earth_orientation: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class LlrAdjustmentResult:
    """Report and products of one nonlinear LLR estimate.

    Attributes
    ----------
    converged : bool
        Whether the estimate converged.
    termination_reason : str
        ``CONVERGED`` or ``MAX_ITERATION_COUNT_REACHED``.
    settings : dict[str, object]
        Settings used by the estimate.
    equation_evaluations : list[dict[str, object]]
        Record of each observation-model evaluation.
    state : dict[str, object]
        Final parametrization state per block.
    gross_rejected : dict[ObsKey, float]
        Gross-error rejections with their reduced pre-fit residuals, in metres.
    accuracy_screening : dict[str, object]
        Summary of the a-priori sigma screening.
    sigma_factors : dict[str, float]
        Final sigma factor per variance component.
    floor_sigmas : dict[str, float]
        Final floor sigma per variance component, in metres (zero with the ``scaled`` model).
    weight_factors : dict[ObsKey, float]
        Final robust weight factor per retained observation.
    sigma_weight_iterations : list[dict[str, object]]
        Records of the sigma/weight iterations.
    adjustment_iterations : list[dict[str, object]]
        Records of the adjustment iterations.
    summary : dict[str, object]
        Counts, rank, condition number, ``sigma0Post`` and timings.
    observation_domain : LlrAdjustmentObservationDomain
        Fixed observation domain.
    parameters : list[dict[str, object]]
        Per-parameter records; empty when not finalized.
    global_residuals : dict[str, object]
        Residual statistics over all observations; empty when not finalized.
    variance_components : list[dict[str, object]]
        Per-component records; empty when not finalized.
    observations : list[dict[str, object]]
        Per-observation records; empty when not finalized.
    normals : NormalEquations
        Normal equations of the final solution; ``normals.x0`` holds the final parameter values.
    remaining_correction : ndarray
        Correction from ``normals.x0`` solved at the final state but not applied.
    cofactor : ndarray
        Cofactor matrix ``N^-1``.
    sigma0_post : float or None
        A-posteriori unit-weight sigma, ``None`` without redundancy.
    earth_orientation : list[dict[str, object]]
        Earth-orientation corrections per station night (see ``AdjustmentState``); empty without them.
    """

    converged: bool
    termination_reason: str
    settings: dict[str, object]
    equation_evaluations: list[dict[str, object]]
    state: dict[str, object]
    gross_rejected: dict[ObsKey, float]
    accuracy_screening: dict[str, object]
    sigma_factors: dict[str, float]
    floor_sigmas: dict[str, float]
    weight_factors: dict[ObsKey, float]
    sigma_weight_iterations: list[dict[str, object]]
    adjustment_iterations: list[dict[str, object]]
    summary: dict[str, object]
    observation_domain: LlrAdjustmentObservationDomain
    parameters: list[dict[str, object]]
    global_residuals: dict[str, object]
    variance_components: list[dict[str, object]]
    observations: list[dict[str, object]]
    normals: NormalEquations
    remaining_correction: np.ndarray
    cofactor: np.ndarray
    sigma0_post: float | None
    earth_orientation: list[dict[str, object]] = field(default_factory=list)

    @property
    def parameter_names(self) -> list[ParameterName]:
        """Parameter names in column order."""
        return self.normals.parameter_names

    def parameter_vector(self):
        """Return the final parameter values (``normals.x0``) with their uncertainties.

        The remaining correction is not applied. Uncertainties are ``PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER``
        times the one-sigma values: ``sigma0_post * sqrt(diag(cofactor))``, or ``sqrt(diag(cofactor))`` when
        ``sigma0_post`` is ``None``.

        Returns
        -------
        ParameterVector
            Names, values, units, uncertainties and the sigma multiplier.
        """
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
        """Return the posterior parameter covariance matrix.

        The matrix is ``sigma0_post**2 * cofactor``, or the cofactor itself when ``sigma0_post`` is ``None``.

        Returns
        -------
        CovarianceMatrix
            Matrix with names, units and kind ``posteriorCovariance`` or ``cofactor``.
        """
        from lunarops.estimation.parameter_products import CovarianceMatrix

        matrix = self.cofactor if self.sigma0_post is None else self.sigma0_post**2 * self.cofactor
        return CovarianceMatrix(
            tuple(self.parameter_names),
            matrix,
            tuple(self.normals.parameter_units),
            "cofactor" if self.sigma0_post is None else "posteriorCovariance",
        )

    def publication_quality(self) -> dict[str, object]:
        """Return whether the result qualifies as a formal product, and the reasons when it does not.

        The result is diagnostic only when the estimate did not converge, there is no posterior redundancy, the
        cofactor is not finite, a variance component had insufficient redundancy, or any reported observation has
        a clamped troposphere elevation.

        Returns
        -------
        dict[str, object]
            ``formalProductEligible`` (bool), ``status`` (``passed`` or ``diagnostic``) and ``reasons``.
        """
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
        """Check that the result qualifies as a formal product.

        Raises
        ------
        ValueError
            If ``publication_quality`` reports reasons.
        """
        quality = self.publication_quality()
        if not quality["formalProductEligible"]:
            raise ValueError(f"Adjustment is diagnostic only: {quality['reasons']}")

    def to_dict(self) -> dict[str, object]:
        """Return the report as a JSON-style dictionary with camelCase keys."""
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
            "floorSigmasM": self.floor_sigmas,
            "weightFactors": self.weight_factors,
            "sigmaWeightIterations": self.sigma_weight_iterations,
            "adjustmentIterations": self.adjustment_iterations,
            "summary": self.summary,
            "parameters": self.parameters,
            "globalResiduals": self.global_residuals,
            "varianceComponents": self.variance_components,
            **({"earthOrientationCorrections": self.earth_orientation} if self.earth_orientation else {}),
            "observations": self.observations,
        }


__all__ = [
    "AdjustmentState",
    "LlrAdjustmentObservationDomain",
    "LlrAdjustmentResult",
]
