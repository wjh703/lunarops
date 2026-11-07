"""Stochastic model of the LLR observations: the variance function of each variance component and its estimation.

Observation ``i`` of variance component ``k`` has the one-way variance

    ``sigma_i^2 = a_k^2 + s_k^2 sigma_rep,i^2``

with reported (a-priori) one-way sigma ``sigma_rep,i``, sigma factor ``s_k`` and floor sigma ``a_k``, in metres. The
adjustment weights the observation with ``f_i / sigma_i^2``, ``f_i`` being its robust weight factor.

Variance models
---------------
``scaled``
    ``a_k = 0``: the reported sigmas are scaled per component, as the LLR analysis centres scale the normal-point
    uncertainties per station and period.
``floorAndScaled``
    ``a_k`` and ``s_k`` are both estimated: an error floor added in quadrature to the scaled reported sigma, as VLBI
    analyses add a noise term in quadrature to the formal delay errors. The two variance components of each group are
    estimated by least-squares variance component estimation (LS-VCE, Teunissen and Amiri-Simkooei 2008, J Geod
    82:65-82), as for multi-component GNSS stochastic models.

Estimation
----------
The variance parameters ``theta = (s_k^2, a_k^2)`` of a component, with cofactors ``q_i = (sigma_rep,i^2, 1)``,
solve the estimating equations

    ``sum_i (q_ij / sigma_i^4) (f_i v_i^2 - kappa rho_i sigma_i^2) = 0``,  ``j = 1, 2``,

over the component's active observations (``f_i > 0``), with residual ``v_i``, residual-variance factor ``rho_i``
(the redundancy number; under robust weights ``(1 - f_i t_i) (1 + (1 - f_i) t_i)``,
see :func:`lunarops.estimation.adjustment_solver.residual_variance_factors`) and the
consistency factor ``kappa`` of the robust weight function. One update is the weighted least-squares fit of the
observed variances ``y_i = f_i v_i^2 / (kappa rho_i)`` by ``q_i theta`` with weights ``rho_i / sigma_i^4`` at the
current ``theta``, constrained to ``theta >= 0``; iterated, it converges to the solution of the equations. With the
single component of ``scaled`` the update is the closed form ``s^2 = sum f v^2 / sigma_rep^2 / (kappa sum rho)``.

Night sessions
--------------
With night sessions (:mod:`lunarops.estimation.robust_vce`) the covariance of a session ``n`` of component ``k`` is
``V_n = D_n + tau_k^2 1 1^T``, so ``tau_k^2`` is a third variance component with the cofactor ``1 1^T`` per session.
Its REML estimating equation is ``sum_n (b_n^2 / tau^4 - r_n / tau^2) = 0`` with offset estimate ``b_n`` and
offset redundancy ``r_n``. White and session components are strongly coupled in short sessions, so separate updates
of each (simplified Helmert) converge slowly. ``session_variance_step`` instead takes the profile Fisher-scoring step
for ``tau^2``: the joint REML (rigorous Helmert, LS-VCE) step of all components of the group, ``delta = H^-1 S`` with
the score ``S`` of the estimating equations above and the Fisher information ``H_jl = tr(W Q_j W Q_l) / 2`` of the
session-block covariance (``W = V^-1``, the projection on the parameters neglected), with the white components then
re-solved by their own estimator. The fixed point is that of the separate updates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from lunarops.estimation.robust_vce import MINIMUM_SESSION_SIGMA_M, Sessions
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition

VARIANCE_MODELS = ("scaled", "floorAndScaled")
#: Bounds of one session-variance step, as factors of the current ``tau^2``.
SESSION_STEP_BOUNDS = (0.1, 100.0)

# Algorithm invariant, not a dataset-tuning parameter: a component needs a total
# redundancy above this value before its variance parameters are updated.
MINIMUM_COMPONENT_REDUNDANCY = 3.0

#: Relative determinant below which the two cofactors of a component are not separable.
SEPARABILITY_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class ObservationVariances:
    """Variance parameters of every component: ``sigma_i^2 = a_k^2 + s_k^2 sigma_rep,i^2`` and session sigma ``tau_k``.

    Attributes
    ----------
    sigma_factors : dict[str, float]
        Sigma factor ``s_k`` per component.
    floor_sigmas : dict[str, float]
        Floor sigma ``a_k`` per component, in metres.
    session_sigmas : dict[str, float]
        Sigma ``tau_k`` of the night-session offsets per component, in metres; empty without night sessions.
    """

    sigma_factors: dict[str, float]
    floor_sigmas: dict[str, float]
    session_sigmas: dict[str, float] = field(default_factory=dict)

    @classmethod
    def initial(
        cls,
        components: Iterable[str],
        sigma_factors: Mapping[str, float] | None = None,
        floor_sigmas: Mapping[str, float] | None = None,
        session_sigma: float | None = None,
        session_sigmas: Mapping[str, float] | None = None,
    ) -> ObservationVariances:
        """Return unit sigma factors and zero floors, overridden by the given values of known components.

        With ``session_sigma`` every component gets that session sigma, overridden by ``session_sigmas``; without
        it the model has no night sessions and ``session_sigmas`` is ignored.
        """
        identifiers = list(components)
        factors = {key: 1.0 for key in identifiers}
        floors = {key: 0.0 for key in identifiers}
        factors.update({key: float(value) for key, value in (sigma_factors or {}).items() if key in factors})
        floors.update({key: float(value) for key, value in (floor_sigmas or {}).items() if key in floors})
        sessions: dict[str, float] = {}
        if session_sigma is not None:
            sessions = {key: float(session_sigma) for key in identifiers}
            sessions.update({key: float(value) for key, value in (session_sigmas or {}).items() if key in sessions})
        return cls(factors, floors, sessions)

    def without_floors(self) -> ObservationVariances:
        """Return the same sigma factors and session sigmas with zero floors."""
        return replace(self, floor_sigmas={key: 0.0 for key in self.floor_sigmas})

    def sigmas(self, component_ids: np.ndarray, apriori_sigmas: np.ndarray) -> np.ndarray:
        """Return ``sqrt(a_k^2 + (s_k sigma_rep,i)^2)`` per observation, in metres.

        It is exactly ``s_k sigma_rep,i`` when ``a_k = 0``.
        """
        factors = np.asarray([self.sigma_factors[key] for key in component_ids], dtype=float)
        floors = np.asarray([self.floor_sigmas[key] for key in component_ids], dtype=float)
        return np.hypot(floors, factors * np.asarray(apriori_sigmas, dtype=float))

    def relative_change(
        self, other: ObservationVariances, component_ids: np.ndarray, apriori_sigmas: np.ndarray
    ) -> float:
        """Largest relative change of an observation sigma from ``self`` to ``other``."""
        if len(component_ids) == 0:
            return 0.0
        before = self.sigmas(component_ids, apriori_sigmas)
        after = other.sigmas(component_ids, apriori_sigmas)
        return float(np.max(np.abs(after - before) / before))


@dataclass(frozen=True)
class VarianceComponentEstimate:
    """Result of one variance-component update.

    Attributes
    ----------
    variances : ObservationVariances
        Updated variance parameters.
    diagnostics : dict[str, dict[str, object]]
        Per component: active count, effective redundancy, weighted square sum, sigma factor and floor before and
        after, variance ratio, ``updateStatus`` and, with ``floorAndScaled``, the active bound.
    """

    variances: ObservationVariances
    diagnostics: dict[str, dict[str, object]]


@dataclass(frozen=True)
class VarianceComponentEstimator:
    """Variance-component estimator of the ``scaled`` or ``floorAndScaled`` model on given residuals.

    Attributes
    ----------
    components : tuple[VarianceComponentDefinition, ...]
        Components to estimate.
    model : str, default "scaled"
        Variance model, one of ``VARIANCE_MODELS``.
    """

    components: tuple[VarianceComponentDefinition, ...]
    model: str = "scaled"

    def __post_init__(self) -> None:
        if self.model not in VARIANCE_MODELS:
            raise ValueError(f"Unknown variance model {self.model!r}; expected one of {VARIANCE_MODELS}.")

    def estimate(
        self,
        *,
        apriori_sigmas: np.ndarray,
        residuals: np.ndarray,
        standardization: np.ndarray,
        component_ids: np.ndarray,
        weight_factors: np.ndarray,
        variances: ObservationVariances,
        consistency: float = 1.0,
    ) -> VarianceComponentEstimate:
        """Update the variance parameters of every component once.

        A component keeps its parameters when it has no observations (``NO_OBSERVATIONS``), its total
        residual-variance factor is at most ``MINIMUM_COMPONENT_REDUNDANCY`` (``INSUFFICIENT_REDUNDANCY``) or its
        weighted square sum is zero (``ZERO_VARIANCE_TARGET``); otherwise it is updated (``UPDATED``).

        Parameters
        ----------
        apriori_sigmas : ndarray, shape (n,)
            Reported one-way sigmas, in metres.
        residuals : ndarray, shape (n,)
            Post-fit one-way residuals, in metres.
        standardization : ndarray, shape (n,)
            Residual-variance factor ``rho_i`` of every observation (redundancy number without robust weights).
        component_ids : ndarray of object, shape (n,)
            Component identifier per observation.
        weight_factors : ndarray, shape (n,)
            Robust weight factors; observations with factor 0 are inactive.
        variances : ObservationVariances
            Current variance parameters; with ``scaled`` their floors must be zero.
        consistency : float, default 1.0
            Consistency factor ``kappa`` of the robust weight function.

        Returns
        -------
        VarianceComponentEstimate
            Updated variance parameters and diagnostics.

        Raises
        ------
        ValueError
            If ``scaled`` is given non-zero floors.
        RuntimeError
            If an estimated variance is negative or not finite.
        """
        if self.model == "scaled" and any(value != 0.0 for value in variances.floor_sigmas.values()):
            raise ValueError("The scaled variance model has no floor sigmas.")
        factors = dict(variances.sigma_factors)
        floors = dict(variances.floor_sigmas)
        diagnostics: dict[str, dict[str, object]] = {}
        active = weight_factors > 0.0
        for component in self.components:
            key = component.id
            factor, floor = float(variances.sigma_factors[key]), float(variances.floor_sigmas[key])
            rows = component_ids == key
            mask = active & rows
            redundancy = float(np.sum(standardization[mask]))
            weighted_square_sum = float(np.sum(weight_factors[mask] * (residuals[mask] / apriori_sigmas[mask]) ** 2))
            new_factor, new_floor, bound, variance_ratio = factor, floor, None, 1.0
            if not np.any(rows):
                status = "NO_OBSERVATIONS"
            elif redundancy <= MINIMUM_COMPONENT_REDUNDANCY:
                status = "INSUFFICIENT_REDUNDANCY"
            elif weighted_square_sum == 0.0:
                status = "ZERO_VARIANCE_TARGET"
            elif self.model == "scaled":
                proposed_variance = weighted_square_sum / (consistency * redundancy)
                if not np.isfinite(proposed_variance) or proposed_variance < 0.0:
                    raise RuntimeError(f"Invalid sigma-factor estimate for {key!r}.")
                new_factor, status = float(np.sqrt(proposed_variance)), "UPDATED"
                variance_ratio = proposed_variance / factor**2
            else:
                theta, bound = _floor_and_scale_update(
                    apriori_sigmas[mask],
                    residuals[mask],
                    standardization[mask],
                    weight_factors[mask],
                    factor,
                    floor,
                    consistency,
                )
                if not np.all(np.isfinite(theta)) or np.any(theta < 0.0):
                    raise RuntimeError(f"Invalid variance-component estimate for {key!r}.")
                new_factor, new_floor, status = float(np.sqrt(theta[0])), float(np.sqrt(theta[1])), "UPDATED"
                variance_ratio = _variance_ratio(apriori_sigmas[rows], factor, floor, new_factor, new_floor)
            factors[key], floors[key] = new_factor, new_floor
            item: dict[str, object] = {
                "activeCount": int(np.count_nonzero(mask)),
                "effectiveRedundancy": redundancy,
                "weightedSquareSum": weighted_square_sum,
                "sigmaFactorBefore": factor,
                "sigmaFactorAfter": new_factor,
                "varianceRatio": float(variance_ratio),
                "updateStatus": status,
            }
            if self.model == "floorAndScaled":
                item.update({"floorSigmaBeforeM": floor, "floorSigmaAfterM": new_floor, "activeBound": bound})
            diagnostics[key] = item
        return VarianceComponentEstimate(replace(variances, sigma_factors=factors, floor_sigmas=floors), diagnostics)


def session_variance_step(
    *,
    model: str,
    sessions: Sessions,
    component_ids: np.ndarray,
    apriori_sigmas: np.ndarray,
    residuals: np.ndarray,
    standardization: np.ndarray,
    weight_factors: np.ndarray,
    offsets: np.ndarray,
    offset_redundancy: np.ndarray,
    variances: ObservationVariances,
    offset_noise_excess: np.ndarray,
    consistency: float = 1.0,
) -> tuple[ObservationVariances, dict[str, dict[str, object]]]:
    """Take one profile Fisher-scoring step for the session variance ``tau_k^2`` of every component.

    See the module docstring. Per component, with active white precisions ``p_i = 1 / sigma_i^2`` (zero for rejected
    rows), ``lambda = 1 / tau^2`` and ``d_n = sum_{i in n} p_i + lambda``:

    * scores ``S_j = sum_i (q_ij / sigma_i^4) (f_i v_i^2 / kappa - rho_i sigma_i^2) / 2`` of the white cofactors and
      ``S_tau = sum_n lambda^2 (b_n^2 - tau^2 r_n - e_n) / 2``, where ``tau^2 r_n`` is the expectation of ``b_n^2``
      with the robust weights ``f_i p_i`` taken as inverse variances and ``e_n`` (``offset_noise_excess``) corrects
      its noise part to the expectation under the robust weights (``kappa_2 = E[w^2 z^2 | w > 0]`` instead of ``f``
      per row; zero without robust weights), as robust REML corrects each estimating equation for the weight
      function; for a session alone in its block ``e_n = (kappa_2 P_n - F_n) / (F_n + lambda)^2``,
      ``F_n = sum f_i p_i``;
    * Fisher information ``H_jl = [sum_i q_ij q_il (p_i^2 - 2 p_i^3 / d_n) + sum_n (sum p^2 q_j)(sum p^2 q_l) / d_n^2] / 2``,
      ``H_j,tau = sum_n (lambda / d_n)^2 sum_{i in n} p_i^2 q_ij / 2`` and ``H_tau,tau = sum_n (lambda P_n / d_n)^2 / 2``;
    * step ``delta = (S_tau - H_tau,w H_ww^+ S_w) / (H_tau,tau - H_tau,w H_ww^+ H_w,tau)``, bounded to
      ``SESSION_STEP_BOUNDS`` times ``tau^2`` and to ``MINIMUM_SESSION_SIGMA_M``.

    Components with at most ``MINIMUM_COMPONENT_REDUNDANCY`` sessions, or a non-positive profile information, keep
    their value.

    Parameters
    ----------
    model : str
        Variance model of the white part, one of ``VARIANCE_MODELS``.
    sessions : Sessions
        Night sessions of the rows.
    component_ids, apriori_sigmas, residuals, standardization, weight_factors : ndarray, shape (n,)
        Component, reported sigma (m), white residual after the session offset (m), residual-variance factor and
        robust weight factor of every row.
    offsets, offset_redundancy : ndarray, shape (s,)
        Estimated offset (m) and redundancy of every session.
    variances : ObservationVariances
        Current variance parameters, with the current ``tau`` per component in ``session_sigmas``.
    offset_noise_excess : ndarray, shape (s,)
        Excess ``e_n`` of the noise variance of every offset estimate under the robust weights, in m^2.
    consistency : float, default 1.0
        Consistency factor ``kappa`` of the robust weight function.

    Returns
    -------
    tuple[ObservationVariances, dict[str, dict[str, object]]]
        ``variances`` with the updated ``tau`` per component, and per-component diagnostics.
    """
    sigma = variances.sigmas(component_ids, apriori_sigmas)
    active = weight_factors > 0.0
    precision = np.where(active, 1.0 / sigma**2, 0.0)
    cofactors = [apriori_sigmas**2] if model == "scaled" else [apriori_sigmas**2, np.ones(len(sigma))]
    score_terms = np.where(
        active, (weight_factors * residuals**2 / consistency - standardization * sigma**2) / sigma**4, 0.0
    )
    updated = dict(variances.session_sigmas)
    diagnostics: dict[str, dict[str, object]] = {}
    for component, before in variances.session_sigmas.items():
        rows = component_ids == component
        session_rows = sessions.component == component
        count = int(np.count_nonzero(session_rows))
        tau2 = max(float(before), MINIMUM_SESSION_SIGMA_M) ** 2
        status = "HELD"
        if count > MINIMUM_COMPONENT_REDUNDANCY:
            expected = tau2 * offset_redundancy[session_rows] + offset_noise_excess[session_rows]
            step = _profile_session_step(
                sessions,
                rows,
                session_rows,
                precision,
                cofactors,
                score_terms,
                offsets[session_rows],
                expected,
                1.0 / tau2,
            )
            if step is not None:
                low, high = SESSION_STEP_BOUNDS
                proposed = tau2 + step
                bounded = min(max(proposed, low * tau2, MINIMUM_SESSION_SIGMA_M**2), high * tau2)
                updated[component] = float(np.sqrt(bounded))
                status = "UPDATED" if bounded == proposed else "BOUNDED"
        diagnostics[component] = {
            "sessionCount": count,
            "sessionRedundancy": float(np.sum(offset_redundancy[session_rows])),
            "sessionSigmaBeforeM": float(before),
            "sessionSigmaM": float(updated[component]),
            "sessionOffsetRmsM": float(np.sqrt(np.mean(offsets[session_rows] ** 2))) if count else None,
            "sessionUpdateStatus": status,
        }
    return replace(variances, session_sigmas=updated), diagnostics


def _profile_session_step(
    sessions: Sessions,
    rows: np.ndarray,
    session_rows: np.ndarray,
    precision: np.ndarray,
    cofactors: list[np.ndarray],
    score_terms: np.ndarray,
    offsets: np.ndarray,
    expected_squares: np.ndarray,
    inverse_tau2: float,
) -> float | None:
    """Profile Fisher-scoring step for ``tau^2`` of one component; ``None`` without positive profile information.

    ``offsets`` and ``expected_squares`` (the expectation of each squared offset) are those of the component's
    sessions.
    """
    p = np.where(rows, precision, 0.0)
    total = sessions.sums(p)[session_rows]
    d = total + inverse_tau2
    d_rows = (sessions.sums(p) + inverse_tau2)[sessions.index]
    squared = p**2
    white_score = np.array([0.5 * np.sum(np.where(rows, q * score_terms, 0.0)) for q in cofactors])
    session_score = 0.5 * inverse_tau2**2 * float(np.sum(offsets**2 - expected_squares))
    block = [sessions.sums(squared * q)[session_rows] for q in cofactors]
    size = len(cofactors)
    white = np.empty((size, size))
    for j in range(size):
        for k in range(size):
            diagonal = np.sum(cofactors[j] * cofactors[k] * (squared - 2.0 * squared * p / d_rows))
            white[j, k] = 0.5 * (diagonal + np.sum(block[j] * block[k] / d**2))
    cross = np.array([0.5 * np.sum((inverse_tau2 / d) ** 2 * block[j]) for j in range(size)])
    session = 0.5 * float(np.sum((inverse_tau2 * total / d) ** 2))
    inverse_white = np.linalg.pinv(white)
    information = session - float(cross @ inverse_white @ cross)
    if not np.isfinite(information) or information <= 0.0:
        return None
    return (session_score - float(cross @ inverse_white @ white_score)) / information


def _variance_ratio(
    apriori_sigmas: np.ndarray, factor: float, floor: float, new_factor: float, new_floor: float
) -> float:
    """Mean observation variance of the component after the update over the mean before."""
    mean_square = float(np.mean(apriori_sigmas**2))
    return (new_floor**2 + new_factor**2 * mean_square) / (floor**2 + factor**2 * mean_square)


def _floor_and_scale_update(
    apriori_sigmas: np.ndarray,
    residuals: np.ndarray,
    standardization: np.ndarray,
    weight_factors: np.ndarray,
    factor: float,
    floor: float,
    consistency: float,
) -> tuple[np.ndarray, str | None]:
    """One LS-VCE update ``theta = (s^2, a^2)`` of a ``floorAndScaled`` component; see the module docstring.

    Returns the non-negative weighted least-squares fit and the bound it lies on (``"scale"`` or ``"floor"`` when
    that parameter is zero, else ``None``).
    """
    usable = standardization > 0.0
    rho = standardization[usable]
    cofactors = np.column_stack((apriori_sigmas[usable] ** 2, np.ones(np.count_nonzero(usable))))
    variance = cofactors @ np.array([factor**2, floor**2])
    weights = rho / variance**2
    observed = weight_factors[usable] * residuals[usable] ** 2 / (consistency * rho)
    normal = cofactors.T @ (weights[:, None] * cofactors)
    rhs = cofactors.T @ (weights * observed)
    return _nonnegative_least_squares(normal, rhs)


def _nonnegative_least_squares(normal: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Minimize ``theta^T H theta - 2 r^T theta`` over ``theta >= 0`` for a 2x2 positive semi-definite ``H``.

    The unconstrained minimum is returned when it is feasible and the system is separable; otherwise the better of
    the two one-parameter minima, the scale-only one unless the floor-only one is better by more than rounding (as
    for inseparable cofactors, where both fit equally).
    """
    scale = np.sqrt(np.diag(normal))
    scaled_normal = normal / np.outer(scale, scale)
    scaled_rhs = rhs / scale
    determinant = float(np.linalg.det(scaled_normal))
    if determinant > SEPARABILITY_TOLERANCE:
        theta = np.linalg.solve(scaled_normal, scaled_rhs)
        if np.all(theta >= 0.0):
            return theta / scale, None
    best: tuple[float, np.ndarray, str] | None = None
    for index, bound in ((0, "floor"), (1, "scale")):
        candidate = np.zeros(2)
        candidate[index] = max(scaled_rhs[index], 0.0)
        objective = float(candidate @ scaled_normal @ candidate - 2.0 * scaled_rhs @ candidate)
        if best is None or objective < best[0] - SEPARABILITY_TOLERANCE * abs(best[0]):
            best = (objective, candidate, bound)
    assert best is not None
    return best[1] / scale, best[2]


__all__ = [
    "MINIMUM_COMPONENT_REDUNDANCY",
    "SESSION_STEP_BOUNDS",
    "VARIANCE_MODELS",
    "ObservationVariances",
    "VarianceComponentEstimate",
    "VarianceComponentEstimator",
    "session_variance_step",
]
