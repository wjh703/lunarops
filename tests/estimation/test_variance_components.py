"""Stochastic model: the ``floorAndScaled`` variance function and its LS-VCE estimation.

Synthetic residuals and linear problems with variance ``a^2 + (s sigma_rep)^2`` must return ``a`` and ``s``; the
estimate must stay on the non-negative bound when a component is absent, and ``scaled`` must remain the closed-form
single-component estimate.
"""

import numpy as np
import pytest

from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.classes.time import Epoch, TimeScale
from lunarops.estimation.adjustment.plan import (
    AccuracyScreeningSettings,
    EstimateControl,
    LlrAdjustmentSettings,
    ResidualScreening,
    RobustWeightSettings,
    VarianceComponentSettings,
)
from lunarops.estimation.adjustment.screening import screen_observations
from lunarops.estimation.adjustment.solver import LlrAdjustmentSolver
from lunarops.estimation.stochastic.sessions import night_sessions
from lunarops.estimation.stochastic.variances import ObservationVariances
from lunarops.estimation.stochastic.vce import VarianceComponentEstimator, session_variance_step
from tests.estimation.test_reweighting import COMPONENT, START_JD, Linear, _run
from tests.estimation.test_reweighting import _problem as _session_problem


def _iterate(apriori, residuals, model="floorAndScaled", start=(1.0, 0.0), count=200):
    """Iterate the estimator on fixed residuals (redundancy 1, full weights) to its fixed point."""
    estimator = VarianceComponentEstimator((COMPONENT,), model)
    variances = ObservationVariances({"A": start[0]}, {"A": start[1]})
    ids = np.full(len(apriori), "A", dtype=object)
    for _ in range(count):
        estimate = estimator.estimate(
            apriori_sigmas=apriori,
            residuals=residuals,
            standardization=np.ones(len(apriori)),
            component_ids=ids,
            weight_factors=np.ones(len(apriori)),
            variances=variances,
        )
        variances = estimate.variances
    return variances, estimate.diagnostics["A"]


def _residuals(floor, factor, n=200_000, seed=1, apriori=None):
    rng = np.random.default_rng(seed)
    apriori = rng.uniform(2.0e-3, 9.0e-3, n) if apriori is None else apriori
    return apriori, np.hypot(floor, factor * apriori) * rng.normal(size=len(apriori))


def test_floor_and_scale_are_recovered_from_residuals():
    apriori, residuals = _residuals(0.010, 2.0)
    variances, diagnostics = _iterate(apriori, residuals)

    assert variances.floor_sigmas["A"] == pytest.approx(0.010, rel=0.03)
    assert variances.sigma_factors["A"] == pytest.approx(2.0, rel=0.03)
    assert diagnostics["activeBound"] is None and diagnostics["updateStatus"] == "UPDATED"


def test_absent_floor_stays_on_its_bound_and_matches_the_scaled_estimate():
    apriori, residuals = _residuals(0.0, 3.0, seed=2)
    floor_model, diagnostics = _iterate(apriori, residuals)
    scaled, _ = _iterate(apriori, residuals, model="scaled", count=1)

    assert floor_model.floor_sigmas["A"] < 1.0e-3
    if diagnostics["activeBound"] == "floor":
        assert floor_model.sigma_factors["A"] == pytest.approx(scaled.sigma_factors["A"], rel=1.0e-9)


def test_constant_noise_drives_the_scale_to_its_bound():
    apriori, residuals = _residuals(0.015, 0.0, seed=3)
    variances, diagnostics = _iterate(apriori, residuals)

    assert diagnostics["activeBound"] == "scale"
    assert variances.sigma_factors["A"] == 0.0
    assert variances.floor_sigmas["A"] == pytest.approx(0.015, rel=0.01)


def test_inseparable_components_fall_back_to_the_scale():
    apriori, residuals = _residuals(0.0, 2.0, seed=4, apriori=np.full(50_000, 5.0e-3))
    variances, diagnostics = _iterate(apriori, residuals)

    assert diagnostics["activeBound"] == "floor"
    assert variances.floor_sigmas["A"] == 0.0
    assert variances.sigma_factors["A"] == pytest.approx(2.0, rel=0.02)


def test_zero_floors_give_exactly_the_scaled_sigmas():
    apriori = np.array([1.0e-3, 3.3e-3, 7.1e-2])
    ids = np.full(3, "A", dtype=object)
    sigmas = ObservationVariances({"A": 1.7}, {"A": 0.0}).sigmas(ids, apriori)

    assert np.array_equal(sigmas, 1.7 * apriori)


def test_scaled_model_rejects_floors():
    estimator = VarianceComponentEstimator((COMPONENT,), "scaled")
    with pytest.raises(ValueError, match="no floor"):
        estimator.estimate(
            apriori_sigmas=np.ones(5),
            residuals=np.ones(5),
            standardization=np.ones(5),
            component_ids=np.full(5, "A", dtype=object),
            weight_factors=np.ones(5),
            variances=ObservationVariances({"A": 1.0}, {"A": 0.01}),
        )


def _problem(floor, factor, nights=150, per_night=20, seed=11):
    rng = np.random.default_rng(seed)
    n = nights * per_night
    jd = START_JD + np.repeat(np.arange(nights), per_night) + np.tile(np.arange(per_night), nights) / 144.0
    phase = 2.0 * np.pi * (jd - START_JD) / 27.3
    rows = np.column_stack([np.ones(n), np.cos(phase), np.sin(phase), (jd - jd.mean()) / 100.0])
    truth = rng.normal(size=4) * 0.05
    apriori = rng.uniform(2.0e-3, 9.0e-3, n)
    values = rows @ truth + np.hypot(floor, factor * apriori) * rng.normal(size=n)
    equations = EquationBatch.concatenate(
        [
            EquationBatch.single(
                oc=float(value),
                sigma=float(sigma),
                observation_id=index,
                station="STA_A",
                reflector="REF",
                epoch=Epoch(float(day), 0.0, TimeScale.UTC),
                receive_epoch=Epoch(float(day), 0.0, TimeScale.UTC),
                wavelength_nm=532.0,
            )
            for index, (value, sigma, day) in enumerate(zip(values, apriori, jd, strict=True))
        ]
    )
    return equations, rows, truth


def _solve(equations, rows, model, **kwargs):
    settings = LlrAdjustmentSettings(
        residual=ResidualScreening(maximumAbsoluteM=None, maximumAbsoluteByStationM=None),
        variance_components=VarianceComponentSettings((COMPONENT,)),
        adjustment=EstimateControl(
            maxIterationCount=3,
            convergenceThreshold=1.0e-6,
            computeResiduals=True,
            estimateVarianceFactors=True,
            estimateRobustWeights=True,
            varianceModel=model,
        ),
        accuracy_screening=AccuracyScreeningSettings(minimumOneWayM=1.0e-3, minimumFractionOfGroupMedian=0.1),
        robust_weights=RobustWeightSettings(model="igg3", k0=1.5, k1=6.0),
    )
    block = Linear(rows)
    parametrization = ParametrizationList([block])
    domain = screen_observations(
        equations,
        parametrization,
        residual=settings.residual,
        reported_sigma=settings.accuracy_screening,
        variance_components=settings.variance_components,
    )
    solver = LlrAdjustmentSolver(
        equation_source=lambda _: equations,
        parametrization=parametrization,
        settings=settings,
        observation_domain=domain,
        **kwargs,
    )
    return solver.run(), block


def test_adjustment_with_robust_weights_recovers_floor_and_scale():
    equations, rows, _ = _problem(0.010, 2.0)
    result, _ = _solve(equations, rows, "floorAndScaled")

    assert result.converged
    assert result.floor_sigmas["A"] == pytest.approx(0.010, rel=0.1)
    assert result.sigma_factors["A"] == pytest.approx(2.0, rel=0.1)
    component = result.variance_components[0]
    assert component["finalFloorSigmaM"] == result.floor_sigmas["A"] and component["varianceModel"] == "floorAndScaled"
    row = result.observations[0]
    assert row["sigma0M"] == pytest.approx(np.hypot(row["floorSigmaM"], row["sigmaFactor"] * row["aprioriSigmaM"]))
    # Under the true model the standardized residuals of the full-weight points are not ordered by reported sigma.
    sigma = np.array([r["aprioriSigmaM"] for r in result.observations])
    standardized = np.abs([r["standardizedResidual"] for r in result.observations])
    low, high = sigma < np.quantile(sigma, 0.2), sigma > np.quantile(sigma, 0.8)
    assert np.median(standardized[low]) == pytest.approx(np.median(standardized[high]), rel=0.15)


def test_scaled_estimate_drops_floors_of_an_earlier_estimate():
    equations, rows, _ = _problem(0.010, 2.0, nights=40)
    result, _ = _solve(equations, rows, "scaled", initial_sigma_factors={"A": 2.0}, initial_floor_sigmas={"A": 0.01})

    assert result.floor_sigmas == {"A": 0.0}
    assert result.variance_components[0]["initialFloorSigmaM"] == 0.0


def _noise_excess(sessions, factors, sigma, tau, kappa2):
    """``(kappa_2 P_n - F_n) / (F_n + 1/tau^2)^2`` of every session alone in its block."""
    precision = np.where(factors > 0.0, 1.0 / sigma**2, 0.0)
    weighted = sessions.sums(factors * precision)
    return (sessions.sums(kappa2 * precision) - weighted) / (weighted + 1.0 / tau**2) ** 2


@pytest.mark.parametrize("model", ["scaled", "floorAndScaled"])
def test_session_step_is_the_profile_of_the_dense_fisher_scoring_step(model):
    """The closed-form step equals the profile step built from dense ``tr(W Q_j W Q_l) / 2`` session blocks."""
    rng = np.random.default_rng(21)
    sizes = rng.integers(1, 6, 40)
    n = int(sizes.sum())
    jd = START_JD + np.repeat(np.arange(len(sizes)), sizes) + np.concatenate([np.arange(k) for k in sizes]) / 144.0
    ids = np.full(n, "A", dtype=object)
    sessions = night_sessions(ids, jd, 4.0)
    apriori = rng.uniform(2.0e-3, 9.0e-3, n)
    residuals, rho = rng.normal(size=n) * 0.01, rng.uniform(0.5, 1.0, n)
    factors = np.where(rng.random(n) < 0.1, 0.0, rng.uniform(0.3, 1.0, n))
    offsets, redundancy = rng.normal(size=sessions.count) * 0.01, rng.uniform(0.1, 0.9, sessions.count)
    tau = 0.012
    variances = ObservationVariances({"A": 1.5}, {"A": 0.008 if model == "floorAndScaled" else 0.0}, {"A": tau})
    updated, diagnostics = session_variance_step(
        model=model,
        sessions=sessions,
        component_ids=ids,
        apriori_sigmas=apriori,
        residuals=residuals,
        standardization=rho,
        weight_factors=factors,
        offsets=offsets,
        offset_redundancy=redundancy,
        variances=variances,
        consistency=0.8,
        offset_noise_excess=_noise_excess(sessions, factors, variances.sigmas(ids, apriori), tau, 0.7),
    )

    sigma2 = variances.sigmas(ids, apriori) ** 2
    active = factors > 0.0
    precision = np.where(active, 1.0 / sigma2, 0.0)
    cofactors = [np.diag(apriori**2)] + ([np.eye(n)] if model == "floorAndScaled" else [])
    membership = np.eye(sessions.count)[sessions.index]
    inverse = np.diag(precision) - (precision[:, None] * membership) @ np.diag(
        1.0 / (membership.T @ precision + 1.0 / tau**2)
    ) @ (membership.T * precision[None, :])
    matrices = [*cofactors, membership @ membership.T]
    fisher = np.array([[0.5 * np.trace(inverse @ a @ inverse @ b) for b in matrices] for a in matrices])
    terms = np.where(active, (factors * residuals**2 / 0.8 - rho * sigma2) / sigma2**2, 0.0)
    # Expected squared offset: tau^2 r with the robust weights as inverse variances, noise part corrected to kappa_2.
    weighted = membership.T @ (factors * precision)
    noise = (membership.T @ (0.7 * precision) - weighted) / (weighted + 1.0 / tau**2) ** 2
    score = np.r_[
        [0.5 * np.sum(np.diag(q) * terms) for q in cofactors],
        0.5 * np.sum(offsets**2 - tau**2 * redundancy - noise) / tau**4,
    ]
    step = np.linalg.lstsq(fisher, score, rcond=None)[0][-1]
    expected = np.clip(tau**2 + step, 0.1 * tau**2, 100.0 * tau**2)

    assert updated.session_sigmas["A"] ** 2 == pytest.approx(expected, rel=1.0e-9)
    assert updated.sigma_factors == variances.sigma_factors and updated.floor_sigmas == variances.floor_sigmas
    assert diagnostics["A"]["sessionCount"] == sessions.count


def test_session_sigmas_belong_to_the_variance_parameters():
    """Session sigmas start at the default, take known warm values and survive the white updates."""
    variances = ObservationVariances.initial(["A", "B"], session_sigma=0.005, session_sigmas={"B": 0.02, "C": 1.0})
    assert variances.session_sigmas == {"A": 0.005, "B": 0.02}
    assert ObservationVariances.initial(["A"], session_sigmas={"A": 0.02}).session_sigmas == {}
    assert variances.without_floors().session_sigmas == variances.session_sigmas

    apriori, residuals = _residuals(0.0, 2.0, n=1000)
    estimate = VarianceComponentEstimator((COMPONENT,), "scaled").estimate(
        apriori_sigmas=apriori,
        residuals=residuals,
        standardization=np.ones(len(apriori)),
        component_ids=np.full(len(apriori), "A", dtype=object),
        weight_factors=np.ones(len(apriori)),
        variances=ObservationVariances({"A": 1.0}, {"A": 0.0}, {"A": 0.007}),
    )
    assert estimate.variances.session_sigmas == {"A": 0.007}


def test_a_warm_session_sigma_is_carried_into_the_next_estimate():
    """The session sigma of one estimate is reported and, as a warm start, needs no further passes to settle."""
    equations, rows, _ = _session_problem(nights=60, per_night=5, sigma=1.0, session_sigma=1.5, seed=31)
    cold, _ = _run(equations, rows, robust=False, sessions=True)
    tau = cold.session_sigmas["A"]
    assert tau == pytest.approx(cold.variance_components[0]["sessionSigmaM"])
    assert cold.to_dict()["sessionSigmasM"] == {"A": tau}

    warm, _ = _run(
        equations,
        rows,
        robust=False,
        sessions=True,
        initial_sigma_factors=dict(cold.sigma_factors),
        initial_session_sigmas={"A": tau},
    )
    assert warm.session_sigmas["A"] == pytest.approx(tau, rel=1.0e-3)
    assert (
        warm.adjustment_iterations[0]["sigmaWeightIterations"] < cold.adjustment_iterations[0]["sigmaWeightIterations"]
    )


def test_session_model_converges_to_the_dense_reml_solution():
    """Without robust weights the converged white and session variances solve the exact REML equations."""
    equations, rows, _ = _session_problem(nights=60, per_night=5, sigma=1.0, session_sigma=1.5, seed=31)
    result, _ = _run(equations, rows, robust=False, sessions=True)
    s, tau = result.sigma_factors["A"], result.variance_components[0]["sessionSigmaM"]

    n = len(rows)
    night = np.repeat(np.arange(60), 5)
    membership = np.eye(60)[night]
    covariance = s**2 * np.eye(n) + tau**2 * membership @ membership.T
    inverse = np.linalg.inv(covariance)
    projector = inverse - inverse @ rows @ np.linalg.solve(rows.T @ inverse @ rows, rows.T @ inverse)
    values = np.asarray(equations.oc, dtype=float)
    for cofactor in (np.eye(n), membership @ membership.T):
        quadratic = values @ projector @ cofactor @ projector @ values
        assert quadratic == pytest.approx(np.trace(projector @ cofactor), rel=1.0e-3)
    assert len(result.sigma_weight_iterations) <= 30
