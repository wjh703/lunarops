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
from lunarops.estimation.adjustment_plan import (
    AccuracyScreeningSettings,
    EstimateControl,
    LlrAdjustmentSettings,
    ResidualScreening,
    RobustWeightSettings,
    VarianceComponentSettings,
)
from lunarops.estimation.adjustment_preprocessing import screen_observations
from lunarops.estimation.adjustment_solver import LlrAdjustmentSolver
from lunarops.estimation.stochastic_model import ObservationVariances, VarianceComponentEstimator
from tests.estimation.test_robust_vce import COMPONENT, START_JD, Linear


def _iterate(apriori, residuals, model="floorAndScaled", start=(1.0, 0.0), count=200):
    """Iterate the estimator on fixed residuals (redundancy 1, full weights) to its fixed point."""
    estimator = VarianceComponentEstimator((COMPONENT,), model)
    variances = ObservationVariances({"A": start[0]}, {"A": start[1]})
    ids = np.full(len(apriori), "A", dtype=object)
    for _ in range(count):
        estimate = estimator.estimate(apriori_sigmas=apriori, residuals=residuals, standardization=np.ones(len(apriori)),
                                      component_ids=ids, weight_factors=np.ones(len(apriori)), variances=variances)
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
        estimator.estimate(apriori_sigmas=np.ones(5), residuals=np.ones(5), standardization=np.ones(5),
                           component_ids=np.full(5, "A", dtype=object), weight_factors=np.ones(5),
                           variances=ObservationVariances({"A": 1.0}, {"A": 0.01}))


def test_floor_model_requires_the_consistent_scheme():
    with pytest.raises(ValueError, match="floorAndScaled"):
        EstimateControl(maxIterationCount=1, convergenceThreshold=0.0, computeResiduals=True,
                        estimateVarianceFactors=True, estimateRobustWeights=True, varianceEstimation="legacy",
                        varianceModel="floorAndScaled")


def _problem(floor, factor, nights=150, per_night=20, seed=11):
    rng = np.random.default_rng(seed)
    n = nights * per_night
    jd = START_JD + np.repeat(np.arange(nights), per_night) + np.tile(np.arange(per_night), nights) / 144.0
    phase = 2.0 * np.pi * (jd - START_JD) / 27.3
    rows = np.column_stack([np.ones(n), np.cos(phase), np.sin(phase), (jd - jd.mean()) / 100.0])
    truth = rng.normal(size=4) * 0.05
    apriori = rng.uniform(2.0e-3, 9.0e-3, n)
    values = rows @ truth + np.hypot(floor, factor * apriori) * rng.normal(size=n)
    equations = EquationBatch.concatenate([
        EquationBatch.single(oc=float(value), sigma=float(sigma), observation_id=index, station="STA_A",
                             reflector="REF", epoch=Epoch(float(day), 0.0, TimeScale.UTC),
                             receive_epoch=Epoch(float(day), 0.0, TimeScale.UTC), wavelength_nm=532.0)
        for index, (value, sigma, day) in enumerate(zip(values, apriori, jd, strict=True))
    ])
    return equations, rows, truth


def _solve(equations, rows, model, **kwargs):
    settings = LlrAdjustmentSettings(
        residual=ResidualScreening(maximumAbsoluteM=None, maximumAbsoluteByStationM=None),
        variance_components=VarianceComponentSettings((COMPONENT,)),
        adjustment=EstimateControl(maxIterationCount=3, convergenceThreshold=1.0e-6, computeResiduals=True,
                                   estimateVarianceFactors=True, estimateRobustWeights=True, varianceModel=model),
        accuracy_screening=AccuracyScreeningSettings(minimumOneWayM=1.0e-3, minimumFractionOfGroupMedian=0.1),
        robust_weights=RobustWeightSettings(model="igg3", k0=1.5, k1=6.0),
    )
    block = Linear(rows)
    parametrization = ParametrizationList([block])
    domain = screen_observations(equations, parametrization, residual=settings.residual,
                                 reported_sigma=settings.accuracy_screening,
                                 variance_components=settings.variance_components)
    solver = LlrAdjustmentSolver(equation_source=lambda _: equations, parametrization=parametrization,
                                 settings=settings, observation_domain=domain, **kwargs)
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
