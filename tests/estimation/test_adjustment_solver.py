"""Iterative LLR adjustment: screening domain, variance factors, robust weights and convergence.

A one-parameter offset model makes every estimate a weighted mean, so results are known analytically.
"""

from typing import Any

import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.base.station_interval import StationInterval
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.base import Parametrization, ParametrizationList
from lunarops.classes.time import Epoch, TimeScale
from lunarops.estimation.adjustment_plan import (
    AccuracyScreeningSettings,
    EstimateControl,
    LlrAdjustmentSettings,
    ResidualScreening,
    RobustWeightSettings,
    VarianceComponentSettings,
)
from lunarops.estimation.adjustment_preprocessing import reject_implausible_apriori_accuracies, screen_observations
from lunarops.estimation.adjustment_solver import SIGMA_WEIGHT_ITERATION_COUNT, LlrAdjustmentSolver
from lunarops.estimation.stochastic_model import ObservationVariances, VarianceComponentEstimator
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition
from tests.tolerances import ESTIMATE_REL, assert_close

EPOCH = Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC)
COMPONENT = VarianceComponentDefinition("A", StationInterval("STA_A", StationInterval.parse("2020-01-01"), None))


def _equations(values, sigmas=None, identities=None) -> EquationBatch:
    sigmas = np.ones(len(values)) if sigmas is None else sigmas
    identities = range(len(values)) if identities is None else identities
    return EquationBatch.concatenate([
        EquationBatch.single(oc=float(value), sigma=float(sigma), reflector_position_pa=np.array([1.0, 0.0, 0.0]),
                             observation_id=identity, station="STA_A", reflector="REF", epoch=EPOCH,
                             wavelength_nm=532.0, receive_epoch=EPOCH)
        for value, sigma, identity in zip(values, sigmas, identities, strict=True)
    ])


class Offset(Parametrization):
    """O - C = value - offset for every observation."""

    TYPE = "Offset"

    def __init__(self):
        self.value = 0.0
        self.names = [ParameterName("test", "position.x")]

    def design(self, batch, out):
        out[:, 0] = 1.0

    def reduce(self, batch):
        return np.full(len(batch), self.value)

    def update(self, delta):
        self.value += float(delta[0])


def _settings(*, iterations=1, threshold=0.0, thresholds=None, residuals=True, variance=True, robust=True,
              accuracy=None, maximum_m=None, scheme="legacy"):
    """Settings of the one-parameter tests; they document the ``legacy`` scheme unless ``scheme`` says otherwise."""
    return LlrAdjustmentSettings(
        residual=ResidualScreening(maximumAbsoluteM=maximum_m, maximumAbsoluteByStationM=None),
        variance_components=VarianceComponentSettings((COMPONENT,)),
        adjustment=EstimateControl(maxIterationCount=iterations, convergenceThreshold=threshold,
                                   convergenceThresholdByParametrizations=thresholds, computeResiduals=residuals,
                                   estimateVarianceFactors=variance, estimateRobustWeights=robust,
                                   varianceEstimation=scheme),
        accuracy_screening=accuracy or AccuracyScreeningSettings(minimumOneWayM=1.0e-3, minimumFractionOfGroupMedian=0.1),
        robust_weights=RobustWeightSettings(model="directRejection", k0=3.0, k1=None),
    )


def _solve(equations, settings, *, offset=None, source=None, domain=None, **kwargs) -> tuple[Any, Offset]:
    offset = Offset() if offset is None else offset
    parametrization = ParametrizationList([offset])
    if domain is None:
        domain = screen_observations(equations, parametrization, residual=settings.residual,
                                     reported_sigma=settings.accuracy_screening,
                                     variance_components=settings.variance_components)
    solver = LlrAdjustmentSolver(equation_source=source or (lambda _: equations), parametrization=parametrization,
                                 settings=settings, observation_domain=domain, **kwargs)
    return solver, offset


def test_full_update_converges_to_the_weighted_mean_in_one_iteration():
    solver, offset = _solve(_equations(np.full(8, 2.0)), _settings(iterations=5, threshold=10.0))
    result = solver.run()

    assert offset.value == pytest.approx(2.0)
    assert result.converged and len(result.adjustment_iterations) == 1


def test_per_block_threshold_overrides_the_default():
    solver, _ = _solve(_equations(np.full(8, 2.0)),
                       _settings(threshold=10.0, thresholds={"Offset": 1.0}))
    result = solver.run(finalize=False)

    assert not result.converged
    assert result.adjustment_iterations[0]["convergenceThresholdByParametrizationM"] == {"Offset": 1.0}


def test_each_outer_iteration_runs_the_fixed_number_of_sigma_weight_updates():
    solver, _ = _solve(_equations(np.arange(8) % 2), _settings(iterations=2))
    result = solver.run()

    outer = [record["adjustmentIteration"] for record in result.sigma_weight_iterations]
    assert len(outer) == 2 * SIGMA_WEIGHT_ITERATION_COUNT
    assert outer == sorted(outer) and [outer.count(value) for value in dict.fromkeys(outer)] == [
        SIGMA_WEIGHT_ITERATION_COUNT] * 2
    assert any(item["purpose"] == "final-state-report" for item in result.equation_evaluations)


def test_an_intermediate_estimate_skips_the_final_state_report():
    solver, _ = _solve(_equations(np.arange(8) % 2), _settings(iterations=2))
    assert all(item["purpose"] != "final-state-report" for item in solver.run(finalize=False).equation_evaluations)


def test_inner_updates_reuse_the_frozen_residuals_and_redundancies_of_their_outer_solve():
    solver, _ = _solve(_equations(np.arange(8) % 3), _settings())
    records = solver.run().sigma_weight_iterations
    inner = [r for r in records if r["adjustmentIteration"] == records[0]["adjustmentIteration"]]

    assert len(inner) == SIGMA_WEIGHT_ITERATION_COUNT
    assert len({r["totalFrozenRedundancy"] for r in inner}) == 1
    assert len({r["candidateWrmsM"] for r in inner}) == 1
    assert inner[0]["totalFrozenRedundancy"] == pytest.approx(inner[0]["expectedTotalRedundancy"])


def test_robust_weights_enter_only_the_next_outer_solve():
    # Seven consistent observations and one 10-sigma outlier; sigma factors held at one.
    values = np.r_[np.linspace(0.9, 1.1, 7), 11.0]
    one, offset_one = _solve(_equations(values), _settings(iterations=1, variance=False))
    two, offset_two = _solve(_equations(values), _settings(iterations=2, variance=False))
    one.run(finalize=False)
    result = two.run(finalize=False)

    # The first solve still uses unit weights; the outlier rejected after it is absent from the second.
    assert_close(offset_one.value, values.mean(), ESTIMATE_REL)
    assert_close(offset_two.value, values[:7].mean(), ESTIMATE_REL)
    assert result.weight_factors[7] == 0.0


@pytest.mark.parametrize(("sigma", "rejected"), [(0.2, True), (0.12, False)])
def test_observations_with_redundancy_below_one_tenth_are_never_rejected(sigma, rejected):
    """One precise observation 2.5 m from seven unit-sigma ones.

    Its redundancy is 1 - h with h = w/(w + 7): 0.22 for sigma 0.2 and 0.09 for sigma 0.12. Its
    standardized residual (5.9 and 6.3) exceeds k0 = 3 in both cases; the others stay below 2.3.
    """
    values = np.r_[np.zeros(7), 2.5]
    solver, _ = _solve(_equations(values, np.r_[np.ones(7), sigma]), _settings(iterations=2, variance=False))
    result = solver.run(finalize=False)

    assert (result.weight_factors[7] == 0.0) is rejected
    assert all(result.weight_factors[index] == 1.0 for index in range(7))


def test_sigma_factors_start_at_one():
    solver, _ = _solve(_equations(100.0 + np.arange(8)), _settings())
    assert solver.run().variance_components[0]["initialSigmaFactor"] == pytest.approx(1.0)


def _variances(factor):
    return ObservationVariances({"A": factor}, {"A": 0.0})


def test_sigma_factor_is_the_frozen_group_rms_without_ratio_bounds():
    estimate = VarianceComponentEstimator((COMPONENT,)).estimate(
        apriori_sigmas=np.ones(5), residuals=np.full(5, 10.0), standardization=np.ones(5),
        component_ids=np.full(5, "A", dtype=object), weight_factors=np.ones(5), variances=_variances(1.0))
    # sqrt(sum w v^2 / sigma^2 / sum r) = 10.
    assert estimate.variances.sigma_factors["A"] == pytest.approx(10.0)
    assert estimate.diagnostics["A"]["varianceRatio"] == pytest.approx(100.0)


def test_sigma_factor_is_held_without_enough_redundancy_and_counts_every_positive_weight():
    estimator = VarianceComponentEstimator((COMPONENT,))
    held = estimator.estimate(apriori_sigmas=np.ones(3), residuals=np.full(3, 10.0), standardization=np.ones(3),
                              component_ids=np.full(3, "A", dtype=object), weight_factors=np.ones(3),
                              variances=_variances(2.0))
    active = estimator.estimate(apriori_sigmas=np.ones(5), residuals=np.ones(5), standardization=np.ones(5),
                                component_ids=np.full(5, "A", dtype=object),
                                weight_factors=np.array([1.0, 1.0, 1.0, 1.0, 1.0e-15]), variances=_variances(1.0))

    assert held.variances.sigma_factors == {"A": 2.0}
    assert held.diagnostics["A"]["updateStatus"] == "INSUFFICIENT_REDUNDANCY"
    assert active.diagnostics["A"]["activeCount"] == 5


def test_screening_rejects_gross_residuals_and_implausibly_small_sigmas_permanently():
    equations = _equations([10.0, 0.0, 0.0], [1.0, 0.01, 1.0], ["gross", "tiny-sigma", "retained"])
    domain = screen_observations(equations, ParametrizationList([Offset()]),
                                 residual=ResidualScreening(maximumAbsoluteM=5.0, maximumAbsoluteByStationM=None),
                                 reported_sigma=AccuracyScreeningSettings(minimumOneWayM=0.05,
                                                                          minimumFractionOfGroupMedian=0.1),
                                 variance_components=VarianceComponentSettings((COMPONENT,)))

    assert domain.gross_rejected == {"gross": 10.0}
    assert domain.accuracy_records["tiny-sigma"]["status"] == "REJECTED"
    assert domain.retained_keys == {"retained"} and domain.assignments == {"retained": "A"}


def test_accuracy_screening_rejects_instead_of_flooring():
    retained, records, groups = reject_implausible_apriori_accuracies(
        _equations([0.0, 0.0, 0.0], [0.01, 1.0, 1.0], ["tiny", "a", "b"]), {"tiny": "A", "a": "A", "b": "A"},
        minimumOneWayM=0.05, minimum_group_median_fraction=0.1)

    assert retained.observation_id.tolist() == ["a", "b"]
    assert records["tiny"]["status"] == "REJECTED"
    assert records["a"]["reportedSigmaM"] == pytest.approx(1.0)
    assert groups["A"]["rejectedCount"] == 1


def test_a_reused_domain_is_not_screened_again():
    equations = _equations(np.arange(8) % 2)
    first, _ = _solve(equations, _settings(maximum_m=5.0))
    domain = first.run(finalize=False).observation_domain
    # Observation 0 is now a gross outlier but belongs to the permanent domain of the first estimate.
    changed = _equations(np.r_[100.0, np.arange(1, 8) % 2])
    second, _ = _solve(changed, _settings(maximum_m=5.0, robust=False), domain=domain)
    result = second.run(finalize=False)

    assert result.summary["retainedObservationCount"] == 8
    assert result.equation_evaluations[0]["fixedDomainReturnedCount"] == 8


def test_accuracy_rejections_persist_across_relinearizations():
    equations = _equations(np.r_[0.0, np.arange(8)], np.r_[0.001, np.ones(8)], ["tiny", *range(8)])
    solver, _ = _solve(equations, _settings(iterations=2, accuracy=AccuracyScreeningSettings(
        minimumOneWayM=0.01, minimumFractionOfGroupMedian=0.1)))
    result = solver.run()

    assert "tiny" not in result.weight_factors
    assert result.accuracy_screening["rejectedCount"] == 1
    assert all(item["fixedDomainReturnedCount"] == 8 for item in result.equation_evaluations[1:])


def test_weights_survive_a_temporary_light_time_failure():
    equations = _equations(np.arange(8) % 2)
    first, _ = _solve(equations, _settings())
    previous = first.run(finalize=False)
    warm = {**previous.weight_factors, 0: 0.25}
    calls = []

    def source(_):
        calls.append(None)
        return equations[np.arange(8) > 0] if len(calls) == 1 else equations

    second, _ = _solve(equations, _settings(iterations=2, residuals=False, variance=False, robust=False),
                       source=source, domain=previous.observation_domain,
                       initial_sigma_factors=previous.sigma_factors, initial_weight_factors=warm)
    result = second.run(finalize=False)

    assert [item["fixedDomainReturnedCount"] for item in result.equation_evaluations[:2]] == [7, 8]
    assert result.weight_factors[0] == pytest.approx(0.25)
