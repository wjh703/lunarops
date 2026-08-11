from typing import Any

import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import Parametrization, ParametrizationList
from lunarops.classes.time import Epoch, TimeScale
from lunarops.estimation.adjustment_preprocessing import reject_implausible_apriori_accuracies
from lunarops.estimation.adjustment_settings import (
    AccuracyScreeningSettings,
    AdjustmentControlSettings,
    LlrAdjustmentSettings,
    RobustWeightSettings,
    VarianceComponentSettings,
)
from lunarops.estimation.adjustment_result_models import LlrAdjustmentStageResult
from lunarops.estimation.adjustment_solver import LlrAdjustmentSolver, SIGMA_WEIGHT_ITERATION_COUNT
from lunarops.estimation.sigma_factor_estimator import SigmaFactorEstimator
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition


def _equation(identity: Any, value: float, *, sigma: float = 1.0) -> ObservationEquation:
    return ObservationEquation(
        observed_minus_computed_one_way_m=value,
        sigma_one_way_m=sigma,
        design_partials={"test": np.array([1.0])},
        observation_id=identity,
        station_key="STA_A",
        reflector_key="REF",
        transmit_epoch_utc=Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC),
        wavelength_nm=532.0,
    )


class OffsetParametrization(Parametrization):
    def __init__(self):
        self.value = 0.0

    def parameter_names(self):
        return [ParameterName("test", "position.x")]

    def design_columns(self, equation):
        return np.array([1.0])

    def reduce_observation(self, equation):
        return self.value

    def apply_update(self, delta):
        self.value += float(delta[0])


def _component():
    return VarianceComponentDefinition("A", "STA_A", "2020-01-01", None)


def _settings(*, max_iterations=1, convergence_threshold=0.0, accuracy=None):
    return LlrAdjustmentSettings(
        variance_components=VarianceComponentSettings((_component(),)),
        adjustment=AdjustmentControlSettings(
            prefit_gross_threshold_m=None,
            max_iteration_count=max_iterations,
            convergence_threshold_m=convergence_threshold,
        ),
        accuracy_screening=accuracy or AccuracyScreeningSettings(),
        robust_weights=RobustWeightSettings(model="directRejection", k0=3.0),
    )


def test_accuracy_screening_rejects_instead_of_flooring():
    equations = [_equation("tiny", 0.0, sigma=0.01), _equation("a", 0.0), _equation("b", 0.0)]
    assignments = {item.observation_id: "A" for item in equations}
    retained, records, groups = reject_implausible_apriori_accuracies(
        equations,
        assignments,
        minimum_one_way_m=0.05,
        minimum_group_median_fraction=0.1,
    )
    assert [item.observation_id for item in retained] == ["a", "b"]
    assert records["tiny"]["status"] == "REJECTED"
    assert records["a"]["reported_sigma_m"] == pytest.approx(1.0)
    assert groups["A"]["rejected_count"] == 1


def test_sigma_factor_estimator_uses_frozen_group_redundancy_without_ratio_bounds():
    estimator = SigmaFactorEstimator((_component(),))
    estimate = estimator.estimate(
        apriori_sigmas=np.ones(5),
        residuals=np.full(5, 10.0),
        redundancies=np.ones(5),
        component_ids=np.full(5, "A", dtype=object),
        weight_factors=np.ones(5),
        sigma_factors={"A": 1.0},
    )
    assert estimate.sigma_factors["A"] == pytest.approx(10.0)
    assert estimate.diagnostics["A"]["variance_ratio"] == pytest.approx(100.0)


def test_sigma_factor_is_held_when_component_redundancy_is_not_above_three():
    estimator = SigmaFactorEstimator((_component(),))
    estimate = estimator.estimate(
        apriori_sigmas=np.ones(3),
        residuals=np.full(3, 10.0),
        redundancies=np.ones(3),
        component_ids=np.full(3, "A", dtype=object),
        weight_factors=np.ones(3),
        sigma_factors={"A": 2.0},
    )
    assert estimate.sigma_factors == {"A": 2.0}
    assert estimate.diagnostics["A"]["update_status"] == "INSUFFICIENT_REDUNDANCY"


def test_solver_runs_exactly_ten_sigma_weight_updates_per_outer_iteration(monkeypatch):
    equations = [_equation(index, float(index % 2)) for index in range(8)]
    solver = LlrAdjustmentSolver(
        equation_source=lambda _: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(max_iterations=2),
    )
    solve_calls = 0
    original = solver._solve_linearized

    def counted(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_solve_linearized", counted)
    result = solver.run()
    assert len(result.sigma_weight_iterations) == 2 * SIGMA_WEIGHT_ITERATION_COUNT
    # One parameter solve per outer iteration plus one final-state report solve.
    assert solve_calls == 3


def test_groops_convergence_applies_full_update_and_stops_immediately():
    equations = [_equation(index, 2.0) for index in range(8)]
    offset = OffsetParametrization()
    solver = LlrAdjustmentSolver(
        equation_source=lambda _: equations,
        parametrization=ParametrizationList([offset]),
        settings=_settings(max_iterations=5, convergence_threshold=10.0),
    )

    result = solver.run()

    assert offset.value == pytest.approx(2.0)
    assert result.converged
    assert len(result.adjustment_iterations) == 1


def test_intermediate_stage_skips_final_state_report_and_extra_solve(monkeypatch):
    equations = [_equation(index, float(index % 2)) for index in range(8)]
    solver = LlrAdjustmentSolver(
        equation_source=lambda _: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(max_iterations=2),
    )
    solve_calls = 0
    original = solver._solve_linearized

    def counted(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_solve_linearized", counted)
    result = solver.run(finalize=False)

    assert isinstance(result, LlrAdjustmentStageResult)
    assert solve_calls == 2
    assert all(item["purpose"] != "final-state-report" for item in result.equation_evaluations)


def test_inner_updates_reuse_the_same_frozen_residuals_and_redundancies(monkeypatch):
    equations = [_equation(index, float(index % 3)) for index in range(8)]
    solver = LlrAdjustmentSolver(
        equation_source=lambda _: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(),
    )
    seen = []
    original = solver._adjust_sigma0

    def recorded(residuals, redundancies, *args, **kwargs):
        seen.append((np.array(residuals, copy=True), np.array(redundancies, copy=True)))
        return original(residuals, redundancies, *args, **kwargs)

    monkeypatch.setattr(solver, "_adjust_sigma0", recorded)
    solver.run()
    inner = seen[:SIGMA_WEIGHT_ITERATION_COUNT]
    assert len(inner) == SIGMA_WEIGHT_ITERATION_COUNT
    assert all(np.array_equal(item[0], inner[0][0]) for item in inner)
    assert all(np.array_equal(item[1], inner[0][1]) for item in inner)


def test_new_weight_factors_first_enter_the_next_outer_solve(monkeypatch):
    equations = [_equation(index, float(index)) for index in range(8)]
    solver = LlrAdjustmentSolver(
        equation_source=lambda _: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(max_iterations=2),
    )

    class RejectFirst:
        @staticmethod
        def target_factors(standardized, keys):
            return {key: (0.0 if key == 0 else 1.0) for key in keys}

    solver.robust_weight_model = RejectFirst()
    used = []
    original = solver._solve_linearized

    def recorded(equations, sigma_factors, weight_factors):
        used.append(dict(weight_factors))
        return original(equations, sigma_factors, weight_factors)

    monkeypatch.setattr(solver, "_solve_linearized", recorded)
    solver.run()
    assert used[0][0] == 1.0
    assert used[1][0] == 0.0


def test_low_redundancy_observation_is_not_robustly_rejected():
    equations = [_equation(index, 0.0) for index in range(7)] + [_equation("outlier", 100.0)]
    solver = LlrAdjustmentSolver(
        equation_source=lambda _: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(),
    )
    solver.parametrization.setup(equations, None)
    solver._names = solver.parametrization.parameter_names()
    solver._assignments = {item.observation_id: "A" for item in equations}
    solver._prepare_linearization(equations)
    standardized, _ = solver._standardized_residuals(
        np.asarray([0.0] * 7 + [100.0]),
        np.asarray([1.0] * 7 + [0.1]),
        {"A": 1.0},
    )
    factors = solver.robust_weight_model.target_factors(standardized, [item.observation_id for item in equations])
    assert factors["outlier"] == 1.0


def test_all_sigma_factors_start_at_one_without_mad():
    equations = [_equation(index, 100.0 + index) for index in range(8)]
    solver = LlrAdjustmentSolver(
        equation_source=lambda _: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(),
    )
    result = solver.run()
    assert result.variance_components[0]["initial_sigma_factor"] == pytest.approx(1.0)


def test_accuracy_rejection_is_permanent_across_relinearizations():
    equations = [_equation("tiny", 0.0, sigma=0.001)] + [_equation(index, float(index)) for index in range(8)]
    solver = LlrAdjustmentSolver(
        equation_source=lambda _: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        settings=_settings(
            max_iterations=2,
            accuracy=AccuracyScreeningSettings(minimum_one_way_m=0.01),
        ),
    )
    result = solver.run()
    assert "tiny" not in result.weight_factors
    assert result.accuracy_screening["rejected_count"] == 1
    assert all(item["fixed_domain_returned_count"] == 8 for item in result.equation_evaluations[1:])
