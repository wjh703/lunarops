"""Consistent robust variance-component estimation and night-session correlation.

Synthetic linear problems with known noise: the consistent scheme must recover the true sigma (where the
legacy scheme is biased low by the consistency factor), stay unbiased with gross outliers, and recover the
sigma of night-session offsets, whose closed-form elimination must equal the explicit offset parameters.
"""

from math import erf, exp, pi, sqrt

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
from lunarops.estimation.adjustment_preprocessing import screen_observations
from lunarops.estimation.adjustment_solver import LlrAdjustmentSolver
from lunarops.estimation.robust_vce import (
    SessionElimination,
    Sessions,
    clustering_diagnostics,
    consistency_factor,
)
from lunarops.estimation.variance_component_groups import VarianceComponentDefinition

COMPONENT = VarianceComponentDefinition("A", StationInterval("STA_A", StationInterval.parse("2000-01-01"), None))
START_JD = 2_451_545.0


class Linear(Parametrization):
    """O - C = design_i . x for a fixed design row per observation identity."""

    TYPE = "Linear"

    def __init__(self, rows):
        self.rows = rows
        self.values = np.zeros(rows.shape[1])
        self.names = [ParameterName(f"p{index}", "position.x") for index in range(rows.shape[1])]

    def design(self, batch, out):
        out[:, :] = self.rows[batch.observation_id.astype(int)]

    def reduce(self, batch):
        return self.rows[batch.observation_id.astype(int)] @ self.values

    def update(self, delta):
        self.values += delta


def _problem(nights=150, per_night=20, parameters=4, sigma=1.0, session_sigma=0.0, outliers=0.0, seed=1):
    """Return equations and design rows: ``nights`` sessions of ``per_night`` points 10 minutes apart."""
    rng = np.random.default_rng(seed)
    n = nights * per_night
    night = np.repeat(np.arange(nights), per_night)
    jd = START_JD + night + np.tile(np.arange(per_night), nights) / 144.0
    phase = 2.0 * np.pi * (jd - START_JD) / 27.3
    rows = np.column_stack([np.ones(n), np.cos(phase), np.sin(phase), (jd - jd.mean()) / 100.0][:parameters])
    truth = rng.normal(size=parameters)
    noise = sigma * rng.normal(size=n) + session_sigma * rng.normal(size=nights)[night]
    bad = rng.random(n) < outliers
    noise[bad] += rng.choice([-1.0, 1.0], bad.sum()) * rng.uniform(10.0, 30.0, bad.sum()) * sigma
    values = rows @ truth + noise
    equations = EquationBatch.concatenate([
        EquationBatch.single(oc=float(value), sigma=1.0, observation_id=index, station="STA_A", reflector="REF",
                             epoch=Epoch(float(day), 0.0, TimeScale.UTC), receive_epoch=Epoch(float(day), 0.0, TimeScale.UTC),
                             wavelength_nm=532.0)
        for index, (value, day) in enumerate(zip(values, jd, strict=True))
    ])
    return equations, rows, truth


def _run(equations, rows, *, scheme="consistent", robust=True, sessions=False, model="igg3"):
    settings = LlrAdjustmentSettings(
        residual=ResidualScreening(maximumAbsoluteM=None, maximumAbsoluteByStationM=None),
        variance_components=VarianceComponentSettings((COMPONENT,)),
        adjustment=EstimateControl(maxIterationCount=3, convergenceThreshold=1.0e-4, computeResiduals=True,
                                   estimateVarianceFactors=True, estimateRobustWeights=robust,
                                   varianceEstimation=scheme, sessionCorrelation=sessions),
        accuracy_screening=AccuracyScreeningSettings(minimumOneWayM=1.0e-3, minimumFractionOfGroupMedian=0.1),
        robust_weights=RobustWeightSettings(model=model, k0=1.5 if model == "igg3" else 3.0,
                                            k1=6.0 if model == "igg3" else None),
    )
    block = Linear(rows)
    parametrization = ParametrizationList([block])
    domain = screen_observations(equations, parametrization, residual=settings.residual,
                                 reported_sigma=settings.accuracy_screening,
                                 variance_components=settings.variance_components)
    solver = LlrAdjustmentSolver(equation_source=lambda _: equations, parametrization=parametrization,
                                 settings=settings, observation_domain=domain)
    return solver.run(), block


def test_consistency_factor_of_direct_rejection_is_the_truncated_gaussian_variance():
    k = 3.0
    density = exp(-0.5 * k * k) / sqrt(2.0 * pi)
    assert consistency_factor("directRejection", k, None) == pytest.approx(1.0 - 2.0 * k * density / erf(k / sqrt(2.0)),
                                                                           rel=1.0e-7)
    assert consistency_factor("igg3", 1.5, 6.0) == pytest.approx(0.786, abs=1.0e-3)
    assert consistency_factor("igg3", 8.0, 9.0) == pytest.approx(1.0, abs=1.0e-9)


def test_consistent_scale_recovers_the_true_sigma_where_legacy_is_biased_low():
    equations, rows, _ = _problem(sigma=2.0)
    consistent, _ = _run(equations, rows)
    legacy, _ = _run(equations, rows, scheme="legacy")

    assert consistent.sigma_factors["A"] == pytest.approx(2.0, rel=0.03)
    # The legacy estimate is low by about sqrt(kappa) = 0.89.
    assert legacy.sigma_factors["A"] < 0.93 * consistent.sigma_factors["A"]


def test_consistent_scheme_is_robust_to_gross_outliers():
    equations, rows, truth = _problem(sigma=1.0, outliers=0.05, seed=2)
    result, block = _run(equations, rows)

    assert result.sigma_factors["A"] == pytest.approx(1.0, rel=0.05)
    assert np.max(np.abs(block.values - truth)) < 0.1
    assert result.converged


def test_consistent_passes_end_at_a_fixed_point_of_the_weights():
    equations, rows, _ = _problem(sigma=1.0, outliers=0.05, seed=3)
    result, _ = _run(equations, rows)

    last = result.sigma_weight_iterations[-1]
    assert last["maximumWeightFactorChange"] <= 1.0e-4
    assert last["maximumObservationSigmaRelativeChange"] <= 1.0e-4
    # The final re-solve proposes the weights that were used.
    proposed = np.array([row["finalStateProposedWeightFactor"] for row in result.observations])
    used = np.array([row["appliedWeightFactor"] for row in result.observations])
    assert np.max(np.abs(proposed - used)) < 1.0e-3


def test_session_sigma_is_recovered_and_the_white_sigma_is_not_inflated():
    equations, rows, _ = _problem(sigma=1.0, session_sigma=1.5, seed=4)
    plain, _ = _run(equations, rows, robust=False)
    sessions, _ = _run(equations, rows, robust=False, sessions=True)

    component = sessions.variance_components[0]
    assert component["sessionSigmaM"] == pytest.approx(1.5, rel=0.15)
    assert sessions.sigma_factors["A"] == pytest.approx(1.0, rel=0.05)
    # Without sessions the night offsets inflate the white sigma to about sqrt(1 + 1.5^2).
    assert plain.sigma_factors["A"] > 1.6
    assert component["sessionClusteringIndex"] is None or component["sessionClusteringIndex"] >= 0.0


def test_white_sigma_does_not_collapse_into_large_offsets_of_small_sessions():
    # Two to three points per night and offsets three times the white noise, with robust weights: the
    # standardization must follow tau, or the white sigma is biased low on every pass and collapses.
    equations, rows, _ = _problem(nights=1500, per_night=2, sigma=1.0, session_sigma=3.0, seed=7)
    result, _ = _run(equations, rows, sessions=True)

    assert result.sigma_factors["A"] == pytest.approx(1.0, rel=0.1)
    assert result.variance_components[0]["sessionSigmaM"] == pytest.approx(3.0, rel=0.1)


def test_session_elimination_equals_explicit_offset_parameters():
    rng = np.random.default_rng(5)
    n, m = 30, 3
    design = rng.normal(size=(n, m))
    observations = rng.normal(size=n)
    weights = rng.uniform(0.5, 2.0, n)
    component = np.full(n, "A", dtype=object)
    jd = START_JD + np.repeat(np.arange(5), 6) + np.tile(np.arange(6), 5) / 144.0
    sessions = Sessions.build(component, jd, 4.0)
    tau = np.full(sessions.count, 0.7)
    elimination = SessionElimination.build(sessions, design, observations, weights, tau)
    normal, rhs, _ = elimination.reduce(design.T @ (weights[:, None] * design), design.T @ (weights * observations), 0.0)
    delta = np.linalg.solve(normal, rhs)
    cofactor = np.linalg.inv(normal)

    # Explicit system: parameters [x, b], observations rows plus pseudo-observations 0 = b with weight 1/tau^2.
    indicator = np.eye(sessions.count)[sessions.index]
    extended = np.block([[design, indicator], [np.zeros((sessions.count, m)), np.eye(sessions.count)]])
    extended_weights = np.r_[weights, 1.0 / tau**2]
    extended_obs = np.r_[observations, np.zeros(sessions.count)]
    full_normal = extended.T @ (extended_weights[:, None] * extended)
    solution = np.linalg.solve(full_normal, extended.T @ (extended_weights * extended_obs))
    full_cofactor = np.linalg.inv(full_normal)
    leverage = extended_weights * np.einsum("ij,jk,ik->i", extended, full_cofactor, extended)

    assert delta == pytest.approx(solution[:m], abs=1.0e-12)
    assert elimination.offsets(delta) == pytest.approx(solution[m:], abs=1.0e-12)
    assert elimination.leverage(sessions, design, weights, cofactor) == pytest.approx(leverage[:n], abs=1.0e-12)
    assert elimination.offset_redundancy(cofactor) == pytest.approx(1.0 - leverage[n:], abs=1.0e-12)
    # Total redundancy of the extended system is n - m.
    assert np.sum(1.0 - leverage) == pytest.approx(n - m)


def test_sessions_split_on_gaps_and_components():
    component = np.array(["A", "A", "A", "B", "B"], dtype=object)
    jd = START_JD + np.array([0.0, 0.1, 0.5, 0.1, 0.11])
    sessions = Sessions.build(component, jd, 4.0)

    assert sessions.count == 3
    assert sessions.index[0] == sessions.index[1] != sessions.index[2]
    assert sessions.index[3] == sessions.index[4]


def test_clustering_index_separates_random_from_night_clustered_outliers():
    rng = np.random.default_rng(6)
    nights, per_night = 200, 20
    component = np.full(nights * per_night, "A", dtype=object)
    jd = START_JD + np.repeat(np.arange(nights), per_night) + np.tile(np.arange(per_night), nights) / 144.0
    sessions = Sessions.build(component, jd, 4.0)
    random = np.where(rng.random(len(jd)) < 0.1, 0.5, 1.0)
    clustered = np.where(np.repeat(rng.random(nights) < 0.1, per_night), 0.5, 1.0)

    def index(factors):
        return clustering_diagnostics(sessions, component, factors, ["A"], (0.13, 0.0))["A"]["sessionClusteringIndex"]

    assert index(random) == pytest.approx(1.0, abs=0.3)
    assert index(clustered) > 10.0
