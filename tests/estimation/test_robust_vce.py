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
from lunarops.estimation.adjustment_solver import LlrAdjustmentSolver, residual_variance_factors
from lunarops.estimation.random_effects import RandomEffectElimination, RandomEffectLayout
from lunarops.estimation.robust_vce import (
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


def _run(equations, rows, *, scheme="consistent", robust=True, sessions=False, model="igg3", **solver_options):
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
                                 settings=settings, observation_domain=domain, **solver_options)
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


@pytest.mark.parametrize("station_nights", [False, True])
def test_random_effect_elimination_equals_explicit_effect_parameters(station_nights):
    """Eliminated random effects equal explicit effect parameters with pseudo-observations ``0 = u``.

    Two components share one station; component B observes only the first three of six nights, so station-night
    blocks carry the three Earth-orientation effects and one or two session offsets (one padded slot).
    """
    rng = np.random.default_rng(5)
    n, m = 36, 3
    design = rng.normal(size=(n, m))
    observations = rng.normal(size=n)
    weights = rng.uniform(0.5, 2.0, n)
    jd = START_JD + np.repeat(np.arange(6), 6) + np.tile(np.arange(6), 6) / 144.0
    component = np.where((np.arange(n) % 2 == 1) & (np.arange(n) < 18), "B", "A").astype(object)
    sessions = Sessions.build(component, jd, 4.0)
    if station_nights:
        nights = Sessions.build(np.full(n, "STA", dtype=object), jd, 4.0)
        layout = RandomEffectLayout.station_nights(nights, rng.normal(size=(n, 3)), sessions)
        assert layout.size == 5 and np.count_nonzero(layout.design[:, 4]) == 9
    else:
        layout = RandomEffectLayout.sessions_only(sessions)
    precision = rng.uniform(0.5, 3.0, (layout.blocks.count, layout.size))
    elimination = RandomEffectElimination.build(layout, design, observations, weights, precision)
    normal, rhs, _ = elimination.reduce(design.T @ (weights[:, None] * design), design.T @ (weights * observations), 0.0)
    delta = np.linalg.solve(normal, rhs)
    cofactor = np.linalg.inv(normal)

    # Explicit system: parameters [x, u], observation rows plus pseudo-observations 0 = u with weight Lambda.
    effects = layout.blocks.count * layout.size
    columns = np.zeros((n, effects))
    for i, block in enumerate(layout.blocks.index):
        columns[i, block * layout.size : (block + 1) * layout.size] = layout.design[i]
    extended = np.block([[design, columns], [np.zeros((effects, m)), np.eye(effects)]])
    extended_weights = np.r_[weights, precision.reshape(-1)]
    extended_obs = np.r_[observations, np.zeros(effects)]
    full_normal = extended.T @ (extended_weights[:, None] * extended)
    solution = np.linalg.solve(full_normal, extended.T @ (extended_weights * extended_obs))
    full_cofactor = np.linalg.inv(full_normal)
    leverage = extended_weights * np.einsum("ij,jk,ik->i", extended, full_cofactor, extended)

    assert delta == pytest.approx(solution[:m], abs=1.0e-12)
    assert elimination.effects(delta).reshape(-1) == pytest.approx(solution[m:], abs=1.0e-12)
    assert elimination.leverage(design, weights, cofactor) == pytest.approx(leverage[:n], abs=1.0e-12)
    assert elimination.effect_redundancy(cofactor).reshape(-1) == pytest.approx(1.0 - leverage[n:], abs=1.0e-12)
    assert elimination.posterior_variance(cofactor).reshape(-1) == pytest.approx(
        np.diag(full_cofactor)[m:], abs=1.0e-12)
    # Total redundancy of the extended system is n - m.
    assert np.sum(1.0 - leverage) == pytest.approx(n - m)
    # Session offsets are picked in session order.
    offsets = elimination.session_values(elimination.effects(delta))
    assert len(offsets) == sessions.count
    assert np.array_equal(layout.session_slot >= (3 if station_nights else 0), np.ones(sessions.count, bool))


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


def test_standardized_residual_does_not_depend_on_the_observations_own_weight():
    """``v / sqrt(residual_variance_factors)`` is Baarda's w-statistic: the same for any own weight factor."""
    rng = np.random.default_rng(12)
    design = rng.normal(size=(12, 3))
    observations = rng.normal(size=12)
    weights = rng.uniform(0.5, 2.0, 12)
    statistics = []
    for own in (1.0, 0.6, 0.2, 0.0):
        factors = np.ones(12)
        factors[0] = own
        p = weights * factors
        normal = design.T @ (p[:, None] * design)
        residual = observations - design @ np.linalg.solve(normal, design.T @ (p * observations))
        ratio = weights[0] * design[0] @ np.linalg.solve(normal, design[0])
        statistics.append(residual[0] * np.sqrt(weights[0]) / np.sqrt(residual_variance_factors(np.array([ratio]),
                                                                                              np.array([own]))[0]))
    assert statistics == pytest.approx([statistics[0]] * 4, rel=1.0e-10)
    # Full weight: internally studentized (redundancy); zero weight: externally studentized (prediction variance).
    assert residual_variance_factors(np.array([0.3]), np.array([1.0]))[0] == pytest.approx(0.7)
    assert residual_variance_factors(np.array([0.3]), np.array([0.0]))[0] == pytest.approx(1.3)


def test_robust_weights_of_short_sessions_converge_without_oscillating():
    equations, rows, _ = _problem(nights=1500, per_night=2, sigma=1.0, session_sigma=3.0, seed=7)
    result, _ = _run(equations, rows, sessions=True)

    last = result.sigma_weight_iterations[-1]
    assert last["maximumWeightFactorChange"] <= 1.0e-4
    assert last["maximumSessionSigmaRelativeChange"] <= 1.0e-4
    assert len(result.sigma_weight_iterations) < 60


def test_session_sigma_of_short_sessions_survives_robust_weights():
    """With three points a night the robust weights would drive tau to zero without its consistency correction."""
    equations, rows, _ = _problem(nights=1000, per_night=3, sigma=1.0, session_sigma=0.5, seed=8)
    result, _ = _run(equations, rows, sessions=True)
    component = result.variance_components[0]

    assert component["sessionSigmaM"] == pytest.approx(0.5, rel=0.2)
    assert result.sigma_factors["A"] == pytest.approx(1.0, rel=0.07)
    assert component["sessionConsistencyFactor"] == pytest.approx(0.687, abs=1.0e-3)


def test_robust_session_sigma_stays_at_zero_without_session_errors():
    equations, rows, _ = _problem(nights=1000, per_night=3, sigma=1.0, session_sigma=0.0, seed=9)
    result, _ = _run(equations, rows, sessions=True)

    assert result.variance_components[0]["sessionSigmaM"] < 0.15
