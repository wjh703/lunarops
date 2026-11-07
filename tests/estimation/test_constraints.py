"""Parameter constraints: a-priori sigmas and minimal (datum) constraints as pseudo-observations."""

from dataclasses import replace

import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.estimation.adjustment_plan import parse_adjustment_plan
from lunarops.estimation.constraints import (
    EARTH_RADIUS_M,
    AprioriSigmaConstraint,
    MinimalConstraint,
    ParameterConstraints,
)
from tests.estimation.test_adjustment_solver import Offset, _equations, _settings, _solve


class AbsoluteOffset(Offset):
    """``Offset`` whose current value is its linearization point, as for the real parametrizations."""

    def x0(self):
        return np.array([self.value])


def _network(count=6, seed=3):
    """Station names, a-priori ITRF positions on the Earth's surface and their position columns."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    positions = EARTH_RADIUS_M * directions / np.linalg.norm(directions, axis=1)[:, None]
    stations = [f"S{k}" for k in range(count)]
    names = [ParameterName(station, f"position.{axis}") for station in stations for axis in "xyz"]
    columns = {station: [3 * k, 3 * k + 1, 3 * k + 2] for k, station in enumerate(stations)}
    return stations, positions.reshape(-1), names, columns


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def test_minimal_constraint_rows_return_the_helmert_parameters_of_the_network():
    _, apriori, names, columns = _network()
    translation, rotation, scale = np.array([0.01, -0.02, 0.005]), np.array([3e-9, -1e-9, 2e-9]), 4e-9
    positions = apriori.reshape(-1, 3)
    moved = positions + translation + np.cross(rotation, positions) + scale * positions
    pseudo = ParameterConstraints(
        (MinimalConstraint(None, ("translation", "rotation", "scale"), 1e-4),)
    ).pseudo_observations(names, apriori, columns)

    helmert = pseudo.residuals(moved.reshape(-1))
    assert pseudo.labels[0] == "minimalConstraints:translation.x" and len(pseudo) == 7
    assert helmert[:3] == pytest.approx(translation, abs=1e-9)
    assert helmert[3:6] == pytest.approx(rotation * EARTH_RADIUS_M, abs=1e-9)
    assert helmert[6] == pytest.approx(scale * EARTH_RADIUS_M, abs=1e-9)
    # A subset of conditions constrains the same parameters of the full transformation, not of a reduced one.
    rotation_only = ParameterConstraints((MinimalConstraint(None, ("rotation",), 1e-4),)).pseudo_observations(
        names, apriori, columns
    )
    assert rotation_only.residuals(moved.reshape(-1)) == pytest.approx(rotation * EARTH_RADIUS_M, abs=1e-9)


def test_minimal_constraints_remove_the_datum_defect_without_distorting_the_network():
    """Distances between stations leave translation and rotation free; the constraints fix them at the a priori."""
    stations, apriori, names, columns = _network()
    positions = apriori.reshape(-1, 3)
    rng = np.random.default_rng(4)
    correction = rng.normal(size=positions.shape) * 0.05
    rows, observed = [], []
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            unit = (positions[i] - positions[j]) / np.linalg.norm(positions[i] - positions[j])
            row = np.zeros(apriori.size)
            row[3 * i : 3 * i + 3], row[3 * j : 3 * j + 3] = unit, -unit
            rows.append(row)
            observed.append(unit @ (correction[i] - correction[j]))
    design, observed = np.array(rows), np.array(observed)
    normal = design.T @ design
    assert np.linalg.matrix_rank(normal) == apriori.size - 6
    # The distances do not conflict with the constraints, so the solution does not depend on their sigma; a sigma
    # near the observations' keeps the test system well conditioned.
    pseudo = ParameterConstraints((MinimalConstraint(None, ("translation", "rotation"), 1e-2),)).pseudo_observations(
        names, apriori, columns
    )
    constrained, rhs, _ = pseudo.reduce(normal, design.T @ observed, 0.0, apriori)
    delta = np.linalg.solve(constrained, rhs)

    assert np.linalg.matrix_rank(constrained) == apriori.size
    assert design @ delta == pytest.approx(observed, abs=1e-9)
    assert pseudo.residuals(apriori + delta) == pytest.approx(np.zeros(6), abs=1e-8)


def test_apriori_sigma_is_a_weighted_pseudo_observation():
    values = np.array([1.0, 2.0, 3.0, 2.5])
    settings = _settings(iterations=3, threshold=1e-12, variance=False, robust=False)
    prior = ParameterConstraints((AprioriSigmaConstraint(("test:position.*",), 0.5),))
    solver, offset = _solve(_equations(values), replace(settings, constraints=prior), offset=AbsoluteOffset())
    result = solver.run()

    # Weighted mean of four unit-sigma observations and the a-priori value 0 with sigma 0.5.
    assert offset.value == pytest.approx(values.sum() / (4 + 4))
    assert result.summary["pseudoObservationCount"] == 1
    record = result.summary["constraints"][0]
    assert record["label"] == "aprioriSigma:test:position.x::" and record["value"] == pytest.approx(offset.value)
    # Degrees of freedom: 4 observations + 1 pseudo-observation - 1 parameter.
    residual_square = np.sum((values - offset.value) ** 2) + (offset.value / 0.5) ** 2
    assert result.sigma0_post == pytest.approx(np.sqrt(residual_square / 4))


def test_plan_parses_the_constraints():
    plan = parse_adjustment_plan(
        {
            "varianceComponents": [{"id": "A", "station": "STA_A", "start": "2000-01-01"}],
            "processingSteps": [],
            "constraints": [
                {"type": "aprioriSigma", "parameters": ["*:rangeBias*"], "sigma": 0.05},
                {"type": "minimalConstraints", "stations": None, "conditions": ["rotation"], "sigmaM": 1e-4},
            ],
        }
    )

    assert plan.constraints.items == (
        AprioriSigmaConstraint(("*:rangeBias*",), 0.05),
        MinimalConstraint(None, ("rotation",), 1e-4),
    )


def test_minimal_constraints_need_their_datum_stations_estimated():
    _, apriori, names, columns = _network(count=4)
    constraint = ParameterConstraints((MinimalConstraint(("S0", "S9"), ("rotation",), 1e-4),))
    with pytest.raises(ValueError, match="S9"):
        constraint.pseudo_observations(names, apriori, columns)
