"""Absolute normal equations N x = W, their strict Cholesky solution and directory format."""

import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName, parameter_unit
from lunarops.estimation.lsq.linearization import DenseLinearization
from lunarops.estimation.lsq.normal_equations import NormalEquations
from lunarops.estimation.lsq.solve import (
    NormalEquationInconsistencyError,
    NormalEquationSingularError,
    solve_normal_equations,
)
from lunarops.fileio.normal_equations import read_normal_equations, write_normal_equations
from lunarops.fileio.yaml_artifact import write_structured_text
from tests.tolerances import COVARIANCE_REL, ESTIMATE_REL, assert_close

X, Y = ParameterName("test", "x"), ParameterName("test", "y")


def _normals(N, W, *, lPl=0.0, obs_count=0, x0=None, names=(X, Y)):
    names = list(names)[: len(W)]
    return NormalEquations(
        names,
        np.asarray(N, float),
        np.asarray(W, float),
        lPl=lPl,
        obs_count=obs_count,
        meta={},
        parameter_units=[parameter_unit(name) for name in names],
        x0=np.zeros(len(W)) if x0 is None else np.asarray(x0, float),
    )


def _linearized(A, l, sigma, x0):
    P = np.diag(1.0 / np.asarray(sigma) ** 2)
    return NormalEquations.from_linearized_statistics(
        [X, Y][: A.shape[1]],
        A.T @ P @ A,
        A.T @ P @ l,
        float(l @ P @ l),
        obs_count=len(l),
        x0=np.asarray(x0, float),
        meta={},
        parameter_units=["m"] * A.shape[1],
    )


def test_weighted_least_squares_solution_covariance_and_sigma0():
    A = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, -1.0]])
    l = np.array([5.0, 11.0, 1.0])
    sigma = np.array([1.0, 2.0, 0.5])
    P = np.diag(1.0 / sigma**2)
    solved = solve_normal_equations(_linearized(A, l, sigma, [0.0, 0.0]))
    residual = l - A @ solved.values

    assert_close(solved.values, np.linalg.solve(A.T @ P @ A, A.T @ P @ l), ESTIMATE_REL)
    assert_close(solved.covariance, np.linalg.inv(A.T @ P @ A), COVARIANCE_REL)
    # sigma0^2 = v^T P v / (n - u).
    assert_close(solved.sigma0_post, np.sqrt(residual @ P @ residual / 1.0), COVARIANCE_REL)


def test_absolute_system_records_x0_and_returns_absolute_values():
    normals = NormalEquations.from_linearized_statistics(
        [X], np.array([[4.0]]), np.array([8.0]), 20.0, obs_count=3, x0=np.array([10.0]), meta={}, parameter_units=["m"]
    )
    solved = solve_normal_equations(normals)

    # W = N x0 + A^T P l = 40 + 8.
    assert_close(normals.W, [48.0], ESTIMATE_REL)
    assert_close(solved.values, [12.0], ESTIMATE_REL)
    assert_close(normals.correction_at_x0(solved.values), [2.0], ESTIMATE_REL)


def test_large_absolute_x0_keeps_the_correction_and_posterior_sigma0():
    reduced = np.array([1.0, 2.0, 3.0, 4.0])
    design = np.ones((4, 1))
    normals = NormalEquations.from_linearized_statistics(
        [X],
        design.T @ design,
        design.T @ reduced,
        float(reduced @ reduced),
        obs_count=4,
        x0=np.array([1.0e9]),
        meta={},
        parameter_units=["m"],
    )
    solved = solve_normal_equations(normals)

    assert solved.values[0] - 1.0e9 == pytest.approx(2.5)
    assert_close(solved.sigma0_post, np.sqrt(np.sum((reduced - 2.5) ** 2) / 3.0), COVARIANCE_REL)


def test_exactly_determined_system_has_no_posterior_variance_factor():
    solved = solve_normal_equations(_normals(np.eye(2), [1.0, 2.0], lPl=5.0, obs_count=2))

    assert_close(solved.values, [1.0, 2.0], ESTIMATE_REL)
    assert_close(solved.covariance, np.eye(2), COVARIANCE_REL)
    assert solved.sigma0_post is None


def test_singular_and_numerically_unobservable_systems_are_rejected_with_diagnostics():
    with pytest.raises(NormalEquationSingularError) as error:
        solve_normal_equations(_normals([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0], obs_count=2))
    assert "rank=1/2" in str(error.value) and "obs_count=2" in str(error.value)
    with pytest.raises(NormalEquationSingularError, match="rank=1/2"):
        solve_normal_equations(_normals(np.diag([1.0, 1.0e-20]), [0.0, 0.0], obs_count=3))


def test_rank_tolerance_is_relative_to_the_matrix_scale():
    solved = solve_normal_equations(_normals(np.diag([1.0e-30, 2.0e-30]), [3.0e-30, 8.0e-30], lPl=4.1e-29, obs_count=3))
    assert_close(solved.values, [3.0, 4.0], ESTIMATE_REL)


def test_negative_residual_quadratic_form_is_rejected():
    with pytest.raises(NormalEquationInconsistencyError, match="negative residual quadratic"):
        solve_normal_equations(_normals([[1.0]], [2.0], lPl=1.0, obs_count=2, names=(X,)))


def test_dense_linearization_forms_exactly_symmetric_normals_for_mixed_scales():
    rng = np.random.default_rng(3)
    design = np.column_stack([rng.normal(size=500), 1.0e-5 * rng.normal(size=500), 1.0e3 * rng.normal(size=500)])
    dense = DenseLinearization(
        equations=tuple(range(500)),
        parameter_names=tuple(ParameterName("p", f"position.{a}") for a in "xyz"),
        design=design,
        reduced_observations=rng.normal(size=500),
        apriori_sigmas=np.ones(500),
        identities=tuple(range(500)),
    )
    weights = rng.uniform(0.1, 10.0, size=500)
    normals = dense.normal_equations(weights, active=np.ones(500, bool), x0=np.zeros(3))

    # 0.5 * (N + N^T) is symmetric by IEEE commutativity, on every platform.
    np.testing.assert_array_equal(normals.N, normals.N.T)
    assert_close(normals.N, design.T @ (weights[:, None] * design), COVARIANCE_REL)


def test_directory_round_trip_keeps_the_absolute_system(tmp_path):
    normals = NormalEquations.from_linearized_statistics(
        [X], np.array([[4.0]]), np.array([8.0]), 20.0, obs_count=3, x0=np.array([10.0]), meta={}, parameter_units=["m"]
    )
    write_normal_equations(normals, tmp_path / "normals")
    loaded = read_normal_equations(tmp_path / "normals")

    # Written and read back, so bit for bit.
    for name in ("N", "W", "x0"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(normals, name))
    assert (loaded.lPl, loaded.obs_count) == (20.0, 3)
    assert (tmp_path / "normals" / "x0.dat.gz").is_file() and (tmp_path / "normals" / "metadata.txt").is_file()
    info = (tmp_path / "normals" / "info.txt").read_text()
    assert info.startswith("lunarops normalEquationInfo") and "parameterCount: 1" in info
    assert not list(tmp_path.glob("*.npz"))


def test_directory_format_rejects_extensions_and_foreign_metadata(tmp_path):
    normals = _normals([[0.0]], [0.0], names=(X,))
    with pytest.raises(ValueError, match="extensionless directory"):
        write_normal_equations(normals, tmp_path / "normals.npz")
    write_normal_equations(normals, tmp_path / "normals")
    write_structured_text(tmp_path / "normals" / "metadata.txt", "other", {})
    with pytest.raises(ValueError, match="Expected LunarOps 'normalEquationMetadata'"):
        read_normal_equations(tmp_path / "normals")


def test_parameter_units_follow_the_parameter_type():
    units = {
        kind: parameter_unit(ParameterName("A", kind))
        for kind in ("position.x", "velocity.x", "rangeBias", "libration.term.cos", "clock", "scale")
    }
    assert units == {
        "position.x": "m",
        "velocity.x": "m/yr",
        "rangeBias": "m",
        "libration.term.cos": "mas",
        "clock": "s",
        "scale": "1",
    }
