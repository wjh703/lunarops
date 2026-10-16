import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName
from lunarops.estimation.normal_equation_solver import (
    NormalEquationInconsistencyError,
    solve_normal_equations,
)
from lunarops.estimation.normal_equations import NormalEquations
from lunarops.fileio.normal_equations import read_normal_equations, write_normal_equations


def test_omitted_x0_is_materialized_as_a_read_only_zero_vector():
    names = [ParameterName("test", "x"), ParameterName("test", "y")]
    normals = NormalEquations(names, np.eye(2), np.ones(2))

    assert normals.x0 == pytest.approx(np.zeros(2))
    assert not normals.x0.flags.writeable


def test_normal_equations_use_W_and_np_solve_convention(tmp_path):
    names = [ParameterName("test", "x"), ParameterName("test", "y")]
    A = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, -1.0]])
    L = np.array([5.0, 11.0, 1.0])
    sigma = np.array([1.0, 2.0, 0.5])
    P = np.diag(1.0 / sigma**2)

    expected_N = A.T @ P @ A
    expected_W = A.T @ P @ L

    normals = NormalEquations.from_linearized_statistics(names, expected_N, expected_W, float(L @ P @ L), obs_count=len(L))

    assert np.allclose(normals.N, expected_N)
    assert np.allclose(normals.W, expected_W)

    solved = solve_normal_equations(normals)
    x = solved.values
    Qxx = solved.covariance
    sigma0 = solved.sigma0_post
    assert Qxx is not None
    assert np.allclose(x, np.linalg.solve(expected_N, expected_W))
    assert np.allclose(Qxx, np.linalg.solve(expected_N, np.eye(2)))
    assert sigma0 is not None
    assert sigma0 >= 0.0

    stem = tmp_path / "normals"
    write_normal_equations(normals, stem)
    loaded = read_normal_equations(stem)
    assert np.allclose(loaded.N, expected_N)
    assert np.allclose(loaded.W, expected_W)
    assert (stem / "info.txt").read_text().startswith("lunarops normalEquationInfo")
    assert not list(tmp_path.glob("*.npz"))


def test_persisted_normal_equations_are_absolute_and_preserve_x0(tmp_path):
    name = ParameterName("test", "x")
    normals = NormalEquations.from_linearized_statistics(
        parameter_names=[name],
        N=np.array([[4.0]]),
        correction_rhs=np.array([8.0]),
        correction_lPl=20.0,
        obs_count=3,
        x0=np.array([10.0]),
    )

    path = tmp_path / "normals"
    write_normal_equations(normals, path)
    loaded = read_normal_equations(path)

    assert loaded.x0 == pytest.approx([10.0])
    assert loaded.W == pytest.approx([48.0])
    assert loaded.lPl == pytest.approx(20.0)
    assert (path / "x0.dat.gz").is_file()
    info = (path / "info.txt").read_text()
    assert "rhsConvention" not in info
    assert "payload x0.dat.gz" in info
    assert (path / "metadata.txt").is_file()

    solved = solve_normal_equations(loaded)
    assert solved.values == pytest.approx([12.0])
    correction_rhs = loaded.W - loaded.N @ loaded.x0
    correction = solved.values - loaded.x0
    assert loaded.lPl - float(correction_rhs @ correction) == pytest.approx(4.0)


def test_normal_equation_metadata_checksum_is_verified(tmp_path):
    path = tmp_path / "normals"
    write_normal_equations(NormalEquations([ParameterName("test", "x")], np.zeros((1, 1)), np.zeros(1)), path)
    with (path / "metadata.txt").open("a", encoding="utf-8") as stream:
        stream.write("corrupt: true\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_normal_equations(path)




def test_exactly_determined_system_has_no_posterior_variance_factor():
    names = [ParameterName("test", "x"), ParameterName("test", "y")]
    normals = NormalEquations(names, np.eye(2), np.array([1.0, 2.0]), lPl=5.0, obs_count=2)

    solved = solve_normal_equations(normals)
    solution = solved.values
    covariance = solved.covariance
    sigma0 = solved.sigma0_post
    assert covariance is not None

    assert np.allclose(solution, [1.0, 2.0])
    assert np.allclose(covariance, np.eye(2))
    assert sigma0 is None


def test_large_absolute_x0_preserves_correction_and_posterior_sigma0():
    name = ParameterName("test", "x")
    design = np.ones((4, 1))
    reduced = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    x0 = np.array([1.0e9])
    normals = NormalEquations.from_linearized_statistics(
        parameter_names=[name],
        N=design.T @ design,
        correction_rhs=design.T @ reduced,
        correction_lPl=float(reduced @ reduced),
        obs_count=4,
        x0=x0,
    )

    solved = solve_normal_equations(normals)

    expected_delta = np.array([2.5])
    expected_residuals = reduced - design @ expected_delta
    assert solved.values - x0 == pytest.approx(expected_delta)
    assert solved.sigma0_post == pytest.approx(
        np.sqrt(np.dot(weights, expected_residuals**2) / 3.0)
    )


def test_inconsistent_residual_quadratic_form_is_rejected():
    normals = NormalEquations([ParameterName("test", "x")], np.zeros((1, 1)), np.zeros(1))
    normals.N[0, 0] = 1.0
    normals.W[0] = 2.0
    normals.lPl = 1.0
    normals.obs_count = 2

    with pytest.raises(NormalEquationInconsistencyError, match="negative residual quadratic"):
        solve_normal_equations(normals)


def test_normal_equation_parameter_names_must_be_unique():
    name = ParameterName("test", "x")
    with pytest.raises(ValueError, match="must be unique"):
        NormalEquations([name, name], np.eye(2), np.zeros(2))
















def test_normal_equation_groups_require_extensionless_directories(tmp_path):
    normals = NormalEquations([ParameterName("test", "x")], np.zeros((1, 1)), np.zeros(1))

    with pytest.raises(ValueError, match="extensionless directory"):
        write_normal_equations(normals, tmp_path / "normals.npz")
