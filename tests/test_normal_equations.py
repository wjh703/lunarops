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

    normals = NormalEquations.zeros(names)
    normals.accumulate(A, L, sigma)

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
    write_normal_equations(NormalEquations.zeros([ParameterName("test", "x")]), path)
    with (path / "metadata.txt").open("a", encoding="utf-8") as stream:
        stream.write("corrupt: true\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_normal_equations(path)


def test_dense_and_sparse_accumulation_publish_absolute_statistics_at_nonzero_x0():
    name = ParameterName("test", "x")
    x0 = np.array([10.0])
    design = np.array([[2.0], [1.0]])
    reduced = np.array([4.0, -1.0])
    sigma = np.array([2.0, 1.0])
    weights = 1.0 / sigma**2
    correction_rhs = design.T @ (weights * reduced)
    matrix = design.T @ (weights[:, None] * design)
    correction_lpl = float(np.dot(weights, reduced**2))
    expected_rhs = correction_rhs + matrix @ x0

    dense = NormalEquations.zeros([name], x0=x0)
    dense.accumulate(design, reduced, sigma)
    sparse = NormalEquations.zeros([name], x0=x0)
    sparse.accumulate_sparse_rows(
        [
            ([(0, 2.0)], 4.0, 0.25),
            ([(0, 1.0)], -1.0, 1.0),
        ]
    )

    for normals in (dense, sparse):
        assert normals.N == pytest.approx(matrix)
        assert normals.W == pytest.approx(expected_rhs)
        assert normals.lPl == pytest.approx(correction_lpl)
        assert solve_normal_equations(normals).values == pytest.approx([10.0 + correction_rhs[0] / matrix[0, 0]])


def test_exactly_determined_system_has_no_posterior_variance_factor():
    names = [ParameterName("test", "x"), ParameterName("test", "y")]
    normals = NormalEquations.zeros(names)
    normals.accumulate(
        np.eye(2),
        np.array([1.0, 2.0]),
        np.ones(2),
    )

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
    normals = NormalEquations.zeros([ParameterName("test", "x")])
    normals.N[0, 0] = 1.0
    normals.W[0] = 2.0
    normals.lPl = 1.0
    normals.obs_count = 2

    with pytest.raises(NormalEquationInconsistencyError, match="negative residual quadratic"):
        solve_normal_equations(normals)


def test_normal_equation_parameter_names_must_be_unique():
    name = ParameterName("test", "x")
    with pytest.raises(ValueError, match="must be unique"):
        NormalEquations.zeros([name, name])


def test_absolute_normal_equation_addition_accepts_different_x0_references():
    name = ParameterName("test", "x")
    first = NormalEquations(
        [name], np.array([[2.0]]), np.array([8.0]), lPl=32.0, x0=np.array([1.0])
    )
    second = NormalEquations(
        [name], np.array([[3.0]]), np.array([15.0]), lPl=75.0, x0=np.array([2.0])
    )

    combined = first.add(second)

    assert combined.x0 == pytest.approx([1.0])
    assert np.allclose(combined.N, [[5.0]])
    assert combined.W == pytest.approx([23.0])
    assert combined.lPl == pytest.approx(128.0)


def test_normal_equation_convention_cannot_be_selected():
    name = ParameterName("test", "x")

    with pytest.raises(TypeError, match="always absolute"):
        NormalEquations.zeros([name], rhs_convention="correction")


def test_sparse_batch_matches_validated_row_accumulation():
    names = [ParameterName("test", f"x{index}") for index in range(5)]
    rows = [
        ([(0, 1.0), (3, -2.0), (0, 0.5)], 4.0, 0.25),
        ([], -3.0, 0.25),
        ([(1, 2.0), (4, 1.5)], 0.5, 3.0),
    ]
    design = np.array([[1.5, 0.0, 0.0, -2.0, 0.0], [0.0] * 5, [0.0, 2.0, 0.0, 0.0, 1.5]])
    observations = np.array([4.0, -3.0, 0.5])
    weights = np.array([0.25, 0.25, 3.0])

    actual = NormalEquations.zeros(names)
    actual.accumulate_sparse_rows(rows)

    assert actual.N == pytest.approx(design.T @ (weights[:, None] * design), rel=0.0, abs=1.0e-15)
    assert actual.W == pytest.approx(design.T @ (weights * observations), rel=0.0, abs=1.0e-15)
    assert actual.lPl == pytest.approx(np.dot(weights, observations**2), rel=0.0, abs=1.0e-15)
    assert actual.obs_count == 3


def test_sparse_batch_validates_all_rows_before_mutating_normals():
    normals = NormalEquations.zeros([ParameterName("test", "x")])
    rows = [
        ([(0, 1.0)], 2.0, 1.0),
        ([(1, 1.0)], 3.0, 1.0),
    ]

    with pytest.raises(ValueError, match="outside"):
        normals.accumulate_sparse_rows(rows)

    assert not np.any(normals.N)
    assert not np.any(normals.W)
    assert normals.lPl == 0.0
    assert normals.obs_count == 0


def test_dense_batch_rejects_nonfinite_values_before_mutating_normals():
    normals = NormalEquations.zeros([ParameterName("test", "x")])

    with pytest.raises(ValueError, match="finite"):
        normals.accumulate(np.array([[np.nan]]), np.array([1.0]), np.array([1.0]))

    assert not np.any(normals.N)
    assert not np.any(normals.W)
    assert normals.lPl == 0.0
    assert normals.obs_count == 0


def test_zero_weight_sparse_rows_do_not_contribute_degrees_of_freedom():
    normals = NormalEquations.zeros([ParameterName("test", "x")])
    normals.accumulate_sparse_rows(
        [
            ([(0, 2.0)], 4.0, 0.0),
            ([(0, 1.0)], 3.0, 1.0),
        ]
    )

    assert np.allclose(normals.N, [[1.0]])
    assert normals.W == pytest.approx([3.0])
    assert normals.lPl == pytest.approx(9.0)
    assert normals.obs_count == 1


def test_sparse_rows_require_integral_column_indices():
    normals = NormalEquations.zeros([ParameterName("test", "x")])

    with pytest.raises(TypeError, match="indices must be integers"):
        normals.accumulate_sparse_rows([([(0.5, 1.0)], 1.0, 1.0)])


def test_normal_equation_groups_require_extensionless_directories(tmp_path):
    normals = NormalEquations.zeros([ParameterName("test", "x")])

    with pytest.raises(ValueError, match="extensionless directory"):
        write_normal_equations(normals, tmp_path / "normals.npz")
