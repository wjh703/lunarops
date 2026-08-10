"""Strict solution and diagnostics for fixed normal-equation systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lunarops.estimation.normal_equations import NormalEquations


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class NormalEquationSolution:
    """Solution of one fixed-linearization normal-equation system."""

    delta: np.ndarray
    covariance: np.ndarray
    sigma0_post: Optional[float]
    method: str


class NormalEquationSingularError(np.linalg.LinAlgError):
    """Raised when a normal-equation matrix cannot be solved strictly."""


class NormalEquationInconsistencyError(np.linalg.LinAlgError):
    """Raised when normal-equation statistics contradict the fitted solution."""


def _spectrum(normals: NormalEquations) -> tuple[np.ndarray, float]:
    if not isinstance(normals, NormalEquations):
        raise TypeError("Normal-matrix diagnostics require a NormalEquations instance.")
    matrix = np.asarray(normals.N, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Normal matrix must be square for spectral diagnostics.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Normal matrix must be finite for spectral diagnostics.")
    if matrix.size == 0:
        return np.empty(0, dtype=float), 0.0
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    scale = float(np.max(np.abs(eigenvalues)))
    tolerance = np.finfo(float).eps * max(1, matrix.shape[0]) * scale
    return eigenvalues, tolerance


def normal_matrix_rank(normals: NormalEquations) -> int:
    """Return the numerical rank using the same scale-aware spectral threshold everywhere."""
    eigenvalues, tolerance = _spectrum(normals)
    return int(np.count_nonzero(eigenvalues > tolerance))


def normal_matrix_condition(normals: NormalEquations) -> Optional[float]:
    """Return the normal-matrix condition number, or ``inf`` when rank-deficient."""
    if not isinstance(normals, NormalEquations):
        raise TypeError("Normal-matrix diagnostics require a NormalEquations instance.")
    if normals.N.size == 0:
        return None
    eigenvalues, tolerance = _spectrum(normals)
    positive = eigenvalues[eigenvalues > tolerance]
    if positive.size == 0:
        return float("inf")
    if positive.size != eigenvalues.size:
        return float("inf")
    return float(positive[-1] / positive[0])


def normal_matrix_rank_diagnostics(normals: NormalEquations) -> str:
    """Return a compact diagnostic string for singular or near-singular ``N``."""
    if not isinstance(normals, NormalEquations):
        raise TypeError("Normal-matrix diagnostics require a NormalEquations instance.")
    matrix = np.asarray(normals.N, dtype=float)
    parameter_count = len(normals.parameter_names)
    if matrix.shape != (parameter_count, parameter_count):
        return f"normal matrix has shape {matrix.shape}, expected {(parameter_count, parameter_count)}."
    if parameter_count == 0:
        return "normal matrix has no parameters."
    eigenvalues, tolerance = _spectrum(normals)
    rank = normal_matrix_rank(normals)
    diagonal = np.diag(matrix)
    zeroish = np.where(np.abs(diagonal) <= tolerance)[0]
    zero_names = [str(normals.parameter_names[index]) for index in zeroish[:10]]
    condition = normal_matrix_condition(normals)
    condition_text = "unknown" if condition is None else f"{condition:.3e}"
    pieces = [
        "normal-equation matrix is singular or numerically singular",
        f"rank={rank}/{parameter_count}",
        f"condition≈{condition_text}",
        f"obs_count={normals.obs_count}",
    ]
    if zero_names:
        suffix = "..." if len(zeroish) > len(zero_names) else ""
        pieces.append(f"zero-diagonal parameters={zero_names!r}{suffix}")
    return "; ".join(pieces) + "."


def solve_normal_equations(normals: NormalEquations) -> NormalEquationSolution:
    """Solve ``N x = W`` strictly with Cholesky factorization.

    A pseudo-inverse would conceal unobservable parameter combinations, so a
    rank-deficient or non-positive-definite system always fails with a
    diagnostic error.
    """
    if not isinstance(normals, NormalEquations):
        raise TypeError("solve_normal_equations requires a NormalEquations instance.")
    matrix = np.asarray(normals.N, dtype=float)
    rhs = np.asarray(normals.W, dtype=float).reshape(-1)
    parameter_count = len(normals.parameter_names)
    if matrix.shape != (parameter_count, parameter_count) or rhs.shape != (parameter_count,):
        raise ValueError("Normal-equation matrix, right-hand side, and parameter names are inconsistent.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)) or not np.isfinite(normals.lPl):
        raise ValueError("Normal-equation matrix, right-hand side, and lPl must be finite.")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("Normal matrix must be symmetric.")
    rank = normal_matrix_rank(normals)
    if parameter_count == 0 or rank != parameter_count:
        raise NormalEquationSingularError(normal_matrix_rank_diagnostics(normals))
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        lower = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as exc:
        raise NormalEquationSingularError(normal_matrix_rank_diagnostics(normals)) from exc
    delta = np.linalg.solve(lower.T, np.linalg.solve(lower, rhs))
    covariance = np.linalg.solve(lower.T, np.linalg.solve(lower, np.eye(parameter_count)))
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(covariance)):
        raise NormalEquationInconsistencyError("Normal-equation solve produced non-finite solution values.")
    covariance = 0.5 * (covariance + covariance.T)
    fitted_quadratic = float(rhs @ delta)
    residual_quadratic = normals.lPl - fitted_quadratic
    if not np.isfinite(residual_quadratic):
        raise NormalEquationInconsistencyError("Normal-equation residual quadratic is non-finite.")
    tolerance = 1.0e-10 * max(1.0, normals.lPl, abs(fitted_quadratic))
    if residual_quadratic < -tolerance:
        raise NormalEquationInconsistencyError(
            "Normal-equation negative residual quadratic is beyond roundoff: "
            f"lPl-W.T@x={residual_quadratic:.6e}."
        )
    residual_quadratic = max(residual_quadratic, 0.0)
    degrees_of_freedom = normals.obs_count - parameter_count
    sigma0 = None if degrees_of_freedom <= 0 else float(np.sqrt(residual_quadratic / degrees_of_freedom))
    delta = np.asarray(delta, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    delta.setflags(write=False)
    covariance.setflags(write=False)
    return NormalEquationSolution(
        delta=delta,
        covariance=covariance,
        sigma0_post=sigma0,
        method="cholesky",
    )


__all__ = [
    "NormalEquationSingularError",
    "NormalEquationInconsistencyError",
    "NormalEquationSolution",
    "normal_matrix_condition",
    "normal_matrix_rank",
    "normal_matrix_rank_diagnostics",
    "solve_normal_equations",
]
