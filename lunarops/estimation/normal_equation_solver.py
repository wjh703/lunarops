"""Strict solution and diagnostics for fixed normal-equation systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.estimation.normal_equations import NormalEquations


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class NormalEquationSolution:
    """Absolute solution vector and diagnostics for ``N x = W``."""

    values: np.ndarray
    covariance: np.ndarray
    sigma0_post: float | None
    method: str
    rank: int
    normals: NormalEquations

    @property
    def condition(self) -> float:
        return self.normals.condition


class NormalEquationSingularError(np.linalg.LinAlgError):
    """Raised when a normal-equation matrix cannot be solved strictly."""


class NormalEquationInconsistencyError(np.linalg.LinAlgError):
    """Raised when normal-equation statistics contradict the fitted solution."""


def normal_matrix_rank_diagnostics(normals: NormalEquations) -> str:
    """Return a compact diagnostic string for singular or near-singular ``N``."""
    matrix = normals.N
    parameter_count = len(normals.parameter_names)
    if parameter_count == 0:
        return "normal matrix has no parameters."
    eigenvalues, tolerance = normals.spectrum
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    diagonal = np.diag(matrix)
    zeroish = np.where(np.abs(diagonal) <= tolerance)[0]
    zero_names = [str(normals.parameter_names[index]) for index in zeroish[:10]]
    condition = normals.condition
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
    """Solve the absolute system ``N x = W`` strictly with Cholesky factorization.

    A pseudo-inverse would conceal unobservable parameter combinations, so a
    rank-deficient or non-positive-definite system always fails with a
    diagnostic error.
    """
    matrix = normals.N
    rhs = normals.W
    parameter_count = len(normals.parameter_names)
    eigenvalues, tolerance = normals.spectrum
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    if parameter_count == 0 or rank != parameter_count:
        raise NormalEquationSingularError(normal_matrix_rank_diagnostics(normals))
    try:
        lower = normals.cholesky
    except np.linalg.LinAlgError as exc:
        raise NormalEquationSingularError(normal_matrix_rank_diagnostics(normals)) from exc
    correction_rhs = rhs - matrix @ normals.x0
    delta = np.linalg.solve(lower.T, np.linalg.solve(lower, correction_rhs))
    values = normals.x0 + delta
    covariance = normals.covariance
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(covariance)):
        raise NormalEquationInconsistencyError("Normal-equation solve produced non-finite solution values.")
    fitted_quadratic = float(correction_rhs @ delta)
    residual_quadratic = normals.lPl - fitted_quadratic
    if not np.isfinite(residual_quadratic):
        raise NormalEquationInconsistencyError("Normal-equation residual quadratic is non-finite.")
    tolerance = 1.0e-10 * max(1.0, normals.lPl, abs(fitted_quadratic))
    if residual_quadratic < -tolerance:
        raise NormalEquationInconsistencyError(
            "Normal-equation negative residual quadratic is beyond roundoff: "
            f"lPl-(W-N@x0).T@(x-x0)={residual_quadratic:.6e}."
        )
    residual_quadratic = max(residual_quadratic, 0.0)
    degrees_of_freedom = normals.obs_count - parameter_count
    sigma0 = None if degrees_of_freedom <= 0 else float(np.sqrt(residual_quadratic / degrees_of_freedom))
    values.setflags(write=False)
    covariance.setflags(write=False)
    return NormalEquationSolution(
        values=values,
        covariance=covariance,
        sigma0_post=sigma0,
        method="cholesky",
        rank=rank,
        normals=normals,
    )


__all__ = [
    "NormalEquationInconsistencyError",
    "NormalEquationSingularError",
    "NormalEquationSolution",
    "normal_matrix_rank_diagnostics",
    "solve_normal_equations",
]
