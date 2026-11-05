"""Strict Cholesky solution and rank diagnostics of absolute normal equations ``N x = W``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.estimation.normal_equations import NormalEquations


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class NormalEquationSolution:
    """Absolute solution vector and diagnostics of ``N x = W``.

    Attributes
    ----------
    values : ndarray, shape (m,)
        Absolute parameter estimate ``x``, in parameter units.
    covariance : ndarray, shape (m, m)
        Cofactor matrix ``N^-1``, not scaled by ``sigma0_post``.
    sigma0_post : float or None
        A-posteriori unit-weight sigma ``sqrt((lPl - (W - N x0)^T dx) / (n - m))``; ``None`` without redundancy.
    method : str
        Solution method, ``"cholesky"``.
    rank : int
        Numerical rank of ``N``.
    normals : NormalEquations
        The solved system.
    """

    values: np.ndarray
    covariance: np.ndarray
    sigma0_post: float | None
    method: str
    rank: int
    normals: NormalEquations

    @property
    def condition(self) -> float:
        """Condition number of ``N`` (largest over smallest eigenvalue)."""
        return self.normals.condition


class NormalEquationSingularError(np.linalg.LinAlgError):
    """Raised when a normal-equation matrix cannot be solved strictly."""


class NormalEquationInconsistencyError(np.linalg.LinAlgError):
    """Raised when normal-equation statistics contradict the fitted solution."""


def normal_matrix_rank_diagnostics(normals: NormalEquations) -> str:
    """Return a compact diagnostic string for a singular or near-singular ``N``.

    Parameters
    ----------
    normals : NormalEquations
        Normal equations to describe.

    Returns
    -------
    str
        Numerical rank, condition number, observation count and up to ten parameters with a zero diagonal.
    """
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
    condition_text = f"{condition:.3e}"
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

    The correction ``dx = N^-1 (W - N x0)`` is solved from the linearization point ``x0`` and ``x = x0 + dx``.
    A pseudo-inverse would conceal unobservable parameter combinations, so a rank-deficient or
    non-positive-definite system always fails with a diagnostic error.

    Parameters
    ----------
    normals : NormalEquations
        System to solve.

    Returns
    -------
    NormalEquationSolution
        Absolute estimate, cofactor matrix, ``sigma0_post`` and rank.

    Raises
    ------
    NormalEquationSingularError
        If the system has no parameters, is numerically rank deficient, or is not positive definite.
    NormalEquationInconsistencyError
        If the residual quadratic ``lPl - (W - N x0)^T dx`` is negative beyond roundoff.
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
    fitted_quadratic = float(correction_rhs @ delta)
    residual_quadratic = normals.lPl - fitted_quadratic
    tolerance = 1.0e-10 * max(1.0, normals.lPl, abs(fitted_quadratic))
    if residual_quadratic < -tolerance:
        raise NormalEquationInconsistencyError(
            "Normal-equation negative residual quadratic is beyond roundoff: "
            f"lPl-(W-N@x0).T@(x-x0)={residual_quadratic:.6e}."
        )
    residual_quadratic = max(residual_quadratic, 0.0)
    degrees_of_freedom = normals.obs_count - parameter_count
    sigma0 = None if degrees_of_freedom <= 0 else float(np.sqrt(residual_quadratic / degrees_of_freedom))
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
