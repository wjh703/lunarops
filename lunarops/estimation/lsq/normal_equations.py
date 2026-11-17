"""Weighted normal equations in the absolute convention ``N x = W``.

The system is stored for the absolute parameter vector ``x`` together with the linearization point ``x0``;
the correction system around ``x0`` is ``N dx = W - N x0``.

Parameters of one adjustment have different units (metres, milliarcseconds, radians per second, m^3/s^2), so
the diagonal of ``N`` can span tens of orders of magnitude. The factorization, the rank test and the
condition number therefore use the symmetrically scaled matrix ``D^-1 N D^-1`` with ``D_jj = sqrt(N_jj)``,
which has a unit diagonal (Jacobi scaling; its condition number is within a factor ``m`` of the smallest that
any diagonal scaling of a symmetric positive definite matrix reaches: van der Sluis 1969, Numer. Math. 14,
14-23; Higham 2002, Accuracy and Stability of Numerical Algorithms, 2nd ed., Theorem 7.5 and Section 10.1).
The scaled matrix, its spectrum and its condition number do not depend on the units of the parameters;
solutions and the cofactor matrix are returned in the parameters' own units.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import numpy as np

from lunarops.base.parameter_name import ParameterName


class NormalEquations:
    """Weighted absolute normal-equation system ``N x = W``.

    ``x0`` records the parameter values used to linearize the observation rows, while ``lPl`` is the
    reduced-observation quadratic at ``x0``. Matrix assembly must finish before its first factorization or
    solve, because the factorization, spectrum and covariance are cached.

    Parameters
    ----------
    parameter_names : list[ParameterName]
        Parameter names in column order.
    N : ndarray, shape (m, m)
        Normal matrix ``A^T P A``.
    W : ndarray, shape (m,)
        Absolute right-hand side.
    lPl : float, default 0.0
        Weighted quadratic ``l^T P l`` of the reduced observations at ``x0``.
    obs_count : int, default 0
        Number of observations accumulated, constraint pseudo-observations included; ``obs_count - m`` are the
        degrees of freedom of the solution.
    meta : dict[str, object]
        Free-form metadata.
    parameter_units : Sequence[str]
        Unit label per parameter.
    x0 : ndarray, shape (m,)
        Linearization point, in parameter units.
    """

    __slots__ = (
        "N",
        "W",
        "__dict__",
        "lPl",
        "meta",
        "obs_count",
        "parameter_names",
        "parameter_units",
        "x0",
    )

    def __init__(
        self,
        parameter_names: list[ParameterName],
        N: np.ndarray,
        W: np.ndarray,
        lPl: float = 0.0,
        obs_count: int = 0,
        *,
        meta: dict[str, object],
        parameter_units: Sequence[str],
        x0: np.ndarray,
    ) -> None:
        self.parameter_names = list(parameter_names)
        self.parameter_units = list(parameter_units)
        self.N = np.asarray(N, dtype=float)
        self.W = np.asarray(W, dtype=float).reshape(-1)
        self.lPl = float(lPl)
        self.obs_count = int(obs_count)
        self.meta = dict(meta)
        self.x0 = np.asarray(x0, dtype=float).reshape(-1)

    @cached_property
    def symmetric_matrix(self) -> np.ndarray:
        """Symmetric part ``(N + N^T) / 2`` of the normal matrix."""
        return 0.5 * (self.N + self.N.T)

    @cached_property
    def column_scale(self) -> np.ndarray:
        """Jacobi scale ``D_jj = sqrt(N_jj)`` of each parameter, shape ``(m,)``.

        It is 1 for a parameter whose diagonal element is not positive and finite.
        """
        diagonal = np.diag(self.symmetric_matrix).astype(float)
        scale = np.ones(len(diagonal))
        usable = np.isfinite(diagonal) & (diagonal > 0.0)
        scale[usable] = np.sqrt(diagonal[usable])
        return scale

    @cached_property
    def scaled_matrix(self) -> np.ndarray:
        """Symmetric normal matrix scaled to a unit diagonal, ``D^-1 N D^-1``."""
        scale = self.column_scale
        return self.symmetric_matrix / scale[:, None] / scale[None, :]

    @cached_property
    def spectrum(self) -> tuple[np.ndarray, float]:
        """Ascending eigenvalues of the scaled normal matrix and the rank tolerance ``eps * m * max|lambda|``.

        The eigenvalues of ``D^-1 N D^-1`` do not depend on the units of the parameters.
        """
        if self.N.size == 0:
            return np.empty(0, dtype=float), 0.0
        eigenvalues = np.linalg.eigvalsh(self.scaled_matrix)
        scale = float(np.max(np.abs(eigenvalues)))
        tolerance = np.finfo(float).eps * max(1, self.N.shape[0]) * scale
        return eigenvalues, tolerance

    @cached_property
    def scaled_cholesky(self) -> np.ndarray:
        """Lower Cholesky factor of the scaled matrix; raises ``LinAlgError`` if not positive definite."""
        return np.linalg.cholesky(self.scaled_matrix)

    @cached_property
    def cholesky(self) -> np.ndarray:
        """Lower Cholesky factor of the symmetric normal matrix; raises ``LinAlgError`` if not positive definite."""
        return self.column_scale[:, None] * self.scaled_cholesky

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Return ``N^-1 rhs`` from the Cholesky factor of the scaled matrix.

        Parameters
        ----------
        rhs : ndarray, shape (m,) or (m, k)
            Right-hand side(s), in the units of ``N x``.

        Returns
        -------
        ndarray
            Solution(s) in parameter units, with the shape of ``rhs``.
        """
        lower = self.scaled_cholesky
        scale = self.column_scale if np.ndim(rhs) == 1 else self.column_scale[:, None]
        scaled = np.linalg.solve(lower.T, np.linalg.solve(lower, np.asarray(rhs, dtype=float) / scale))
        return scaled / scale

    @cached_property
    def covariance(self) -> np.ndarray:
        """Cofactor matrix ``N^-1`` from the Cholesky factor, symmetrized."""
        covariance = self.solve(np.eye(len(self.parameter_names)))
        return 0.5 * (covariance + covariance.T)

    @cached_property
    def condition(self) -> float:
        """Condition number of the scaled matrix ``D^-1 N D^-1`` (largest over smallest eigenvalue).

        Infinity when the matrix is empty or numerically rank deficient.
        """
        eigenvalues, tolerance = self.spectrum
        full_rank = eigenvalues.size and np.count_nonzero(eigenvalues > tolerance) == eigenvalues.size
        return float(eigenvalues[-1] / eigenvalues[0]) if full_rank else float("inf")

    def __repr__(self) -> str:
        """Return a summary with the parameter and observation counts."""
        return (
            "NormalEquations("
            f"parameter_count={len(self.parameter_names)}, "
            f"obs_count={self.obs_count}, lPl={self.lPl!r}, "
            f"meta_keys={tuple(self.meta)!r})"
        )

    @classmethod
    def from_linearized_statistics(
        cls,
        parameter_names: Sequence[ParameterName],
        N: np.ndarray,
        correction_rhs: np.ndarray,
        correction_lPl: float,
        *,
        obs_count: int,
        x0: np.ndarray,
        meta: dict[str, object],
        parameter_units: Sequence[str],
    ) -> NormalEquations:
        """Build an absolute system from statistics at a linearization point.

        Linearized observation rows form the correction system ``N dx = Wc`` around ``x0``. The equivalent
        absolute system ``N x = Wc + N x0`` is stored. ``N`` is symmetrized first.

        Parameters
        ----------
        parameter_names : Sequence[ParameterName]
            Parameter names in column order.
        N : ndarray, shape (m, m)
            Normal matrix ``A^T P A``.
        correction_rhs : ndarray, shape (m,)
            Correction right-hand side ``Wc = A^T P l``.
        correction_lPl : float
            Weighted quadratic ``l^T P l`` of the reduced observations.
        obs_count : int
            Number of observations accumulated.
        x0 : ndarray, shape (m,)
            Linearization point, in parameter units.
        meta : dict[str, object]
            Free-form metadata.
        parameter_units : Sequence[str]
            Unit label per parameter.

        Returns
        -------
        NormalEquations
            Absolute system.
        """
        # Weighted normal matrices are symmetric despite roundoff across m/mas columns.
        normal_matrix = 0.5 * (N + N.T)
        normals = cls(
            parameter_names=list(parameter_names),
            parameter_units=parameter_units,
            N=normal_matrix,
            W=correction_rhs + normal_matrix @ x0,
            lPl=correction_lPl,
            obs_count=obs_count,
            meta=meta,
            x0=x0,
        )

        normals.symmetric_matrix = normal_matrix
        return normals

    def correction_at_x0(self, values: np.ndarray) -> np.ndarray:
        """Return the correction ``values - x0`` from this system's linearization point.

        Raises
        ------
        ValueError
            If ``values`` is not finite or does not have the shape of ``x0``.
        """
        estimate = np.asarray(values, dtype=float).reshape(-1)
        reference = self.x0
        if estimate.shape != reference.shape or not np.all(np.isfinite(estimate)):
            raise ValueError("Absolute parameter estimate must be finite and align with x0.")
        return estimate - reference


__all__ = ["NormalEquations"]
