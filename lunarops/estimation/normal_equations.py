"""Absolute normal-equation convention: N x = W."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import numpy as np

from lunarops.base.parameter_name import ParameterName, parameter_unit


class NormalEquations:
    """Weighted absolute normal-equation system ``N x = W``.

    ``x0`` records the parameter values used to linearize the observation
    rows, while ``lPl`` is the reduced-observation quadratic at ``x0``.  An
    omitted ``x0`` means the zero vector.  The constructed object's ``x0`` is
    a parameter vector recording the linearization reference.
    Matrix assembly must finish before its first factorization or solve.
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
        meta: dict[str, object] | None = None,
        parameter_units: Sequence[str] | None = None,
        *,
        x0: np.ndarray | None = None,
    ) -> None:
        names = list(parameter_names)
        units = [
            str(unit).strip()
            for unit in ([parameter_unit(name) for name in names] if parameter_units is None else parameter_units)
        ]

        parameter_count = len(names)
        normal_matrix = np.asarray(N, dtype=float)
        right_hand_side = np.asarray(W, dtype=float).reshape(-1)
        reference_values = (
            np.zeros(parameter_count, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        )
        self.parameter_names = names
        self.parameter_units = units
        self.N = normal_matrix
        self.W = right_hand_side
        self.lPl = float(lPl)
        self.obs_count = int(obs_count)
        self.meta = {} if meta is None else dict(meta)
        self.x0 = reference_values

    @cached_property
    def symmetric_matrix(self) -> np.ndarray:
        return 0.5 * (self.N + self.N.T)

    @cached_property
    def spectrum(self) -> tuple[np.ndarray, float]:
        if self.N.size == 0:
            return np.empty(0, dtype=float), 0.0
        eigenvalues = np.linalg.eigvalsh(self.symmetric_matrix)
        scale = float(np.max(np.abs(eigenvalues)))
        tolerance = np.finfo(float).eps * max(1, self.N.shape[0]) * scale
        return eigenvalues, tolerance

    @cached_property
    def cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(self.symmetric_matrix)

    @cached_property
    def covariance(self) -> np.ndarray:
        lower = self.cholesky
        covariance = np.linalg.solve(lower.T, np.linalg.solve(lower, np.eye(len(self.parameter_names))))
        return 0.5 * (covariance + covariance.T)

    @cached_property
    def condition(self) -> float:
        eigenvalues, tolerance = self.spectrum
        full_rank = eigenvalues.size and np.count_nonzero(eigenvalues > tolerance) == eigenvalues.size
        return float(eigenvalues[-1] / eigenvalues[0]) if full_rank else float("inf")

    def __repr__(self) -> str:
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
        x0: np.ndarray | None = None,
        meta: dict[str, object] | None = None,
        parameter_units: Sequence[str] | None = None,
    ) -> NormalEquations:
        """Build an absolute system from statistics at a linearization point.

        Linearized observation rows naturally form ``N dx = Wc`` around
        ``x0``.  LunarOps stores only the equivalent absolute system
        ``N x = Wc + N x0``.
        """
        reference_values = np.zeros(len(parameter_names), dtype=float) if x0 is None else x0
        # Weighted normal matrices are symmetric despite roundoff across m/mas columns.
        normal_matrix = 0.5 * (N + N.T)
        normals = cls(
            parameter_names=list(parameter_names),
            parameter_units=parameter_units,
            N=normal_matrix,
            W=correction_rhs + normal_matrix @ reference_values,
            lPl=correction_lPl,
            obs_count=obs_count,
            meta=meta,
            x0=reference_values,
        )

        normals.symmetric_matrix = normal_matrix
        return normals

    def correction_at_x0(self, values: np.ndarray) -> np.ndarray:
        """Return the update from this system's recorded linearization point."""
        estimate = np.asarray(values, dtype=float).reshape(-1)
        reference = self.x0
        if estimate.shape != reference.shape or not np.all(np.isfinite(estimate)):
            raise ValueError("Absolute parameter estimate must be finite and align with x0.")
        return estimate - reference


__all__ = ["NormalEquations"]
