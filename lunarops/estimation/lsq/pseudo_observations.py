"""Weighted pseudo-observations ``c x = t`` of the parameters, added to the normal equations.

A row ``c`` with target ``t`` and sigma ``s`` adds ``N += c^T c / s^2`` and, around the linearization point ``x0``,
the reduced pseudo-observation ``t - c x0``. Pseudo-observations count as observations in the degrees of freedom of
the a-posteriori unit-weight sigma, and their weighted squared residuals enter it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PseudoObservations:
    """Pseudo-observations ``rows x = targets`` with ``weights`` over the current parameters.

    Attributes
    ----------
    rows : ndarray, shape (k, m)
        Constraint rows, in pseudo-observation unit per parameter unit.
    targets : ndarray, shape (k,)
        Target values of ``rows x``.
    weights : ndarray, shape (k,)
        Weights ``1 / sigma^2``.
    labels : tuple[str, ...]
        Label of each pseudo-observation.
    """

    rows: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    labels: tuple[str, ...]

    def __len__(self) -> int:
        """Return the number of pseudo-observations."""
        return len(self.targets)

    def reduce(self, normal_matrix: np.ndarray, rhs: np.ndarray, lpl: float, x0: np.ndarray):
        """Add the pseudo-observations to the correction system ``N dx = rhs`` around ``x0``.

        Returns the new ``N``, ``rhs`` and ``l^T P l``.
        """
        if not len(self):
            return normal_matrix, rhs, lpl
        reduced = self.targets - self.rows @ x0
        weighted = self.weights[:, None] * self.rows
        return (
            normal_matrix + self.rows.T @ weighted,
            rhs + weighted.T @ reduced,
            lpl + float(np.sum(self.weights * reduced**2)),
        )

    def residuals(self, values: np.ndarray) -> np.ndarray:
        """Return ``rows x - targets`` at the parameter values ``values``."""
        return self.rows @ values - self.targets if len(self) else np.zeros(0)

    def records(self, values: np.ndarray, cofactor: np.ndarray, sigma0: float | None) -> list[dict[str, object]]:
        """Return per pseudo-observation its label, value ``rows x - targets``, a-priori and a-posteriori sigma."""
        scale = 1.0 if sigma0 is None else float(sigma0)
        formal = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", self.rows, cofactor, self.rows), 0.0)) if len(self) else []
        return [
            {
                "label": label,
                "value": float(value),
                "aprioriSigma": float(1.0 / np.sqrt(weight)),
                "aposterioriSigma": float(scale * sigma),
            }
            for label, value, weight, sigma in zip(
                self.labels, self.residuals(values), self.weights, formal, strict=True
            )
        ]


__all__ = ["PseudoObservations"]
