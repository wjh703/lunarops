"""Absolute normal-equation system.

Persistence lives in :mod:`lunarops.fileio.normal_equations`; strict solution
and diagnostics live in :mod:`lunarops.estimation.normal_equation_solver`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from lunarops.base.parameter_name import ParameterName, parameter_unit


class NormalEquations:
    """Mutable weighted absolute normal-equation system ``N x = W``.

    ``x0`` records the parameter values used to linearize the observation
    rows, while ``lPl`` is the reduced-observation quadratic at ``x0``.  An
    omitted ``x0`` means the zero vector.  The constructed object's ``x0`` is
    always a validated read-only array; it is provenance, not an alternative
    right-hand-side convention.
    """

    __slots__ = (
        "parameter_names",
        "parameter_units",
        "N",
        "W",
        "lPl",
        "obs_count",
        "meta",
        "x0",
    )

    def __init__(
        self,
        parameter_names: List[ParameterName],
        N: np.ndarray,
        W: np.ndarray,
        lPl: float = 0.0,
        obs_count: int = 0,
        meta: Optional[Dict[str, object]] = None,
        parameter_units: Optional[Sequence[str]] = None,
        *,
        x0: Optional[np.ndarray] = None,
    ) -> None:
        names = list(parameter_names)
        if not all(isinstance(name, ParameterName) for name in names):
            raise TypeError("Normal-equation parameter names must be ParameterName objects.")
        if len(set(names)) != len(names):
            raise ValueError("Normal-equation parameter names must be unique.")
        units = [
            str(unit).strip()
            for unit in ([parameter_unit(name) for name in names] if parameter_units is None else parameter_units)
        ]
        if len(units) != len(names) or any(not unit for unit in units):
            raise ValueError("Normal-equation parameter units must align and be non-empty.")

        parameter_count = len(names)
        normal_matrix = np.asarray(N, dtype=float)
        right_hand_side = np.asarray(W, dtype=float).reshape(-1)
        reference_values = (
            np.zeros(parameter_count, dtype=float) if x0 is None else np.array(x0, dtype=float, copy=True).reshape(-1)
        )
        if normal_matrix.shape != (parameter_count, parameter_count):
            raise ValueError(
                f"Normal matrix has shape {normal_matrix.shape}, expected {(parameter_count, parameter_count)}."
            )
        if right_hand_side.shape != (parameter_count,):
            raise ValueError(
                f"Normal right-hand side has shape {right_hand_side.shape}, expected {(parameter_count,)}."
            )
        if reference_values.shape != (parameter_count,):
            raise ValueError(
                f"Normal-equation x0 has shape {reference_values.shape}, expected {(parameter_count,)}."
            )
        if (
            not np.all(np.isfinite(normal_matrix))
            or not np.all(np.isfinite(right_hand_side))
            or not np.all(np.isfinite(reference_values))
        ):
            raise ValueError("Normal equations contain non-finite matrix values.")
        if not np.allclose(normal_matrix, normal_matrix.T, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("Normal matrix must be symmetric.")
        lPl = float(lPl)
        if not np.isfinite(lPl) or lPl < 0.0:
            raise ValueError("Normal-equation lPl must be finite and non-negative.")
        obs_count = int(obs_count)
        if obs_count < 0:
            raise ValueError("Normal-equation observation count must be a non-negative integer.")
        metadata = {} if meta is None else dict(meta)
        if any(not isinstance(key, str) or not key for key in metadata):
            raise ValueError("Normal-equation metadata keys must be non-empty strings.")
        reference_values.setflags(write=False)

        self.parameter_names = names
        self.parameter_units = units
        self.N = normal_matrix
        self.W = right_hand_side
        self.lPl = lPl
        self.obs_count = obs_count
        self.meta = metadata
        self.x0 = reference_values

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
        x0: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, object]] = None,
        parameter_units: Optional[Sequence[str]] = None,
    ) -> "NormalEquations":
        """Build an absolute system from statistics at a linearization point.

        Linearized observation rows naturally form ``N dx = Wc`` around
        ``x0``.  LunarOps stores only the equivalent absolute system
        ``N x = Wc + N x0``.
        """
        reference_values = np.zeros(len(parameter_names), dtype=float) if x0 is None else x0
        return cls(
            parameter_names=list(parameter_names),
            parameter_units=parameter_units,
            N=N,
            W=correction_rhs + N @ reference_values,
            lPl=correction_lPl,
            obs_count=obs_count,
            meta=meta,
            x0=reference_values,
        )

    def correction_at_x0(self, values: np.ndarray) -> np.ndarray:
        """Return the update from this system's recorded linearization point."""
        estimate = np.asarray(values, dtype=float).reshape(-1)
        reference = self.x0
        if estimate.shape != reference.shape or not np.all(np.isfinite(estimate)):
            raise ValueError("Absolute parameter estimate must be finite and align with x0.")
        return estimate - reference


__all__ = ["NormalEquations"]
