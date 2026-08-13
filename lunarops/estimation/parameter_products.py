"""Validated solution-vector and covariance products emitted by estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.base.parameter_name import ParameterName


@dataclass(frozen=True, slots=True, eq=False)
class ParameterVector:
    parameter_names: tuple[ParameterName, ...]
    values: np.ndarray
    units: tuple[str, ...]
    uncertainties: np.ndarray | None = None
    uncertainty_sigma_multiplier: float | None = None

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        if not all(isinstance(name, ParameterName) for name in names):
            raise TypeError("Parameter vector names must be ParameterName objects.")
        units = tuple(str(unit) for unit in self.units)
        values = np.array(self.values, dtype=float, copy=True).reshape(-1)
        if len(names) != len(units) or len(names) != values.size:
            raise ValueError("Parameter vector names, units, and values must have equal length.")
        if len(set(names)) != len(names):
            raise ValueError("Parameter vector names must be unique.")
        if any(not unit for unit in units):
            raise ValueError("Parameter vector units must not be empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Parameter vector values must be finite.")
        uncertainties = self.uncertainties
        multiplier = self.uncertainty_sigma_multiplier
        if uncertainties is not None:
            uncertainties = np.array(uncertainties, dtype=float, copy=True).reshape(-1)
            if (
                uncertainties.size != values.size
                or not np.all(np.isfinite(uncertainties))
                or np.any(uncertainties < 0.0)
            ):
                raise ValueError("Parameter uncertainties must be finite, non-negative, and aligned.")
            if multiplier is None:
                raise ValueError("Parameter uncertainties require an uncertainty sigma multiplier.")
            multiplier = float(multiplier)
            if not np.isfinite(multiplier) or multiplier <= 0.0:
                raise ValueError("Parameter uncertainty sigma multiplier must be positive and finite.")
            uncertainties.setflags(write=False)
        elif multiplier is not None:
            raise ValueError("Parameter uncertainty sigma multiplier requires uncertainties.")
        values.setflags(write=False)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "uncertainties", uncertainties)
        object.__setattr__(self, "uncertainty_sigma_multiplier", multiplier)


@dataclass(frozen=True, slots=True, eq=False)
class CovarianceMatrix:
    parameter_names: tuple[ParameterName, ...]
    matrix: np.ndarray
    units: tuple[str, ...]
    kind: str = "cofactor"

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        if not all(isinstance(name, ParameterName) for name in names):
            raise TypeError("Covariance names must be ParameterName objects.")
        units = tuple(str(unit) for unit in self.units)
        matrix = np.array(self.matrix, dtype=float, copy=True)
        if len(names) != len(units) or matrix.shape != (len(names), len(names)):
            raise ValueError("Covariance names, units, and square matrix are inconsistent.")
        if len(set(names)) != len(names) or not np.all(np.isfinite(matrix)):
            raise ValueError("Covariance parameters must be unique and values finite.")
        if any(not unit for unit in units):
            raise ValueError("Covariance units must not be empty.")
        if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("Covariance matrix must be symmetric.")
        tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(matrix), initial=0.0)))
        if matrix.size and float(np.min(np.linalg.eigvalsh(matrix))) < -tolerance:
            raise ValueError("Covariance matrix must be positive semidefinite.")
        matrix.setflags(write=False)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "matrix", matrix)
        kind = str(self.kind)
        if kind not in {"cofactor", "posteriorCovariance"}:
            raise ValueError(f"Unsupported covariance kind {kind!r}.")
        object.__setattr__(self, "kind", kind)


__all__ = ["CovarianceMatrix", "ParameterVector"]
