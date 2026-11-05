"""Parameter-vector and covariance-matrix products of an estimate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.base.parameter_name import ParameterName

# Published parameter uncertainties use three-sigma by project convention.
PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER = 3.0


@dataclass(frozen=True, slots=True, eq=False)
class ParameterVector:
    """Estimated parameter values with optional uncertainties.

    Attributes
    ----------
    parameter_names : tuple[ParameterName, ...]
        Parameter names.
    values : ndarray, shape (m,)
        Parameter values, in the units given by ``units``.
    units : tuple[str, ...]
        Unit label per parameter.
    uncertainties : ndarray, shape (m,), optional
        Uncertainties, ``uncertainty_sigma_multiplier`` times the one-sigma values.
    uncertainty_sigma_multiplier : float, optional
        Multiple of one sigma that ``uncertainties`` represents.
    """

    parameter_names: tuple[ParameterName, ...]
    values: np.ndarray
    units: tuple[str, ...]
    uncertainties: np.ndarray | None = None
    uncertainty_sigma_multiplier: float | None = None



@dataclass(frozen=True, slots=True, eq=False)
class CovarianceMatrix:
    """Parameter covariance or cofactor matrix.

    Attributes
    ----------
    parameter_names : tuple[ParameterName, ...]
        Parameter names in row and column order.
    matrix : ndarray, shape (m, m)
        Matrix values, in products of the parameter units.
    units : tuple[str, ...]
        Unit label per parameter.
    kind : str
        ``cofactor`` (unscaled ``N^-1``) or ``posteriorCovariance`` (scaled by ``sigma0_post**2``).
    """

    parameter_names: tuple[ParameterName, ...]
    matrix: np.ndarray
    units: tuple[str, ...]
    kind: str = "cofactor"



__all__ = ["PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER", "CovarianceMatrix", "ParameterVector"]
