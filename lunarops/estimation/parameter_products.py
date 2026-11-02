from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.base.parameter_name import ParameterName

# Published parameter uncertainties use three-sigma by project convention.
PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER = 3.0


@dataclass(frozen=True, slots=True, eq=False)
class ParameterVector:
    parameter_names: tuple[ParameterName, ...]
    values: np.ndarray
    units: tuple[str, ...]
    uncertainties: np.ndarray | None = None
    uncertainty_sigma_multiplier: float | None = None



@dataclass(frozen=True, slots=True, eq=False)
class CovarianceMatrix:
    parameter_names: tuple[ParameterName, ...]
    matrix: np.ndarray
    units: tuple[str, ...]
    kind: str = "cofactor"



__all__ = ["PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER", "CovarianceMatrix", "ParameterVector"]
