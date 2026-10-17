"""Dense linearization for repeated reweighting.

``LlrProcessing`` controls nonlinear Gauss--Newton iteration, outlier handling,
convergence, update absorption, and optional normal-equation output.  The dense
linearization materializes the design matrix once per relinearization so the
sigma/weight iterations can rebuild normal equations cheaply.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.normal_equations import NormalEquations


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class DenseLinearization:
    """Materialized fixed-linearization system for repeated reweighting."""

    equations: tuple[ObservationEquation, ...]
    parameter_names: tuple[ParameterName, ...]
    design: np.ndarray
    reduced_observations: np.ndarray
    apriori_sigmas: np.ndarray
    identities: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        if self.design.shape != (len(self.equations), len(self.parameter_names)):
            raise ValueError("Dense linearization design shape is inconsistent.")

    @classmethod
    def build(
        cls,
        equations: Sequence[ObservationEquation],
        parametrization: ParametrizationList,
        parameter_names: Sequence[ParameterName],
    ) -> "DenseLinearization":
        rows = tuple(equations)
        design = np.vstack([parametrization.design_row(eq) for eq in rows])
        reduced = np.asarray([parametrization.reduced_observation(eq) for eq in rows], dtype=float)
        sigmas = np.asarray([eq.sigma_one_way_m for eq in rows], dtype=float)
        return cls(
            equations=rows,
            parameter_names=tuple(parameter_names),
            design=design,
            reduced_observations=reduced,
            apriori_sigmas=sigmas,
            identities=tuple(eq.observation_id for eq in rows),
        )

    def normal_equations(
        self,
        weights: np.ndarray,
        *,
        active: np.ndarray,
        x0: np.ndarray,
    ) -> NormalEquations:
        mask = active & (weights > 0.0)
        A = self.design[mask]
        observations = self.reduced_observations[mask]
        w = weights[mask]
        weighted_A = w[:, None] * A
        # A.T @ (w[:, None] * A) is symmetric mathematically; tiny floating
        # point asymmetry is accepted by NormalEquations and handled at solve.
        normal_matrix = A.T @ weighted_A
        return NormalEquations.from_linearized_statistics(
            parameter_names=list(self.parameter_names),
            N=normal_matrix,
            correction_rhs=A.T @ (w * observations),
            correction_lPl=float(np.dot(w, observations * observations)),
            obs_count=int(np.count_nonzero(mask)),
            x0=x0,
        )


__all__ = ["DenseLinearization"]
