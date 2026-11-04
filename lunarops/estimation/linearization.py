"""A fixed linearization shares its design matrix across sigma/weight iterations."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from lunarops.base.parameter_name import ParameterName, parameter_unit
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.normal_equations import NormalEquations


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class DenseLinearization:

    equations: EquationBatch
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
        equations: EquationBatch,
        parametrization: ParametrizationList,
        parameter_names: Sequence[ParameterName],
    ) -> DenseLinearization:
        return cls(
            equations=equations,
            parameter_names=tuple(parameter_names),
            design=parametrization.design(equations),
            reduced_observations=parametrization.reduce(equations),
            apriori_sigmas=equations.sigma,
            identities=tuple(equations.observation_id),
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
        normal_matrix = A.T @ weighted_A
        return NormalEquations.from_linearized_statistics(
            parameter_names=list(self.parameter_names),
            N=normal_matrix,
            correction_rhs=A.T @ (w * observations),
            correction_lPl=float(np.dot(w, observations * observations)),
            obs_count=int(np.count_nonzero(mask)),
            x0=x0,
            meta={}, parameter_units=[parameter_unit(name) for name in self.parameter_names],
        )


__all__ = ["DenseLinearization"]
