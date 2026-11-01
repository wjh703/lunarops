"""A fixed linearization shares its design matrix across sigma/weight iterations."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from lunarops.base.parameter_name import ParameterName
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
        rows = equations
        design = parametrization.design(rows)
        reduced = parametrization.reduce(rows)
        sigmas = rows.sigma
        return cls(
            equations=rows,
            parameter_names=tuple(parameter_names),
            design=design,
            reduced_observations=reduced,
            apriori_sigmas=sigmas,
            identities=tuple(rows.observation_id),
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
        # Weighted normal matrices are symmetric despite roundoff across m/mas columns.
        normal_matrix = A.T @ weighted_A
        normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)
        return NormalEquations.from_linearized_statistics(
            parameter_names=list(self.parameter_names),
            N=normal_matrix,
            correction_rhs=A.T @ (w * observations),
            correction_lPl=float(np.dot(w, observations * observations)),
            obs_count=int(np.count_nonzero(mask)),
            x0=x0,
        )


__all__ = ["DenseLinearization"]
