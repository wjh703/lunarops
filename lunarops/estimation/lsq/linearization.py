"""Dense linearization of the observation equations at one parameter state.

The design matrix and reduced observations are computed once per linearization point and shared by the
sigma-factor and weight iterations that follow.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from lunarops.base.parameter_name import ParameterName, parameter_unit
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.lsq.normal_equations import NormalEquations


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class DenseLinearization:
    """Design matrix and reduced observations of a batch at one linearization point.

    Attributes
    ----------
    equations : EquationBatch
        Linearized observation equations.
    parameter_names : tuple[ParameterName, ...]
        Parameter names in column order.
    design : ndarray, shape (n, m)
        Partials of the one-way range, in metres per parameter unit.
    reduced_observations : ndarray, shape (n,)
        One-way O-C minus the linear parametrization contributions, in metres.
    apriori_sigmas : ndarray, shape (n,)
        A-priori one-way sigmas, in metres.
    identities : tuple[Hashable, ...]
        Observation identities in row order.

    Raises
    ------
    ValueError
        If ``design`` does not have shape ``(len(equations), len(parameter_names))``.
    """

    equations: EquationBatch
    parameter_names: tuple[ParameterName, ...]
    design: np.ndarray
    reduced_observations: np.ndarray
    apriori_sigmas: np.ndarray
    identities: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        """Check the design-matrix shape."""
        if self.design.shape != (len(self.equations), len(self.parameter_names)):
            raise ValueError("Dense linearization design shape is inconsistent.")

    @classmethod
    def build(
        cls,
        equations: EquationBatch,
        parametrization: ParametrizationList,
        parameter_names: Sequence[ParameterName],
    ) -> DenseLinearization:
        """Linearize ``equations`` with the parametrization's current state.

        Parameters
        ----------
        equations : EquationBatch
            Observation equations.
        parametrization : ParametrizationList
            Set-up parametrization providing the design matrix and reduced observations.
        parameter_names : Sequence[ParameterName]
            Parameter names in column order.

        Returns
        -------
        DenseLinearization
            The linearization.
        """
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
        """Accumulate the weighted normal equations of the active rows.

        With design ``A``, weights ``P`` and reduced observations ``l`` of the rows that are active and have
        positive weight, the correction system is ``N dx = A^T P l`` with ``N = A^T P A``; it is stored as the
        absolute system ``N x = W`` with ``W = A^T P l + N x0``.

        Parameters
        ----------
        weights : ndarray, shape (n,)
            Observation weights, in 1/m^2.
        active : ndarray of bool, shape (n,)
            Rows to include.
        x0 : ndarray, shape (m,)
            Linearization point, in parameter units.

        Returns
        -------
        NormalEquations
            Absolute normal equations with ``lPl = l^T P l`` and the observation count.
        """
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
            meta={},
            parameter_units=[parameter_unit(name) for name in self.parameter_names],
        )


__all__ = ["DenseLinearization"]
