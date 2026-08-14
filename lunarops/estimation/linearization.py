"""Dense and streaming linearization utilities.

``LlrProcessing`` controls nonlinear Gauss--Newton iteration, outlier handling,
convergence, update absorption, and optional normal-equation output.

The streaming path accumulates rows directly into ``N, W, lPl``.  The dense
path materializes the design matrix once when repeated reweighting makes that
time-memory tradeoff worthwhile.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.normal_equations import NormalEquations, SparseNormalRow


_STREAMING_BATCH_SIZE = 4096


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
        equations = tuple(self.equations)
        names = tuple(self.parameter_names)
        identities = tuple(self.identities)
        if not equations:
            raise ValueError("Dense linearization requires at least one observation equation.")
        if not all(isinstance(equation, ObservationEquation) for equation in equations):
            raise TypeError("Dense linearization equations must be ObservationEquation objects.")
        if not names:
            raise ValueError("Dense linearization requires at least one parameter.")
        if not all(isinstance(name, ParameterName) for name in names) or len(set(names)) != len(names):
            raise ValueError("Dense linearization parameter names must be unique ParameterName objects.")
        if not all(isinstance(identity, Hashable) for identity in identities):
            raise TypeError("Dense linearization observation IDs must be hashable.")
        if len(identities) != len(equations) or len(set(identities)) != len(identities):
            raise ValueError("Dense linearization observation IDs must be unique and align with its equations.")
        if identities != tuple(equation.observation_id for equation in equations):
            raise ValueError("Dense linearization observation IDs must match its equations.")
        design = np.array(self.design, dtype=float, copy=True)
        reduced = np.array(self.reduced_observations, dtype=float, copy=True).reshape(-1)
        apriori_sigmas = np.array(self.apriori_sigmas, dtype=float, copy=True).reshape(-1)
        if design.shape != (len(equations), len(names)):
            raise ValueError("Dense linearization design shape is inconsistent.")
        if reduced.size != len(equations) or apriori_sigmas.size != len(equations):
            raise ValueError("Dense linearization row vectors must align with its equations.")
        if not np.all(np.isfinite(design)) or not np.all(np.isfinite(reduced)):
            raise ValueError("Dense linearization design and observations must be finite.")
        if not np.all(np.isfinite(apriori_sigmas)) or np.any(apriori_sigmas <= 0.0):
            raise ValueError("Dense linearization a-priori sigmas must be positive and finite.")
        for array in (design, reduced, apriori_sigmas):
            array.setflags(write=False)
        object.__setattr__(self, "equations", equations)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "reduced_observations", reduced)
        object.__setattr__(self, "apriori_sigmas", apriori_sigmas)
        object.__setattr__(self, "identities", identities)

    @classmethod
    def build(
        cls,
        equations: Sequence[ObservationEquation],
        parametrization: ParametrizationList,
        parameter_names: Sequence[ParameterName],
    ) -> "DenseLinearization":
        rows = tuple(equations)
        if not rows:
            raise ValueError("Cannot build a dense linearization from an empty observation sequence.")
        if not parameter_names:
            raise ValueError("Cannot build a dense linearization without parameters.")
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
        active: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ) -> NormalEquations:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != len(self.equations):
            raise ValueError("Dense weights do not match the observation count.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Dense weights must be finite and non-negative.")
        if active is None:
            requested_mask = weights > 0.0
        else:
            raw_active = np.asarray(active)
            if raw_active.dtype != np.bool_:
                raise TypeError("Dense active mask must contain booleans.")
            requested_mask = raw_active
        if requested_mask.shape != weights.shape:
            raise ValueError("Dense active mask does not match the observation count.")
        mask = requested_mask & (weights > 0.0)
        A = self.design[mask]
        observations = self.reduced_observations[mask]
        w = weights[mask]
        weighted_A = w[:, None] * A
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


def build_normal_equations_streaming(
    equations: Iterable[ObservationEquation],
    parametrization: ParametrizationList,
    *,
    parameter_names: Optional[Sequence[ParameterName]] = None,
    weight_for: Optional[Callable[[ObservationEquation], float]] = None,
    x0: Optional[np.ndarray] = None,
    **meta,
) -> NormalEquations:
    """Build normal equations by streaming over observation equations.

    Parameters
    ----------
    equations
        Linearized observation equations at one fixed model state.
    parametrization
        Concatenated parameter blocks that provide design rows and reduced
        observations.
    parameter_names
        Optional explicit name list.  Supplying it avoids recomputing names and
        guarantees the same column order across iterations/programs.
    meta
        Metadata stored in the resulting :class:`NormalEquations` object.
    x0
        Absolute parameter values at the linearization point, in column order.
        If omitted, the zero vector is used.
    """
    names = list(parameter_names if parameter_names is not None else parametrization.parameter_names())
    normals = NormalEquations.zeros(
        names,
        x0=x0,
        **meta,
    )
    batch: list[SparseNormalRow] = []
    for eq in equations:
        entries = parametrization.design_entries(eq)
        reduced = parametrization.reduced_observation(eq)
        if weight_for is None:
            sigma = float(eq.sigma_one_way_m)
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"Observation sigma must be positive and finite, got {sigma!r}.")
            weight = 1.0 / (sigma * sigma)
        else:
            weight = float(weight_for(eq))
        batch.append((entries, reduced, weight))
        if len(batch) == _STREAMING_BATCH_SIZE:
            normals.accumulate_sparse_rows(batch)
            batch.clear()
    normals.accumulate_sparse_rows(batch)
    return normals


__all__ = ["DenseLinearization", "build_normal_equations_streaming"]
