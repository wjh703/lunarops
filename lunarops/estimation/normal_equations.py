"""Normal-equation accumulation and combination.

Persistence lives in :mod:`lunarops.fileio.normal_equations`; this module
contains the scientific object and normal-equation construction operations.
Strict solution and diagnostics live in :mod:`lunarops.estimation.normal_equation_solver`.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

import lunarops._normal_equations_core as _normal_equations_core
from lunarops.base.parameter_name import ParameterName, parameter_unit


SparseNormalRow = tuple[Iterable[tuple[int, float]], float, float]


class NormalEquations:
    """Mutable weighted absolute normal-equation system ``N x = W``.

    ``x0`` records the parameter values used to linearize the observation
    rows, while ``lPl`` is the reduced-observation quadratic at ``x0``.  An
    omitted ``x0`` means the zero vector.  The constructed object's ``x0`` is
    always a validated array; it is provenance, not an alternative
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
        self.parameter_names = parameter_names
        self.parameter_units = (
            [parameter_unit(name) for name in parameter_names] if parameter_units is None else list(parameter_units)
        )
        self.N = N
        self.W = W
        self.lPl = lPl
        self.obs_count = obs_count
        self.meta = {} if meta is None else meta
        self.x0 = np.zeros(len(parameter_names), dtype=float) if x0 is None else x0
        self.__post_init__()

    def __repr__(self) -> str:
        return (
            "NormalEquations("
            f"parameter_count={len(self.parameter_names)}, "
            f"obs_count={self.obs_count}, lPl={self.lPl!r}, "
            f"meta_keys={tuple(self.meta)!r})"
        )

    def __post_init__(self) -> None:
        names = list(self.parameter_names)
        if not all(isinstance(name, ParameterName) for name in names):
            raise TypeError("Normal-equation parameter names must be ParameterName objects.")
        if len(set(names)) != len(names):
            raise ValueError("Normal-equation parameter names must be unique.")
        units = [str(unit).strip() for unit in self.parameter_units]
        if len(units) != len(names) or any(not unit for unit in units):
            raise ValueError("Normal-equation parameter units must align and be non-empty.")

        normal_matrix = np.asarray(self.N, dtype=float)
        right_hand_side = np.asarray(self.W, dtype=float).reshape(-1)
        parameter_count = len(names)
        reference_values = np.array(self.x0, dtype=float, copy=True).reshape(-1)
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
        if isinstance(self.lPl, bool) or not isinstance(self.lPl, Real):
            raise TypeError("Normal-equation lPl must be a real number.")
        if not np.isfinite(self.lPl) or float(self.lPl) < 0.0:
            raise ValueError("Normal-equation lPl must be finite and non-negative.")
        if isinstance(self.obs_count, bool) or not isinstance(self.obs_count, Integral) or int(self.obs_count) < 0:
            raise ValueError("Normal-equation observation count must be a non-negative integer.")
        self.parameter_names = names
        self.parameter_units = units
        self.N = normal_matrix
        self.W = right_hand_side
        self.lPl = float(self.lPl)
        self.obs_count = int(self.obs_count)
        reference_values.setflags(write=False)
        self.x0 = reference_values
        if not isinstance(self.meta, dict):
            raise TypeError("Normal-equation metadata must be a dictionary.")
        metadata: dict[str, object] = {}
        for key, value in dict(self.meta).items():
            if not isinstance(key, str) or not key:
                raise ValueError("Normal-equation metadata keys must be non-empty strings.")
            metadata[key] = value
        self.meta = metadata

    @classmethod
    def zeros(
        cls,
        parameter_names: Sequence[ParameterName],
        *,
        parameter_units: Optional[Sequence[str]] = None,
        x0: Optional[np.ndarray] = None,
        **meta,
    ) -> "NormalEquations":
        if "rhs_convention" in meta:
            raise TypeError("Normal equations are always absolute; rhs_convention is not supported.")
        names = list(parameter_names)
        count = len(names)
        reference_values = np.zeros(count, dtype=float) if x0 is None else x0
        return cls(
            parameter_names=names,
            parameter_units=parameter_units,
            N=np.zeros((count, count), dtype=float),
            W=np.zeros(count, dtype=float),
            lPl=0.0,
            obs_count=0,
            meta=dict(meta),
            x0=reference_values,
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
        provisional = cls(
            parameter_names=list(parameter_names),
            parameter_units=parameter_units,
            N=N,
            W=correction_rhs,
            lPl=correction_lPl,
            obs_count=obs_count,
            meta=meta,
            x0=x0,
        )
        absolute_rhs = provisional.W + provisional.N @ provisional.x0
        return cls(
            parameter_names=provisional.parameter_names,
            parameter_units=provisional.parameter_units,
            N=provisional.N,
            W=absolute_rhs,
            lPl=provisional.lPl,
            obs_count=provisional.obs_count,
            meta=provisional.meta,
            x0=provisional.x0,
        )

    def correction_at_x0(self, values: np.ndarray) -> np.ndarray:
        """Return the update from this system's recorded linearization point."""
        estimate = np.asarray(values, dtype=float).reshape(-1)
        reference = self.x0
        if estimate.shape != reference.shape or not np.all(np.isfinite(estimate)):
            raise ValueError("Absolute parameter estimate must be finite and align with x0.")
        return estimate - reference

    def _normalize_sparse_row(
        self,
        entries: Iterable[tuple[int, float]],
        observation: float,
        weight: float,
    ) -> tuple[list[int], list[float], float, float]:
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Observation weight must be finite and non-negative, got {weight!r}.")
        observation = float(observation)
        if not np.isfinite(observation):
            raise ValueError("Reduced observation must be finite.")

        coalesced: dict[int, float] = {}
        parameter_count = len(self.parameter_names)
        for raw_index, raw_value in entries:
            if isinstance(raw_index, (bool, np.bool_)) or not isinstance(raw_index, (int, np.integer)):
                raise TypeError("Sparse design column indices must be integers.")
            index = int(raw_index)
            if index < 0 or index >= parameter_count:
                raise ValueError(f"Sparse design column {index} is outside [0, {parameter_count}).")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError("Sparse design values must be finite.")
            if value:
                coalesced[index] = coalesced.get(index, 0.0) + value
        indices = list(coalesced)
        values = [coalesced[index] for index in indices]
        if not np.all(np.isfinite(values)):
            raise ValueError("Coalesced sparse design values must remain finite.")
        return indices, values, observation, weight

    def accumulate_sparse_rows(self, rows: Iterable[SparseNormalRow]) -> None:
        """Accumulate sparse rows linearized at ``x0`` into ``N x = W``."""
        offsets = [0]
        indices: list[int] = []
        values: list[float] = []
        observations: list[float] = []
        weights: list[float] = []
        for entries, observation, weight in rows:
            row_indices, row_values, value, weight_value = self._normalize_sparse_row(
                entries,
                observation,
                weight,
            )
            if weight_value == 0.0:
                continue
            indices.extend(row_indices)
            values.extend(row_values)
            offsets.append(len(indices))
            observations.append(value)
            weights.append(weight_value)
        if not observations:
            return
        reference = self.x0
        matrix = self.N.copy()
        correction_rhs_increment = np.zeros_like(self.W)
        lpl_increment = float(
            _normal_equations_core.accumulate_sparse_batch(
                matrix,
                correction_rhs_increment,
                np.asarray(offsets, dtype=np.intp),
                np.asarray(indices, dtype=np.intp),
                np.asarray(values, dtype=np.float64),
                np.asarray(observations, dtype=np.float64),
                np.asarray(weights, dtype=np.float64),
            )
        )
        normal_increment = matrix - self.N
        rhs = self.W + correction_rhs_increment + normal_increment @ reference
        next_lpl = self.lPl + lpl_increment
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)) or not np.isfinite(next_lpl):
            raise FloatingPointError("Sparse normal-equation accumulation produced non-finite values.")
        self.N[:] = matrix
        self.W[:] = rhs
        self.lPl = float(next_lpl)
        self.obs_count += len(observations)

    def accumulate(self, A: np.ndarray, reduced_observations: np.ndarray, sigma: np.ndarray) -> None:
        """Accumulate dense rows linearized at ``x0`` into ``N x = W``."""
        design = np.asarray(A, dtype=float)
        observations = np.asarray(reduced_observations, dtype=float).reshape(-1)
        sigmas = np.asarray(sigma, dtype=float).reshape(-1)
        if design.ndim != 2:
            raise ValueError("Design matrix A must be two-dimensional.")
        if design.shape[1] != len(self.parameter_names):
            raise ValueError(f"Design matrix has {design.shape[1]} columns, expected {len(self.parameter_names)}.")
        if design.shape[0] != observations.size or observations.size != sigmas.size:
            raise ValueError("A, l and sigma dimensions are inconsistent.")
        if not np.all(np.isfinite(design)):
            raise ValueError("Design matrix values must be finite.")
        if not np.all(np.isfinite(observations)):
            raise ValueError("Reduced observation values must be finite.")
        if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            raise ValueError("Observation sigmas must be positive and finite.")
        reference = self.x0
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            weights = 1.0 / sigmas**2
            normal_increment = design.T @ (weights[:, None] * design)
            correction_rhs_increment = design.T @ (weights * observations)
            correction_lpl_increment = float(np.dot(weights, observations**2))
        matrix = self.N + normal_increment
        rhs = self.W + correction_rhs_increment + normal_increment @ reference
        next_lpl = self.lPl + correction_lpl_increment
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)) or not np.isfinite(next_lpl):
            raise FloatingPointError("Dense normal-equation accumulation produced non-finite values.")
        self.N[:] = matrix
        self.W[:] = rhs
        self.lPl = float(next_lpl)
        self.obs_count += observations.size

    def _assert_compatible(self, other: "NormalEquations") -> None:
        left = self.meta.get("compatibility")
        right = other.meta.get("compatibility")
        if left != right:
            raise ValueError("Normal equations have incompatible scientific conventions.")
        left_units = dict(zip(self.parameter_names, self.parameter_units))
        for name, unit in zip(other.parameter_names, other.parameter_units):
            if name in left_units and left_units[name] != unit:
                raise ValueError(f"Parameter {name} has incompatible units {left_units[name]!r} and {unit!r}.")

    def add(self, other: "NormalEquations") -> "NormalEquations":
        if not isinstance(other, NormalEquations):
            raise TypeError("Can only add another NormalEquations object.")
        self._assert_compatible(other)
        union: List[ParameterName] = list(self.parameter_names)
        unit_by_name = dict(zip(self.parameter_names, self.parameter_units))
        index = {name: position for position, name in enumerate(union)}
        for name, unit in zip(other.parameter_names, other.parameter_units):
            if name not in index:
                index[name] = len(union)
                union.append(name)
                unit_by_name[name] = unit
        count = len(union)
        matrix = np.zeros((count, count), dtype=float)
        rhs = np.zeros(count, dtype=float)
        self_x0 = self.x0
        other_x0 = other.x0
        x0_by_name = dict(zip(self.parameter_names, self_x0))
        for name, value in zip(other.parameter_names, other_x0):
            # In an absolute system, x0 is a reported reference vector, not
            # part of the equation itself.  Retain the left-hand reference for
            # shared columns and use the right-hand reference only for new
            # columns, yielding a deterministic valid reference for the sum.
            x0_by_name.setdefault(name, value)

        combined_lpl = 0.0

        def scatter(source: "NormalEquations") -> None:
            nonlocal combined_lpl
            indices = np.array([index[name] for name in source.parameter_names], dtype=int)
            matrix[np.ix_(indices, indices)] += source.N
            rhs[indices] += source.W
            target_reference = np.asarray([x0_by_name[name] for name in source.parameter_names], dtype=float)
            reference_shift = source.x0 - target_reference
            correction_rhs = source.W - source.N @ source.x0
            combined_lpl += float(
                source.lPl
                + 2.0 * (reference_shift @ correction_rhs)
                + reference_shift @ source.N @ reference_shift
            )

        scatter(self)
        scatter(other)
        return NormalEquations(
            parameter_names=union,
            parameter_units=[unit_by_name[name] for name in union],
            N=matrix,
            W=rhs,
            lPl=combined_lpl,
            obs_count=self.obs_count + other.obs_count,
            meta={**other.meta, **self.meta},
            x0=np.asarray([x0_by_name[name] for name in union], dtype=float),
        )

__all__ = ["NormalEquations", "SparseNormalRow"]
