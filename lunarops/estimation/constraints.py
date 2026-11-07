"""Parameter constraints of the LLR adjustment, applied as weighted pseudo-observations.

A constraint row ``c`` with target ``t`` and sigma ``s`` adds the pseudo-observation ``c x = t`` with weight
``1 / s^2`` to the normal equations: ``N += c^T c / s^2`` and, around the linearization point ``x0``, the reduced
pseudo-observation ``t - c x0``. Pseudo-observations count as observations in the degrees of freedom of the
a-posteriori unit-weight sigma, and their weighted squared residuals enter it; they do not belong to a variance
component.

``aprioriSigma``
    Each parameter whose name matches one of the patterns (``fnmatch`` on ``object:type:temporal:interval``) is
    tied to its a-priori value with the given sigma, in the parameter's unit (soft constraint, stochastic prior).
``minimalConstraints``
    Minimum constraints of the station network to its a-priori coordinates, as in the realization of the ITRF
    (Altamimi et al. 2002, J Geophys Res 107(B10); Altamimi et al. 2016, J Geophys Res 121): of the 7-parameter
    Helmert transformation ``theta = (C^T C)^-1 C^T (X - X_apriori)`` between the estimated and the a-priori
    positions of the datum stations, the selected parameters (translation: no-net-translation, rotation:
    no-net-rotation, scale) are constrained to zero with the given sigma; the others stay free. Rotations and scale are
    expressed as displacements at ``EARTH_RADIUS_M``, so that one sigma in metres serves every condition. They
    define the datum without distorting the network, in place of fixing the coordinates of one station.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase

import numpy as np

from lunarops.base.parameter_name import ParameterName

#: Radius at which rotations and scale of the minimal constraints are expressed as displacements, in metres.
EARTH_RADIUS_M = 6_378_137.0
MINIMAL_CONSTRAINT_CONDITIONS = ("translation", "rotation", "scale")
_AXES = ("x", "y", "z")
_CONDITION_SIZE = {"translation": 3, "rotation": 3, "scale": 1}


@dataclass(frozen=True)
class AprioriSigmaConstraint:
    """Soft constraint of the matching parameters to their a-priori values.

    Attributes
    ----------
    parameters : tuple[str, ...]
        ``fnmatch`` patterns of the constrained parameter names.
    sigma : float
        A-priori sigma, in the unit of the parameters.
    """

    parameters: tuple[str, ...]
    sigma: float

    def __post_init__(self) -> None:
        if not self.parameters or not self.sigma > 0.0:
            raise ValueError("aprioriSigma needs parameter patterns and a positive sigma.")


@dataclass(frozen=True)
class MinimalConstraint:
    """Minimum constraints of the estimated station positions to their a-priori values.

    Attributes
    ----------
    stations : tuple[str, ...] or None
        Datum stations; ``None`` uses every station whose position is estimated.
    conditions : tuple[str, ...]
        Helmert parameters constrained to zero, from ``MINIMAL_CONSTRAINT_CONDITIONS``.
    sigma_m : float
        Sigma of each constrained Helmert parameter, in metres (rotations and scale at ``EARTH_RADIUS_M``).
    """

    stations: tuple[str, ...] | None
    conditions: tuple[str, ...]
    sigma_m: float

    def __post_init__(self) -> None:
        unknown = set(self.conditions) - set(MINIMAL_CONSTRAINT_CONDITIONS)
        if not self.conditions or unknown or len(set(self.conditions)) != len(self.conditions):
            raise ValueError(f"minimalConstraints conditions must be distinct items of {MINIMAL_CONSTRAINT_CONDITIONS}.")
        if not self.sigma_m > 0.0:
            raise ValueError("minimalConstraints needs a positive sigma.")


@dataclass(frozen=True)
class PseudoObservations:
    """Pseudo-observations ``rows x = targets`` with ``weights`` over the current parameters.

    Attributes
    ----------
    rows : ndarray, shape (k, m)
        Constraint rows, in pseudo-observation unit per parameter unit.
    targets : ndarray, shape (k,)
        Target values of ``rows x``.
    weights : ndarray, shape (k,)
        Weights ``1 / sigma^2``.
    labels : tuple[str, ...]
        Label of each pseudo-observation.
    """

    rows: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    labels: tuple[str, ...]

    def __len__(self) -> int:
        """Return the number of pseudo-observations."""
        return len(self.targets)

    def reduce(self, normal_matrix: np.ndarray, rhs: np.ndarray, lpl: float, x0: np.ndarray):
        """Add the pseudo-observations to the correction system ``N dx = rhs`` around ``x0``.

        Returns the new ``N``, ``rhs`` and ``l^T P l``.
        """
        if not len(self):
            return normal_matrix, rhs, lpl
        reduced = self.targets - self.rows @ x0
        weighted = self.weights[:, None] * self.rows
        return (
            normal_matrix + self.rows.T @ weighted,
            rhs + weighted.T @ reduced,
            lpl + float(np.sum(self.weights * reduced**2)),
        )

    def residuals(self, values: np.ndarray) -> np.ndarray:
        """Return ``rows x - targets`` at the parameter values ``values``."""
        return self.rows @ values - self.targets if len(self) else np.zeros(0)

    def records(self, values: np.ndarray, cofactor: np.ndarray, sigma0: float | None) -> list[dict[str, object]]:
        """Return per pseudo-observation its label, value ``rows x - targets``, a-priori and a-posteriori sigma."""
        scale = 1.0 if sigma0 is None else float(sigma0)
        formal = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", self.rows, cofactor, self.rows), 0.0)) if len(self) else []
        return [
            {"label": label, "value": float(value), "aprioriSigma": float(1.0 / np.sqrt(weight)),
             "aposterioriSigma": float(scale * sigma)}
            for label, value, weight, sigma in zip(self.labels, self.residuals(values), self.weights, formal, strict=True)
        ]


@dataclass(frozen=True)
class ParameterConstraints:
    """Constraints of an adjustment plan.

    Attributes
    ----------
    items : tuple[AprioriSigmaConstraint | MinimalConstraint, ...]
        Constraints, applied to the parameters each estimate has.
    """

    items: tuple[AprioriSigmaConstraint | MinimalConstraint, ...] = field(default=())

    def pseudo_observations(
        self,
        names: Sequence[ParameterName],
        apriori: np.ndarray,
        station_columns: Mapping[str, Sequence[int]] | None = None,
    ) -> PseudoObservations:
        """Return the pseudo-observations of every constraint over the parameters ``names``.

        Constraints without any of their parameters among ``names`` add nothing.

        Parameters
        ----------
        names : Sequence[ParameterName]
            Parameter names in column order.
        apriori : ndarray, shape (m,)
            A-priori parameter values.
        station_columns : Mapping[str, Sequence[int]], optional
            Columns of the X, Y, Z position of every estimated station, for the minimal constraints.

        Raises
        ------
        ValueError
            If a minimal constraint has fewer datum stations than it needs, or a configured datum station is not
            estimated.
        """
        size = len(names)
        rows: list[np.ndarray] = []
        targets: list[float] = []
        weights: list[float] = []
        labels: list[str] = []
        text = [str(name) for name in names]
        for item in self.items:
            if isinstance(item, AprioriSigmaConstraint):
                for j, name in enumerate(text):
                    if any(fnmatchcase(name, pattern) for pattern in item.parameters):
                        row = np.zeros(size)
                        row[j] = 1.0
                        rows.append(row)
                        targets.append(float(apriori[j]))
                        weights.append(1.0 / item.sigma**2)
                        labels.append(f"aprioriSigma:{name}")
            else:
                helmert = _minimal_constraint_rows(item, size, apriori, station_columns or {})
                if helmert is not None:
                    block, block_labels = helmert
                    rows.extend(block)
                    targets.extend(block @ apriori)
                    weights.extend([1.0 / item.sigma_m**2] * len(block_labels))
                    labels.extend(block_labels)
        if not rows:
            return PseudoObservations(np.zeros((0, size)), np.zeros(0), np.zeros(0), ())
        return PseudoObservations(np.vstack(rows), np.asarray(targets), np.asarray(weights), tuple(labels))


def _minimal_constraint_rows(
    item: MinimalConstraint, size: int, apriori: np.ndarray, station_columns: Mapping[str, Sequence[int]]
) -> tuple[np.ndarray, list[str]] | None:
    """Rows ``(C^T C)^-1 C^T`` of the constrained Helmert parameters over the datum-station position columns."""
    estimated = {station: list(columns) for station, columns in station_columns.items()}
    if not estimated:
        return None
    stations = sorted(estimated) if item.stations is None else list(item.stations)
    missing = [station for station in stations if station not in estimated]
    if missing:
        raise ValueError(f"minimalConstraints datum station(s) without estimated positions: {missing}")
    if len(stations) < 3:
        raise ValueError(f"minimalConstraints need at least three datum stations, got {len(stations)}.")
    # All seven Helmert parameters are estimated; only the selected ones are constrained, as in the ITRF.
    design_blocks = []
    for station in stations:
        position = apriori[estimated[station]]
        # Small rotation theta of the frame: dX = theta x X = -[X]x theta; theta and scale in metres at EARTH_RADIUS_M.
        design_blocks.append(np.hstack([np.eye(3), -_skew(position) / EARTH_RADIUS_M, position[:, None] / EARTH_RADIUS_M]))
    design = np.vstack(design_blocks)
    projector = np.linalg.solve(design.T @ design, design.T)
    offsets = {"translation": 0, "rotation": 3, "scale": 6}
    selected = [offsets[condition] + k for condition in item.conditions for k in range(_CONDITION_SIZE[condition])]
    rows = np.zeros((len(selected), size))
    for k, station in enumerate(stations):
        rows[:, estimated[station]] = projector[selected, 3 * k : 3 * k + 3]
    labels = []
    for condition in item.conditions:
        labels.extend([f"minimalConstraints:{condition}.{axis}" for axis in _AXES] if condition != "scale"
                      else ["minimalConstraints:scale"])
    return rows, labels


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


__all__ = [
    "EARTH_RADIUS_M",
    "MINIMAL_CONSTRAINT_CONDITIONS",
    "AprioriSigmaConstraint",
    "MinimalConstraint",
    "ParameterConstraints",
    "PseudoObservations",
]
