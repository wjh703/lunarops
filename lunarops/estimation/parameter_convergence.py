"""Convergence policies for nonlinear parameter updates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real


@dataclass(frozen=True)
class ParameterConvergenceEvaluation:
    converged: bool
    tolerances_m: dict[str, float]
    normalized_updates: dict[str, float]


@dataclass(frozen=True)
class ParameterConvergencePolicy:
    """Evaluate each parametrization block against its own metric tolerance."""

    default_tolerance_m: float
    tolerance_by_block_m: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.default_tolerance_m, bool) or not isinstance(self.default_tolerance_m, Real):
            raise TypeError("Default parameter tolerance must be a real number.")
        default_tolerance = float(self.default_tolerance_m)
        if not isfinite(default_tolerance) or default_tolerance < 0.0:
            raise ValueError("Default parameter tolerance must be finite and non-negative.")
        if not isinstance(self.tolerance_by_block_m, Mapping):
            raise TypeError("Block parameter tolerances must be a mapping.")
        block_tolerances: dict[str, float] = {}
        for raw_label, raw_value in self.tolerance_by_block_m.items():
            if not isinstance(raw_label, str) or not raw_label.strip():
                raise ValueError("Block parameter tolerance labels must be non-empty strings.")
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError("Block parameter tolerances must be real numbers.")
            label = raw_label.strip()
            value = float(raw_value)
            if not isfinite(value) or value < 0.0:
                raise ValueError("Block parameter tolerances must be finite and non-negative.")
            if label in block_tolerances:
                raise ValueError("Block parameter tolerance labels must be unique after trimming.")
            block_tolerances[label] = value
        object.__setattr__(self, "default_tolerance_m", default_tolerance)
        object.__setattr__(self, "tolerance_by_block_m", block_tolerances)

    def tolerance_for(self, label: str) -> float:
        return float(self.tolerance_by_block_m.get(label, self.default_tolerance_m))

    def evaluate(self, updates_m: Mapping[str, float]) -> ParameterConvergenceEvaluation:
        if not isinstance(updates_m, Mapping):
            raise TypeError("Parameter update norms must be a mapping.")
        updates: dict[str, float] = {}
        for raw_label, raw_value in updates_m.items():
            if not isinstance(raw_label, str) or not raw_label.strip():
                raise ValueError("Parameter update labels must be non-empty strings.")
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError("Parameter update norms must be real numbers.")
            label = raw_label.strip()
            value = float(raw_value)
            if not isfinite(value) or value < 0.0:
                raise ValueError("Parameter update norms must be finite and non-negative.")
            if label in updates:
                raise ValueError("Parameter update labels must be unique after trimming.")
            updates[label] = value
        tolerances = {label: self.tolerance_for(label) for label in updates}
        ratios = {
            label: (0.0 if value == 0.0 else float("inf"))
            if tolerances[label] == 0.0
            else float(value) / tolerances[label]
            for label, value in updates.items()
        }
        return ParameterConvergenceEvaluation(
            converged=all(ratio <= 1.0 for ratio in ratios.values()),
            tolerances_m=tolerances,
            normalized_updates=ratios,
        )


__all__ = ["ParameterConvergenceEvaluation", "ParameterConvergencePolicy"]
