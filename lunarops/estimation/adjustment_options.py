"""Validated numerical options for nonlinear LLR adjustment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Mapping, Optional

import numpy as np

from lunarops.estimation.variance_components import VarianceComponentDefinition
from lunarops.estimation.robust_weights import (
    DIRECT_REJECTION_MODEL,
    IGG3_MODEL,
    ROBUST_WEIGHT_MODELS,
)


# Published parameter uncertainties use three-sigma by project convention.
PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER = 3.0


@dataclass(frozen=True)
class LlrAdjustmentOptions:
    components: tuple[VarianceComponentDefinition, ...]
    prefit_gross_threshold_m: Optional[float] = 20.0
    prefit_gross_threshold_by_station_m: Optional[Mapping[str, Optional[float]]] = None
    maximum_linearizations: int = 20
    parameter_update_factor: float = 1.0
    uncertainty_floor_minimum_m: float = 0.0
    uncertainty_floor_group_median_fraction: float = 0.0
    update_tolerance_m: float = 1.0e-3
    update_tolerance_by_block_m: Optional[Mapping[str, float]] = None
    required_consecutive_converged_linearizations: int = 2
    maximum_stochastic_iterations: int = 8
    robust_model: str = IGG3_MODEL
    k0: float = 1.5
    k1: Optional[float] = None
    minimum_one_minus_leverage: float = 1.0e-8
    minimum_nonzero_robust_factor: float = 1.0e-12
    minimum_robust_factor_for_convergence: float = 1.0e-3
    minimum_mad_count: int = 10
    minimum_initial_scale: float = 1.0
    bias_weight_cap: float = 1.0e12
    bias_maximum_iterations: int = 30
    minimum_effective_redundancy: float = 20.0
    scale_log_tolerance: float = 2.5e-2
    robust_factor_change_tolerance: float = 2.0e-2
    robust_factor_change_quantile: float = 0.999
    active_set_change_tolerance: float = 1.0e-3
    minimum_variance_ratio_per_iteration: float = 0.25
    maximum_variance_ratio_per_iteration: float = 4.0

    def __post_init__(self) -> None:
        if not isinstance(self.robust_model, str):
            raise TypeError("robust_model must be a string.")
        if self.robust_model not in ROBUST_WEIGHT_MODELS:
            raise ValueError(f"robust_model must be one of {sorted(ROBUST_WEIGHT_MODELS)}, got {self.robust_model!r}.")
        if self.robust_model == IGG3_MODEL and self.k1 is None:
            object.__setattr__(self, "k1", 6.0)
        integer_fields = {
            "maximum_linearizations": self.maximum_linearizations,
            "maximum_stochastic_iterations": self.maximum_stochastic_iterations,
            "required_consecutive_converged_linearizations": (self.required_consecutive_converged_linearizations),
            "minimum_mad_count": self.minimum_mad_count,
            "bias_maximum_iterations": self.bias_maximum_iterations,
        }
        invalid_integer = next(
            (
                name
                for name, value in integer_fields.items()
                if isinstance(value, bool) or not isinstance(value, int) or value < 1
            ),
            None,
        )
        if invalid_integer is not None:
            raise ValueError(f"{invalid_integer} must be a positive integer.")
        numeric_fields = {
            "parameter_update_factor": self.parameter_update_factor,
            "uncertainty_floor_minimum_m": self.uncertainty_floor_minimum_m,
            "uncertainty_floor_group_median_fraction": (self.uncertainty_floor_group_median_fraction),
            "update_tolerance_m": self.update_tolerance_m,
            "k0": self.k0,
            "minimum_one_minus_leverage": self.minimum_one_minus_leverage,
            "minimum_nonzero_robust_factor": self.minimum_nonzero_robust_factor,
            "minimum_robust_factor_for_convergence": (self.minimum_robust_factor_for_convergence),
            "minimum_initial_scale": self.minimum_initial_scale,
            "bias_weight_cap": self.bias_weight_cap,
            "minimum_effective_redundancy": self.minimum_effective_redundancy,
            "scale_log_tolerance": self.scale_log_tolerance,
            "robust_factor_change_tolerance": (self.robust_factor_change_tolerance),
            "robust_factor_change_quantile": self.robust_factor_change_quantile,
            "active_set_change_tolerance": self.active_set_change_tolerance,
            "minimum_variance_ratio_per_iteration": (self.minimum_variance_ratio_per_iteration),
            "maximum_variance_ratio_per_iteration": (self.maximum_variance_ratio_per_iteration),
        }
        if self.k1 is not None:
            numeric_fields["k1"] = self.k1
        invalid_numeric_type = next(
            (name for name, value in numeric_fields.items() if isinstance(value, bool) or not isinstance(value, Real)),
            None,
        )
        if invalid_numeric_type is not None:
            raise TypeError(f"{invalid_numeric_type} must be a real number.")
        nonfinite = next(
            (name for name, value in numeric_fields.items() if not np.isfinite(float(value))),
            None,
        )
        if nonfinite is not None:
            raise ValueError(f"{nonfinite} must be finite.")
        if self.robust_model == IGG3_MODEL:
            if self.k1 is None or not 0.0 < self.k0 < self.k1:
                raise ValueError("IGGIII thresholds must satisfy 0 < k0 < k1.")
        elif self.k1 is not None:
            raise ValueError("directRejection uses k0 only; omit k1.")
        elif self.k0 <= 0.0:
            raise ValueError("Direct-rejection threshold k0 must be positive.")
        if not 0.0 < self.parameter_update_factor <= 1.0:
            raise ValueError("Parameter update factor must be in (0, 1].")
        if not self.components:
            raise ValueError("At least one variance component is required.")
        component_ids = [component.id for component in self.components]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("Variance-component IDs must be unique.")
        if self.prefit_gross_threshold_m is not None and (
            isinstance(self.prefit_gross_threshold_m, bool) or not isinstance(self.prefit_gross_threshold_m, Real)
        ):
            raise TypeError("Prefit gross threshold must be a real number or null.")
        if self.prefit_gross_threshold_m is not None and (
            not np.isfinite(self.prefit_gross_threshold_m) or self.prefit_gross_threshold_m < 0.0
        ):
            raise ValueError("Prefit gross threshold must be finite and non-negative.")
        for station, threshold in (self.prefit_gross_threshold_by_station_m or {}).items():
            if not str(station).strip():
                raise ValueError("Prefit station threshold keys must be non-empty.")
            if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, Real)):
                raise TypeError("Prefit station thresholds must be real numbers or null.")
            if threshold is not None and (not np.isfinite(threshold) or threshold < 0.0):
                raise ValueError("Prefit station thresholds must be finite and non-negative.")
        if self.uncertainty_floor_minimum_m < 0.0:
            raise ValueError("Uncertainty floor minimum must be non-negative.")
        if not 0.0 <= self.uncertainty_floor_group_median_fraction <= 1.0:
            raise ValueError("Uncertainty floor group-median fraction must be in [0, 1].")
        for block, tolerance in (self.update_tolerance_by_block_m or {}).items():
            if not str(block).strip():
                raise ValueError("Block tolerance keys must be non-empty.")
            if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
                raise TypeError("Block parameter tolerances must be real numbers.")
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("Block parameter tolerances must be finite and non-negative.")
        if self.update_tolerance_m < 0.0:
            raise ValueError("Parameter update tolerance must be non-negative.")
        if not 0.0 < self.robust_factor_change_quantile <= 1.0:
            raise ValueError("Robust factor change quantile must be in (0, 1].")
        if self.scale_log_tolerance < 0.0 or self.robust_factor_change_tolerance < 0.0:
            raise ValueError("Stochastic convergence tolerances must be non-negative.")
        if not 0.0 <= self.active_set_change_tolerance <= 1.0:
            raise ValueError("Active-set change tolerance must be in [0, 1].")
        if not 0.0 < self.minimum_one_minus_leverage <= 1.0:
            raise ValueError("Minimum one-minus-leverage must be in (0, 1].")
        if not 0.0 < self.minimum_nonzero_robust_factor < 1.0:
            raise ValueError("Active robust-factor threshold must be in (0, 1).")
        if not 0.0 <= self.minimum_robust_factor_for_convergence <= 1.0:
            raise ValueError("Robust-factor convergence floor must be in [0, 1].")
        if self.minimum_initial_scale <= 0.0:
            raise ValueError("Minimum initial scale must be positive.")
        if self.bias_weight_cap <= 0.0:
            raise ValueError("Bias weight cap must be positive.")
        if self.minimum_effective_redundancy < 0.0:
            raise ValueError("Minimum effective redundancy must be non-negative.")
        if not (0.0 < self.minimum_variance_ratio_per_iteration <= self.maximum_variance_ratio_per_iteration):
            raise ValueError("VCE variance-ratio limits must be positive and ordered.")

    def settings(self) -> dict[str, object]:
        """Return the validated numerical settings used by the solver."""
        values = asdict(self)
        values.pop("components")
        return values


__all__ = ["LlrAdjustmentOptions", "PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER"]
