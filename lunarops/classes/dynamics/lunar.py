"""Stable contracts for a future lunar orbit/rotation propagator.

The current production ephemeris remains authoritative.  These interfaces
make the later propagator and variational-equation implementation explicit
without changing observation calculations yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from lunarops.base.array_validation import finite_array
from lunarops.classes.time import Epoch, TimeScale


@dataclass(frozen=True, slots=True)
class LunarDynamicState:
    epoch_tdb: Epoch
    position_bcrs_m: np.ndarray
    velocity_bcrs_mps: np.ndarray
    attitude_pa2lcrs: np.ndarray
    angular_velocity_lcrs_radps: np.ndarray

    def __post_init__(self) -> None:
        self.epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
        object.__setattr__(self, "position_bcrs_m", finite_array(self.position_bcrs_m, size=3, name="position_bcrs_m", copy=True, readonly=True))
        object.__setattr__(self, "velocity_bcrs_mps", finite_array(self.velocity_bcrs_mps, size=3, name="velocity_bcrs_mps", copy=True, readonly=True))
        object.__setattr__(self, "attitude_pa2lcrs", finite_array(self.attitude_pa2lcrs, shape=(3, 3), name="attitude_pa2lcrs", copy=True, readonly=True))
        object.__setattr__(self, "angular_velocity_lcrs_radps", finite_array(self.angular_velocity_lcrs_radps, size=3, name="angular_velocity_lcrs_radps", copy=True, readonly=True))


@dataclass(frozen=True, slots=True)
class LunarVariationalState:
    """State transition and parameter sensitivity blocks for future estimation."""

    state_transition: np.ndarray
    parameter_sensitivity: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_transition", finite_array(self.state_transition, name="state_transition", copy=True, readonly=True))
        object.__setattr__(self, "parameter_sensitivity", finite_array(self.parameter_sensitivity, name="parameter_sensitivity", copy=True, readonly=True))


class LunarDynamics(ABC):
    """Provider for propagated lunar translational and rotational state."""

    @abstractmethod
    def state(self, epoch_tdb: Epoch) -> LunarDynamicState:
        """Evaluate the propagated state at a TDB epoch."""

    def variational_state(self, epoch_tdb: Epoch) -> LunarVariationalState:
        raise NotImplementedError("This lunar dynamics provider has no variational equations yet.")

    def close(self) -> None:
        return


class UnsupportedLunarDynamics(LunarDynamics):
    """Explicit placeholder until a numerical propagator is implemented."""

    def state(self, epoch_tdb: Epoch) -> LunarDynamicState:
        raise NotImplementedError("Numerical lunar orbit and rotation dynamics are not implemented yet.")


__all__ = ["LunarDynamicState", "LunarDynamics", "LunarVariationalState", "UnsupportedLunarDynamics"]
