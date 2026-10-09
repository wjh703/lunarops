"""Array-oriented epoch and force-model data records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from lunarops.base.array_validation import rotation_matrix, state_matrix
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.time import Epoch

from .forces import PointMassGravityCache

StateHistoryProvider = Callable[[Sequence[str], Epoch], np.ndarray]


@dataclass(frozen=True, slots=True)
class DynamicsEpochData:
    """Ephemeris, rotation matrices, and fixed force data at one TDB epoch.

    earth_fixed2inertial_matrix rotates Earth gravity axes to inertial axes.
    The dynamics orientation model need not be ITRF.
    moon_fixed2inertial_matrix rotates lunar fixed (principal) axes to
    inertial LCRS axes: v_inertial = moon_fixed2inertial_matrix @ v_fixed.
    """

    epoch_tdb: Epoch
    body_names: tuple[str, ...]
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    earth_fixed2inertial_matrix: np.ndarray
    moon_fixed2inertial_matrix: np.ndarray
    ephemeris_earth_acceleration_mps2: np.ndarray
    point_mass_gravity_cache: PointMassGravityCache | None = None

    def __post_init__(self) -> None:
        names = tuple(body_name(name) for name in self.body_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("DynamicsEpochData body names must be unique and non-empty")
        count = len(names)
        positions = state_matrix(self.positions_m, rows=count, name="positions_m")
        velocities = state_matrix(self.velocities_mps, rows=count, name="velocities_mps")
        positions.setflags(write=False)
        velocities.setflags(write=False)
        object.__setattr__(self, "body_names", names)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "velocities_mps", velocities)
        for name in ("earth_fixed2inertial_matrix", "moon_fixed2inertial_matrix"):
            object.__setattr__(self, name, rotation_matrix(getattr(self, name), name=name))
        earth_acceleration = np.asarray(self.ephemeris_earth_acceleration_mps2, dtype=float)
        if earth_acceleration.shape != (3,) or not np.all(np.isfinite(earth_acceleration)):
            raise ValueError("ephemeris_earth_acceleration_mps2 must be a finite three-vector")
        earth_acceleration = np.ascontiguousarray(earth_acceleration)
        earth_acceleration.setflags(write=False)
        object.__setattr__(self, "ephemeris_earth_acceleration_mps2", earth_acceleration)


class ForceEvaluationContext:
    """Mutable single-evaluation context used by lunar force models."""

    def __init__(self, body_names: Sequence[str], gravitational_parameters_m3_s2) -> None:
        names = tuple(body_name(name) for name in body_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("ForceEvaluationContext body names must be unique and non-empty")
        gm = np.asarray(gravitational_parameters_m3_s2, dtype=float)
        if gm.shape != (len(names),) or not np.all(np.isfinite(gm)) or np.any(gm <= 0):
            raise ValueError("gravitational_parameters_m3_s2 must contain one positive value per body")
        self.body_names = names
        self.body_indices = MappingProxyType({name: index for index, name in enumerate(names)})
        self.gravitational_parameters_m3_s2 = np.ascontiguousarray(gm)
        self.gravitational_parameters_m3_s2.setflags(write=False)
        self.positions_m = np.empty((len(names), 3), dtype=float)
        self.velocities_mps = np.empty_like(self.positions_m)
        self.history: StateHistoryProvider | None = None
        self.evaluation_cache: dict[object, object] | None = None
        self.epoch_data: DynamicsEpochData | None = None

    def load_epoch_data(self, epoch_data: DynamicsEpochData) -> None:
        if epoch_data.body_names != self.body_names:
            raise ValueError("DynamicsEpochData body order does not match ForceEvaluationContext")
        if epoch_data is not self.epoch_data:
            np.copyto(self.positions_m, epoch_data.positions_m)
            np.copyto(self.velocities_mps, epoch_data.velocities_mps)
            self.epoch_data = epoch_data

    def load_relative_state(self, relative_state) -> None:
        state = np.asarray(relative_state, dtype=float)
        self.positions_m[1] = self.positions_m[0] + state[:3]
        self.velocities_mps[1] = self.velocities_mps[0] + state[3:6]

    def load_evaluation(self, epoch_data, relative_state, history, evaluation_cache):
        self.load_epoch_data(epoch_data)
        self.load_relative_state(relative_state)
        self.history = history
        self.evaluation_cache = evaluation_cache
        return self


@dataclass(frozen=True, slots=True)
class AccelerationComponents:
    """Diagnostic total and per-force acceleration arrays."""

    body_names: tuple[str, ...]
    total_accelerations_mps2: np.ndarray
    accelerations_by_force_mps2: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        count = len(self.body_names)
        accelerations = state_matrix(
            self.total_accelerations_mps2,
            rows=count,
            name="total_accelerations_mps2",
        )
        accelerations.setflags(write=False)
        object.__setattr__(self, "total_accelerations_mps2", accelerations)
        terms: dict[str, np.ndarray] = {}
        for name, value in self.accelerations_by_force_mps2.items():
            item = state_matrix(value, rows=count, name=f"accelerations_by_force_mps2[{name!r}]")
            item.setflags(write=False)
            terms[str(name)] = item
        object.__setattr__(self, "accelerations_by_force_mps2", MappingProxyType(terms))
