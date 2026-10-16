"""Array-oriented epoch and force-model data records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from lunarops.base.array_validation import state_matrix
from lunarops.classes.time import Epoch

from .forces import PointMassGravityCache, PointMassGravityEvaluation

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
    gravitational_parameters_m3_s2: np.ndarray
    point_mass_gravity_cache: PointMassGravityCache | None = None


class DynamicsEvaluationContext:
    """Current candidate state and caches shared by forces and torques."""

    def __init__(self, body_names: Sequence[str], gravitational_parameters_m3_s2: np.ndarray) -> None:
        names = tuple(body_names)
        self.body_names = names
        self.body_indices = MappingProxyType({name: index for index, name in enumerate(names)})
        self.moon_index = self.body_indices["MOON"]
        self.moon_targets = np.asarray([self.moon_index], dtype=np.intp)
        self.moon_targets.setflags(write=False)
        self.gravitational_parameters_m3_s2 = gravitational_parameters_m3_s2
        self.positions_m = np.empty((len(names), 3), dtype=float)
        self.velocities_mps = np.empty_like(self.positions_m)
        self.history: StateHistoryProvider | None = None
        self.evaluation_cache: dict[object, object] | None = None
        self.epoch_data: DynamicsEpochData
        self.newtonian_evaluation: PointMassGravityEvaluation | None = None
        self.moon_fixed2inertial_matrix = np.eye(3)
        self._figure_cache: dict = {}
        self.lunar_inertia = None
        self._loaded_epoch_data = None

    def load_epoch_data(self, epoch_data: DynamicsEpochData) -> None:
        if epoch_data is not self._loaded_epoch_data:
            np.copyto(self.positions_m, epoch_data.positions_m)
            np.copyto(self.velocities_mps, epoch_data.velocities_mps)
            self.epoch_data = epoch_data
            self._loaded_epoch_data = epoch_data
            self.gravitational_parameters_m3_s2 = epoch_data.gravitational_parameters_m3_s2

    def load_relative_state(self, relative_state) -> None:
        state = np.asarray(relative_state, dtype=float)
        self.positions_m[1] = self.positions_m[0] + state[:3]
        self.velocities_mps[1] = self.velocities_mps[0] + state[3:6]

    def load_evaluation(self, epoch_data, relative_state, history, evaluation_cache, *, moon_matrix=None):
        self.load_epoch_data(epoch_data)
        self.load_relative_state(relative_state)
        self.history = history
        self.evaluation_cache = evaluation_cache
        self.newtonian_evaluation = None
        self.moon_fixed2inertial_matrix = epoch_data.moon_fixed2inertial_matrix if moon_matrix is None else moon_matrix
        # Corrector iterations at one epoch have different candidate states.
        self._figure_cache.clear()
        self.lunar_inertia = None
        return self

    def figure_accelerations(self, source, targets, field):
        """PA/fixed positions and direct non-spherical accelerations.

        Cache each target separately so a force's batch can serve individual
        torque models. Entries live for exactly one candidate evaluation.
        """
        cache = self._figure_cache.setdefault((source, id(field)), {})
        missing = [index for index in targets if index not in cache]
        if missing:
            R = self.epoch_data.earth_fixed2inertial_matrix if source == 0 else self.moon_fixed2inertial_matrix
            relative = (self.positions_m[missing] - self.positions_m[source]) @ R
            accelerations = field.nonspherical_accelerations(relative)
            cache.update({index: (r, a) for index, r, a in zip(missing, relative, accelerations, strict=True)})
        return tuple(np.asarray([cache[index][component] for index in targets]) for component in (0, 1))


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
