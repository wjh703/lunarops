"""Array-oriented epoch and force-model data records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from lunarops.base.array_validation import state_matrix
from lunarops.classes.time import Epoch

from .deformation import LunarInertiaEvaluation
from .epoch_data import DynamicsEpochData
from .forces import PointMassGravityEvaluation

StateHistoryProvider = Callable[[Sequence[str], Epoch], np.ndarray]


class DynamicsEvaluationContext:
    """Current candidate state and caches shared by forces and torques."""

    def __init__(self, body_names: Sequence[str], gravitational_parameters_m3_s2: np.ndarray) -> None:
        names = tuple(body_names)
        self.body_names = names
        self.body_indices = MappingProxyType({name: index for index, name in enumerate(names)})
        self.earth_index = self.body_indices["EARTH"]
        self.moon_index = self.body_indices["MOON"]
        self.moon_targets = np.asarray([self.moon_index], dtype=np.intp)
        self.moon_targets.setflags(write=False)
        self.gravitational_parameters_m3_s2 = gravitational_parameters_m3_s2
        self.positions_m = np.empty((len(names), 3), dtype=float)
        self.velocities_mps = np.empty_like(self.positions_m)
        self.history: StateHistoryProvider | None = None
        self.step_payloads: Mapping[str, object] | None = None
        self.epoch_data: DynamicsEpochData
        self.newtonian_evaluation: PointMassGravityEvaluation | None = None
        self.moon_fixed2inertial_matrix = np.eye(3)
        self._figure_cache: dict = {}
        # id(field) -> coefficient array of this evaluation (degree 2 or J2(t) applied).
        self.coefficient_overrides: dict[int, np.ndarray] = {}
        self.lunar_inertia = None
        self._loaded_epoch_data: DynamicsEpochData | None = None

    def load_epoch_data(self, epoch_data: DynamicsEpochData) -> None:
        if epoch_data is not self._loaded_epoch_data:
            np.copyto(self.positions_m, epoch_data.positions_m)
            np.copyto(self.velocities_mps, epoch_data.velocities_mps)
            self.epoch_data = epoch_data
            self._loaded_epoch_data = epoch_data
            self.gravitational_parameters_m3_s2 = epoch_data.gravitational_parameters_m3_s2

    def load_relative_state(self, relative_state) -> None:
        state = np.asarray(relative_state, dtype=float)
        self.positions_m[self.moon_index] = self.positions_m[self.earth_index] + state[:3]
        self.velocities_mps[self.moon_index] = self.velocities_mps[self.earth_index] + state[3:6]

    def load_evaluation(self, epoch_data, relative_state, history, *, moon_matrix=None, step_payloads=None):
        self.load_epoch_data(epoch_data)
        self.load_relative_state(relative_state)
        self.history = history
        self.step_payloads = step_payloads
        self.newtonian_evaluation = None
        self.moon_fixed2inertial_matrix = epoch_data.moon_fixed2inertial_matrix if moon_matrix is None else moon_matrix
        # Corrector iterations at one epoch have different candidate states.
        self._figure_cache.clear()
        self.coefficient_overrides.clear()
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
            R = self.epoch_data.earth_fixed2inertial_matrix if source == self.earth_index else self.moon_fixed2inertial_matrix
            relative = (self.positions_m[missing] - self.positions_m[source]) @ R
            accelerations = field.nonspherical_accelerations(relative, self.coefficient_overrides.get(id(field)))
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


@dataclass(frozen=True, slots=True)
class StepData:
    """Candidate-independent data shared by every right-hand-side evaluation of one ABM step."""

    force_payloads: Mapping[str, object]
    lunar_inertia: LunarInertiaEvaluation | None = None
