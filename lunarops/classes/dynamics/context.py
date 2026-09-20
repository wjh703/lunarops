"""Array-oriented epoch and force-model data records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.time import Epoch

HistoryProvider = Callable[[Sequence[str], Epoch], np.ndarray]


def _state_matrix(value, *, rows: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (rows, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite ({rows},3) array")
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class PreparedEpoch:
    """Time-dependent data shared by all state channels at one epoch."""

    epoch_tdb: Epoch
    body_names: tuple[str, ...]
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    frames: Mapping[str, np.ndarray]
    force_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(body_name(name) for name in self.body_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("PreparedEpoch body names must be unique and non-empty")
        count = len(names)
        positions = _state_matrix(self.positions_m, rows=count, name="positions_m")
        velocities = _state_matrix(self.velocities_mps, rows=count, name="velocities_mps")
        positions.setflags(write=False)
        velocities.setflags(write=False)
        frames = {body_name(name): np.asarray(matrix, dtype=float) for name, matrix in self.frames.items()}
        for name, matrix in frames.items():
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"Frame {name!r} must be a finite 3x3 matrix")
            matrix = np.ascontiguousarray(matrix)
            matrix.setflags(write=False)
            frames[name] = matrix
        object.__setattr__(self, "body_names", names)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "velocities_mps", velocities)
        object.__setattr__(self, "frames", MappingProxyType(frames))
        object.__setattr__(self, "force_data", MappingProxyType(dict(self.force_data)))


class DynamicsContext:
    """Reusable numeric workspace bound to a :class:`PreparedEpoch`."""

    def __init__(self, body_names: Sequence[str], gravitational_parameters_m3_s2) -> None:
        names = tuple(body_name(name) for name in body_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("DynamicsContext body names must be unique and non-empty")
        gm = np.asarray(gravitational_parameters_m3_s2, dtype=float)
        if gm.shape != (len(names),) or not np.all(np.isfinite(gm)) or np.any(gm <= 0):
            raise ValueError("gravitational_parameters_m3_s2 must contain one positive value per body")
        self.body_names = names
        self.body_indices = MappingProxyType({name: index for index, name in enumerate(names)})
        self.gravitational_parameters_m3_s2 = np.ascontiguousarray(gm)
        self.gravitational_parameters_m3_s2.setflags(write=False)
        self.positions_m = np.empty((len(names), 3), dtype=float)
        self.velocities_mps = np.empty_like(self.positions_m)
        self.epoch_tdb: Epoch | None = None
        self.frames: Mapping[str, np.ndarray] = MappingProxyType({})
        self.history: HistoryProvider | None = None
        self.force_data: Mapping[str, Any] = MappingProxyType({})
        self._prepared: PreparedEpoch | None = None

    def bind_prepared(self, prepared: PreparedEpoch) -> None:
        if prepared.body_names != self.body_names:
            raise ValueError("PreparedEpoch body order does not match DynamicsContext")
        if prepared is not self._prepared:
            np.copyto(self.positions_m, prepared.positions_m)
            np.copyto(self.velocities_mps, prepared.velocities_mps)
            self._prepared = prepared
        self.epoch_tdb = prepared.epoch_tdb
        self.frames = prepared.frames
        self.force_data = prepared.force_data

    def bind_history(self, history: HistoryProvider | None) -> None:
        self.history = history


@dataclass(frozen=True, slots=True)
class AccelerationBreakdown:
    """Diagnostic acceleration arrays produced outside the propagation fast path."""

    body_names: tuple[str, ...]
    accelerations_mps2: np.ndarray
    emb_mps2: np.ndarray
    relative_mps2: np.ndarray
    terms_mps2: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        count = len(self.body_names)
        accelerations = _state_matrix(self.accelerations_mps2, rows=count, name="accelerations_mps2")
        accelerations.setflags(write=False)
        object.__setattr__(self, "accelerations_mps2", accelerations)
        for name in ("emb_mps2", "relative_mps2"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite three-vector")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        terms: dict[str, np.ndarray] = {}
        for name, value in self.terms_mps2.items():
            item = _state_matrix(value, rows=count, name=f"terms_mps2[{name!r}]")
            item.setflags(write=False)
            terms[str(name)] = item
        object.__setattr__(self, "terms_mps2", MappingProxyType(terms))


@dataclass(frozen=True, slots=True)
class AccelerationBatch:
    """Body, EMB, and relative accelerations for channels sharing one epoch."""

    body_names: tuple[str, ...]
    accelerations_mps2: np.ndarray
    emb_mps2: np.ndarray
    relative_mps2: np.ndarray

    def __post_init__(self) -> None:
        body = np.asarray(self.accelerations_mps2, dtype=float)
        emb = np.asarray(self.emb_mps2, dtype=float)
        relative = np.asarray(self.relative_mps2, dtype=float)
        channels = body.shape[0] if body.ndim == 3 else -1
        if (
            body.shape != (channels, len(self.body_names), 3)
            or emb.shape != (channels, 3)
            or relative.shape != (channels, 3)
        ):
            raise ValueError("Invalid acceleration batch shapes")
        if not np.all(np.isfinite(body)) or not np.all(np.isfinite(emb)) or not np.all(np.isfinite(relative)):
            raise ValueError("Acceleration batch values must be finite")
        object.__setattr__(self, "accelerations_mps2", np.ascontiguousarray(body))
        object.__setattr__(self, "emb_mps2", np.ascontiguousarray(emb))
        object.__setattr__(self, "relative_mps2", np.ascontiguousarray(relative))
