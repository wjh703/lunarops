"""Parametrization: lunar reflector PA-frame coordinates (3 per reflector)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.observation.resolver import ObservationCatalogState
from lunarops.config.schema import ConfigSchema, sequence

from .base import Parametrization

_AXES = ("x", "y", "z")


class ReflectorPositionParametrization(Parametrization):
    TYPE = "reflectorPosition"
    SCHEMA = ConfigSchema(
        fields=(sequence("reflectors", item_kind="string", unique=True, min_items=1, non_empty=True),),
        type_name="reflectorPosition",
    )

    def __init__(self, *, model_state: ObservationCatalogState, reflectors: Sequence[str] | None = None) -> None:
        requested = None if reflectors is None else list(reflectors)
        self.requested = requested
        self.keys: list[str] = []
        self.names: list[ParameterName] = []
        self._model_state = model_state

    @classmethod
    def from_config(cls, config: dict, context, *, model_state) -> ReflectorPositionParametrization:
        return cls(model_state=model_state, reflectors=config.get("reflectors"))

    def setup(self, batch: EquationBatch) -> list[ParameterName]:
        catalog = self._model_state.reflector_catalog
        observed = sorted(set(batch.reflector.tolist()))
        if self.requested is None:
            self.keys = [key for key in observed if key in catalog]
        else:
            requested = set(self.requested)
            missing = requested - set(catalog)
            if missing:
                raise KeyError(f"reflectorPosition: unknown reflector key(s) {sorted(missing)}")
            unobserved = requested - set(observed)
            if unobserved:
                raise ValueError(f"reflectorPosition requested reflector(s) have no observations: {sorted(unobserved)}")
            self.keys = list(self.requested)
        self.names = [ParameterName(key, f"position.{axis}") for key in self.keys for axis in _AXES]

        return self.names

    def x0(self) -> np.ndarray:
        return np.asarray(
            [
                float(np.asarray(self._model_state.reflector_catalog[key].moon_fixed_xyz_m, dtype=float)[axis])
                for key in self.keys
                for axis in range(3)
            ],
            dtype=float,
        )

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        for j, key in enumerate(self.keys):
            rows = batch.reflector == key
            out[rows, 3 * j : 3 * j + 3] += batch.reflector_position_pa[rows]

    def reduce(self, batch: EquationBatch) -> np.ndarray:
        return np.zeros(len(batch))

    def update(self, delta: np.ndarray) -> None:
        delta = np.asarray(delta, float)
        positions = {}
        for j, key in enumerate(self.keys):
            record = self._model_state.reflector_catalog[key]
            positions[key] = np.asarray(record.moon_fixed_xyz_m, dtype=float) + delta[3 * j : 3 * j + 3]
        self._model_state.apply_reflector_positions_pa_m(positions)

    def max_update_norm(self, delta: np.ndarray) -> float:
        if not len(delta):
            return 0.0
        return max(float(np.linalg.norm(delta[3 * j : 3 * j + 3])) for j in range(len(self.keys)))

    def state(self) -> dict[str, object]:
        return {
            key: [
                float(value)
                for value in np.asarray(
                    self._model_state.reflector_catalog[key].moon_fixed_xyz_m,
                    dtype=float,
                )
            ]
            for key in self.keys
        }
