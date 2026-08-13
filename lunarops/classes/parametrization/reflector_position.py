"""Parametrization: lunar reflector PA-frame coordinates (3 per reflector)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunarops.base.array_validation import parameter_vector
from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.observation.resolver import ObservationCatalogState
from lunarops.config.registry import register
from lunarops.config.schema import ConfigSchema, sequence

from .base import Parametrization

_AXES = ("x", "y", "z")


@register(
    "parametrization",
    "reflectorPosition",
    schema=ConfigSchema(
        fields=(sequence("reflectors", item_kind="string", min_items=1, non_empty=True),),
        type_name="reflectorPosition",
    ),
)
class ReflectorPositionParametrization(Parametrization):
    """Estimate corrections to reflector moon-fixed (PA) coordinates.

    Options
    -------
    reflectors :
        Optional explicit list of reflector catalog keys to estimate; default
        is every reflector present in the observation set.
    The explicit :class:`ObservationCatalogState` supplied to :meth:`setup`
    owns the coordinates updated between nonlinear iterations.
    """

    def __init__(self, *, reflectors: Sequence[str] | None = None) -> None:
        if isinstance(reflectors, (str, bytes)):
            raise TypeError("reflectorPosition reflectors must be a sequence of strings.")
        if reflectors is not None and not isinstance(reflectors, Sequence):
            raise TypeError("reflectorPosition reflectors must be a sequence of strings.")
        if reflectors is None:
            requested = None
        else:
            requested = []
            for reflector in reflectors:
                if not isinstance(reflector, str):
                    raise TypeError("reflectorPosition reflectors must contain only strings.")
                key = reflector.strip()
                if not key:
                    raise ValueError("reflectorPosition reflector keys must not be empty.")
                if key in requested:
                    raise ValueError(f"reflectorPosition reflector key {key!r} was provided more than once.")
                requested.append(key)
            if not requested:
                raise ValueError(
                    "reflectorPosition reflectors must not be empty; omit it to estimate all observed keys."
                )
        self.requested = requested
        self.keys: list[str] = []
        self._index_by_key: dict[str, int] = {}
        self._names: list[ParameterName] = []
        self._model_state: ObservationCatalogState | None = None

    @classmethod
    def from_config(cls, config: dict, context) -> ReflectorPositionParametrization:
        return cls(reflectors=config.get("reflectors"))

    def setup(
        self,
        equations: Sequence[ObservationEquation],
        model_state: ObservationCatalogState,
    ) -> None:
        if not isinstance(model_state, ObservationCatalogState):
            raise TypeError("reflectorPosition requires an ObservationCatalogState.")
        self._model_state = model_state
        catalog = model_state.reflector_catalog
        observed = sorted({eq.reflector_key for eq in equations})
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
        self._index_by_key = {key: index for index, key in enumerate(self.keys)}
        self._names = [ParameterName(key, f"position.{axis}") for key in self.keys for axis in _AXES]

    def parameter_names(self) -> list[ParameterName]:
        return list(self._names)

    def reference_values(self) -> np.ndarray:
        if self._model_state is None:
            raise RuntimeError("reflectorPosition has not been set up.")
        return np.asarray(
            [
                float(np.asarray(self._model_state.reflector_catalog[key].moon_fixed_xyz_m, dtype=float)[axis])
                for key in self.keys
                for axis in range(3)
            ],
            dtype=float,
        )

    def _partial_block(self, eq: ObservationEquation) -> tuple[int | None, np.ndarray | None]:
        j = self._index_by_key.get(eq.reflector_key)
        if j is None:
            return None, None
        block = eq.design_partials.get("reflector_position_pa")
        if block is None:
            raise KeyError(
                "Observation equation lacks the 'reflector_position_pa' partial "
                "block; run the forward model with "
                "include_reflector_position_partials=True "
                "(or config includeReflectorDesign=true)."
            )
        return j, np.asarray(block, dtype=float).reshape(3)

    def design_columns(self, eq: ObservationEquation) -> np.ndarray:
        cols = np.zeros(3 * len(self.keys), dtype=float)
        j, block = self._partial_block(eq)
        if j is not None and block is not None:
            cols[3 * j : 3 * j + 3] = block
        return cols

    def design_entries(self, eq: ObservationEquation) -> list[tuple[int, float]]:
        j, block = self._partial_block(eq)
        if j is None or block is None:
            return []
        start = 3 * j
        return [(start + axis, float(value)) for axis, value in enumerate(block) if float(value)]

    def apply_update(self, delta: np.ndarray) -> None:
        delta = parameter_vector(delta, expected_size=3 * len(self.keys), name="reflectorPosition update")
        if self._model_state is None:
            raise RuntimeError("reflectorPosition has not been set up.")
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
        if self._model_state is None:
            return {}
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
