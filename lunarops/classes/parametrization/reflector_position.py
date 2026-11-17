"""Parametrization of lunar reflector coordinates in the Moon principal-axis frame."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.observation.resolver import ObservationCatalogState
from lunarops.config.schema import ConfigSchema, sequence

from .base import Parametrization
from .catalog_position import AXES, catalog_snapshot, float_list, largest_position_update, select_catalog_keys


class ReflectorPositionParametrization(Parametrization):
    """Lunar reflector coordinates in the Moon principal-axis frame, three parameters per reflector.

    The parameters are the PA coordinates x, y, z in metres, named ``<reflector>:position.<axis>``. Updates are
    written to the shared reflector catalog, so the forward model is re-evaluated with them.

    Parameters
    ----------
    model_state : ObservationCatalogState
        Shared catalogs holding the reflector positions.
    reflectors : Sequence[str], optional
        Reflector keys to estimate. ``None`` estimates every observed reflector present in the catalog.
    """

    TYPE = "reflectorPosition"
    SCHEMA = ConfigSchema(
        fields=(
            sequence(
                "reflectors",
                item_kind="string",
                unique=True,
                min_items=1,
                non_empty=True,
                description="Reflector catalog keys to estimate. Defaults to every observed reflector in the catalog.",
            ),
        ),
        type_name="reflectorPosition",
    )

    def __init__(self, *, model_state: ObservationCatalogState, reflectors: Sequence[str] | None = None) -> None:
        requested = None if reflectors is None else list(reflectors)
        self.requested = requested
        self.keys: list[str] = []
        self.names: list[ParameterName] = []
        self._model_state = model_state
        # A-priori catalog coordinates, before any estimate or restored state changes the shared catalog.
        self._apriori = catalog_snapshot(model_state.reflector_catalog, "moon_fixed_xyz_m")

    @classmethod
    def from_config(cls, config: dict, context, *, model_state) -> ReflectorPositionParametrization:
        """Create the parametrization from a validated configuration."""
        return cls(model_state=model_state, reflectors=config.get("reflectors"))

    def setup(self, batch: EquationBatch) -> list[ParameterName]:
        """Select the estimated reflectors and return their parameter names.

        Raises
        ------
        KeyError
            If a requested reflector is not in the catalog.
        ValueError
            If a requested reflector has no observations in ``batch``.
        """
        catalog = self._model_state.reflector_catalog
        observed = sorted(set(batch.reflector.tolist()))
        self.keys = select_catalog_keys(self.requested, observed, catalog, block=self.TYPE, noun="reflector")
        self.names = [ParameterName(key, f"position.{axis}") for key in self.keys for axis in AXES]

        return self.names

    def x0(self) -> np.ndarray:
        """Return the current catalog PA coordinates of the estimated reflectors, in metres."""
        catalog = self._model_state.reflector_catalog
        return np.asarray(
            [value for key in self.keys for value in float_list(catalog[key].moon_fixed_xyz_m)], dtype=float
        )

    def apriori(self) -> np.ndarray:
        """Return the a-priori catalog PA coordinates of the estimated reflectors, in metres."""
        return np.concatenate([self._apriori[key][0] for key in self.keys]) if self.keys else np.zeros(0)

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        """Add the ``reflector_position_pa`` partials (m/m) of each reflector's rows to its three columns."""
        for j, key in enumerate(self.keys):
            rows = batch.reflector == key
            out[rows, 3 * j : 3 * j + 3] += batch.reflector_position_pa[rows]

    def update(self, delta: np.ndarray) -> None:
        """Add the PA coordinate corrections (metres) to the reflector positions in the shared catalog."""
        delta = np.asarray(delta, float)
        positions = {}
        for j, key in enumerate(self.keys):
            record = self._model_state.reflector_catalog[key]
            positions[key] = np.asarray(record.moon_fixed_xyz_m, dtype=float) + delta[3 * j : 3 * j + 3]
        self._model_state.apply_reflector_positions_pa_m(positions)

    def max_update_norm(self, delta: np.ndarray) -> float:
        """Return the largest 3-D position correction of any reflector, in metres."""
        return largest_position_update(delta, len(self.keys))

    def state(self) -> dict[str, object]:
        """Return the current PA position ``[x, y, z]`` in metres per estimated reflector."""
        catalog = self._model_state.reflector_catalog
        return {key: float_list(catalog[key].moon_fixed_xyz_m) for key in self.keys}
