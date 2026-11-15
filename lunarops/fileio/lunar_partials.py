"""Read and write ``lunarDynamicsPartials`` tables: partials of the lunar integration state.

A ``lunarDynamicsPartials`` artifact is a ``LUNARTBL`` numeric table (see
:mod:`lunarops.fileio.numeric_table`) with one row per epoch, in increasing time. Its columns are

1. ``offset_tdb_s``: offset from the initial epoch, in TDB seconds;
2. the ``d`` components of the integration state at that epoch, named by :data:`STATE_NAMES` (the first
   ``d`` of them: 6 for the orbit, 12 with the mantle rotation, 15 with a fluid core);
3. the ``d x q`` partials ``∂y_i/∂p_j``, row-major (``i`` slowest), named ``d_<state>_d_<quantity>``.

The quantities are lunar dynamics parameter names (see :mod:`lunarops.classes.dynamics.parameters`): the
initial values of the state, then dynamical parameters. A partial has the unit of its state component divided
by the unit of its quantity.

The header metadata holds ``timeScale`` (``TDB``), ``initialTdbJd1`` and ``initialTdbJd2`` (the epoch of
offset 0), ``stateNames``, ``stateUnits``, ``quantityNames``, ``quantityUnits`` and ``producer`` (how the
partials were computed, for example ``variational`` or ``finiteDifference``), and any further keys of the
producer.

Attributes
----------
ARTIFACT_TYPE : str
    Artifact type of the table.
STATE_NAMES : tuple of str
    Names of the 15 integration state components, in state order.
STATE_UNITS : tuple of str
    Units of the state components.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .numeric_table import read_numeric_table, write_numeric_table

ARTIFACT_TYPE = "lunarDynamicsPartials"

STATE_NAMES = (
    "r_x",
    "r_y",
    "r_z",
    "v_x",
    "v_y",
    "v_z",
    "phi",
    "theta",
    "psi",
    "wm_x",
    "wm_y",
    "wm_z",
    "wc_x",
    "wc_y",
    "wc_z",
)
STATE_UNITS = ("m",) * 3 + ("m/s",) * 3 + ("rad",) * 3 + ("rad/s",) * 6

_RESERVED_METADATA = frozenset(
    (
        "timeScale",
        "initialTdbJd1",
        "initialTdbJd2",
        "stateNames",
        "stateUnits",
        "quantityNames",
        "quantityUnits",
        "producer",
    )
)


def partial_column_name(state: str, quantity: str) -> str:
    """Return the column name of the partial of a state component with respect to a quantity."""
    return f"d_{state}_d_{quantity}"


@dataclass(frozen=True, eq=False)
class LunarDynamicsPartials:
    """Partials of the lunar integration state at a set of epochs.

    Parameters
    ----------
    epoch_tdb_jd1, epoch_tdb_jd2 : float
        Two-part TDB Julian date of offset 0.
    offsets_s : array_like
        Offsets of the epochs from the initial epoch, in TDB seconds, strictly increasing, shape ``(n,)``.
    states : array_like
        Integration state at each epoch, shape ``(n, d)``.
    partials : array_like
        ``∂y_i/∂p_j`` at each epoch, shape ``(n, d, q)``.
    quantity_names : sequence of str
        Names of the ``q`` quantities, in column order.
    quantity_units : sequence of str
        Their units.
    producer : str
        How the partials were computed, for example ``"variational"``.
    metadata : Mapping, optional
        Further JSON-serialisable metadata of the producer; it must not repeat the keys of the layout.

    Attributes
    ----------
    state_names : tuple of str
        Names of the ``d`` state components, the first ``d`` of :data:`STATE_NAMES`.
    state_units : tuple of str
        Their units.

    Raises
    ------
    ValueError
        If the shapes do not agree, ``d`` is not 6, 12 or 15, the offsets do not increase, the quantity names
        repeat, a value is not finite, or ``metadata`` repeats a key of the layout.
    """

    epoch_tdb_jd1: float
    epoch_tdb_jd2: float
    offsets_s: np.ndarray
    states: np.ndarray
    partials: np.ndarray
    quantity_names: tuple[str, ...]
    quantity_units: tuple[str, ...]
    producer: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the layout and store read-only arrays."""
        offsets = np.array(self.offsets_s, dtype=float)
        states = np.array(self.states, dtype=float)
        partials = np.array(self.partials, dtype=float)
        names, units = tuple(map(str, self.quantity_names)), tuple(map(str, self.quantity_units))
        if offsets.ndim != 1 or states.ndim != 2 or partials.ndim != 3:
            raise ValueError("Partials need offsets (n,), states (n, d) and partials (n, d, q)")
        count, dimension = states.shape
        if len(offsets) != count or partials.shape[:2] != (count, dimension):
            raise ValueError("The offsets, states and partials of a partials table must have matching shapes")
        if dimension not in (6, 12, 15):
            raise ValueError(f"The state dimension of a partials table must be 6, 12 or 15, not {dimension}")
        if len(names) != partials.shape[2] or len(units) != len(names):
            raise ValueError("A partials table needs one quantity name and unit per partials column")
        if len(set(names)) != len(names) or not all(names):
            raise ValueError("The quantity names of a partials table must be unique and non-empty")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("The offsets of a partials table must increase")
        if not (np.all(np.isfinite(offsets)) and np.all(np.isfinite(states)) and np.all(np.isfinite(partials))):
            raise ValueError("The values of a partials table must be finite")
        if reserved := sorted(_RESERVED_METADATA & set(self.metadata)):
            raise ValueError(f"Partials metadata must not set the layout keys {reserved}")
        for array in (offsets, states, partials):
            array.setflags(write=False)
        object.__setattr__(self, "epoch_tdb_jd1", float(self.epoch_tdb_jd1))
        object.__setattr__(self, "epoch_tdb_jd2", float(self.epoch_tdb_jd2))
        object.__setattr__(self, "offsets_s", offsets)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "partials", partials)
        object.__setattr__(self, "quantity_names", names)
        object.__setattr__(self, "quantity_units", units)
        object.__setattr__(self, "producer", str(self.producer))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def dimension(self) -> int:
        """int: State dimension ``d``."""
        return self.states.shape[1]

    @property
    def state_names(self) -> tuple[str, ...]:
        """Tuple of str: Names of the state components."""
        return STATE_NAMES[: self.dimension]

    @property
    def state_units(self) -> tuple[str, ...]:
        """Tuple of str: Units of the state components."""
        return STATE_UNITS[: self.dimension]

    def column(self, quantity: str) -> np.ndarray:
        """Return the partials with respect to one quantity, shape ``(n, d)``.

        Raises
        ------
        KeyError
            If ``quantity`` is not one of :attr:`quantity_names`.
        """
        try:
            index = self.quantity_names.index(quantity)
        except ValueError:
            raise KeyError(f"The partials table has no quantity {quantity!r}") from None
        return self.partials[:, :, index]

    def columns(self) -> tuple[str, ...]:
        """Return the column names of the table."""
        return (
            "offset_tdb_s",
            *self.state_names,
            *(partial_column_name(state, quantity) for state in self.state_names for quantity in self.quantity_names),
        )


def write_lunar_dynamics_partials(path: str | Path, partials: LunarDynamicsPartials) -> Path:
    """Write a ``lunarDynamicsPartials`` numeric table atomically.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.dat`` or ``.dat.gz``.
    partials : LunarDynamicsPartials
        Table to write.

    Returns
    -------
    Path
        The written path.
    """
    count = len(partials.offsets_s)
    values = np.column_stack((partials.offsets_s, partials.states, partials.partials.reshape(count, -1)))
    metadata = {
        "timeScale": "TDB",
        "initialTdbJd1": partials.epoch_tdb_jd1,
        "initialTdbJd2": partials.epoch_tdb_jd2,
        "stateNames": list(partials.state_names),
        "stateUnits": list(partials.state_units),
        "quantityNames": list(partials.quantity_names),
        "quantityUnits": list(partials.quantity_units),
        "producer": partials.producer,
        **partials.metadata,
    }
    return write_numeric_table(path, ARTIFACT_TYPE, partials.columns(), values, metadata=metadata)


def read_lunar_dynamics_partials(path: str | Path) -> LunarDynamicsPartials:
    """Read and validate a ``lunarDynamicsPartials`` numeric table.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.dat`` or ``.dat.gz``.

    Returns
    -------
    LunarDynamicsPartials
        The table; keys of the header metadata beyond the layout keys are in ``metadata``.

    Raises
    ------
    ValueError
        If the artifact type is not ``lunarDynamicsPartials``, the time scale is not TDB, or the columns, state
        names or state units differ from the layout the metadata describes.
    KeyError
        If a layout key is missing from the metadata.
    """
    _, columns, values, metadata = read_numeric_table(path, ARTIFACT_TYPE)
    if metadata.get("timeScale") != "TDB":
        raise ValueError(f"lunarDynamicsPartials must be on the TDB time scale, not {metadata.get('timeScale')!r}")
    state_names = tuple(metadata["stateNames"])
    quantity_names = tuple(metadata["quantityNames"])
    dimension, quantities = len(state_names), len(quantity_names)
    if state_names != STATE_NAMES[:dimension] or tuple(metadata["stateUnits"]) != STATE_UNITS[:dimension]:
        raise ValueError("lunarDynamicsPartials state names or units differ from the integration state layout")
    if len(columns) != 1 + dimension * (1 + quantities):
        raise ValueError("lunarDynamicsPartials columns differ from the layout of its metadata")
    count = len(values)
    table = LunarDynamicsPartials(
        metadata["initialTdbJd1"],
        metadata["initialTdbJd2"],
        values[:, 0],
        values[:, 1 : 1 + dimension],
        values[:, 1 + dimension :].reshape(count, dimension, quantities),
        quantity_names,
        tuple(metadata["quantityUnits"]),
        metadata["producer"],
        {key: value for key, value in metadata.items() if key not in _RESERVED_METADATA},
    )
    if tuple(columns) != table.columns():
        raise ValueError("lunarDynamicsPartials columns differ from the layout of its metadata")
    return table


__all__ = [
    "ARTIFACT_TYPE",
    "STATE_NAMES",
    "STATE_UNITS",
    "LunarDynamicsPartials",
    "partial_column_name",
    "read_lunar_dynamics_partials",
    "write_lunar_dynamics_partials",
]
