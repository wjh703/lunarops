"""Parametrization base class (GROOPS ``parametrization`` analogue).

A parametrization owns a contiguous block of design-matrix columns.  It

* declares structured :class:`ParameterName`s,
* fills its columns of an observation-equation row from the equation's
  named partial blocks,
* maps solved corrections back into model state (catalogs, bias tables,
  force-model coefficients, integrator initial conditions, ...),
* reports its current values for output.

The estimator (:mod:`lunarops.estimation.adjustment_solver`) and the
normal-equation builder are generic over the parametrization list — adding EOP,
Love-number or orbit-state parameters never touches them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunarops.base.array_validation import parameter_vector
from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation


class Parametrization:
    _registry_type: str
    """One block of estimated parameters."""

    #: overridden by subclasses
    category = "parametrization"

    @property
    def block_id(self) -> str:
        return type(self)._registry_type

    def setup(self, equations: Sequence[ObservationEquation], model_state) -> None:
        """Inspect the dataset once before the first iteration (e.g. discover
        which reflectors / stations actually occur).  Default: no-op."""

    def parameter_names(self) -> list[ParameterName]:
        raise NotImplementedError

    def reference_values(self) -> np.ndarray:
        """Return the current values of this block in ``parameter_names`` order.

        These values are the ``x0`` linearization reference written with a
        persisted normal-equation system.  Blocks whose natural reference is
        non-zero should override this method; zero is the GROOPS-inspired default.
        """
        return np.zeros(self.parameter_count, dtype=float)

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names())

    def design_entries(self, eq: ObservationEquation) -> list[tuple[int, float]]:
        """Return non-zero local design entries as ``(column, value)`` pairs."""
        raise NotImplementedError

    def reduce_observation(self, eq: ObservationEquation) -> float:
        """Amount to subtract from ``eq.observed_minus_computed_one_way_m`` for the *current* parameter
        values (linearization point), e.g. the currently accumulated station
        bias.  Default 0."""
        return 0.0

    def apply_update(self, delta: np.ndarray) -> None:
        """Absorb solved corrections (same order as :meth:`parameter_names`)."""
        raise NotImplementedError

    def max_update_norm(self, delta: np.ndarray) -> float:
        """Convergence metric for this block; default max |delta_i|."""
        return float(np.max(np.abs(delta))) if len(delta) else 0.0

    def state(self) -> dict[str, object]:
        """Current parameter values for reporting."""
        return {}

class ParametrizationList:
    """Concatenation of parametrization blocks into one design matrix.

    The block layout is cached after :meth:`setup`, so streaming normal-equation
    accumulation does not repeatedly rebuild parameter-name lists or search for
    block slice boundaries row by row.
    """

    def __init__(
        self,
        blocks: Sequence[Parametrization],
        *,
        reduction_blocks: Sequence[Parametrization] | None = None,
    ) -> None:
        normalized = tuple(blocks)
        self._blocks: tuple[Parametrization, ...] = normalized
        reductions = normalized if reduction_blocks is None else tuple(reduction_blocks)
        if any(block not in reductions for block in normalized):
            raise ValueError("Every estimated parametrization block must also reduce the observations.")
        self._reduction_blocks: tuple[Parametrization, ...] = reductions
        self._parameter_names: list[ParameterName] | None = None
        self._slices: list[slice] = []

    @property
    def blocks(self) -> tuple[Parametrization, ...]:
        """The ordered blocks, exposed as an immutable tuple.

        The global column layout is cached, so allowing callers to mutate the
        block collection directly would leave the cached slices inconsistent
        with the actual blocks.
        """
        return self._blocks

    def _ensure_layout(self) -> None:
        if self._parameter_names is not None:
            return
        names: list[ParameterName] = []
        slices: list[slice] = []
        offset = 0
        for block in self.blocks:
            block_names = list(block.parameter_names())
            names.extend(block_names)
            next_offset = offset + len(block_names)
            slices.append(slice(offset, next_offset))
            offset = next_offset
        block_ids = [block.block_id for block in self.blocks]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError(f"Parametrization block IDs must be unique: {block_ids!r}.")
        if len(set(names)) != len(names):
            raise ValueError("Parametrization parameter names must be unique.")
        self._parameter_names = names
        self._slices = slices

    def setup(self, equations: Sequence[ObservationEquation], model_state) -> None:
        self._parameter_names = None
        self._slices = []
        for block in self._reduction_blocks:
            block.setup(equations, model_state)
        self._ensure_layout()

    def parameter_names(self) -> list[ParameterName]:
        self._ensure_layout()
        return list(self._parameter_names or [])

    def reference_values(self) -> np.ndarray:
        """Return the current global parameter vector in canonical column order."""
        self._ensure_layout()
        values = [block.reference_values() for block in self.blocks]
        return np.concatenate(values) if values else np.zeros(0, dtype=float)

    def reference_values_for(self, names: Sequence[ParameterName]) -> np.ndarray:
        """Return current values aligned to an arbitrary subset of known names."""
        reference_by_name = dict(zip(self.parameter_names(), self.reference_values()))
        missing = [name for name in names if name not in reference_by_name]
        if missing:
            raise KeyError(f"No x0 value is available for parameter(s) {missing!r}.")
        return np.asarray([reference_by_name[name] for name in names], dtype=float)

    def select_blocks(self, selectors: Sequence[str]) -> ParametrizationList:
        """Return a view over selected parameter blocks, reusing block state.

        Selectors are stable registry type names, for example
        ``reflectorPosition``. An empty selector list is invalid
        so a processing step cannot silently solve a zero-parameter system.
        """
        if isinstance(selectors, (str, bytes)):
            raise TypeError("Parametrization block selectors must be a sequence of strings.")
        requested_values: list[str] = []
        for value in selectors:
            if not isinstance(value, str):
                raise TypeError("Parametrization block selectors must contain only strings.")
            selector = value.strip()
            if not selector:
                raise ValueError("Parametrization block selectors must not be empty.")
            if selector in requested_values:
                raise ValueError(f"Parametrization block selector {selector!r} was provided more than once.")
            requested_values.append(selector)
        requested = set(requested_values)
        if not requested_values:
            raise ValueError("At least one parametrization block selector is required.")
        selected = [block for block in self.blocks if block.block_id in requested]
        found = {block.block_id for block in selected}
        missing = requested - found
        if missing:
            raise KeyError(f"Unknown parametrization block selector(s): {sorted(missing)}")
        return ParametrizationList(selected, reduction_blocks=self._reduction_blocks)

    @property
    def parameter_count(self) -> int:
        self._ensure_layout()
        return len(self._parameter_names or [])

    def design_entries(self, eq: ObservationEquation) -> list[tuple[int, float]]:
        self._ensure_layout()
        return [
            (block_slice.start + index, value)
            for block, block_slice in zip(self.blocks, self._slices)
            for index, value in block.design_entries(eq)
        ]

    def design_row(self, eq: ObservationEquation) -> np.ndarray:
        self._ensure_layout()
        row = np.zeros(self.parameter_count, dtype=float)
        for index, value in self.design_entries(eq):
            row[index] += value
        return row

    def reduced_observation(self, eq: ObservationEquation) -> float:
        return float(eq.observed_minus_computed_one_way_m) - sum(
            block.reduce_observation(eq) for block in self._reduction_blocks
        )

    def split(self, delta: np.ndarray) -> list[np.ndarray]:
        self._ensure_layout()
        values = parameter_vector(delta, expected_size=self.parameter_count, name="delta")
        return [values[block_slice] for block_slice in self._slices]

    def update_norms(self, delta: np.ndarray) -> dict[str, float]:
        return {
            block.block_id: float(block.max_update_norm(block_delta))
            for block, block_delta in zip(self.blocks, self.split(delta))
        }

    def apply_update(self, delta: np.ndarray) -> dict[str, float]:
        """Apply all block updates; returns per-block max update norms."""
        block_updates = self.split(delta)
        norms = {
            block.block_id: float(block.max_update_norm(block_delta))
            for block, block_delta in zip(self.blocks, block_updates)
        }
        for block, block_delta in zip(self.blocks, block_updates):
            block.apply_update(block_delta)
        return norms

    def state(self) -> dict[str, object]:
        return {block.block_id: block.state() for block in self.blocks}

    def matched_parameter_names(self, eq: ObservationEquation) -> list[str]:
        names = self.parameter_names()
        return [str(names[index]) for index, _ in self.design_entries(eq)]
