"""Parametrization interface and the ordered list of parameter blocks.

Each parametrization owns a contiguous block of design-matrix columns. Design columns are partial derivatives
of the computed one-way range, in metres per parameter unit, so that the reduced one-way O-C residual
satisfies ``reduced ~ A dx``. Parametrizations whose values enter the forward model (reflector and station
coordinates) write updates to the shared catalog state; the others (range biases, empirical librations) add
their current contribution linearly, and it is subtracted from O-C to form the reduced observations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch


class Parametrization:
    """Base class of a block of estimated parameters.

    Subclasses set ``TYPE`` (the configuration type name and block identifier) and ``names``, and implement
    ``design`` and ``update``; parametrizations with a linear contribution to the observations implement
    ``_contributions``.

    Attributes
    ----------
    TYPE : str
        Configuration type name of the parametrization.
    category : str
        Registry category, ``"parametrization"``.
    names : list[ParameterName]
        Parameter names of the block, in column order.
    """

    TYPE: str
    category = "parametrization"
    names: list[ParameterName]

    @property
    def block_id(self) -> str:
        """Identifier of the block, equal to ``TYPE``."""
        return type(self).TYPE

    def setup(self, batch: EquationBatch) -> list[ParameterName]:
        """Select the parameters for the observations in ``batch`` and return their names."""
        return self.names

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        """Add the block's design columns to ``out``.

        Parameters
        ----------
        batch : EquationBatch
            Observation equations.
        out : ndarray, shape (n, len(names))
            View of the design-matrix columns of this block; partials of the one-way range in metres per
            parameter unit are added in place.
        """
        raise NotImplementedError

    def x0(self) -> np.ndarray:
        """Return the current parameter values, the block's linearization point; zeros in the base class."""
        return np.zeros(len(self.names))

    def reduce(self, batch: EquationBatch) -> np.ndarray:
        """Return the block's current one-way contribution to each observation, in metres, shape ``(n,)``.

        Fixed (not estimated) coefficients are included.
        """
        # Sum in physical term order, including fixed coefficients.
        return sum(self._contributions(batch), start=np.zeros(len(batch)))

    def _contributions(self, batch: EquationBatch):
        raise NotImplementedError

    def update(self, delta: np.ndarray) -> None:
        """Add the correction ``delta`` (in parameter units, column order) to the current values."""
        raise NotImplementedError

    def max_update_norm(self, delta: np.ndarray) -> float:
        """Return the size of a correction used for the convergence test: the largest absolute component."""
        return float(np.max(np.abs(delta))) if len(delta) else 0.0

    def state(self) -> dict[str, object]:
        """Return the current parameter values in a serializable form."""
        return {}


class ParametrizationList:
    """Ordered parameter blocks forming the columns of one adjustment.

    Parameters
    ----------
    blocks : Sequence[Parametrization]
        Estimated blocks, in column order.
    reduction_blocks : Sequence[Parametrization], optional
        Blocks whose current contributions are subtracted from O-C and that are set up on every ``setup``.
        Defaults to ``blocks``; ``select_blocks`` keeps the full list here so deselected blocks still apply
        their values.
    """

    def __init__(self, blocks: Sequence[Parametrization], *, reduction_blocks=None) -> None:
        self.blocks = tuple(blocks)
        self._reduction_blocks = self.blocks if reduction_blocks is None else tuple(reduction_blocks)
        self.names: list[ParameterName] = []
        self._slices: list[slice] = []

    def setup(self, batch: EquationBatch) -> list[ParameterName]:
        """Set up every block for ``batch`` and return the parameter names in column order."""
        for block in self._reduction_blocks:
            block.setup(batch)
        self.names = []
        self._slices = []
        for block in self.blocks:
            start = len(self.names)
            self.names.extend(block.names)
            self._slices.append(slice(start, len(self.names)))
        return self.names

    def x0(self) -> np.ndarray:
        """Return the concatenated current parameter values, the linearization point ``x0``."""
        return np.concatenate([block.x0() for block in self.blocks])

    def select_blocks(self, selectors: Sequence[str]) -> ParametrizationList:
        """Return a list estimating only the blocks whose ``block_id`` is in ``selectors``.

        All blocks of this list remain reduction blocks.
        """
        return ParametrizationList(
            [block for block in self.blocks if block.block_id in selectors],
            reduction_blocks=self._reduction_blocks,
        )

    def design(self, batch: EquationBatch) -> np.ndarray:
        """Return the design matrix, ndarray of shape ``(n, len(names))``, in metres per parameter unit."""
        out = np.zeros((len(batch), len(self.names)))
        for block, columns in zip(self.blocks, self._slices):
            block.design(batch, out[:, columns])
        return out

    def reduce(self, batch: EquationBatch) -> np.ndarray:
        """Return the reduced observations: one-way O-C minus the reduction blocks' contributions, in metres."""
        return batch.oc - sum(block.reduce(batch) for block in self._reduction_blocks)

    def update_norms(self, delta: np.ndarray) -> dict[str, float]:
        """Return each block's ``max_update_norm`` of its part of ``delta``, keyed by block identifier."""
        return {block.block_id: block.max_update_norm(delta[columns])
                for block, columns in zip(self.blocks, self._slices)}

    def update(self, delta: np.ndarray) -> dict[str, float]:
        """Apply the correction ``delta`` to every block and return the update norms per block."""
        norms = self.update_norms(delta)
        for block, columns in zip(self.blocks, self._slices):
            block.update(delta[columns])
        return norms

    def state(self) -> dict[str, object]:
        """Return the state of every estimated block, keyed by block identifier."""
        return {block.block_id: block.state() for block in self.blocks}
