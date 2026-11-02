"""Each parametrization owns a contiguous block of design-matrix columns."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch


class Parametrization:
    TYPE: str
    category = "parametrization"
    names: list[ParameterName]

    @property
    def block_id(self) -> str:
        return type(self).TYPE

    def setup(self, batch: EquationBatch) -> list[ParameterName]:
        return self.names

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        raise NotImplementedError

    def x0(self) -> np.ndarray:
        return np.zeros(len(self.names))

    def reduce(self, batch: EquationBatch) -> np.ndarray:
        # Sum in physical term order, including fixed coefficients.
        return sum(self._contributions(batch), start=np.zeros(len(batch)))

    def _contributions(self, batch: EquationBatch):
        raise NotImplementedError

    def update(self, delta: np.ndarray) -> None:
        raise NotImplementedError

    def max_update_norm(self, delta: np.ndarray) -> float:
        return float(np.max(np.abs(delta))) if len(delta) else 0.0

    def state(self) -> dict[str, object]:
        return {}


class ParametrizationList:
    def __init__(self, blocks: Sequence[Parametrization], *, reduction_blocks=None) -> None:
        self.blocks = tuple(blocks)
        self._reduction_blocks = self.blocks if reduction_blocks is None else tuple(reduction_blocks)
        self.names: list[ParameterName] = []
        self._slices: list[slice] = []

    def setup(self, batch: EquationBatch) -> list[ParameterName]:
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
        return np.concatenate([block.x0() for block in self.blocks])

    def select_blocks(self, selectors: Sequence[str]) -> ParametrizationList:
        return ParametrizationList(
            [block for block in self.blocks if block.block_id in selectors],
            reduction_blocks=self._reduction_blocks,
        )

    def design(self, batch: EquationBatch) -> np.ndarray:
        out = np.zeros((len(batch), len(self.names)))
        for block, columns in zip(self.blocks, self._slices):
            block.design(batch, out[:, columns])
        return out

    def reduce(self, batch: EquationBatch) -> np.ndarray:
        return batch.oc - sum(block.reduce(batch) for block in self._reduction_blocks)

    def update_norms(self, delta: np.ndarray) -> dict[str, float]:
        return {block.block_id: block.max_update_norm(delta[columns])
                for block, columns in zip(self.blocks, self._slices)}

    def update(self, delta: np.ndarray) -> dict[str, float]:
        norms = self.update_norms(delta)
        for block, columns in zip(self.blocks, self._slices):
            block.update(delta[columns])
        return norms

    def state(self) -> dict[str, object]:
        return {block.block_id: block.state() for block in self.blocks}
