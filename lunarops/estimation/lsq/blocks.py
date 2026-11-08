"""Partition of the rows of a linear system into blocks of consecutive rows of one key."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False, slots=True)
class RowBlocks:
    """Blocks of rows: rows of one key, ordered by time, without a gap longer than a given maximum.

    Attributes
    ----------
    index : ndarray of int, shape (n,)
        Block of each row.
    key : ndarray of object, shape (b,)
        Key of each block.
    order : ndarray of int, shape (n,)
        Rows sorted by block.
    starts : ndarray of int, shape (b,)
        First position of each block in ``order``.
    """

    index: np.ndarray
    key: np.ndarray
    order: np.ndarray
    starts: np.ndarray

    @property
    def count(self) -> int:
        """Number of blocks."""
        return len(self.key)

    @classmethod
    def build(cls, keys: np.ndarray, times: np.ndarray, maximum_gap: float) -> RowBlocks:
        """Split the rows into blocks: same key, no gap between consecutive times longer than ``maximum_gap``.

        Parameters
        ----------
        keys : ndarray of object, shape (n,)
            Key per row.
        times : ndarray, shape (n,)
            Time per row.
        maximum_gap : float
            Largest gap between consecutive rows of one block, in the unit of ``times``.
        """
        n = len(keys)
        if n == 0:
            empty = np.empty(0, dtype=int)
            return cls(empty, np.empty(0, dtype=object), empty, empty)
        _, key_code = np.unique(keys.astype(str), return_inverse=True)
        order = np.lexsort((times, key_code))
        sorted_code = key_code[order]
        sorted_time = times[order]
        new = np.ones(n, dtype=bool)
        new[1:] = (sorted_code[1:] != sorted_code[:-1]) | (np.diff(sorted_time) > maximum_gap)
        sorted_block = np.cumsum(new) - 1
        index = np.empty(n, dtype=int)
        index[order] = sorted_block
        starts = np.flatnonzero(new)
        key = np.asarray(keys, dtype=object)[order[starts]]
        return cls(index, key, order, starts)

    def sums(self, values: np.ndarray) -> np.ndarray:
        """Sum ``values`` (rows first) over the rows of each block."""
        return np.add.reduceat(values[self.order], self.starts, axis=0)


__all__ = ["RowBlocks"]
