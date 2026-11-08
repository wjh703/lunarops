"""Random effects eliminated block by block from the normal equations.

The rows are partitioned into blocks (:class:`lunarops.estimation.lsq.blocks.RowBlocks`). Block ``b`` has random
effects ``u_b ~ N(0, Lambda_b^-1)`` (``Lambda_b`` diagonal) that enter its rows through the design rows ``z_i``:
``l_i = a_i x + z_i u_b + e_i``. Eliminating ``u_b`` (equivalently, adding it as parameters with the
pseudo-observations ``0 = u_b``, weight ``Lambda_b``) gives, with weights ``p_i``,

    ``D_b = sum p_i z_i^T z_i + Lambda_b``,  ``G_b = sum p_i z_i^T a_i``,  ``h_b = sum p_i z_i^T l_i``,
    ``N = A^T P A - sum G_b^T D_b^-1 G_b``,  ``A^T P l - sum G_b^T D_b^-1 h_b``,  ``l^T P l - sum h_b^T D_b^-1 h_b``,

and back-substitution ``u_b = D_b^-1 (h_b - G_b dx)``. The fitted value of row ``i`` has the variance
``q_i = c_i N^-1 c_i^T + z_i D_b^-1 z_i^T`` with ``c_i = a_i - z_i D_b^-1 G_b`` (its leverage is ``p_i q_i``), and the
pseudo-observation of effect ``k`` the redundancy ``1 - Lambda_k ((D_b^-1)_kk + (D_b^-1 G_b N^-1 G_b^T D_b^-1)_kk)``.

Two kinds of effects are supported: effects with given partials ``z_i`` shared by the rows of a block, and one offset
(``z_i = 1``) per sub-block of a finer partition. Blocks are padded to a common size with inert effects (``z = 0``,
prior precision 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.estimation.lsq.blocks import RowBlocks


@dataclass(frozen=True, eq=False, slots=True)
class RandomEffectLayout:
    """Blocks and random-effect design of the rows of one linearization.

    Attributes
    ----------
    blocks : RowBlocks
        Partition of the rows into blocks (``blocks.index`` is the block of each row).
    design : ndarray, shape (n, q)
        Design row ``z_i`` of every row over the ``q`` effect slots of its block.
    offset_block, offset_slot : ndarray of int, shape (s,)
        Block and slot of the offset of every sub-block (empty without offsets).
    partial_slots : tuple[int, ...]
        Slots of the effects with given partials (empty without them).
    """

    blocks: RowBlocks
    design: np.ndarray
    offset_block: np.ndarray
    offset_slot: np.ndarray
    partial_slots: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        """Number of effect slots per block."""
        return self.design.shape[1]

    @classmethod
    def offsets(cls, blocks: RowBlocks) -> RandomEffectLayout:
        """One offset per block."""
        count = blocks.count
        return cls(blocks, np.ones((len(blocks.index), 1)), np.arange(count), np.zeros(count, dtype=int))

    @classmethod
    def with_partials(
        cls,
        blocks: RowBlocks,
        partials: np.ndarray,
        offsets: RowBlocks | None = None,
    ) -> RandomEffectLayout:
        """Effects with the given partials per block, plus one offset per sub-block of ``offsets``.

        Parameters
        ----------
        blocks : RowBlocks
            Blocks of the rows.
        partials : ndarray, shape (n, k)
            Partials of every row with respect to the ``k`` effects of its block.
        offsets : RowBlocks, optional
            Finer partition whose every sub-block gets an offset; each must lie within one block.

        Raises
        ------
        ValueError
            If a sub-block spans several blocks.
        """
        partials = np.asarray(partials, float)
        rows, count = partials.shape
        shared = tuple(range(count))
        if offsets is None:
            return cls(blocks, partials, np.zeros(0, int), np.zeros(0, int), shared)
        first_row = offsets.order[offsets.starts]
        offset_block = blocks.index[first_row]
        if np.any(blocks.index != offset_block[offsets.index]):
            raise ValueError("A sub-block spans several blocks.")
        order = np.lexsort((np.arange(offsets.count), offset_block))
        slot = np.empty(offsets.count, dtype=int)
        starts = np.r_[0, np.flatnonzero(np.diff(offset_block[order])) + 1]
        rank = np.arange(offsets.count) - np.repeat(starts, np.diff(np.r_[starts, offsets.count]))
        slot[order] = count + rank
        design = np.zeros((rows, count + int(rank.max(initial=-1)) + 1))
        design[:, :count] = partials
        design[np.arange(rows), slot[offsets.index]] = 1.0
        return cls(blocks, design, offset_block, slot, shared)


@dataclass(frozen=True, eq=False, slots=True)
class RandomEffectElimination:
    """Block statistics of a weighted system, used to eliminate the random effects (see the module docstring).

    Attributes
    ----------
    layout : RandomEffectLayout
        Blocks and effect design.
    inverse : ndarray, shape (B, q, q)
        ``D_b^-1``.
    gain : ndarray, shape (B, q, m)
        ``K_b = D_b^-1 G_b``.
    gradient : ndarray, shape (B, q, m)
        ``G_b``.
    observation_sum : ndarray, shape (B, q)
        ``h_b``.
    precision : ndarray, shape (B, q)
        Prior precision ``Lambda_b`` of every effect.
    """

    layout: RandomEffectLayout
    inverse: np.ndarray
    gain: np.ndarray
    gradient: np.ndarray
    observation_sum: np.ndarray
    precision: np.ndarray

    @classmethod
    def build(
        cls,
        layout: RandomEffectLayout,
        design: np.ndarray,
        observations: np.ndarray,
        weights: np.ndarray,
        precision: np.ndarray,
    ) -> RandomEffectElimination:
        """Accumulate the block statistics of ``weights``-weighted rows; ``precision`` has shape ``(B, q)``."""
        blocks, z = layout.blocks, layout.design
        weighted = weights[:, None] * z
        normal = blocks.sums(weighted[:, :, None] * z[:, None, :])
        normal[:, np.arange(layout.size), np.arange(layout.size)] += precision
        inverse = np.linalg.inv(normal)
        gradient = np.stack([blocks.sums(weighted[:, [k]] * design) for k in range(layout.size)], axis=1)
        return cls(
            layout=layout,
            inverse=inverse,
            gain=inverse @ gradient,
            gradient=gradient,
            observation_sum=blocks.sums(weighted * observations[:, None]),
            precision=precision,
        )

    def reduce(self, normal_matrix: np.ndarray, rhs: np.ndarray, lpl: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ``N``, ``A^T P l`` and ``l^T P l`` with the random effects eliminated."""
        solved = np.einsum("bqr,br->bq", self.inverse, self.observation_sum)
        return (
            normal_matrix - np.einsum("bqm,bqn->mn", self.gradient, self.gain),
            rhs - np.einsum("bqm,bq->m", self.gradient, solved),
            lpl - float(np.sum(self.observation_sum * solved)),
        )

    def effects(self, delta: np.ndarray) -> np.ndarray:
        """Back-substitute the random effects ``u_b = D_b^-1 h_b - K_b dx``, shape ``(B, q)``."""
        return np.einsum("bqr,br->bq", self.inverse, self.observation_sum) - self.gain @ delta

    def effect_rows(self, effects: np.ndarray) -> np.ndarray:
        """Return ``z_i u_b``, the contribution of the effects to every row, in metres."""
        return np.einsum("iq,iq->i", self.layout.design, effects[self.layout.blocks.index])

    def prior_quadratic(self, effects: np.ndarray) -> float:
        """Return ``sum u^T Lambda u``, the weighted squares of the effects' pseudo-observations."""
        return float(np.sum(self.precision * effects**2))

    def prediction_variance(self, design: np.ndarray, cofactor: np.ndarray) -> np.ndarray:
        """Return ``q_i = c_i N^-1 c_i^T + z_i D_b^-1 z_i^T``, the variance of the fitted value of every row."""
        z, block = self.layout.design, self.layout.blocks.index
        centred = design - np.einsum("iq,iqm->im", z, self.gain[block])
        return np.einsum("ij,ij->i", centred @ cofactor, centred) + np.einsum("iq,iqr,ir->i", z, self.inverse[block], z)

    def posterior_variance(self, cofactor: np.ndarray) -> np.ndarray:
        """Return ``(D_b^-1)_kk + (K_b N^-1 K_b^T)_kk``, the cofactor of every effect estimate, shape ``(B, q)``."""
        diagonal = np.einsum("bkk->bk", self.inverse)
        return diagonal + np.einsum("bkm,mn,bkn->bk", self.gain, cofactor, self.gain)

    def posterior_covariance(self, cofactor: np.ndarray, slots: list[int]) -> np.ndarray:
        """Return ``D_b^-1 + K_b N^-1 K_b^T`` over the effect ``slots``, the cofactor of their estimates, ``(B, k, k)``."""
        gain = self.gain[:, slots]
        return self.inverse[:, slots][:, :, slots] + np.einsum("bkm,mn,bln->bkl", gain, cofactor, gain)

    def effect_redundancy(self, cofactor: np.ndarray) -> np.ndarray:
        """Return the redundancy ``1 - Lambda_k posterior_variance_k`` of every effect's pseudo-observation."""
        return np.clip(1.0 - self.precision * self.posterior_variance(cofactor), 0.0, 1.0)

    def noise_variance(self, row_variances: np.ndarray) -> np.ndarray:
        """Return ``diag(D_b^-1 (sum c_i z_i^T z_i) D_b^-1)``, shape ``(B, q)``.

        It is the variance of every effect estimate from noise terms ``e_i`` with ``Var(p_i e_i) = c_i``, with the
        parameters held fixed.
        """
        z = self.layout.design
        moment = self.layout.blocks.sums(row_variances[:, None, None] * z[:, :, None] * z[:, None, :])
        return np.einsum("bkq,bqr,bkr->bk", self.inverse, moment, self.inverse)

    def offset_values(self, values: np.ndarray) -> np.ndarray:
        """Pick the values ``(B, q)`` of the sub-block offsets, in sub-block order."""
        return values[self.layout.offset_block, self.layout.offset_slot]


def fitted_value_variance(
    design: np.ndarray, cofactor: np.ndarray, elimination: RandomEffectElimination | None
) -> np.ndarray:
    """Return the variance ``q_i`` of the fitted value of every row, with or without eliminated random effects."""
    if elimination is None:
        return np.einsum("ij,ij->i", design @ cofactor, design)
    return elimination.prediction_variance(design, cofactor)


__all__ = ["RandomEffectElimination", "RandomEffectLayout", "fitted_value_variance"]
