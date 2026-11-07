"""Random effects of the LLR adjustment, eliminated block by block from the normal equations.

The observations are partitioned into blocks (night sessions of one variance component, or station nights). Block
``b`` has random effects ``u_b ~ N(0, Lambda_b^-1)`` (``Lambda_b`` diagonal) that enter its rows through the design
rows ``z_i``: ``l_i = a_i x + z_i u_b + e_i``. Eliminating ``u_b`` (equivalently, adding it as parameters with the
pseudo-observations ``0 = u_b``, weight ``Lambda_b``) gives, with weights ``p_i``,

    ``D_b = sum p_i z_i^T z_i + Lambda_b``,  ``G_b = sum p_i z_i^T a_i``,  ``h_b = sum p_i z_i^T l_i``,
    ``N = A^T P A - sum G_b^T D_b^-1 G_b``,  ``A^T P l - sum G_b^T D_b^-1 h_b``,  ``l^T P l - sum h_b^T D_b^-1 h_b``,

and back-substitution ``u_b = D_b^-1 (h_b - G_b dx)``. The fitted value of row ``i`` has the variance
``q_i = c_i N^-1 c_i^T + z_i D_b^-1 z_i^T`` with ``c_i = a_i - z_i D_b^-1 G_b`` (its leverage is ``p_i q_i``), and the
pseudo-observation of effect ``k`` the redundancy ``1 - Lambda_k ((D_b^-1)_kk + (D_b^-1 G_b N^-1 G_b^T D_b^-1)_kk)``.

Effects used here: one offset per night session (``z_i = 1``), and per station night the corrections of UT1 and the
polar motion (``z_i`` their partials, :mod:`lunarops.estimation.earth_orientation_effects`). Blocks are padded to a
common size with inert effects (``z = 0``, prior precision 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.estimation.robust_vce import Sessions


@dataclass(frozen=True, eq=False, slots=True)
class RandomEffectLayout:
    """Blocks and random-effect design of the rows of one linearization.

    Attributes
    ----------
    blocks : Sessions
        Partition of the rows into blocks (``blocks.index`` is the block of each row).
    design : ndarray, shape (n, q)
        Design row ``z_i`` of every row over the ``q`` effect slots of its block.
    session_block, session_slot : ndarray of int, shape (s,)
        Block and slot of the offset of every night session (empty without session offsets).
    eop_slots : tuple[int, ...]
        Slots of the Earth-orientation corrections (empty without them).
    """

    blocks: Sessions
    design: np.ndarray
    session_block: np.ndarray
    session_slot: np.ndarray
    eop_slots: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        """Number of effect slots per block."""
        return self.design.shape[1]

    @classmethod
    def sessions_only(cls, sessions: Sessions) -> RandomEffectLayout:
        """One offset per night session: the blocks are the sessions."""
        count = sessions.count
        return cls(sessions, np.ones((len(sessions.index), 1)), np.arange(count), np.zeros(count, dtype=int))

    @classmethod
    def station_nights(
        cls,
        station_nights: Sessions,
        eop_partials: np.ndarray,
        sessions: Sessions | None,
    ) -> RandomEffectLayout:
        """Earth-orientation corrections per station night, plus the offsets of its sessions if ``sessions``.

        Every session must lie within one station night (true when both use the same gap).
        """
        rows = len(station_nights.index)
        if sessions is None:
            return cls(station_nights, np.asarray(eop_partials, float), np.zeros(0, int), np.zeros(0, int), (0, 1, 2))
        first_row = sessions.order[sessions.starts]
        session_block = station_nights.index[first_row]
        if np.any(station_nights.index != session_block[sessions.index]):
            raise ValueError("A night session spans several station nights.")
        order = np.lexsort((np.arange(sessions.count), session_block))
        slot = np.empty(sessions.count, dtype=int)
        starts = np.r_[0, np.flatnonzero(np.diff(session_block[order])) + 1]
        rank = np.arange(sessions.count) - np.repeat(starts, np.diff(np.r_[starts, sessions.count]))
        slot[order] = 3 + rank
        design = np.zeros((rows, 3 + int(rank.max(initial=-1)) + 1))
        design[:, :3] = eop_partials
        design[np.arange(rows), slot[sessions.index]] = 1.0
        return cls(station_nights, design, session_block, slot, (0, 1, 2))


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

    def leverage(self, design: np.ndarray, weights: np.ndarray, cofactor: np.ndarray) -> np.ndarray:
        """Return the leverage ``p_i q_i`` of every row."""
        return weights * self.prediction_variance(design, cofactor)

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

    def session_values(self, values: np.ndarray) -> np.ndarray:
        """Pick the values ``(B, q)`` of the session offsets, in session order."""
        return values[self.layout.session_block, self.layout.session_slot]


__all__ = ["RandomEffectElimination", "RandomEffectLayout"]
