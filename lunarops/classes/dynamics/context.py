"""Interfaces and records shared by the integrator and the right-hand side.

:class:`StateHistoryProvider` is the protocol through which delayed force
models read barycentric body states, and :class:`StepData` carries the data
that one integration step computes once and reuses for every corrector
evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lunarops.classes.time import Epoch

from .deformation import LunarInertiaEvaluation


class StateHistoryProvider(Protocol):
    """Source of barycentric body states at arbitrary TDB epochs.

    Implementations are :class:`~lunarops.classes.dynamics.history.EphemerisHistory`
    (ephemeris only) and
    :class:`~lunarops.classes.dynamics.history.DelayedHistory` (integrated
    Earth-Moon relative state about the ephemeris barycentre, ephemeris for
    every other body).
    """

    def states(self, body_names: Sequence[str], epoch: Epoch) -> np.ndarray:
        """Return BCRS states of several bodies at one epoch.

        Parameters
        ----------
        body_names : sequence of str
            Canonical body identifiers.
        epoch : Epoch
            Evaluation epoch in TDB.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(body_names), 6)``: BCRS position in m and
            velocity in m s⁻¹ with ICRF axes, one row per requested body.
        """
        ...


@dataclass(frozen=True, slots=True)
class StepData:
    """Candidate-independent data of one integration step.

    The data depend on the epoch and on accepted history only, so the
    predictor and every corrector evaluation of the step share one instance.

    Attributes
    ----------
    force_payloads : Mapping[str, object]
        Per-force step payloads keyed by force name, for example the delayed
        Earth-tide raiser vectors (``"tide"``) or the prescribed lunar
        degree-2 coefficients (``"figure_moon"``).
    lunar_inertia : LunarInertiaEvaluation or None
        Delayed whole-Moon inertia evaluation when the integrated rotation has
        a deformation model, otherwise ``None``.
    """

    force_payloads: Mapping[str, object]
    lunar_inertia: LunarInertiaEvaluation | None = None
