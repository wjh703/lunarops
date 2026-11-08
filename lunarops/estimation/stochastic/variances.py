"""Variance parameters of the LLR observations: the variance function of each variance component.

Observation ``i`` of variance component ``k`` has the one-way variance

    ``sigma_i^2 = a_k^2 + s_k^2 sigma_rep,i^2``

with reported (a-priori) one-way sigma ``sigma_rep,i``, sigma factor ``s_k`` and floor sigma ``a_k``, in metres. The
adjustment weights the observation with ``f_i / sigma_i^2``, ``f_i`` being its robust weight factor.

Variance models
---------------
``scaled``
    ``a_k = 0``: the reported sigmas are scaled per component, as the LLR analysis centres scale the normal-point
    uncertainties per station and period.
``floorAndScaled``
    ``a_k`` and ``s_k`` are both estimated: an error floor added in quadrature to the scaled reported sigma, as VLBI
    analyses add a noise term in quadrature to the formal delay errors. The two variance components of each group are
    estimated by least-squares variance component estimation (LS-VCE, Teunissen and Amiri-Simkooei 2008, J Geod
    82:65-82), as for multi-component GNSS stochastic models.

The estimation of these parameters is in :mod:`lunarops.estimation.stochastic.vce`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np

VARIANCE_MODELS = ("scaled", "floorAndScaled")


@dataclass(frozen=True)
class ObservationVariances:
    """Variance parameters of every component: ``sigma_i^2 = a_k^2 + s_k^2 sigma_rep,i^2`` and session sigma ``tau_k``.

    Attributes
    ----------
    sigma_factors : dict[str, float]
        Sigma factor ``s_k`` per component.
    floor_sigmas : dict[str, float]
        Floor sigma ``a_k`` per component, in metres.
    session_sigmas : dict[str, float]
        Sigma ``tau_k`` of the night-session offsets per component, in metres; empty without night sessions.
    """

    sigma_factors: dict[str, float]
    floor_sigmas: dict[str, float]
    session_sigmas: dict[str, float] = field(default_factory=dict)

    @classmethod
    def initial(
        cls,
        components: Iterable[str],
        sigma_factors: Mapping[str, float] | None = None,
        floor_sigmas: Mapping[str, float] | None = None,
        session_sigma: float | None = None,
        session_sigmas: Mapping[str, float] | None = None,
    ) -> ObservationVariances:
        """Return unit sigma factors and zero floors, overridden by the given values of known components.

        With ``session_sigma`` every component gets that session sigma, overridden by ``session_sigmas``; without
        it the model has no night sessions and ``session_sigmas`` is ignored.
        """
        identifiers = list(components)
        factors = {key: 1.0 for key in identifiers}
        floors = {key: 0.0 for key in identifiers}
        factors.update({key: float(value) for key, value in (sigma_factors or {}).items() if key in factors})
        floors.update({key: float(value) for key, value in (floor_sigmas or {}).items() if key in floors})
        sessions: dict[str, float] = {}
        if session_sigma is not None:
            sessions = {key: float(session_sigma) for key in identifiers}
            sessions.update({key: float(value) for key, value in (session_sigmas or {}).items() if key in sessions})
        return cls(factors, floors, sessions)

    def without_floors(self) -> ObservationVariances:
        """Return the same sigma factors and session sigmas with zero floors."""
        return replace(self, floor_sigmas={key: 0.0 for key in self.floor_sigmas})

    def toward(self, other: ObservationVariances, fraction: float) -> ObservationVariances:
        """Return the variances ``fraction`` of the way from ``self`` to ``other`` in ``s^2`` and ``a^2``.

        Session sigmas are taken from ``other``.
        """

        def blend(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
            return {
                key: float(np.sqrt((1.0 - fraction) * before[key] ** 2 + fraction * value**2))
                for key, value in after.items()
            }

        return replace(
            other,
            sigma_factors=blend(self.sigma_factors, other.sigma_factors),
            floor_sigmas=blend(self.floor_sigmas, other.floor_sigmas),
        )

    def sigmas(self, component_ids: np.ndarray, apriori_sigmas: np.ndarray) -> np.ndarray:
        """Return ``sqrt(a_k^2 + (s_k sigma_rep,i)^2)`` per observation, in metres.

        It is exactly ``s_k sigma_rep,i`` when ``a_k = 0``.
        """
        factors = np.asarray([self.sigma_factors[key] for key in component_ids], dtype=float)
        floors = np.asarray([self.floor_sigmas[key] for key in component_ids], dtype=float)
        return np.hypot(floors, factors * np.asarray(apriori_sigmas, dtype=float))

    def relative_change(
        self, other: ObservationVariances, component_ids: np.ndarray, apriori_sigmas: np.ndarray
    ) -> float:
        """Largest relative change of an observation sigma from ``self`` to ``other``."""
        if len(component_ids) == 0:
            return 0.0
        before = self.sigmas(component_ids, apriori_sigmas)
        after = other.sigmas(component_ids, apriori_sigmas)
        return float(np.max(np.abs(after - before) / before))


__all__ = ["VARIANCE_MODELS", "ObservationVariances"]
