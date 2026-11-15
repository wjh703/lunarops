"""Parametrization of the lunar displacement Love number h2.

The solid-tide displacement of a retroreflector is linear in ``h2`` (Pavlov, Williams & Suvorkin 2016, Celest.
Mech. Dyn. Astron. 126, 61-88, Eq. 24), so the partial of the one-way range is the ``lunar_h2`` column of the
observation equations: the range gradient with respect to the reflector position projected on the displacement
of unit ``h2``. The correction enters linearly: the forward model keeps the configured ``h2`` and
``lunar_h2 * correction`` is subtracted from O-C, which is exact because the model is linear in ``h2``.
"""

from __future__ import annotations

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch
from lunarops.config.schema import ConfigSchema

from .base import Parametrization


class LunarLoveNumberParametrization(Parametrization):
    """Correction to the lunar displacement Love number h2 of the lunar solid tide.

    The parameter ``moon:loveNumber.h2`` is the dimensionless correction to the configured ``h2``, starting at
    zero.

    Parameters
    ----------
    nominal_h2 : float, optional
        ``h2`` of the lunar solid tide model of the observation model; ``None`` when it is not known.
    """

    TYPE = "lunarLoveNumber"
    SCHEMA = ConfigSchema(fields=(), type_name="lunarLoveNumber")

    def __init__(self, *, nominal_h2: float | None = None) -> None:
        self.nominal_h2 = None if nominal_h2 is None else float(nominal_h2)
        self.names = [ParameterName("moon", "loveNumber.h2")]
        # Correction to the nominal h2.
        self.values = {"h2": 0.0}
        self._column_rms: float | None = None

    @classmethod
    def from_config(cls, config: dict, context, *, lunar_solid_tide=None) -> LunarLoveNumberParametrization:
        """Create the parametrization; ``lunar_solid_tide`` is the tide model of the observation model.

        Raises
        ------
        ValueError
            If the observation model has no lunar solid tide.
        """
        if lunar_solid_tide is None:
            raise ValueError("lunarLoveNumber needs reflectorDisplacement of type lunarSolidTide.")
        return cls(nominal_h2=lunar_solid_tide.h2)

    def _column(self, batch: EquationBatch) -> np.ndarray:
        if batch.lunar_h2.shape != (len(batch), 1):
            raise ValueError("lunarLoveNumber needs the lunar h2 partial of the observation equations.")
        return batch.lunar_h2[:, 0]

    def x0(self) -> np.ndarray:
        """Return the current correction to h2."""
        return np.asarray([self.values["h2"]], dtype=float)

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        """Add the partial of the one-way range with respect to h2, in metres.

        Raises
        ------
        ValueError
            If the batch lacks the h2 partial.
        """
        column = self._column(batch)
        out[:, 0] += column
        if len(column):
            self._column_rms = float(np.sqrt(np.mean(column**2)))

    def _contributions(self, batch: EquationBatch):
        yield self._column(batch) * self.values["h2"]

    def update(self, delta: np.ndarray) -> None:
        """Add the correction ``delta`` to h2."""
        self.values["h2"] += float(np.asarray(delta, dtype=float)[0])

    def max_update_norm(self, delta: np.ndarray) -> float:
        """Return the effect of the correction on the one-way range, in metres.

        The correction is multiplied by the root mean square of the design column of the last linearization.
        """
        size = float(np.max(np.abs(delta))) if len(delta) else 0.0
        return size if self._column_rms is None else size * self._column_rms

    def state(self) -> dict[str, object]:
        """Return the correction and the nominal and updated h2."""
        correction = float(self.values["h2"])
        return {
            "values": {"h2": correction},
            "h2": {
                "nominal": self.nominal_h2,
                "correction": correction,
                "updated": None if self.nominal_h2 is None else self.nominal_h2 + correction,
            },
        }


__all__ = ["LunarLoveNumberParametrization"]
