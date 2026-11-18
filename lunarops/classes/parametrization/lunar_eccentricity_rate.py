"""Parametrization of an extra (unmodelled) eccentricity rate of the lunar orbit.

Tidal dissipation in Earth and Moon makes the eccentricity of the lunar orbit grow secularly; LLR solutions
have repeatedly found an eccentricity rate beyond the modelled one (Williams & Boggs 2009; Williams et al.
2013, the DE430 report, "LLR fits after generating DE430 find an extra eccentricity rate of (5±2)e-12 /yr").
Pavlov, Williams & Suvorkin (2016, Celest. Mech. Dyn. Astron. 126, 61-88, Sect. 5.2.2) estimate it with a
kinematic partial derived from the Earth-Moon distance series of Chapront-Touzé & Chapront (1998),

    A = 385000.5 - 20905.4 cos l - 3699.1 cos(2D - l) - ...   km,

where the l and 2D - l terms carry a hidden factor e = 0.0549, so that

    dA/de ~ -380791 cos l - 67379 cos(2D - l)   km                                        (Pavlov et al. Eq. 29)

and the one-way range changes by ``dA/de * de_dt * (t - t0)``. The parameter is the extra rate in units of
1e-12 per Julian year; ``t0`` is the reference epoch at which the extra eccentricity vanishes (a constant
eccentricity offset is absorbed by the initial lunar state). The correction enters linearly: the forward
model is not re-evaluated, and ``reduce`` subtracts the current contribution from O-C.
"""

from __future__ import annotations

import numpy as np

from lunarops.base.constants import DAYS_PER_JULIAN_YEAR, J2000_JD
from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch
from lunarops.config.schema import ConfigSchema, number

from .base import SingleColumnParametrization

#: One-way range change per unit eccentricity, metres: coefficients of cos l and cos(2D - l).
DISTANCE_PER_ECCENTRICITY_M = (-380_791.0e3, -67_379.0e3)
#: Parameter unit: 1e-12 per Julian year.
RATE_UNIT_PER_YEAR = 1.0e-12


class LunarEccentricityRateParametrization(SingleColumnParametrization):
    """Extra eccentricity rate of the lunar orbit, in units of 1e-12 per year (Pavlov et al. 2016, Eq. 29).

    The parameter ``moon:orbit.extraEccentricityRate`` is the total extra rate; it starts at the configured
    a-priori value, which is applied to the observations whether or not it is estimated.

    Parameters
    ----------
    value : float, optional
        A-priori extra eccentricity rate, in 1e-12 per year.
    reference_epoch_jd : float, optional
        Julian date (UTC) at which the extra eccentricity is zero.
    """

    TYPE = "lunarEccentricityRate"
    SCHEMA = ConfigSchema(
        fields=(
            number("value", default=0.0, description="A-priori extra eccentricity rate, in 1e-12 per Julian year."),
            number(
                "referenceEpochJd",
                default=J2000_JD,
                description="Julian date (UTC) at which the extra eccentricity vanishes.",
            ),
        ),
        type_name="lunarEccentricityRate",
    )

    def __init__(self, *, value: float = 0.0, reference_epoch_jd: float = J2000_JD) -> None:
        self.apriori_value = float(value)
        self.reference_epoch_jd = float(reference_epoch_jd)
        self.values = {"rate": self.apriori_value}
        self.names = [ParameterName("moon", "orbit.extraEccentricityRate")]

    @classmethod
    def from_config(cls, config: dict, context) -> LunarEccentricityRateParametrization:
        """Create the parametrization from a validated configuration."""
        return cls(value=config["value"], reference_epoch_jd=config["referenceEpochJd"])

    def column(self, batch: EquationBatch) -> np.ndarray:
        """Return the partial of the one-way range with respect to the rate, in metres per 1e-12/yr."""
        l, _, _, d, _ = batch.fundamental_arguments
        years = ((batch.epoch_jd1 + batch.epoch_jd2) - self.reference_epoch_jd) / DAYS_PER_JULIAN_YEAR
        per_e = DISTANCE_PER_ECCENTRICITY_M[0] * np.cos(l) + DISTANCE_PER_ECCENTRICITY_M[1] * np.cos(2.0 * d - l)
        return per_e * years * RATE_UNIT_PER_YEAR

    def x0(self) -> np.ndarray:
        """Return the current extra eccentricity rate, in 1e-12 per year."""
        return np.asarray([self.values["rate"]], dtype=float)

    def apriori(self) -> np.ndarray:
        """Return the configured a-priori rate, in 1e-12 per year."""
        return np.asarray([self.apriori_value], dtype=float)

    def _contributions(self, batch: EquationBatch):
        yield self.column(batch) * self.values["rate"]

    def update(self, delta: np.ndarray) -> None:
        """Add the correction ``delta`` (1e-12 per year) to the rate."""
        self.values["rate"] += float(np.asarray(delta, dtype=float)[0])

    def state(self) -> dict[str, object]:
        """Return the a-priori and current rate (1e-12 per year) and the reference epoch."""
        return {
            "values": {"rate": float(self.values["rate"])},
            "apriori": self.apriori_value,
            "referenceEpochJd": self.reference_epoch_jd,
        }


__all__ = ["LunarEccentricityRateParametrization"]
