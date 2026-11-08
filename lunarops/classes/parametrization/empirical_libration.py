"""Parametrization of empirical periodic lunar orientation (libration) corrections.

Each term adds a small rotation about one axis,

    d_angle(t) = c * cos(arg(t)) + s * sin(arg(t))      [mas]

Axes ``x``, ``y`` and ``psi`` are rotations about the x, y and z axes of the lunar PA frame. Axes ``rho``,
``Isigma`` and ``tau`` are corrections to the physical librations referred to the J2000 mean ecliptic, the
convention used for empirical libration terms of lunar ephemerides: rho changes the inclination of the lunar
equator, I*sigma its node (with the mean inclination I = 1.543 deg) and tau the longitude. ``tau`` and ``psi``
are the same body-z rotation to first order. A latitude term written in rho/I*sigma with argument ``a``
appears on x/y at ``a +/- F``, so the two axis sets should not be mixed for the same signal.

The argument is either an integer combination of the Delaunay arguments (l, l', F, D, Omega) of
IERS Conventions (2003), or ``2 pi (JD - 2451545.0) / periodDays``. Both use the Julian date of the transmit
epoch in UTC.

The correction enters the observation equations linearly through the ``lunar_orientation_pa`` and
``lunar_ecliptic_libration`` partials (m/rad): the forward model is not re-evaluated with the rotated frame,
which is accurate for mas-level corrections (1 mas ~ 8 mm at the lunar surface). Terms with
``estimate: false`` are applied with their given values and are not estimated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import EquationBatch
from lunarops.config.schema import ConfigSchema, boolean, mapping, number, sequence, string

from .base import Parametrization

_MAS_TO_RAD = np.deg2rad(1.0 / 3_600_000.0)
# axis -> (design-partial block, component)
_AXES = {
    "x": ("lunar_orientation_pa", 0),
    "y": ("lunar_orientation_pa", 1),
    "psi": ("lunar_orientation_pa", 2),
    "rho": ("lunar_ecliptic_libration", 0),
    "Isigma": ("lunar_ecliptic_libration", 1),
    "tau": ("lunar_ecliptic_libration", 2),
}
_DELAUNAY = ("l", "lp", "F", "D", "Om")
_J2000_JD = 2_451_545.0


@dataclass(frozen=True, slots=True)
class EmpiricalLibrationTerm:
    """One periodic libration term.

    Attributes
    ----------
    name : str
        Unique term name; it forms the parameter names.
    axis : str
        Rotation axis: ``x``, ``y``, ``psi``, ``rho``, ``Isigma`` or ``tau``.
    delaunay : tuple[int, int, int, int, int] or None
        Integer multipliers of (l, l', F, D, Omega), or ``None`` when ``period_days`` is given.
    period_days : float or None
        Period of the argument, in days, or ``None`` when ``delaunay`` is given.
    cos_mas : float
        A-priori cosine amplitude, in milliarcseconds.
    sin_mas : float
        A-priori sine amplitude, in milliarcseconds.
    estimate_cos : bool
        Whether the cosine amplitude is estimated.
    estimate_sin : bool
        Whether the sine amplitude is estimated.
    """

    name: str
    axis: str
    delaunay: tuple[int, int, int, int, int] | None
    period_days: float | None
    cos_mas: float
    sin_mas: float
    estimate_cos: bool
    estimate_sin: bool


def _parse_term(item: dict) -> EmpiricalLibrationTerm:
    name = item["name"].strip()
    argument = item.get("delaunay")
    period = item.get("periodDays")
    if (argument is None) == (period is None):
        raise ValueError(f"empiricalLibration term {name!r}: give exactly one of delaunay or periodDays.")
    delaunay = None
    if argument is not None:
        unknown = set(argument) - set(_DELAUNAY)
        if unknown:
            raise ValueError(f"empiricalLibration term {name!r}: unknown Delaunay argument(s) {sorted(unknown)}.")
        l, lp, f, d, om = (int(argument.get(key, 0)) for key in _DELAUNAY)
        delaunay = (l, lp, f, d, om)
        if not any(delaunay):
            raise ValueError(f"empiricalLibration term {name!r}: Delaunay combination must not be zero.")
    if period == 0:
        raise ValueError(f"empiricalLibration term {name!r}: periodDays must be non-zero.")
    return EmpiricalLibrationTerm(
        name=name,
        axis=item["axis"],
        delaunay=delaunay,
        period_days=period,
        cos_mas=item["cosMas"],
        sin_mas=item["sinMas"],
        estimate_cos=item["estimate"],
        estimate_sin=item["estimate"] and item["estimateSin"],
    )


_TERM_SCHEMA = ConfigSchema(
    fields=(
        string("name", required=True, non_empty=True, description="Unique term name; it forms the parameter names."),
        string(
            "axis",
            default="psi",
            choices=tuple(sorted(_AXES)),
            description=(
                "Rotation axis of the term: `x`, `y`, `psi` (PA axes), or `rho`, `Isigma`, `tau` (physical librations"
                " referred to the J2000 mean ecliptic)."
            ),
        ),
        mapping(
            "delaunay",
            description=(
                "Integer multipliers of the Delaunay arguments `l`, `lp`, `F`, `D`, `Om` that form the argument. Give"
                " either this or `periodDays`."
            ),
        ),
        number(
            "periodDays",
            description=(
                "Period of the argument, in days, counted from JD 2451545.0 of the UTC transmit epoch. "
                "Give either this or `delaunay`."
            ),
        ),
        number("cosMas", default=0.0, description="A-priori cosine amplitude, in milliarcseconds."),
        number("sinMas", default=0.0, description="A-priori sine amplitude, in milliarcseconds."),
        boolean(
            "estimate",
            default=True,
            description="Estimate the term. If false, the a-priori amplitudes are applied and kept fixed.",
        ),
        boolean(
            "estimateSin",
            default=True,
            description="Estimate the sine amplitude as well as the cosine amplitude.",
        ),
    )
)


class EmpiricalLibrationParametrization(Parametrization):
    """Empirical periodic lunar libration corrections, cosine and sine amplitudes in milliarcseconds per term.

    Parameter names are ``moon:libration.<term>.cos`` and ``moon:libration.<term>.sin``; only estimated
    amplitudes get parameters, while every term's current amplitudes contribute to the reduced observations.

    Parameters
    ----------
    terms : Sequence[EmpiricalLibrationTerm]
        Terms with unique names.

    Raises
    ------
    ValueError
        If term names are not unique or no amplitude is estimated.
    """

    TYPE = "empiricalLibration"
    SCHEMA = ConfigSchema(
        fields=(
            sequence(
                "terms",
                required=True,
                item_kind="mapping",
                item_nested=_TERM_SCHEMA,
                min_items=1,
                description="Periodic libration terms.",
            ),
        ),
        type_name="empiricalLibration",
    )

    def __init__(self, *, terms: Sequence[EmpiricalLibrationTerm]) -> None:
        self.terms = list(terms)
        names = [term.name for term in self.terms]
        if len(set(names)) != len(names):
            raise ValueError("empiricalLibration term names must be unique.")
        # values[k] = [cos, sin] in mas, starting from the configured a priori values
        self.values = {term.name: [term.cos_mas, term.sin_mas] for term in self.terms}
        self._slots: list[tuple[int, int]] = []
        for k, term in enumerate(self.terms):
            if term.estimate_cos:
                self._slots.append((k, 0))
            if term.estimate_sin:
                self._slots.append((k, 1))
        if not self._slots:
            raise ValueError("empiricalLibration has no estimated coefficients.")

        self.names = [
            ParameterName("moon", f"libration.{self.terms[k].name}.{'cos' if c == 0 else 'sin'}")
            for k, c in self._slots
        ]

    @classmethod
    def from_config(cls, config: dict, context) -> EmpiricalLibrationParametrization:
        """Create the parametrization from a validated configuration."""
        return cls(terms=[_parse_term(item) for item in config["terms"]])

    def x0(self) -> np.ndarray:
        """Return the current estimated amplitudes, in milliarcseconds."""
        return np.asarray([self.values[self.terms[k].name][c] for k, c in self._slots], dtype=float)

    def apriori(self) -> np.ndarray:
        """Return the configured a-priori amplitudes of the estimated coefficients, in milliarcseconds."""
        return np.asarray([(term := self.terms[k]).cos_mas if c == 0 else term.sin_mas for k, c in self._slots])

    def _basis(self, batch: EquationBatch) -> np.ndarray:
        """Return the per-term cos/sin design basis in m/mas, shape (n, n_terms, 2), cached on the batch."""
        key = tuple(self.terms)
        if key in batch.libration_bases:
            return batch.libration_bases[key]
        days = (batch.epoch_jd1 + batch.epoch_jd2) - _J2000_JD
        basis = np.empty((len(batch), len(self.terms), 2))
        for k, term in enumerate(self.terms):
            argument = (
                2.0 * np.pi * days / term.period_days
                if term.period_days is not None
                else sum(
                    (
                        coefficient * value
                        for coefficient, value in zip(cast(tuple[int, ...], term.delaunay), batch.fundamental_arguments)
                    ),
                    start=np.zeros(len(batch)),
                )
            )
            block_name, component = _AXES[term.axis]
            scale = getattr(batch, block_name)[:, component] * _MAS_TO_RAD
            basis[:, k, 0] = scale * np.fromiter((np.cos(a) for a in argument), dtype=float, count=len(batch))
            basis[:, k, 1] = scale * np.fromiter((np.sin(a) for a in argument), dtype=float, count=len(batch))
        batch.libration_bases[key] = basis
        return basis

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        """Add the estimated amplitudes' columns: axis partial times cos or sin of the argument, in m/mas."""
        basis = self._basis(batch)
        for index, (k, c) in enumerate(self._slots):
            out[:, index] += basis[:, k, c]

    def _contributions(self, batch: EquationBatch):
        basis = self._basis(batch)
        for k, term in enumerate(self.terms):
            yield basis[:, k, 0] * self.values[term.name][0] + basis[:, k, 1] * self.values[term.name][1]

    def update(self, delta: np.ndarray) -> None:
        """Add the amplitude corrections, in milliarcseconds."""
        delta = np.asarray(delta, float)
        for (k, c), value in zip(self._slots, delta):
            self.values[self.terms[k].name][c] += float(value)

    def state(self) -> dict[str, object]:
        """Return axis, current cosine and sine amplitudes (mas) and the estimate flag per term."""
        return {
            term.name: {
                "axis": term.axis,
                "cosMas": float(self.values[term.name][0]),
                "sinMas": float(self.values[term.name][1]),
                "estimated": term.estimate_cos,
            }
            for term in self.terms
        }


__all__ = ["EmpiricalLibrationParametrization", "EmpiricalLibrationTerm"]
