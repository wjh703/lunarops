"""Parametrization: empirical periodic lunar orientation (libration) corrections.

Each term adds a small rotation of the lunar PA frame about one PA axis,

    d_angle(t) = c * cos(arg(t)) + s * sin(arg(t))      [mas]

Axes ``rho``, ``Isigma`` and ``tau`` are corrections to the physical librations
referred to the J2000 mean ecliptic (the convention of the DE/INPOP/EPM
empirical terms): rho changes the equator inclination, I*sigma its node and tau
the longitude.  ``tau`` and ``psi`` are the same body-z rotation to first
order.  ``x`` / ``y`` are rotations about fixed PA axes; a latitude term written
in rho/I*sigma with argument a appears on x/y at a +/- F, so do not mix them.  The argument is either an integer combination of
the IERS 2003 Delaunay arguments (l, l', F, D, Omega) or a pure period in days
counted from J2000.

The correction enters the observation equations linearly through the
``lunar_orientation_pa`` partial: the forward model is not re-evaluated with the
rotated frame, which is accurate for mas-level corrections (1 mas ~ 8 mm at the
lunar surface).  Terms with ``estimate: false`` are applied with their given
values and are not estimated.
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
    name: str
    axis: str
    delaunay: tuple[int, int, int, int, int] | None
    period_days: float | None
    cos_mas: float
    sin_mas: float
    estimate_cos: bool
    estimate_sin: bool


def _parse_term(item: dict) -> EmpiricalLibrationTerm:
    name = str(item["name"]).strip()
    axis = str(item.get("axis", "psi")).strip()
    if axis not in _AXES:
        raise ValueError(f"empiricalLibration term {name!r}: axis must be one of {sorted(_AXES)}.")
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
    if period is not None and (not np.isfinite(float(period)) or float(period) == 0.0):
        raise ValueError(f"empiricalLibration term {name!r}: periodDays must be finite and non-zero.")
    estimate = bool(item.get("estimate", True))
    return EmpiricalLibrationTerm(
        name=name,
        axis=axis,
        delaunay=delaunay,
        period_days=None if period is None else float(period),
        cos_mas=float(item.get("cosMas", 0.0)),
        sin_mas=float(item.get("sinMas", 0.0)),
        estimate_cos=estimate,
        estimate_sin=estimate and bool(item.get("estimateSin", True)),
    )


_TERM_SCHEMA = ConfigSchema(
    fields=(
        string("name", required=True, non_empty=True),
        string("axis", default="psi", choices=tuple(sorted(_AXES))),
        mapping("delaunay"),
        number("periodDays"),
        number("cosMas", default=0.0),
        number("sinMas", default=0.0),
        boolean("estimate", default=True),
        boolean("estimateSin", default=True),
    )
)


class EmpiricalLibrationParametrization(Parametrization):
    """Estimate periodic PA-frame rotation corrections (mas)."""

    TYPE = "empiricalLibration"
    SCHEMA = ConfigSchema(
        fields=(sequence("terms", required=True, item_kind="mapping", item_nested=_TERM_SCHEMA, min_items=1),),
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
        return cls(terms=[_parse_term(item) for item in config["terms"]])

    def x0(self) -> np.ndarray:
        return np.asarray([self.values[self.terms[k].name][c] for k, c in self._slots], dtype=float)

    def _basis(self, batch: EquationBatch) -> np.ndarray:
        key = tuple(self.terms)
        if key in batch.libration_bases:
            return batch.libration_bases[key]
        days = (batch.epoch_jd1 + batch.epoch_jd2) - _J2000_JD
        basis = np.empty((len(batch), len(self.terms), 2))
        for k, term in enumerate(self.terms):
            argument = (2.0 * np.pi * days / term.period_days if term.period_days is not None else
                        sum((coefficient * value for coefficient, value in zip(cast(tuple[int, ...], term.delaunay), batch.fundamental_arguments)), start=np.zeros(len(batch))))
            block_name, component = _AXES[term.axis]
            scale = getattr(batch, block_name)[:, component] * _MAS_TO_RAD
            basis[:, k, 0] = scale * np.fromiter((np.cos(a) for a in argument), dtype=float, count=len(batch))
            basis[:, k, 1] = scale * np.fromiter((np.sin(a) for a in argument), dtype=float, count=len(batch))
        batch.libration_bases[key] = basis
        return basis

    def design(self, batch: EquationBatch, out: np.ndarray) -> None:
        basis = self._basis(batch)
        for index, (k, c) in enumerate(self._slots):
            out[:, index] += basis[:, k, c]

    def _contributions(self, batch: EquationBatch):
        basis = self._basis(batch)
        for k, term in enumerate(self.terms):
            yield basis[:, k, 0] * self.values[term.name][0] + basis[:, k, 1] * self.values[term.name][1]

    def update(self, delta: np.ndarray) -> None:
        delta = np.asarray(delta, float)
        for (k, c), value in zip(self._slots, delta):
            self.values[self.terms[k].name][c] += float(value)

    def state(self) -> dict[str, object]:
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
