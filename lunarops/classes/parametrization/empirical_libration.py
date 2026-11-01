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

import erfa
import numpy as np

from lunarops.base.array_validation import parameter_vector
from lunarops.base.parameter_name import ParameterName
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.config.registry import register
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
_JULIAN_CENTURY_DAYS = 36_525.0


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

    def argument_rad(self, jd_tt: float) -> float:
        days = jd_tt - _J2000_JD
        if self.period_days is not None:
            return 2.0 * np.pi * days / self.period_days
        t = days / _JULIAN_CENTURY_DAYS
        fundamental = (erfa.fal03(t), erfa.falp03(t), erfa.faf03(t), erfa.fad03(t), erfa.faom03(t))
        return float(sum(k * value for k, value in zip(self.delaunay or (), fundamental)))


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
        string("name", required=True, non_empty=True, allow_none=False),
        string("axis", default="psi", choices=tuple(sorted(_AXES)), allow_none=False),
        mapping("delaunay"),
        number("periodDays"),
        number("cosMas", default=0.0, allow_none=False),
        number("sinMas", default=0.0, allow_none=False),
        boolean("estimate", default=True, allow_none=False),
        boolean("estimateSin", default=True, allow_none=False),
    )
)


@register(
    "parametrization",
    "empiricalLibration",
    schema=ConfigSchema(
        fields=(sequence("terms", required=True, item_kind="mapping", item_nested=_TERM_SCHEMA, min_items=1),),
        type_name="empiricalLibration",
    ),
)
class EmpiricalLibrationParametrization(Parametrization):
    """Estimate periodic PA-frame rotation corrections (mas)."""

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

    @classmethod
    def from_config(cls, config: dict, context) -> EmpiricalLibrationParametrization:
        return cls(terms=[_parse_term(item) for item in config["terms"]])

    def setup(self, equations: Sequence[ObservationEquation], model_state) -> None:
        return None

    def parameter_names(self) -> list[ParameterName]:
        return [
            ParameterName("moon", f"libration.{self.terms[k].name}.{'cos' if c == 0 else 'sin'}")
            for k, c in self._slots
        ]

    def reference_values(self) -> np.ndarray:
        return np.asarray([self.values[self.terms[k].name][c] for k, c in self._slots], dtype=float)

    @staticmethod
    def _partial(eq: ObservationEquation, block_name: str) -> np.ndarray:
        block = eq.design_partials.get(block_name)
        if block is None:
            raise KeyError(
                f"Observation equation lacks the {block_name!r} partial block; "
                "run the forward model with include_lunar_orientation_partials=True."
            )
        return np.asarray(block, dtype=float).reshape(3)

    def _basis(self, eq: ObservationEquation) -> np.ndarray:
        """Per-term [cos-column, sin-column] values in range metres per mas."""
        partials = {name: self._partial(eq, name) for name in {block for block, _ in (_AXES[t.axis] for t in self.terms)}}
        epoch = eq.transmit_epoch_utc
        # TT-UTC (~69 s) is irrelevant for periods of weeks to years.
        jd = epoch.jd1 + epoch.jd2
        basis = np.empty((len(self.terms), 2), dtype=float)
        for k, term in enumerate(self.terms):
            argument = term.argument_rad(jd)
            block_name, component = _AXES[term.axis]
            scale = partials[block_name][component] * _MAS_TO_RAD
            basis[k, 0] = scale * np.cos(argument)
            basis[k, 1] = scale * np.sin(argument)
        return basis

    def design_entries(self, eq: ObservationEquation) -> list[tuple[int, float]]:
        basis = self._basis(eq)
        return [(index, float(basis[k, c])) for index, (k, c) in enumerate(self._slots) if basis[k, c]]

    def reduce_observation(self, eq: ObservationEquation) -> float:
        basis = self._basis(eq)
        return float(sum(basis[k, 0] * self.values[t.name][0] + basis[k, 1] * self.values[t.name][1]
                         for k, t in enumerate(self.terms)))

    def apply_update(self, delta: np.ndarray) -> None:
        delta = parameter_vector(delta, expected_size=len(self._slots), name="empiricalLibration update")
        for (k, c), value in zip(self._slots, delta):
            self.values[self.terms[k].name][c] += float(value)

    def max_update_norm(self, delta: np.ndarray) -> float:
        # Convergence metric in mas.
        return float(np.max(np.abs(delta))) if len(delta) else 0.0

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
