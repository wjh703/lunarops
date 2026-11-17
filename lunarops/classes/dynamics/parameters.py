"""Initial values and dynamical parameters of the lunar orbit and rotation.

The quantities that a lunar-laser-ranging adjustment estimates from the
integrated orbit and rotation are named here once: the 15 initial values of
the integration state and 14 parameters of its right-hand side
(:data:`PARAMETER_DEFINITIONS`). Three of them, the orbital delays of the Earth
tide (``tau_o0``, ``tau_o1``, ``tau_o2``), are optional: they are estimated only
when listed, so the default set stays the 11 of the literature (Williams et al.
2013; Pavlov et al. 2016). A :class:`LunarDynamicsParameters` holds
values for any subset of them together with the epoch of the initial values,
and is stored as a ``lunarDynamicsParameters`` structured text artifact
(:func:`read_lunar_dynamics_parameters`,
:func:`write_lunar_dynamics_parameters`).

The functions below are the only place where a parameter value is turned into
model input. :func:`~lunarops.classes.dynamics.assembly.build_lunar_propagation_problem`
calls them, so a parameter file, a perturbed sibling model
(:func:`build_sibling_dynamics`) and a re-integration after an adjustment all
change the model in the same way:

- :func:`apply_to_config` rewrites the configuration values (rotational
  initial state, tide and deformation delays, inertia, core);
- :func:`earth_moon_gm` rescales the Earth and Moon GM;
- :func:`initial_orbit` replaces components of the initial orbit state;
- :func:`apply_to_moon_field` replaces Moon gravity coefficients.

A listed value that equals the value the configuration already implies
changes nothing, bit for bit; so a file written by :func:`nominal_parameters`
reproduces the unmodified model exactly.

Attributes
----------
PARAMETER_DEFINITIONS : tuple of ParameterDefinition
    The 29 quantities in their fixed order: initial values first, in state
    order, then the dynamical parameters.
PARAMETER_NAMES, INITIAL_VALUE_NAMES, DYNAMICAL_PARAMETER_NAMES : tuple of str
    Names of all quantities, of the 15 initial values and of the 11 default
    dynamical parameters.
OPTIONAL_DYNAMICAL_PARAMETER_NAMES, ESTIMABLE_DYNAMICAL_PARAMETER_NAMES : tuple of str
    The three optional orbital tide delays, and all 14 dynamical parameters.

Notes
-----
Coupled quantities:

``gm_emb``
    ``GM_E + GM_M``. The Earth and the Moon GM are scaled by the same factor,
    so their ratio (EMRAT) is unchanged; every model that uses either GM (point
    masses, the Earth tide, the lunar deformation, both gravity fields, the
    torques) is built from the scaled values.
``beta``, ``gamma``
    Moment ratios ``(C - A) / B`` and ``(B - A) / C`` of the undistorted Moon.
    The principal moments are rebuilt from them at fixed ``J2 = C - (A + B) / 2``
    (Park et al. 2021, Eqs. 47-51, as in
    :meth:`~lunarops.classes.dynamics.deformation.LunarDegree2GravityParameters.undistorted_inertia`).
    The core moment follows ``C`` through ``polarMomentFraction``; the friction
    coefficient ``viscosityPerS`` is rescaled with ``C`` so that ``kv_ct`` keeps
    its value unless it is changed itself.
``kv_ct``
    ``k_v / C_T`` in 1/d. The configuration value is
    ``viscosityPerS = kv_ct * C / 86400`` with ``C`` the polar moment
    normalised by ``M R^2``.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np

from lunarops.fileio.yaml_artifact import read_structured_text, write_structured_text

from .epoch_data import EpochBatch
from .gravity import GravityCoefficients

ARTIFACT_TYPE = "lunarDynamicsParameters"
"""Artifact type of a parameter file."""

_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    """One estimated quantity of the lunar dynamics.

    Attributes
    ----------
    name : str
        Stable identifier, used in parameter files and as a column name.
    unit : str
        Unit of the value.
    kind : str
        ``"initial"`` for an initial value of the integration state,
        ``"dynamical"`` for a parameter of the right-hand side.
    step : float
        Suggested half-width of a central difference in ``unit``, for
        finite-difference sensitivities.
    description : str
        Meaning of the quantity.
    """

    name: str
    unit: str
    kind: str
    step: float
    description: str


def _definitions() -> tuple[ParameterDefinition, ...]:
    axes = ("x", "y", "z")
    initial = [
        *(
            ParameterDefinition(f"r0_{axis}", "m", "initial", 1.0e-2, f"Initial Moon-minus-Earth position, ICRF {axis}")
            for axis in axes
        ),
        *(
            ParameterDefinition(
                f"v0_{axis}", "m/s", "initial", 1.0e-8, f"Initial Moon-minus-Earth velocity, ICRF {axis}"
            )
            for axis in axes
        ),
        *(
            ParameterDefinition(f"{angle}0", "rad", "initial", 1.0e-9, f"Initial mantle 3-1-3 Euler angle {angle}")
            for angle in ("phi", "theta", "psi")
        ),
        *(
            ParameterDefinition(
                f"wm0_{axis}", "rad/s", "initial", 1.0e-15, f"Initial mantle angular velocity, PA {axis}"
            )
            for axis in axes
        ),
        *(
            ParameterDefinition(
                f"wc0_{axis}", "rad/s", "initial", 1.0e-13, f"Initial fluid-core angular velocity, PA {axis}"
            )
            for axis in axes
        ),
    ]
    dynamical = [
        ParameterDefinition("gm_emb", "m3/s2", "dynamical", 4.0e5, "GM of the Earth-Moon system, GM_E + GM_M"),
        ParameterDefinition("tau_r1", "d", "dynamical", 1.0e-5, "Earth-tide rotational delay, diurnal"),
        ParameterDefinition("tau_r2", "d", "dynamical", 1.0e-5, "Earth-tide rotational delay, semidiurnal"),
        ParameterDefinition("tau_m", "d", "dynamical", 1.0e-4, "Lunar deformation delay"),
        ParameterDefinition("beta", "1", "dynamical", 1.0e-9, "Lunar moment ratio (C - A) / B"),
        ParameterDefinition("gamma", "1", "dynamical", 1.0e-9, "Lunar moment ratio (B - A) / C"),
        ParameterDefinition("c32", "1", "dynamical", 1.0e-10, "Fully normalised lunar gravity coefficient C32"),
        ParameterDefinition("s32", "1", "dynamical", 1.0e-10, "Fully normalised lunar gravity coefficient S32"),
        ParameterDefinition("c33", "1", "dynamical", 1.0e-10, "Fully normalised lunar gravity coefficient C33"),
        ParameterDefinition("kv_ct", "1/d", "dynamical", 1.0e-10, "Core-mantle friction k_v / C_T"),
        ParameterDefinition("fc", "1", "dynamical", 1.0e-6, "Core-mantle boundary flattening"),
        *(
            ParameterDefinition(f"tau_o{order}", "d", "dynamical", 1.0e-5, f"Earth-tide orbital delay, {band}")
            for order, band in enumerate(("zonal", "diurnal", "semidiurnal"))
        ),
    ]
    return (*initial, *dynamical)


PARAMETER_DEFINITIONS = _definitions()
PARAMETER_NAMES = tuple(definition.name for definition in PARAMETER_DEFINITIONS)
INITIAL_VALUE_NAMES = tuple(d.name for d in PARAMETER_DEFINITIONS if d.kind == "initial")
OPTIONAL_DYNAMICAL_PARAMETER_NAMES = ("tau_o0", "tau_o1", "tau_o2")
ESTIMABLE_DYNAMICAL_PARAMETER_NAMES = tuple(d.name for d in PARAMETER_DEFINITIONS if d.kind == "dynamical")
DYNAMICAL_PARAMETER_NAMES = tuple(
    name for name in ESTIMABLE_DYNAMICAL_PARAMETER_NAMES if name not in OPTIONAL_DYNAMICAL_PARAMETER_NAMES
)
_BY_NAME = MappingProxyType({definition.name: definition for definition in PARAMETER_DEFINITIONS})

_ORBIT_NAMES = INITIAL_VALUE_NAMES[:6]
_EULER_NAMES = INITIAL_VALUE_NAMES[6:9]
_MANTLE_NAMES = INITIAL_VALUE_NAMES[9:12]
_CORE_NAMES = INITIAL_VALUE_NAMES[12:15]
# Coefficient index (row C or S, degree, order) of each Moon gravity parameter.
_MOON_COEFFICIENTS = MappingProxyType({"c32": (0, 3, 2), "s32": (1, 3, 2), "c33": (0, 3, 3)})


def parameter_definition(name: str) -> ParameterDefinition:
    """Return the definition of a quantity.

    Parameters
    ----------
    name : str
        Name of the quantity.

    Returns
    -------
    ParameterDefinition
        Its definition.

    Raises
    ------
    KeyError
        If ``name`` is not one of :data:`PARAMETER_NAMES`.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown lunar dynamics parameter {name!r}; known: {', '.join(PARAMETER_NAMES)}") from None


@dataclass(frozen=True)
class LunarDynamicsParameters:
    """Values of initial values and dynamical parameters of the lunar dynamics.

    Any subset of :data:`PARAMETER_NAMES` may be present. The values are kept
    in the order of :data:`PARAMETER_DEFINITIONS`.

    Parameters
    ----------
    values : Mapping of str to float
        Value of each listed quantity, in the unit of its definition.
    epoch_tdb_jd1, epoch_tdb_jd2 : float
        Two-part TDB Julian date the initial values refer to.
    name : str, optional
        Free description of the set.

    Raises
    ------
    KeyError
        If a name is not one of :data:`PARAMETER_NAMES`.
    ValueError
        If a value or the epoch is not finite.
    """

    values: Mapping[str, float]
    epoch_tdb_jd1: float
    epoch_tdb_jd2: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        """Validate the names and store the values in definition order."""
        given = dict(self.values)
        for key in given:
            parameter_definition(key)
        ordered = {key: float(given[key]) for key in PARAMETER_NAMES if key in given}
        if not all(np.isfinite(value) for value in ordered.values()):
            raise ValueError("Lunar dynamics parameter values must be finite")
        jd1, jd2 = float(self.epoch_tdb_jd1), float(self.epoch_tdb_jd2)
        if not (np.isfinite(jd1) and np.isfinite(jd2)):
            raise ValueError("The epoch of lunar dynamics parameters must be finite")
        object.__setattr__(self, "values", MappingProxyType(ordered))
        object.__setattr__(self, "epoch_tdb_jd1", jd1)
        object.__setattr__(self, "epoch_tdb_jd2", jd2)
        object.__setattr__(self, "name", str(self.name))

    def __reduce__(self):
        """Pickle by constructor arguments (the read-only value mapping does not pickle)."""
        return (type(self), (dict(self.values), self.epoch_tdb_jd1, self.epoch_tdb_jd2, self.name))

    def __contains__(self, name: str) -> bool:
        """Return whether ``name`` has a value."""
        return name in self.values

    def __getitem__(self, name: str) -> float:
        """Return the value of ``name``."""
        return self.values[name]

    def __len__(self) -> int:
        """Return the number of listed quantities."""
        return len(self.values)

    @property
    def names(self) -> tuple[str, ...]:
        """Tuple of str: Listed names in definition order."""
        return tuple(self.values)

    def get(self, name: str, default: float | None = None) -> float | None:
        """Return the value of ``name``, or ``default`` when it is not listed."""
        return self.values.get(name, default)

    def vector(self, names: Iterable[str] | None = None) -> np.ndarray:
        """Return values as an array.

        Parameters
        ----------
        names : iterable of str or None, optional
            Quantities to return; all listed ones when ``None``.

        Returns
        -------
        numpy.ndarray
            Values in the order of ``names``.

        Raises
        ------
        KeyError
            If a requested quantity is not listed.
        """
        selected = self.names if names is None else tuple(names)
        return np.array([self.values[name] for name in selected], dtype=float)

    def with_value(self, name: str, value: float) -> LunarDynamicsParameters:
        """Return a copy with one value set or added.

        Parameters
        ----------
        name : str
            Quantity to set.
        value : float
            Its new value.

        Returns
        -------
        LunarDynamicsParameters
            New set; this one is unchanged.
        """
        return self.with_values({name: value})

    def with_values(self, values: Mapping[str, float]) -> LunarDynamicsParameters:
        """Return a copy with several values set or added.

        Parameters
        ----------
        values : Mapping of str to float
            New values by name.

        Returns
        -------
        LunarDynamicsParameters
            New set; this one is unchanged.
        """
        return dataclasses.replace(self, values={**self.values, **values})

    def is_at(self, jd1: float, jd2: float = 0.0, tolerance_days: float = 1.0e-9) -> bool:
        """Return whether the epoch of the set equals a two-part TDB Julian date."""
        return abs((self.epoch_tdb_jd1 - jd1) + (self.epoch_tdb_jd2 - jd2)) <= tolerance_days


def write_lunar_dynamics_parameters(parameters: LunarDynamicsParameters, path: str | Path) -> Path:
    """Write a ``lunarDynamicsParameters`` structured text artifact.

    The file lists, for each quantity, its name, value and unit; values are
    written with the shortest decimal form that reads back to the same double.

    Parameters
    ----------
    parameters : LunarDynamicsParameters
        Set to write.
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    Path
        The written path.
    """
    payload = {
        "name": parameters.name,
        "epochTdbJd1": parameters.epoch_tdb_jd1,
        "epochTdbJd2": parameters.epoch_tdb_jd2,
        "parameterCount": len(parameters),
        "parameters": [
            {"name": name, "value": value, "unit": _BY_NAME[name].unit} for name, value in parameters.values.items()
        ],
    }
    return write_structured_text(path, ARTIFACT_TYPE, payload)


def read_lunar_dynamics_parameters(path: str | Path) -> LunarDynamicsParameters:
    """Read and validate a ``lunarDynamicsParameters`` structured text artifact.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    LunarDynamicsParameters
        The listed quantities; a file may list any subset.

    Raises
    ------
    ValueError
        If ``parameterCount`` differs from the number of entries, a name is
        repeated, a unit differs from the unit of the definition, or a value
        is not a finite number.
    KeyError
        If a name is unknown.
    TypeError
        If ``parameters`` is not a list of mappings.
    """
    payload = read_structured_text(path, ARTIFACT_TYPE)
    entries = payload.get("parameters")
    if not isinstance(entries, list):
        raise TypeError("lunarDynamicsParameters parameters must be a sequence")
    values: dict[str, float] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"lunarDynamicsParameters parameters[{index}] must be a mapping")
        name = entry["name"]
        definition = parameter_definition(name)
        if name in values:
            raise ValueError(f"lunarDynamicsParameters lists {name!r} twice")
        if entry.get("unit") != definition.unit:
            raise ValueError(
                f"lunarDynamicsParameters {name!r} must be given in {definition.unit!r}, not {entry.get('unit')!r}"
            )
        value = entry["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"lunarDynamicsParameters {name!r} value must be a number")  # noqa: TRY004 -- invalid file content
        values[name] = float(value)
    if payload.get("parameterCount") != len(values):
        raise ValueError("lunarDynamicsParameters parameterCount does not match its entries")
    return LunarDynamicsParameters(
        values,
        float(cast(float, payload["epochTdbJd1"])),
        float(cast(float, payload.get("epochTdbJd2", 0.0))),
        str(payload.get("name", "")),
    )


def moment_ratios(inertia_moments) -> tuple[float, float, float]:
    """Return ``(J2, beta, gamma)`` of principal moments ``(A, B, C)``.

    Parameters
    ----------
    inertia_moments : array_like
        Principal moments ``(A, B, C)`` normalised by ``M R^2``.

    Returns
    -------
    j2 : float
        ``C - (A + B) / 2``.
    beta : float
        ``(C - A) / B``.
    gamma : float
        ``(B - A) / C``.
    """
    a, b, c = (float(value) for value in inertia_moments)
    # The differences of the nearly equal moments are exact; summing A + B first would round J2 at 1e-13.
    return ((c - a) + (c - b)) / 2, (c - a) / b, (b - a) / c


def inertia_from_ratios(j2: float, beta: float, gamma: float) -> tuple[float, float, float]:
    """Return the principal moments ``(A, B, C)`` of ``(J2, beta, gamma)``.

    The inverse of :func:`moment_ratios` (Park et al. 2021, Eqs. 47-51).

    Parameters
    ----------
    j2 : float
        Unnormalised ``J2 = C - (A + B) / 2`` of the undistorted Moon.
    beta, gamma : float
        Moment ratios ``(C - A) / B`` and ``(B - A) / C``.

    Returns
    -------
    tuple of float
        ``(A, B, C)`` normalised by ``M R^2``.

    Raises
    ------
    ValueError
        Unless ``0 < gamma < beta < 1`` and ``j2 > 0``.
    """
    if not (0 < gamma < beta < 1) or j2 <= 0:
        raise ValueError("Lunar moments require 0 < gamma < beta < 1 and positive J2")
    factor = 2 * j2 / (2 * beta - gamma + beta * gamma)
    return factor * (1 - beta * gamma), factor * (1 + gamma), factor * (1 + beta)


def _tide_config(config):
    tide = config.get("earthTide")
    return tide if tide is not None and "tide" in config.get("bodyForceTerms", ("tide",)) else None


def config_parameters(config) -> dict[str, float]:
    """Return the values a configuration implies for the quantities it holds.

    These are the rotational initial values, the tide and deformation delays,
    the moment ratios and the core parameters; a quantity whose model is not
    configured is absent. The orbit initial values, ``gm_emb`` and the Moon
    gravity coefficients come from the ephemeris, the mass catalog and the
    gravity-field file (see :func:`nominal_parameters`).

    Parameters
    ----------
    config : Mapping
        ``LunarOrbitPropagation`` configuration.

    Returns
    -------
    dict of str to float
        Values by name.
    """
    values: dict[str, float] = {}
    rotation = config.get("lunarRotation")
    if rotation is not None:
        values.update(zip(_EULER_NAMES, map(float, rotation["initialEulerAnglesRad"]), strict=True))
        values.update(zip(_MANTLE_NAMES, map(float, rotation["initialAngularVelocityRadS"]), strict=True))
        core = rotation.get("core")
        if core is not None:
            values.update(zip(_CORE_NAMES, map(float, rotation["initialCoreAngularVelocityRadS"]), strict=True))
    tide = _tide_config(config)
    if tide is not None:
        values["tau_r1"], values["tau_r2"] = float(tide["tauRotDays"][1]), float(tide["tauRotDays"][2])
        values.update((f"tau_o{order}", float(tide["tauOrbDays"][order])) for order in range(3))
    if rotation is not None:
        if rotation.get("deformation") is not None:
            values["tau_m"] = float(rotation["deformation"]["delayDays"])
        _, values["beta"], values["gamma"] = moment_ratios(rotation["inertiaMoments"])
        core = rotation.get("core")
        if core is not None:
            polar = float(rotation["inertiaMoments"][2])
            values["kv_ct"] = float(core["viscosityPerS"]) * _SECONDS_PER_DAY / polar
            values["fc"] = float(core["flattening"])
    return values


def _changed(parameters: LunarDynamicsParameters | None, nominal: Mapping[str, float], names) -> dict[str, float]:
    """Return the listed values among ``names`` that differ from the nominal ones.

    Raises
    ------
    ValueError
        If a listed quantity has no nominal value, that is, its model is not configured.
    """
    if parameters is None:
        return {}
    changed = {}
    for name in names:
        if name not in parameters:
            continue
        if name not in nominal:
            raise ValueError(f"Lunar dynamics parameter {name!r} is given, but its model is not configured")
        if parameters[name] != nominal[name]:
            changed[name] = parameters[name]
    return changed


_CONFIG_NAMES = (
    *_EULER_NAMES,
    *_MANTLE_NAMES,
    *_CORE_NAMES,
    "tau_r1",
    "tau_r2",
    *OPTIONAL_DYNAMICAL_PARAMETER_NAMES,
    "tau_m",
    "beta",
    "gamma",
    "kv_ct",
    "fc",
)


def apply_to_config(config, parameters: LunarDynamicsParameters | None):
    """Return a configuration with the configuration-level parameters applied.

    Rewrites ``lunarRotation.initial*``, ``earthTide.tauRotDays[1:]``, ``earthTide.tauOrbDays``,
    ``lunarRotation.deformation.delayDays``, ``lunarRotation.inertiaMoments``
    and ``lunarRotation.core.{viscosityPerS, flattening}``. See the module
    notes for the coupling of ``beta``, ``gamma`` and ``kv_ct``.

    Parameters
    ----------
    config : Mapping
        ``LunarOrbitPropagation`` configuration; not modified.
    parameters : LunarDynamicsParameters or None
        Values to apply; ``None`` applies nothing.

    Returns
    -------
    Mapping
        ``config`` itself when nothing changes, otherwise a copy with the
        changed blocks replaced.

    Raises
    ------
    ValueError
        If a listed quantity belongs to a model that is not configured, or
        the moment ratios are not ``0 < gamma < beta < 1``.
    """
    nominal = config_parameters(config)
    changed = _changed(parameters, nominal, _CONFIG_NAMES)
    if not changed:
        return config
    config = dict(config)
    tide_names = {"tau_r1", "tau_r2", *OPTIONAL_DYNAMICAL_PARAMETER_NAMES}
    if tide_names & set(changed):
        tide = dict(config["earthTide"])
        delays = list(tide["tauRotDays"])
        delays[1] = changed.get("tau_r1", delays[1])
        delays[2] = changed.get("tau_r2", delays[2])
        tide["tauRotDays"] = delays
        tide["tauOrbDays"] = [changed.get(f"tau_o{order}", tide["tauOrbDays"][order]) for order in range(3)]
        config["earthTide"] = tide
    rotation_names = set(changed) - tide_names
    if rotation_names:
        rotation = copy.deepcopy(dict(config["lunarRotation"]))
        for key, names in (
            ("initialEulerAnglesRad", _EULER_NAMES),
            ("initialAngularVelocityRadS", _MANTLE_NAMES),
            ("initialCoreAngularVelocityRadS", _CORE_NAMES),
        ):
            if any(name in changed for name in names):
                rotation[key] = [changed.get(name, nominal[name]) for name in names]
        if "tau_m" in changed:
            rotation["deformation"] = {**rotation["deformation"], "delayDays": changed["tau_m"]}
        polar = float(rotation["inertiaMoments"][2])
        if "beta" in changed or "gamma" in changed:
            j2, beta, gamma = moment_ratios(rotation["inertiaMoments"])
            moments = inertia_from_ratios(j2, changed.get("beta", beta), changed.get("gamma", gamma))
            rotation["inertiaMoments"] = list(moments)
            polar = moments[2]
        if rotation.get("core") is not None and ({"beta", "gamma", "kv_ct", "fc"} & set(changed)):
            core = dict(rotation["core"])
            if {"beta", "gamma", "kv_ct"} & set(changed):
                # k_v / C_T keeps its value when only the polar moment changes.
                core["viscosityPerS"] = changed.get("kv_ct", nominal["kv_ct"]) * polar / _SECONDS_PER_DAY
            if "fc" in changed:
                core["flattening"] = changed["fc"]
            rotation["core"] = core
        config["lunarRotation"] = rotation
    return config


def earth_moon_gm(
    parameters: LunarDynamicsParameters | None, earth_gm_m3_s2: float, moon_gm_m3_s2: float
) -> tuple[float, float]:
    """Return the Earth and Moon GM for a ``gm_emb`` value.

    Parameters
    ----------
    parameters : LunarDynamicsParameters or None
        Values; ``gm_emb`` is used when listed.
    earth_gm_m3_s2, moon_gm_m3_s2 : float
        Catalog GM of the Earth and the Moon in m^3/s^2.

    Returns
    -------
    tuple of float
        Earth and Moon GM in m^3/s^2, both scaled by
        ``gm_emb / (earth + moon)``; the catalog values when ``gm_emb`` is not
        listed or equals their sum.

    Raises
    ------
    ValueError
        If ``gm_emb`` is not positive.
    """
    total = earth_gm_m3_s2 + moon_gm_m3_s2
    if parameters is None or "gm_emb" not in parameters or parameters["gm_emb"] == total:
        return earth_gm_m3_s2, moon_gm_m3_s2
    if parameters["gm_emb"] <= 0:
        raise ValueError("gm_emb must be positive")
    scale = parameters["gm_emb"] / total
    return earth_gm_m3_s2 * scale, moon_gm_m3_s2 * scale


def initial_orbit(parameters: LunarDynamicsParameters | None, ephemeris_state) -> np.ndarray:
    """Return the initial Moon-minus-Earth state with the listed components replaced.

    Parameters
    ----------
    parameters : LunarDynamicsParameters or None
        Values; ``r0_*`` and ``v0_*`` are used when listed.
    ephemeris_state : array_like
        Ephemeris Moon-minus-Earth position (m) and velocity (m/s) with ICRF
        axes, shape ``(6,)``.

    Returns
    -------
    numpy.ndarray
        Initial state of shape ``(6,)``.
    """
    state = np.array(ephemeris_state, dtype=float)[:6].copy()
    if parameters is not None:
        for index, name in enumerate(_ORBIT_NAMES):
            if name in parameters:
                state[index] = parameters[name]
    return state


def apply_to_moon_field(
    parameters: LunarDynamicsParameters | None, coefficients: GravityCoefficients
) -> GravityCoefficients:
    """Return Moon gravity coefficients with the listed ones replaced.

    Parameters
    ----------
    parameters : LunarDynamicsParameters or None
        Values; ``c32``, ``s32`` and ``c33`` are used when listed.
    coefficients : GravityCoefficients
        Loaded Moon field; not modified.

    Returns
    -------
    GravityCoefficients
        ``coefficients`` itself when nothing changes, otherwise a copy whose
        ``source_metadata`` records the replaced values under
        ``parameterOverrides``.

    Raises
    ------
    ValueError
        If a listed coefficient lies beyond the degree of the field.
    """
    if parameters is None:
        return coefficients
    values = coefficients.coeffs.copy()
    overrides = {}
    for name, (row, degree, order) in _MOON_COEFFICIENTS.items():
        if name not in parameters:
            continue
        if degree >= values.shape[1]:
            raise ValueError(f"Lunar dynamics parameter {name!r} needs a Moon gravity field of degree {degree}")
        if parameters[name] != values[row, degree, order]:
            values[row, degree, order] = overrides[name] = parameters[name]
    if not overrides:
        return coefficients
    metadata = {**(coefficients.source_metadata or {}), "parameterOverrides": overrides}
    return dataclasses.replace(coefficients, _coeffs=values, source_metadata=metadata)


def check_applicable(parameters: LunarDynamicsParameters | None, *, has_moon_field: bool) -> None:
    """Raise ``ValueError`` if Moon gravity coefficients are listed without a Moon field."""
    if parameters is None or has_moon_field:
        return
    listed = [name for name in _MOON_COEFFICIENTS if name in parameters]
    if listed:
        raise ValueError(f"Lunar dynamics parameters {listed} are given, but no Moon gravity field is configured")


def nominal_parameters(
    config,
    *,
    epoch_tdb_jd1: float,
    epoch_tdb_jd2: float,
    initial_orbit_state,
    earth_gm_m3_s2: float,
    moon_gm_m3_s2: float,
    moon_coefficients: GravityCoefficients | None,
    name: str = "",
) -> LunarDynamicsParameters:
    """Return every quantity of a built model with its current value.

    Parameters
    ----------
    config : Mapping
        Configuration of the model, with any parameters already applied.
    epoch_tdb_jd1, epoch_tdb_jd2 : float
        Two-part TDB Julian date of the initial epoch.
    initial_orbit_state : array_like
        Initial Moon-minus-Earth position (m) and velocity (m/s), shape ``(6,)``.
    earth_gm_m3_s2, moon_gm_m3_s2 : float
        Earth and Moon GM of the model in m^3/s^2.
    moon_coefficients : GravityCoefficients or None
        Moon gravity field of the model.
    name : str, optional
        Description of the set.

    Returns
    -------
    LunarDynamicsParameters
        All quantities whose model is configured: 26 with a deforming Moon
        with a core, the Earth tide and a Moon field of degree 3 or more.
    """
    values = dict(zip(_ORBIT_NAMES, np.asarray(initial_orbit_state, dtype=float)[:6], strict=True))
    values["gm_emb"] = earth_gm_m3_s2 + moon_gm_m3_s2
    values.update(config_parameters(config))
    if moon_coefficients is not None:
        coeffs = moon_coefficients.coeffs
        for key, (row, degree, order) in _MOON_COEFFICIENTS.items():
            if degree < coeffs.shape[1]:
                values[key] = float(coeffs[row, degree, order])
    return LunarDynamicsParameters(values, epoch_tdb_jd1, epoch_tdb_jd2, name)


def build_sibling_dynamics(problem, context, values: Mapping[str, float]):
    """Build the dynamics of a problem with some parameter values changed.

    The sibling shares the ephemeris of ``context`` and the prepropagated
    Kuiper-belt ring of ``problem``; every other model is built anew from the
    changed parameters, through the same code as the problem itself. Sharing
    the ring ignores the dependence of the ring trajectories on the Earth and
    Moon GM, which is far below the rounding of the lunar orbit.

    Parameters
    ----------
    problem : LunarPropagationProblem
        Nominal problem, built by
        :func:`~lunarops.classes.dynamics.assembly.build_lunar_propagation_problem`.
    context : object
        The program context ``problem`` was built with.
    values : Mapping of str to float
        New values by name; the other quantities keep the values of ``problem``.

    Returns
    -------
    LunarDynamics
        Dynamics with the changed parameters. Changed initial values do not
        enter the dynamics.
    """
    from .assembly import build_lunar_propagation_problem

    sibling = build_lunar_propagation_problem(
        problem.config,
        context,
        parameters=problem.parameters.with_values(values),
        kuiper_belt_ring=problem.dynamics.kuiper_belt_ring,
    )
    return sibling.dynamics


LOCAL_INCREMENTS = MappingProxyType(
    {
        "gm_emb": 4.0e8,
        "tau_r1": 1.0e-3,
        "tau_r2": 1.0e-3,
        "tau_m": 1.0e-2,
        "beta": 1.0e-7,
        "gamma": 1.0e-7,
        "c32": 1.0e-6,
        "s32": 1.0e-6,
        "c33": 1.0e-6,
        "kv_ct": 1.0e-9,
        "fc": 1.0e-5,
        "tau_o0": 1.0e-3,
        "tau_o1": 1.0e-3,
        "tau_o2": 1.0e-3,
    }
)
"""Half-widths of the local central differences of the right-hand side, by dynamical parameter.

Unlike ``ParameterDefinition.step`` (for differences of whole propagations) they only have to resolve the
change of the right-hand side at one epoch, which is linear or nearly so in every parameter. Except for ``tau_m``
(the delayed epochs of its deformation) and ``gm_emb`` (its point-mass pair), the chain rule of
``_ParameterForcing`` does not use them: they set the sibling differences its check compares with, which must stay
within 1e-3 of the orbit block. Those round against the whole sibling forcing and truncate as the parameter
enters it, so each increment sits where both are small, measured as the largest error over the allowed one in 150
checks of a 100-year arc at 5400 s:

- ``c32``, ``s32``, ``c33``: 1e-6. The forcing is linear in them, and the orbit block rounds against the figure
  forces: 15 times the allowed error at 1e-8, 0.06 at 1e-6.
- ``tau_r1``, ``tau_r2``: 1e-3 d. The order-1 and order-2 tide forces are four orders of magnitude below the
  whole tide force the siblings round against: ``tau_r1`` reached 120 times the allowed error at 1e-4 d and
  about 1 at 1e-3 d, where its orbit column passes close to zero (see ``_ParameterForcing.CONFIRMATIONS``),
  while the truncation (m ω_E δ)² / 6 of the rotation of the tide vectors is 7e-6 for order 1 and 3e-5 for
  order 2.
- ``tau_m``: 1e-2 d. Its sibling inertias differ by 1e-12 of an undeformed tensor of 0.4, and the delayed epochs
  t - tau_m -+ delta of the two siblings take different history stencils: 6e-3 of the orbit block at 1e-4 d,
  3e-5 at 1e-2 d, while the chain-rule column moved by 3e-7, the truncation (n delta)^2 / 6 of the monthly tides.
  It is capped at :data:`TAU_M_INCREMENT_FRACTION` of the nominal ``tau_m``.
- ``tau_o0``, ``tau_o1``, ``tau_o2``: 1e-3 d. Their columns are the sibling differences themselves (the delayed
  raiser positions move with the epoch, see ``_ParameterForcing``), taken in double-double; the truncation
  (n δ)² / 6 of the monthly motion of the delayed Moon is 1e-8.
"""

TAU_M_INCREMENT_FRACTION = 0.1
"""Largest local increment of ``tau_m`` as a fraction of its nominal value."""


def build_parameter_variations(problem, context, names: Iterable[str] | None = None, increments=None):
    """Build the parameter variations of the variational equations of a problem.

    Parameters
    ----------
    problem : LunarPropagationProblem
        Nominal problem.
    context : object
        The program context ``problem`` was built with.
    names : iterable of str or None, optional
        Dynamical parameters, in column order; every default dynamical parameter
        (:data:`DYNAMICAL_PARAMETER_NAMES`) the model of ``problem`` has when ``None``.
    increments : Mapping of str to float or None, optional
        Half-widths replacing those of :data:`LOCAL_INCREMENTS` (and the cap of ``tau_m``).

    Returns
    -------
    list of ParameterVariation
        One variation per parameter, each with the two sibling dynamics on either side of the nominal value.

    Raises
    ------
    ValueError
        If a name is not a dynamical parameter of the model of ``problem``.
    """
    from .variational import ParameterVariation

    available = [name for name in ESTIMABLE_DYNAMICAL_PARAMETER_NAMES if name in problem.parameters]
    names = [name for name in available if name in DYNAMICAL_PARAMETER_NAMES] if names is None else list(names)
    variations = []
    for name in names:
        if name not in available:
            raise ValueError(f"{name!r} is not a dynamical parameter of this model")
        nominal = problem.parameters[name]
        if increments and name in increments:
            increment = float(increments[name])
        elif name == "tau_m":
            # Both siblings keep a delay of at least 0.9 tau_m, inside the history of a delay shorter than a step.
            increment = min(LOCAL_INCREMENTS[name], TAU_M_INCREMENT_FRACTION * abs(float(nominal)))
        else:
            increment = LOCAL_INCREMENTS[name]
        variations.append(
            ParameterVariation(
                name,
                increment,
                build_sibling_dynamics(problem, context, {name: nominal + increment}),
                build_sibling_dynamics(problem, context, {name: nominal - increment}),
            )
        )
    return variations


def batch_with_earth_moon_gm(batch: EpochBatch, earth_gm_m3_s2: float, moon_gm_m3_s2: float) -> EpochBatch:
    """Return an epoch batch whose Earth and Moon GM columns are replaced.

    The native kernel reads every GM from the batch, so a model with another
    ``gm_emb`` must be evaluated on such a batch. All other arrays are shared
    with ``batch``; they hold data of the external bodies only and do not
    depend on the Earth or the Moon GM.

    Parameters
    ----------
    batch : EpochBatch
        Prescribed inputs.
    earth_gm_m3_s2, moon_gm_m3_s2 : float
        Earth and Moon GM in m^3/s^2 (body columns 0 and 1).

    Returns
    -------
    EpochBatch
        Batch with a new ``gravitational_parameters_m3_s2`` array.
    """
    gm = batch.gravitational_parameters_m3_s2.copy()
    gm[:, 0] = earth_gm_m3_s2
    gm[:, 1] = moon_gm_m3_s2
    return dataclasses.replace(batch, gravitational_parameters_m3_s2=gm)


__all__ = [
    "ARTIFACT_TYPE",
    "DYNAMICAL_PARAMETER_NAMES",
    "ESTIMABLE_DYNAMICAL_PARAMETER_NAMES",
    "INITIAL_VALUE_NAMES",
    "LOCAL_INCREMENTS",
    "OPTIONAL_DYNAMICAL_PARAMETER_NAMES",
    "PARAMETER_DEFINITIONS",
    "PARAMETER_NAMES",
    "LunarDynamicsParameters",
    "ParameterDefinition",
    "apply_to_config",
    "apply_to_moon_field",
    "batch_with_earth_moon_gm",
    "build_parameter_variations",
    "build_sibling_dynamics",
    "check_applicable",
    "config_parameters",
    "earth_moon_gm",
    "inertia_from_ratios",
    "initial_orbit",
    "moment_ratios",
    "nominal_parameters",
    "parameter_definition",
    "read_lunar_dynamics_parameters",
    "write_lunar_dynamics_parameters",
]
