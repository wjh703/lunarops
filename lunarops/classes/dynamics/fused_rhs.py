"""Compilation of force and torque records into the native right-hand side.

:func:`compile_fused_rhs` translates the force and torque records of a
:class:`~lunarops.classes.dynamics.lunar_dynamics.LunarDynamics` into the
configuration of :class:`lunarops._dynamics_core.LunarRhsKernel`.
:class:`FusedRhs` drives that kernel: it loads epoch and step data only when
they change and maps the kernel's term rows back to force names.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

from lunarops._dynamics_core import KERNEL_EVAL_FULL, LunarRhsKernel
from lunarops.base.constants import C

from .force_models import (
    EarthFigureForce,
    EarthTideForce,
    EihPointMassForce,
    LenseThirringForce,
    MoonFigureForce,
    NewtonianPointMassForce,
    SolarJ2Force,
    SolarRadiationPressureForce,
)
from .torques import EarthFigureTorque, LunarFigureTorque, PointMassGravityTorque

# Order matches the eight force_terms rows in _lunar_rhs.pxi.
_FORCE_TYPES = (
    NewtonianPointMassForce,
    EihPointMassForce,
    SolarJ2Force,
    EarthFigureForce,
    MoonFigureForce,
    EarthTideForce,
    LenseThirringForce,
    SolarRadiationPressureForce,
)


def _tide_step(model, epoch, batch, index, history):
    matrix = np.ascontiguousarray(batch.earth_fixed2inertial_matrix[index].T)
    return model.delayed_vectors(epoch, matrix, history)


def _degree2_step(model, epoch, batch, index, history):
    def relative_position(epoch):
        # Earth minus Moon, from the relative state rather than two barycentric positions.
        return -history.geocentric_positions(("MOON",), (epoch,))[0, 0]

    return model.degree2_coefficients(epoch, relative_position)


def _field_config(field) -> dict:
    coefficients = np.ascontiguousarray(field.coefficients.coeffs, dtype=float).copy()
    return {
        "coefficients": coefficients,
        "norm": np.ascontiguousarray(field.normalization_table, dtype=float).copy(),
        "gm": field.gm_m3_s2,
        "radius": field.radius_m,
        "degree": coefficients.shape[1] - 1,
    }


class FusedRhs:
    """Driver of a compiled :class:`~lunarops._dynamics_core.LunarRhsKernel`.

    Instances are created by :func:`compile_fused_rhs`.

    Parameters
    ----------
    kernel : LunarRhsKernel
        Compiled native kernel.
    moon_field : GravityField or None
        Moon field whose degree 2 is replaced per step, or ``None`` when the
        kernel has no Moon figure targets.
    earth_c20 : tuple of float
        ``(C̄20, reference_jd_tdb, linear_per_year, quadratic_per_year2)`` of
        the Earth figure force.
    earth_j2 : tuple of float
        ``(J2, reference_jd_tdb, linear_per_year, quadratic_per_year2)`` of
        the Earth figure torque.
    uses_tide : bool
        Whether the Earth tide is active.
    rotation_inertia : numpy.ndarray
        Undeformed whole-Moon inertia used when no deformation is evaluated,
        shape ``(3, 3)``.
    moon_degree2_force : str or None
        Name of the force whose step payload is the prescribed lunar degree 2.
    force_indices : list of tuple of (str, int)
        Kernel force-term row of every active force name.
    step_models : list of tuple of (str, Callable)
        Per-step payload builders ``f(epoch, batch, index, history)`` keyed
        by force name.
    kernel_config : dict or None, optional
        Configuration the kernel was built from.
    """

    def __init__(
        self,
        kernel,
        moon_field,
        earth_c20: tuple[float, float, float, float],
        earth_j2: tuple[float, float, float, float],
        uses_tide: bool,
        rotation_inertia,
        moon_degree2_force: str | None,
        force_indices: list[tuple[str, int]],
        step_models: list[tuple[str, Callable]],
        kernel_config: dict | None = None,
    ):
        self.step_models = step_models
        self.kernel_config = {} if kernel_config is None else kernel_config
        self.force_indices = force_indices
        self.kernel = kernel
        self._moon_field = moon_field
        self.earth_c20 = earth_c20
        self.earth_j2 = earth_j2
        self._uses_tide = uses_tide
        self._rotation_inertia = rotation_inertia
        self._moon_degree2_force = moon_degree2_force
        self._batch = None
        self._index = -1
        self._step = None
        self._zero_matrix = np.zeros((3, 3))
        self._empty_vectors = np.zeros((0, 3, 3))
        self._empty_gm = np.zeros(0)
        self._no_coefficients = np.zeros((2, 1, 1))

    def prepare_step(self, epoch, batch, index, history) -> dict[str, object]:
        """Return the per-step force payloads.

        Parameters
        ----------
        epoch : Epoch
            TDB epoch of the step.
        batch : EpochBatch
            Prescribed inputs containing ``epoch``.
        index : int
            Row of ``epoch`` in ``batch``.
        history : StateHistoryProvider
            Accepted history.

        Returns
        -------
        dict
            Payload per force name, for example the delayed tide vectors.
        """
        return {name: prepare(epoch, batch, index, history) for name, prepare in self.step_models}

    def step_inputs(self, step):
        """Return the kernel step inputs of ``step``: tide vectors, raiser GM, inertia, its derivative, Moon coefficients."""
        if self._uses_tide:
            vectors, raiser_gm = step.force_payloads["tide"]
        else:
            vectors, raiser_gm = self._empty_vectors, self._empty_gm
        inertia = step.lunar_inertia
        if inertia is None:
            total, derivative = self._rotation_inertia, self._zero_matrix
        else:
            total, derivative = inertia.total, inertia.derivative
        if self._moon_field is None:
            coefficients = self._no_coefficients
        else:
            degree2 = None if inertia is None else inertia.degree2
            if degree2 is None and self._moon_degree2_force is not None:
                degree2 = step.force_payloads[self._moon_degree2_force]
            field = self._moon_field
            coefficients = field.coefficients.coeffs if degree2 is None else field.coefficients_with_degree2(degree2)
        return vectors, raiser_gm, total, derivative, coefficients

    def _load_step(self, step) -> None:
        self.kernel.set_step(*self.step_inputs(step))
        self._step = step

    def input_vector(self, step, earth_moon_gm) -> np.ndarray:
        """Return the step inputs of ``step`` and the model inputs, flat in the kernel's ``input_layout``.

        These are the inputs :meth:`input_differences` perturbs: the step payloads, the core viscosity and
        diagonal, and the Earth and Moon GM of the epoch (``earth_moon_gm``, the first two GM of the epoch batch)
        with those of the Earth tide and of the Earth and Moon fields.
        """
        vectors, raiser_gm, total, derivative, coefficients = self.step_inputs(step)
        config = self.kernel_config
        core = (config.get("rotation") or {}).get("core")
        tide = config.get("tide") or {}
        gm = [
            *np.asarray(earth_moon_gm, dtype=float)[:2],
            tide.get("mu_earth", 0.0),
            tide.get("mu_moon", 0.0),
            (config.get("earth_figure") or {}).get("gm", 0.0),
            (config.get("moon_field") or {}).get("gm", 0.0),
        ]
        parts = []
        if self._uses_tide:
            parts += [np.ravel(vectors), np.ravel(raiser_gm)]
        parts += [np.ravel(total), np.ravel(derivative)]
        if self._moon_field is not None:
            parts.append(np.ravel(coefficients))
        parts += [
            np.asarray([0.0 if core is None else core["viscosity"]]),
            np.zeros(3) if core is None else np.asarray(core["diagonal"], dtype=float),
            np.asarray(gm, dtype=float),
        ]
        vector = np.concatenate([np.asarray(part, dtype=float) for part in parts])
        if vector.size != self.input_size:
            raise ValueError("Step inputs do not match the kernel input layout")
        return vector

    @property
    def input_size(self) -> int:
        """int: Length of :meth:`input_vector`."""
        _, offset, size = self.kernel.input_layout[-1]
        return offset + size

    def input_differences(self, batch, index, step, state, plus, minus, increments, modes) -> np.ndarray:
        """Return central differences of ``f`` over shifts of the inputs (see LunarRhsKernel), ``(d, m)``."""
        self._load(batch, index, step)
        return self.kernel.input_differences(
            np.ascontiguousarray(state, dtype=float),
            np.ascontiguousarray(plus, dtype=float),
            np.ascontiguousarray(minus, dtype=float),
            np.ascontiguousarray(increments, dtype=float),
            np.ascontiguousarray(modes, dtype=np.intc),
        )

    def _load(self, batch, index, step) -> None:
        if batch is not self._batch or index != self._index:
            self.kernel.set_epoch(batch, index)
            self._batch, self._index = batch, index
        if step is not self._step:
            self._load_step(step)

    def split_derivative(
        self, batch, index, step, state, state_low, mode: int = KERNEL_EVAL_FULL
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the state derivative as a double-double pair.

        Parameters
        ----------
        batch : EpochBatch
            Prescribed inputs.
        index : int
            Row of the epoch in ``batch``.
        step : StepData
            Candidate-independent step data.
        state, state_low : array_like
            High and low parts of the integration state, shape ``(d,)``.

        Returns
        -------
        derivative, derivative_low : numpy.ndarray
            High and low parts of the state derivative, shape ``(d,)``.

        Raises
        ------
        ValueError
            If the kernel reports a degenerate state.
        """
        self._load(batch, index, step)
        return self.kernel.split_derivative(
            np.ascontiguousarray(state, dtype=float), np.ascontiguousarray(state_low, dtype=float), mode
        )

    def state_jacobian(self, batch, index, step, state, increments) -> np.ndarray:
        """Return ∂f/∂y at fixed step data by central differences (see LunarRhsKernel.state_jacobian)."""
        self._load(batch, index, step)
        return self.kernel.state_jacobian(
            np.ascontiguousarray(state, dtype=float), np.ascontiguousarray(increments, dtype=float)
        )

    def tide_jacobian(self, batch, index, step, state, shifts, increment: float) -> np.ndarray:
        """Return central differences of f for shifts of the step's tide vectors (see LunarRhsKernel)."""
        self._load(batch, index, step)
        return self.kernel.tide_jacobian(
            np.ascontiguousarray(state, dtype=float), np.ascontiguousarray(shifts, dtype=float), float(increment)
        )

    def terms(self, batch, index, step, state):
        """Return the derivative and every individual term.

        Parameters
        ----------
        batch : EpochBatch
            Prescribed inputs.
        index : int
            Row of the epoch in ``batch``.
        step : StepData
            Candidate-independent step data.
        state : array_like
            Integration state, shape ``(d,)``.

        Returns
        -------
        dict
            Kernel term dictionary (see
            :meth:`~lunarops.classes.dynamics.lunar_dynamics.LunarDynamics.terms`)
            with ``"forces"`` and ``"earth_forces"`` keyed by force name.

        Raises
        ------
        ValueError
            If the kernel reports a degenerate state.
        """
        self._load(batch, index, step)
        result = self.kernel.terms(np.ascontiguousarray(state, dtype=float))
        for key in ("forces", "earth_forces"):
            result[key] = {name: result[key][slot].copy() for name, slot in self.force_indices}
        return result


def compile_fused_rhs(dynamics) -> FusedRhs:
    """Compile the force and torque models of a dynamics into a native kernel.

    Parameters
    ----------
    dynamics : LunarDynamics
        Dynamics whose active forces, rotation and torques are compiled.

    Returns
    -------
    FusedRhs
        Driver of the compiled kernel.

    Raises
    ------
    ValueError
        If a force or torque type is unsupported, a model is duplicated,
        the Sun is missing for a solar term, or the Moon field differs
        between the figure force, the torques and the deformation.
    """
    models = dynamics.force_group.active_models
    if any(not isinstance(model, _FORCE_TYPES) for model in models):
        raise ValueError("unsupported: force model")
    rotation = dynamics.rotation
    names = dynamics.body_names
    sun = names.index("SUN") if "SUN" in names else -1
    config: dict = {
        "n_bodies": len(names),
        "sun_index": sun,
        "speed_of_light_squared": C**2,
        "point_mass": False,
        "eih": False,
        "solar_j2": None,
        "earth_figure": None,
        "moon_figure_slots": None,
        "moon_field": None,
        "moon_targets": [],
        "tide": None,
        "lense_thirring": None,
        "radiation_pressure": None,
        "rotation": None,
        "torque_kind": [],
        "torque_arg": [],
    }
    targets: list[int] = []

    def slot(body: int) -> int:
        if body not in targets:
            targets.append(body)
        return targets.index(body)

    earth_c20 = (0.0, 2451545.0, 0.0, 0.0)
    earth_j2 = (0.0, 2451545.0, 0.0, 0.0)
    moon_field = None
    moon_degree2_force = None
    force_partner_bodies: list[int] | None = None
    uses_tide = False
    step_models: list[tuple[str, Callable]] = []
    for model in models:
        if isinstance(model, NewtonianPointMassForce):
            config["point_mass"] = True
        elif isinstance(model, EihPointMassForce):
            config["eih"] = True
        elif isinstance(model, SolarJ2Force):
            if sun < 0 or config["solar_j2"] is not None:
                raise ValueError("unsupported: solar J2 needs a unique field and SUN")
            config["solar_j2"] = {
                **_field_config(model.gravity_field),
                "matrix": np.ascontiguousarray(model.inertial2solar_fixed_matrix, dtype=float),
            }
        elif isinstance(model, EarthFigureForce):
            if config["earth_figure"] is not None:
                raise ValueError("unsupported: duplicate Earth figure")
            if "EARTH" in model.partners or len(set(model.partners)) != len(model.partners):
                raise ValueError("unsupported: Earth figure partners must be distinct bodies other than EARTH")
            config["earth_figure"] = {
                **_field_config(model.field),
                "partners": [names.index(name) for name in model.partners],
            }
            earth_c20 = (
                float(model.field.coefficients.coeffs[0, 2, 0]),
                model.reference_jd_tdb,
                model.j2_rate_per_year,
                model.j2_quadratic_per_year2,
            )
        elif isinstance(model, MoonFigureForce):
            if moon_field is not None:
                raise ValueError("unsupported: duplicate Moon figure")
            moon_field = model.field
            force_partner_bodies = [names.index(name) for name in model.partners]
            if model.lunar_degree2_gravity_model is not None:
                moon_degree2_force = model.name
                step_models.append((model.name, partial(_degree2_step, model.lunar_degree2_gravity_model)))
        elif isinstance(model, EarthTideForce):
            tide = model.tide_model
            step_models.append((model.name, partial(_tide_step, tide)))
            p = tide.tide_parameters
            uses_tide = True
            config["tide"] = {
                "mu_earth": tide.earth_gravitational_parameter_m3_s2,
                "mu_moon": tide.moon_gravitational_parameter_m3_s2,
                "radius": tide.earth_radius_m,
                "k20": p.k20,
                "k21": p.k21,
                "k22": p.k22,
                "acceleration_scale": tide.earth_gravitational_parameter_m3_s2
                / (tide.earth_gravitational_parameter_m3_s2 + tide.moon_gravitational_parameter_m3_s2),
                "raisers": len(tide.tide_raisers),
            }
        elif isinstance(model, LenseThirringForce):
            if sun < 0:
                raise ValueError("unsupported: Lense-Thirring needs SUN")
            params = model.solar_parameters
            config["lense_thirring"] = {
                "angular_momentum": np.asarray(params._angular_momentum, dtype=float),
                "gamma": params.gamma,
                "gm": params.gravitational_parameter_m3_s2,
            }
        elif isinstance(model, SolarRadiationPressureForce):
            if sun < 0:
                raise ValueError("unsupported: radiation pressure needs SUN")
            config["radiation_pressure"] = {
                "epsilon": model.epsilon,
                "earth_epsilon": model.earth_epsilon,
                "gm": model.solar_parameters.gravitational_parameter_m3_s2,
            }

    rotation_inertia = np.zeros((3, 3))
    if rotation is not None:
        core = rotation.core
        earth_torque_radius = None
        for model in rotation.torque_group.models:
            if isinstance(model, LunarFigureTorque):
                if moon_field is not None and model.field is not moon_field:
                    raise ValueError("unsupported: torque uses a different Moon field")
                moon_field = model.field
                config["torque_kind"].append(0)
                config["torque_arg"].append(slot(names.index(model.source)))
            elif isinstance(model, PointMassGravityTorque):
                if not np.array_equal(model.total_inertia, rotation.total_inertia):
                    raise ValueError("unsupported: quadrupole torque inertia differs from rotation")
                config["torque_kind"].append(1)
                config["torque_arg"].append(names.index(model.source))
            elif isinstance(model, EarthFigureTorque):
                if earth_torque_radius is not None or not np.array_equal(model.total_inertia, rotation.total_inertia):
                    raise ValueError("unsupported: duplicate Earth torque or mismatched inertia")
                earth_torque_radius = model.earth_radius_m
                config["torque_kind"].append(2)
                config["torque_arg"].append(0)
                earth_j2 = (
                    model.earth_j2,
                    model.reference_jd_tdb,
                    model.j2_rate_per_year,
                    model.j2_quadratic_per_year2,
                )
            else:
                raise ValueError("unsupported: torque model")  # noqa: TRY004
        if rotation.deformation is not None and rotation.deformation.field is not moon_field:
            raise ValueError("unsupported: deformation field differs from Moon field")
        if rotation.geodetic_precession and sun < 0:
            raise ValueError("unsupported: geodetic precession needs SUN")
        config["rotation"] = {
            "core": None
            if core is None
            else {
                "diagonal": np.diag(rotation.core_inertia).copy(),
                "viscosity": core.viscosity_per_s,
            },
            "geodetic": bool(rotation.geodetic_precession),
            "earth_torque_radius": earth_torque_radius,
        }
        rotation_inertia = np.ascontiguousarray(rotation.total_inertia, dtype=float)

    if force_partner_bodies is not None:
        config["moon_figure_slots"] = [slot(body) for body in force_partner_bodies]
    config["moon_targets"] = targets
    if targets:
        if moon_field is None:
            raise ValueError("unsupported: Moon figure targets need a field")
        config["moon_field"] = _field_config(moon_field)
    kernel = LunarRhsKernel(config)
    force_indices = [
        (model.name, next(i for i, kind in enumerate(_FORCE_TYPES) if isinstance(model, kind))) for model in models
    ]
    return FusedRhs(
        kernel,
        moon_field if targets else None,
        earth_c20,
        earth_j2,
        uses_tide,
        rotation_inertia,
        moon_degree2_force,
        force_indices,
        step_models,
        config,
    )
