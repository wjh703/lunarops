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
from typing import cast

import numpy as np

from lunarops._dynamics_core import (
    KERNEL_EVAL_FULL,
    KERNEL_TORQUE_EARTH_J2,
    KERNEL_TORQUE_MOON_FIGURE,
    KERNEL_TORQUE_POINT_MASS,
    LunarRhsKernel,
)
from lunarops.base.constants import J2000_JD, C

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
#: Kernel force rows of the Moon figure force and of the Earth tide force.
MOON_FIGURE_FORCE_ROW = _FORCE_TYPES.index(MoonFigureForce)
EARTH_TIDE_FORCE_ROW = _FORCE_TYPES.index(EarthTideForce)


def _tide_step(model, epoch, batch, index, history):
    matrix = np.ascontiguousarray(batch.earth_fixed2inertial_matrix[index].T)
    return model.delayed_vectors(epoch, matrix, history)


def _tide_steps(model, epochs, batch, indices, history):
    matrices = [np.ascontiguousarray(batch.earth_fixed2inertial_matrix[index].T) for index in indices]
    return model.delayed_vectors_many(epochs, matrices, history)


def _degree2_step(model, epoch, batch, index, history):
    if history is None:
        raise ValueError("lunarDegree2Gravity reads the delayed Earth-Moon position: it needs a state history")

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
    tide_name : str or None
        Name of the Earth tide force, whose step payload holds the tide vectors; ``None`` without the
        Earth tide.
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
    step_batches : dict
        Builders ``f(epochs, batch, indices, history)`` of the payloads of a force at several epochs at
        once, where one exists; the same payloads as its step model.
    kernel_config : dict
        Configuration the kernel was built from.
    """

    def __init__(
        self,
        kernel,
        *,
        moon_field,
        earth_c20: tuple[float, float, float, float],
        earth_j2: tuple[float, float, float, float],
        tide_name: str | None,
        rotation_inertia,
        moon_degree2_force: str | None,
        force_indices: list[tuple[str, int]],
        step_models: list[tuple[str, Callable]],
        step_batches: dict[str, Callable],
        kernel_config: dict,
    ):
        self.step_models = step_models
        self.step_batches = step_batches
        self.kernel_config = kernel_config
        self.force_indices = force_indices
        self.kernel = kernel
        self._moon_field = moon_field
        self.earth_c20 = earth_c20
        self.earth_j2 = earth_j2
        self._tide_name = tide_name
        self._rotation_inertia = rotation_inertia
        self._moon_degree2_force = moon_degree2_force
        self._batch = None
        self._index = -1
        self._step = None
        self._zero_matrix = np.zeros((3, 3))
        self._empty_vectors = np.zeros((0, 3, 3))
        self._empty_gm = np.zeros(0)
        self._no_coefficients = np.zeros((2, 1, 1))

    @property
    def moon_field(self):
        """GravityField or None: Moon field whose degree 2 is replaced per step; ``None`` without Moon figure targets."""
        return self._moon_field

    @property
    def tide_name(self) -> str | None:
        """Name of the Earth tide force, the key of its step payload; ``None`` without the Earth tide."""
        return self._tide_name

    @property
    def uses_tide(self) -> bool:
        """bool: Whether the Earth tide is active."""
        return self._tide_name is not None

    @property
    def state_dimension(self) -> int:
        """int: Length of the integration state of the kernel."""
        return int(self.kernel.state_dimension)

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

    def prepare_steps(self, epochs, batch, indices, history) -> list[dict[str, object]]:
        """Return :meth:`prepare_step` at several epochs of ``batch`` that share one history.

        Forces with a builder in ``step_batches`` take one history query for all the epochs.

        Parameters
        ----------
        epochs : sequence of Epoch
            TDB epochs.
        batch : EpochBatch
            Prescribed inputs containing the epochs.
        indices : sequence of int
            Row of each epoch in ``batch``.
        history : StateHistoryProvider
            Accepted history, for every epoch.

        Returns
        -------
        list of dict
            Payloads per force name, one dict per epoch.
        """
        payloads: list[dict[str, object]] = [{} for _ in epochs]
        for name, prepare in self.step_models:
            many = self.step_batches.get(name)
            values = (
                many(epochs, batch, indices, history)
                if many is not None
                else [prepare(epoch, batch, index, history) for epoch, index in zip(epochs, indices, strict=True)]
            )
            for payload, value in zip(payloads, values, strict=True):
                payload[name] = value
        return payloads

    def step_inputs(self, step):
        """Return the kernel step inputs of ``step``: tide vectors, raiser GM, inertia, its derivative, Moon coefficients."""
        if self._tide_name is not None:
            vectors, raiser_gm = step.force_payloads[self._tide_name]
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
        if self._tide_name is not None:
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

    def _set_eih_bodies(self, eih_bodies: int | None) -> None:
        """Make the EIH sums of ``KERNEL_EVAL_NO_POINT_MASS`` take the first ``eih_bodies`` bodies (all when ``None``)."""
        self.kernel.set_reduced_bodies(self.kernel_config["n_bodies"] if eih_bodies is None else eih_bodies)

    def input_differences(
        self, batch, index, step, state, plus, minus, increments, modes, *, eih_bodies: int | None = None
    ) -> np.ndarray:
        """Return central differences of ``f`` over shifts of the inputs (see LunarRhsKernel), ``(d, m)``.

        Columns in ``KERNEL_EVAL_NO_POINT_MASS`` take the EIH sums over the first ``eih_bodies`` kernel bodies,
        every body when ``None``.
        """
        self.load(batch, index, step)
        self._set_eih_bodies(eih_bodies)
        return self.kernel.input_differences(
            np.ascontiguousarray(state, dtype=float),
            np.ascontiguousarray(plus, dtype=float),
            np.ascontiguousarray(minus, dtype=float),
            np.ascontiguousarray(increments, dtype=float),
            np.ascontiguousarray(modes, dtype=np.intc),
        )

    def load(self, batch, index, step) -> None:
        """Load the epoch data of row ``index`` of ``batch`` and the step data ``step`` into the kernel.

        Either is loaded only when it differs from the one loaded last.
        """
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
        mode : int, default KERNEL_EVAL_FULL
            Terms to evaluate (see ``LunarRhsKernel.split_derivative``).

        Returns
        -------
        derivative, derivative_low : numpy.ndarray
            High and low parts of the state derivative, shape ``(d,)``.

        Raises
        ------
        ValueError
            If the kernel reports a degenerate state.
        """
        self.load(batch, index, step)
        return self.kernel.split_derivative(
            np.ascontiguousarray(state, dtype=float), np.ascontiguousarray(state_low, dtype=float), mode
        )

    def state_jacobian(self, batch, index, step, state, increments, *, eih_bodies: int | None = None) -> np.ndarray:
        """Return ∂f/∂y at fixed step data by central differences (see LunarRhsKernel.state_jacobian).

        With ``eih_bodies`` the Newtonian point masses of the orbit columns are taken in closed form and the
        EIH sums of the differenced terms over the first ``eih_bodies`` kernel bodies; every term is differenced
        when ``None``.
        """
        self.load(batch, index, step)
        self._set_eih_bodies(eih_bodies)
        return self.kernel.state_jacobian(
            np.ascontiguousarray(state, dtype=float),
            np.ascontiguousarray(increments, dtype=float),
            eih_bodies is not None,
        )

    def tide_jacobian(self, batch, index, step, state, shifts, increment: float) -> np.ndarray:
        """Return central differences of f for shifts of the step's tide vectors (see LunarRhsKernel)."""
        self.load(batch, index, step)
        return self.kernel.tide_jacobian(
            np.ascontiguousarray(state, dtype=float), np.ascontiguousarray(shifts, dtype=float), float(increment)
        )

    def raw_terms(self, batch, index, step, state, mode: int = KERNEL_EVAL_FULL) -> dict:
        """Return the kernel term dictionary, with the force terms as kernel rows (see ``LunarRhsKernel.terms``).

        The arrays may be views of kernel buffers that the next evaluation overwrites.
        """
        self.load(batch, index, step)
        return self.kernel.terms(np.ascontiguousarray(state, dtype=float), mode)

    def terms(self, batch, index, step, state, mode: int = KERNEL_EVAL_FULL):
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
        mode : int, default KERNEL_EVAL_FULL
            ``KERNEL_EVAL_FULL``, or ``KERNEL_EVAL_FIGURE`` for the Moon figure force and the rotation alone
            (see ``LunarRhsKernel.terms``).

        Returns
        -------
        dict
            Kernel term dictionary (see
            :meth:`~lunarops.classes.dynamics.lunar_dynamics.LunarDynamics.terms`)
            with ``"forces"`` and ``"earth_forces"`` keyed by force name; without them in
            ``KERNEL_EVAL_FIGURE``.

        Raises
        ------
        ValueError
            If the kernel reports a degenerate state or ``mode`` is another mode.
        """
        result = self.raw_terms(batch, index, step, state, mode)
        if mode == KERNEL_EVAL_FULL:
            for key in ("forces", "earth_forces"):
                result[key] = {name: result[key][slot].copy() for name, slot in self.force_indices}
        return result


def _model_entry(table: dict[type, Callable], model) -> tuple[type, Callable] | None:
    """Return the entry of ``table`` for the type of ``model`` or its nearest base class; ``None`` without one."""
    for kind in type(model).__mro__:
        builder = table.get(kind)
        if builder is not None:
            return kind, builder
    return None


class _KernelBuild:
    """Kernel configuration and the :class:`FusedRhs` inputs, filled in by the builder of each force and torque.

    Parameters
    ----------
    names : tuple of str
        Kernel bodies, as ``LunarDynamics.body_names``.
    """

    def __init__(self, names) -> None:
        self.names = names
        self.sun = names.index("SUN") if "SUN" in names else -1
        self.config: dict = {
            "n_bodies": len(names),
            "sun_index": self.sun,
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
        self.targets: list[int] = []
        self.earth_c20 = (0.0, J2000_JD, 0.0, 0.0)
        self.earth_j2 = (0.0, J2000_JD, 0.0, 0.0)
        self.moon_field = None
        self.moon_degree2_force: str | None = None
        self.force_partner_bodies: list[int] | None = None
        self.tide_name: str | None = None
        self.step_models: list[tuple[str, Callable]] = []
        self.step_batches: dict[str, Callable] = {}
        self.earth_torque_radius: float | None = None
        self.rotation_inertia = np.zeros((3, 3))

    def slot(self, body: int) -> int:
        """Return the Moon figure target slot of kernel body ``body``, adding it when new."""
        if body not in self.targets:
            self.targets.append(body)
        return self.targets.index(body)

    def add_torque(self, kind: int, argument: int) -> None:
        self.config["torque_kind"].append(kind)
        self.config["torque_arg"].append(argument)


def _point_mass_force(build: _KernelBuild, model: NewtonianPointMassForce) -> None:
    build.config["point_mass"] = True


def _eih_force(build: _KernelBuild, model: EihPointMassForce) -> None:
    build.config["eih"] = True


def _solar_j2_force(build: _KernelBuild, model: SolarJ2Force) -> None:
    if build.sun < 0 or build.config["solar_j2"] is not None:
        raise ValueError("unsupported: solar J2 needs a unique field and SUN")
    build.config["solar_j2"] = {
        **_field_config(model.gravity_field),
        "matrix": np.ascontiguousarray(model.inertial2solar_fixed_matrix, dtype=float),
    }


def _earth_figure_force(build: _KernelBuild, model: EarthFigureForce) -> None:
    if build.config["earth_figure"] is not None:
        raise ValueError("unsupported: duplicate Earth figure")
    if "EARTH" in model.partners or len(set(model.partners)) != len(model.partners):
        raise ValueError("unsupported: Earth figure partners must be distinct bodies other than EARTH")
    build.config["earth_figure"] = {
        **_field_config(model.field),
        "partners": [build.names.index(name) for name in model.partners],
    }
    build.earth_c20 = (
        float(model.field.coefficients.coeffs[0, 2, 0]),
        model.reference_jd_tdb,
        model.j2_rate_per_year,
        model.j2_quadratic_per_year2,
    )


def _moon_figure_force(build: _KernelBuild, model: MoonFigureForce) -> None:
    if build.moon_field is not None:
        raise ValueError("unsupported: duplicate Moon figure")
    build.moon_field = model.field
    build.force_partner_bodies = [build.names.index(name) for name in model.partners]
    if model.lunar_degree2_gravity_model is not None:
        build.moon_degree2_force = model.name
        build.step_models.append((model.name, partial(_degree2_step, model.lunar_degree2_gravity_model)))


def _earth_tide_force(build: _KernelBuild, model: EarthTideForce) -> None:
    if build.tide_name is not None:
        raise ValueError("unsupported: duplicate Earth tide")
    tide = model.tide_model
    build.step_models.append((model.name, partial(_tide_step, tide)))
    build.step_batches[model.name] = partial(_tide_steps, tide)
    p = tide.tide_parameters
    build.tide_name = model.name
    build.config["tide"] = {
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


def _lense_thirring_force(build: _KernelBuild, model: LenseThirringForce) -> None:
    if build.sun < 0:
        raise ValueError("unsupported: Lense-Thirring needs SUN")
    params = model.solar_parameters
    build.config["lense_thirring"] = {
        "angular_momentum": np.asarray(params._angular_momentum, dtype=float),
        "gamma": params.gamma,
        "gm": params.gravitational_parameter_m3_s2,
    }


def _radiation_pressure_force(build: _KernelBuild, model: SolarRadiationPressureForce) -> None:
    if build.sun < 0:
        raise ValueError("unsupported: radiation pressure needs SUN")
    build.config["radiation_pressure"] = {
        "epsilon": model.epsilon,
        "earth_epsilon": model.earth_epsilon,
        "gm": model.solar_parameters.gravitational_parameter_m3_s2,
    }


_FORCE_BUILDERS: dict[type, Callable[[_KernelBuild, object], None]] = {
    NewtonianPointMassForce: _point_mass_force,
    EihPointMassForce: _eih_force,
    SolarJ2Force: _solar_j2_force,
    EarthFigureForce: _earth_figure_force,
    MoonFigureForce: _moon_figure_force,
    EarthTideForce: _earth_tide_force,
    LenseThirringForce: _lense_thirring_force,
    SolarRadiationPressureForce: _radiation_pressure_force,
}
"""Builder of each force type of :data:`_FORCE_TYPES`."""


def _moon_figure_torque(build: _KernelBuild, model: LunarFigureTorque, rotation) -> None:
    if build.moon_field is not None and model.field is not build.moon_field:
        raise ValueError("unsupported: torque uses a different Moon field")
    build.moon_field = model.field
    build.add_torque(KERNEL_TORQUE_MOON_FIGURE, build.slot(build.names.index(model.source)))


def _point_mass_torque(build: _KernelBuild, model: PointMassGravityTorque, rotation) -> None:
    if not np.array_equal(model.total_inertia, rotation.total_inertia):
        raise ValueError("unsupported: quadrupole torque inertia differs from rotation")
    build.add_torque(KERNEL_TORQUE_POINT_MASS, build.names.index(model.source))


def _earth_figure_torque(build: _KernelBuild, model: EarthFigureTorque, rotation) -> None:
    if build.earth_torque_radius is not None or not np.array_equal(model.total_inertia, rotation.total_inertia):
        raise ValueError("unsupported: duplicate Earth torque or mismatched inertia")
    build.earth_torque_radius = model.earth_radius_m
    build.add_torque(KERNEL_TORQUE_EARTH_J2, 0)
    build.earth_j2 = (
        model.earth_j2,
        model.reference_jd_tdb,
        model.j2_rate_per_year,
        model.j2_quadratic_per_year2,
    )


_TORQUE_BUILDERS: dict[type, Callable[[_KernelBuild, object, object], None]] = {
    LunarFigureTorque: _moon_figure_torque,
    PointMassGravityTorque: _point_mass_torque,
    EarthFigureTorque: _earth_figure_torque,
}


def _add_rotation(build: _KernelBuild, rotation) -> None:
    """Add the torques, the core and the geodetic precession of the integrated rotation."""
    core = rotation.core
    for model in rotation.torque_group.models:
        entry = _model_entry(_TORQUE_BUILDERS, model)
        if entry is None:
            raise ValueError("unsupported: torque model")
        entry[1](build, model, rotation)
    if rotation.deformation is not None and rotation.deformation.field is not build.moon_field:
        raise ValueError("unsupported: deformation field differs from Moon field")
    if rotation.geodetic_precession and build.sun < 0:
        raise ValueError("unsupported: geodetic precession needs SUN")
    build.config["rotation"] = {
        "core": None
        if core is None
        else {
            "diagonal": np.diag(rotation.core_inertia).copy(),
            "viscosity": core.viscosity_per_s,
        },
        "geodetic": bool(rotation.geodetic_precession),
        "earth_torque_radius": build.earth_torque_radius,
    }
    build.rotation_inertia = np.ascontiguousarray(rotation.total_inertia, dtype=float)


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
    entries = [_model_entry(_FORCE_BUILDERS, model) for model in models]
    if any(entry is None for entry in entries):
        raise ValueError("unsupported: force model")
    build = _KernelBuild(dynamics.body_names)
    force_indices = []
    for model, (kind, builder) in zip(models, cast(list, entries), strict=True):
        builder(build, model)
        force_indices.append((model.name, _FORCE_TYPES.index(kind)))
    if dynamics.rotation is not None:
        _add_rotation(build, dynamics.rotation)
    config = build.config
    if build.force_partner_bodies is not None:
        config["moon_figure_slots"] = [build.slot(body) for body in build.force_partner_bodies]
    config["moon_targets"] = build.targets
    if build.targets:
        if build.moon_field is None:
            raise ValueError("unsupported: Moon figure targets need a field")
        config["moon_field"] = _field_config(build.moon_field)
    return FusedRhs(
        LunarRhsKernel(config),
        moon_field=build.moon_field if build.targets else None,
        earth_c20=build.earth_c20,
        earth_j2=build.earth_j2,
        tide_name=build.tide_name,
        rotation_inertia=build.rotation_inertia,
        moon_degree2_force=build.moon_degree2_force,
        force_indices=force_indices,
        step_models=build.step_models,
        step_batches=build.step_batches,
        kernel_config=config,
    )
