"""Compile a :class:`LunarDynamics` into the fused Cython right-hand side.

The compiled kernel evaluates every supported force and torque in one native
call. Models it does not recognise (for example ad-hoc test forces) make
:func:`compile_fused_rhs` return ``None`` and the Python reference path is used.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lunarops._dynamics_core import LunarRhsKernel
from lunarops.base.constants import C

from .earth_figure_orientation import earth_j2_increment
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


EpochScalar = Callable[..., float]


def _zero(epoch) -> float:
    return 0.0


def _torque_j2(model: EarthFigureTorque) -> EpochScalar:
    def evaluate(epoch) -> float:
        return model.earth_j2 + earth_j2_increment(
            epoch, model.reference_jd_tdb, model.j2_rate_per_year, model.j2_quadratic_per_year2
        )

    return evaluate


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
    """A compiled kernel plus the per-epoch and per-step scalars it needs from Python."""

    def __init__(
        self,
        kernel,
        moon_field,
        earth_c20: EpochScalar,
        earth_j2: EpochScalar,
        uses_tide: bool,
        rotation_inertia,
        moon_degree2_force: str | None,
    ):
        self.kernel = kernel
        self._moon_field = moon_field
        self._earth_c20 = earth_c20
        self._earth_j2 = earth_j2
        self._uses_tide = uses_tide
        self._rotation_inertia = rotation_inertia
        self._moon_degree2_force = moon_degree2_force
        self._epoch = None
        self._step = None
        self._zero_matrix = np.zeros((3, 3))
        self._empty_vectors = np.zeros((0, 3, 3))
        self._empty_gm = np.zeros(0)
        self._no_coefficients = np.zeros((2, 1, 1))

    def _load_epoch(self, epoch_data) -> None:
        epoch = epoch_data.epoch_tdb
        cache = epoch_data.point_mass_gravity_cache
        bodies = epoch_data.positions_m.shape[0]
        if cache is None:
            accelerations, potentials = np.zeros((bodies - 2, 3)), np.zeros(bodies - 2)
        else:
            accelerations, potentials = cache.prescribed_body_accelerations_mps2, cache.prescribed_body_potentials_m2_s2
        self.kernel.set_epoch(
            epoch_data.positions_m,
            epoch_data.velocities_mps,
            epoch_data.gravitational_parameters_m3_s2,
            accelerations,
            potentials,
            epoch_data.earth_fixed2inertial_matrix,
            epoch_data.moon_fixed2inertial_matrix,
            epoch_data.ephemeris_earth_acceleration_mps2,
            self._earth_c20(epoch),
            self._earth_j2(epoch),
        )
        self._epoch = epoch_data

    def _load_step(self, step) -> None:
        if self._uses_tide:
            _, vectors, raiser_gm = step.force_payloads["tide"]
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
        self.kernel.set_step(vectors, raiser_gm, total, derivative, coefficients)
        self._step = step

    def derivative(self, epoch_data, step, state) -> np.ndarray:
        if epoch_data is not self._epoch:
            self._load_epoch(epoch_data)
        if step is not self._step:
            self._load_step(step)
        return self.kernel.derivative(np.ascontiguousarray(state, dtype=float))


def compile_fused_rhs(dynamics) -> FusedRhs | None:
    """Return the compiled right-hand side of ``dynamics`` or ``None`` if a model is unsupported."""
    models = dynamics.force_group.active_models
    if any(not isinstance(model, _FORCE_TYPES) for model in models):
        return None
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

    earth_c20: EpochScalar = _zero
    earth_j2: EpochScalar = _zero
    moon_field = None
    moon_degree2_force = None
    force_partner_bodies: list[int] | None = None
    uses_tide = False
    for model in models:
        if isinstance(model, NewtonianPointMassForce):
            config["point_mass"] = True
        elif isinstance(model, EihPointMassForce):
            config["eih"] = True
        elif isinstance(model, SolarJ2Force):
            if sun < 0 or config["solar_j2"] is not None:
                return None
            config["solar_j2"] = {
                **_field_config(model.gravity_field),
                "matrix": np.ascontiguousarray(model.inertial2solar_fixed_matrix, dtype=float),
            }
        elif isinstance(model, EarthFigureForce):
            if config["earth_figure"] is not None:
                return None
            config["earth_figure"] = _field_config(model.field)
            earth_c20 = model.c20
        elif isinstance(model, MoonFigureForce):
            if moon_field is not None:
                return None
            moon_field = model.field
            force_partner_bodies = [names.index(name) for name in model.partners]
            if model.lunar_degree2_gravity_model is not None:
                moon_degree2_force = model.name
        elif isinstance(model, EarthTideForce):
            tide = model.tide_model
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
                return None
            params = model.solar_parameters
            config["lense_thirring"] = {
                "angular_momentum": np.asarray(params._angular_momentum, dtype=float),
                "gamma": params.gamma,
                "gm": params.gravitational_parameter_m3_s2,
            }
        elif isinstance(model, SolarRadiationPressureForce):
            if sun < 0:
                return None
            config["radiation_pressure"] = {
                "epsilon": model.epsilon,
                "gm": model.solar_parameters.gravitational_parameter_m3_s2,
            }

    rotation_inertia = np.zeros((3, 3))
    if rotation is not None:
        core = rotation.core
        earth_torque_radius = None
        for model in rotation.torque_group.models:
            if isinstance(model, LunarFigureTorque):
                if moon_field is not None and model.field is not moon_field:
                    return None
                moon_field = model.field
                config["torque_kind"].append(0)
                config["torque_arg"].append(slot(names.index(model.source)))
            elif isinstance(model, PointMassGravityTorque):
                if not np.array_equal(model.total_inertia, rotation.total_inertia):
                    return None
                config["torque_kind"].append(1)
                config["torque_arg"].append(names.index(model.source))
            elif isinstance(model, EarthFigureTorque):
                if earth_torque_radius is not None or not np.array_equal(model.total_inertia, rotation.total_inertia):
                    return None
                earth_torque_radius = model.earth_radius_m
                config["torque_kind"].append(2)
                config["torque_arg"].append(0)
                earth_j2 = _torque_j2(model)
            else:
                return None
        if rotation.deformation is not None and rotation.deformation.field is not moon_field:
            return None
        if rotation.geodetic_precession and sun < 0:
            return None
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
            return None
        config["moon_field"] = _field_config(moon_field)
    kernel = LunarRhsKernel(config)
    return FusedRhs(
        kernel, moon_field if targets else None, earth_c20, earth_j2, uses_tide, rotation_inertia, moon_degree2_force
    )
