"""NumPy physical oracle for the fused native RHS.

Forces, torques, rotation equations, quaternion kinematics, the de Sitter rate,
the Earth-tide delays and the inertia-to-degree-2 mapping are written here from
the published equations. Shared inputs still come from production code: the
epoch batch (ephemeris states, orientation matrices, GM drift), the lunar
degree-2 and deformation inertia of ``prepare_step``, and the spherical-harmonic
kernel ``GravityField.nonspherical_accelerations``; the latter is validated
independently in tests/dynamics/test_gravity_field.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np

from lunarops.base.constants import C
from lunarops.classes.dynamics import force_models as parameters
from lunarops.classes.dynamics import forces as sources
from lunarops.classes.dynamics import torques as torque_parameters
from lunarops.classes.dynamics.context import StateHistoryProvider, StepData
from lunarops.classes.dynamics.deformation import LunarInertiaEvaluation
from lunarops.classes.dynamics.epoch_data import EpochBatch
from lunarops.classes.dynamics.gravity import GravityCoefficients
from lunarops.classes.dynamics.lunar_dynamics import LunarDynamics
from lunarops.classes.dynamics.rotation import LunarRotationDynamics as RotationParameters
from lunarops.classes.time import Epoch

cross3 = np.cross
SECONDS_PER_DAY = 86400.0


def quaternion_matrix(quaternion) -> np.ndarray:
    """Active rotation of the Hamilton quaternion (scalar first), normalized for the evaluation."""
    a, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array(
        [
            [a * a + x * x - y * y - z * z, 2 * (x * y - a * z), 2 * (x * z + a * y)],
            [2 * (x * y + a * z), a * a - x * x + y * y - z * z, 2 * (y * z - a * x)],
            [2 * (x * z - a * y), 2 * (y * z + a * x), a * a - x * x - y * y + z * z],
        ]
    )


def earth_j2_increment(epoch, reference_jd_tdb, rate_per_year, quadratic_per_year2):
    """Polynomial J2(t) - J2(reference) in Julian years of TDB."""
    years = (epoch.jd1 + epoch.jd2 - reference_jd_tdb) / 365.25
    return rate_per_year * years + quadratic_per_year2 * years**2


def degree2_coefficients(inertia, *, c21_static=0.0, s21_static=0.0, s22_static=0.0):
    """Fully normalized (C2m, S2m) from MacCullagh's formulas for an inertia per M R^2.

    Unnormalized C20 = (A + B)/2 - C, C21 = -Ixz, S21 = -Iyz, C22 = (B - A)/4, S22 = -Ixy/2;
    Pavlov Eq. (16) static offsets are unnormalized additions.
    """
    i = np.asarray(inertia, dtype=float)
    unnormalized = np.array(
        [
            [(i[0, 0] + i[1, 1]) / 2 - i[2, 2], -i[0, 2] + c21_static, (i[1, 1] - i[0, 0]) / 4],
            [0.0, -i[1, 2] + s21_static, -i[0, 1] / 2 + s22_static],
        ]
    )
    # N_2m = sqrt((2 - delta_m0)(2n + 1)(n - m)!/(n + m)!) for n = 2.
    return unnormalized / np.array([np.sqrt(5.0), np.sqrt(5.0 / 3.0), np.sqrt(5.0 / 12.0)])


def passive_rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def delayed_tide_vectors(tide, epoch, inertial2fixed_matrix, history):
    """JPL Eq. (35)-(36): raiser vectors at the orbital delays, rotated by the Earth-rotation lags.

    Every vector is expressed in the current Earth-fixed basis.
    """
    p = tide.tide_parameters
    raisers = tide.tide_raisers
    gm = np.array([tide.tide_raiser_gm[name] for name in raisers], dtype=float)
    if tide.solar_gm_variation is not None and "SUN" in raisers:
        gm[raisers.index("SUN")] *= tide.solar_gm_variation.factor(epoch)
    vectors = np.empty((len(raisers), 3, 3))
    for order in range(3):
        states = history.states(("EARTH", *raisers), epoch.shifted(-p.tau_orb_days[order] * SECONDS_PER_DAY))
        lag = passive_rz(-p.earth_rotation_rate_rad_s * p.tau_rot_days[order] * SECONDS_PER_DAY)
        for row in range(len(raisers)):
            vectors[row, order] = lag @ inertial2fixed_matrix @ (states[1 + row, :3] - states[0, :3])
    return vectors, gm


class SolarSourceParameters(sources.SolarSourceParameters):
    def lense_thirring_acceleration(self, r, v):
        d = np.linalg.norm(r)
        omega = (
            (1.0 + self.gamma)
            / (2.0 * C**2 * d**3)
            * (-self._angular_momentum + 3.0 * np.dot(self._angular_momentum, r) * r / d**2)
        )
        return 2.0 * cross3(omega, v)

    def radiation_pressure_acceleration(self, r, epsilon):
        d = np.linalg.norm(r)
        return epsilon * self.gravitational_parameter_m3_s2 * r / d**3


class EarthTideModel(sources.EarthTideModel):
    def acceleration_from_vectors(self, moon_minus_earth, inertial2fixed_matrix, delayed_vectors, raiser_gm):
        """Relative Earth-Moon tide acceleration for a candidate state and precomputed delayed vectors."""
        p = self.tide_parameters
        fixed_acceleration = earth_tide_relative_acceleration(
            inertial2fixed_matrix @ moon_minus_earth,
            delayed_vectors,
            raiser_gm,
            self.earth_gravitational_parameter_m3_s2,
            self.moon_gravitational_parameter_m3_s2,
            self.earth_radius_m,
            p.k20,
            p.k21,
            p.k22,
        )
        return inertial2fixed_matrix.T @ fixed_acceleration

    def relative_acceleration(self, epoch, earth, moon, current_inertial2fixed_matrix, history):
        """Evaluate delayed Earth tides in the current Earth-fixed frame."""
        matrix = np.asarray(current_inertial2fixed_matrix, dtype=float)
        vectors, raiser_gm = delayed_tide_vectors(self, epoch, matrix, history)
        return self.acceleration_from_vectors(moon - earth, matrix, vectors, raiser_gm)


class NewtonianPointMassForce(parameters.NewtonianPointMassForce):
    def evaluate(self, inputs):
        return evaluate_newtonian_point_mass_system(inputs.positions_m, inputs.gravitational_parameters_m3_s2)

    def acceleration(self, inputs):
        return self.evaluate(inputs).accelerations_mps2[inputs.moon_index]

    def earth_acceleration(self, inputs):
        return self.evaluate(inputs).accelerations_mps2[inputs.earth_index]


class EihPointMassForce(parameters.EihPointMassForce):
    def _correction(self, inputs, index):
        return evaluate_eih_correction(
            inputs.positions_m,
            inputs.velocities_mps,
            inputs.gravitational_parameters_m3_s2,
            inputs.newtonian_evaluation,
            [index],
        )[index]

    def acceleration(self, inputs):
        return self._correction(inputs, inputs.moon_index)

    def earth_acceleration(self, inputs):
        return self._correction(inputs, inputs.earth_index)


class SolarJ2Force(parameters.SolarJ2Force):
    def _on(self, inputs, index):
        sun = inputs.body_indices["SUN"]
        relative = (inputs.positions_m[[index]] - inputs.positions_m[sun]) @ self.inertial2solar_fixed_matrix.T
        scale = inputs.gravitational_parameters_m3_s2[sun] / self.gravity_field.gm_m3_s2
        return (scale * (self.gravity_field.nonspherical_accelerations(relative) @ self.inertial2solar_fixed_matrix))[0]

    def acceleration(self, inputs):
        return self._on(inputs, inputs.moon_index)

    def earth_acceleration(self, inputs):
        return self._on(inputs, inputs.earth_index)


class LenseThirringForce(parameters.LenseThirringForce):
    def _on(self, inputs, index):
        sun = inputs.body_indices["SUN"]
        out = SolarSourceParameters.lense_thirring_acceleration(
            self.solar_parameters,
            inputs.positions_m[index] - inputs.positions_m[sun],
            inputs.velocities_mps[index] - inputs.velocities_mps[sun],
        )
        out *= inputs.gravitational_parameters_m3_s2[sun] / self.solar_parameters.gravitational_parameter_m3_s2
        return out

    def acceleration(self, inputs):
        return self._on(inputs, inputs.moon_index)

    def earth_acceleration(self, inputs):
        return self._on(inputs, inputs.earth_index)


class SolarRadiationPressureForce(parameters.SolarRadiationPressureForce):
    def acceleration(self, inputs):
        sun = inputs.body_indices["SUN"]
        moon = inputs.moon_index
        out = SolarSourceParameters.radiation_pressure_acceleration(
            self.solar_parameters,
            inputs.positions_m[moon] - inputs.positions_m[sun],
            self.epsilon,
        )
        out *= inputs.gravitational_parameters_m3_s2[sun] / self.solar_parameters.gravitational_parameter_m3_s2
        return out

    def earth_acceleration(self, inputs):
        # Radiation pressure on the Earth is not modelled.
        return np.zeros(3)


class EarthTideForce(parameters.EarthTideForce):
    def _relative(self, inputs):
        matrix = inputs.earth_fixed2inertial_matrix.T
        vectors, raiser_gm = delayed_tide_vectors(self.tide_model, inputs.epoch_tdb, matrix, inputs.history)
        return EarthTideModel.acceleration_from_vectors(self.tide_model, inputs.relative_m, matrix, vectors, raiser_gm)

    def acceleration(self, inputs):
        # JPL's tidal kernel returns Moon-minus-Earth acceleration and leaves the EMB unchanged.
        mu_earth = self.tide_model.earth_gravitational_parameter_m3_s2
        mu_moon = self.tide_model.moon_gravitational_parameter_m3_s2
        return (mu_earth / (mu_earth + mu_moon)) * self._relative(inputs)

    def earth_acceleration(self, inputs):
        mu_earth = self.tide_model.earth_gravitational_parameter_m3_s2
        mu_moon = self.tide_model.moon_gravitational_parameter_m3_s2
        return -(mu_moon / (mu_earth + mu_moon)) * self._relative(inputs)


class EarthFigureForce(parameters.EarthFigureForce):
    def c20(self, epoch) -> float:
        delta = earth_j2_increment(epoch, self.reference_jd_tdb, self.j2_rate_per_year, self.j2_quadratic_per_year2)
        return float(self.field.coefficients.coeffs[0, 2, 0]) - delta / np.sqrt(5.0)

    def _direct(self, inputs):
        inputs.coefficient_overrides[id(self.field)] = self.field.coefficients_with_c20(self.c20(inputs.epoch_tdb))
        targets = np.asarray([inputs.body_indices[name] for name in self.partners], dtype=int)
        _, direct = inputs.figure_accelerations(inputs.earth_index, targets, self.field)
        return targets, direct @ inputs.earth_fixed2inertial_matrix.T

    def acceleration(self, inputs):
        targets, direct = self._direct(inputs)
        return np.sum(direct[targets == inputs.moon_index], axis=0)

    def earth_acceleration(self, inputs):
        # Park et al. (2021), Eq. 29: indirect acceleration -GM_partner/GM_E times each direct one.
        targets, direct = self._direct(inputs)
        mu = inputs.gravitational_parameters_m3_s2
        return -np.sum((mu[targets] / mu[inputs.earth_index])[:, None] * direct, axis=0)


class MoonFigureForce(parameters.MoonFigureForce):
    def acceleration(self, inputs):
        if self.lunar_degree2_gravity_model is not None:
            degree2 = inputs.step_payloads[self.name]
            inputs.coefficient_overrides[id(self.field)] = self.field.coefficients_with_degree2(degree2)
        moon = inputs.moon_index
        mu = inputs.gravitational_parameters_m3_s2
        targets = np.asarray([inputs.body_indices[name] for name in self.partners], dtype=int)
        _, direct = inputs.figure_accelerations(moon, targets, self.field)
        direct_inertial = direct @ inputs.moon_fixed2inertial_matrix.T
        out = np.zeros(3)
        out -= np.sum((mu[targets] / mu[moon])[:, None] * direct_inertial, axis=0)
        return out

    def earth_acceleration(self, inputs):
        if "EARTH" not in self.partners:
            return np.zeros(3)
        self.acceleration(inputs)  # applies the degree-2 override and fills the figure cache
        _, direct = inputs.figure_accelerations(inputs.moon_index, (inputs.earth_index,), self.field)
        return (direct @ inputs.moon_fixed2inertial_matrix.T)[0]


class PointMassGravityTorque(torque_parameters.PointMassGravityTorque):
    def torque(self, inputs):
        source = inputs.body_indices[self.source]
        r = inputs.moon_fixed2inertial_matrix.T @ (inputs.positions_m[source] - inputs.positions_m[inputs.moon_index])
        distance = np.linalg.norm(r)
        u = r / distance
        K = self.total_inertia if inputs.lunar_inertia is None else inputs.lunar_inertia.total
        return 3 * inputs.gravitational_parameters_m3_s2[source] / distance**3 * cross3(u, K @ u)


class LunarFigureTorque(torque_parameters.LunarFigureTorque):
    def torque(self, inputs):
        source = inputs.body_indices[self.source]
        # Same target/field evaluation as MoonFigureForce. This is the direct
        # acceleration of the point mass, not the Moon reaction acceleration.
        relative, direct = inputs.figure_accelerations(inputs.moon_index, (source,), self.field)
        reaction = (
            -inputs.gravitational_parameters_m3_s2[source]
            / inputs.gravitational_parameters_m3_s2[inputs.moon_index]
            * direct[0]
        )
        return cross3(relative[0], reaction) / self.field.radius_m**2


class EarthFigureTorque(torque_parameters.EarthFigureTorque):
    def torque(self, inputs):
        R = inputs.moon_fixed2inertial_matrix
        r = R.T @ (inputs.positions_m[inputs.earth_index] - inputs.positions_m[inputs.moon_index])
        distance = np.linalg.norm(r)
        u = r / distance
        p = R.T @ inputs.earth_fixed2inertial_matrix[:, 2]
        s = u @ p
        K = self.total_inertia if inputs.lunar_inertia is None else inputs.lunar_inertia.total
        j2 = self.earth_j2 + earth_j2_increment(
            inputs.epoch_tdb, self.reference_jd_tdb, self.j2_rate_per_year, self.j2_quadratic_per_year2
        )
        terms = (
            (1 - 7 * s * s) * cross3(u, K @ u) + 2 * s * (cross3(u, K @ p) + cross3(p, K @ u)) - 0.4 * cross3(p, K @ p)
        )
        return (
            7.5
            * inputs.gravitational_parameters_m3_s2[inputs.earth_index]
            * self.earth_radius_m**2
            * j2
            / distance**5
            * terms
        )


@dataclass(frozen=True, slots=True)
class PointMassGravityEvaluation:
    """Complete Newtonian auxiliary state required by the EIH equations."""

    accelerations_mps2: np.ndarray
    potentials_m2_s2: np.ndarray


class DynamicsEvaluationContext:
    """Current candidate state and caches shared by forces and torques."""

    def __init__(self, body_names: Sequence[str], gravitational_parameters_m3_s2: np.ndarray) -> None:
        names = tuple(body_names)
        self.body_names = names
        self.body_indices = MappingProxyType({name: index for index, name in enumerate(names)})
        self.earth_index = self.body_indices["EARTH"]
        self.moon_index = self.body_indices["MOON"]
        self.moon_targets = np.asarray([self.moon_index], dtype=np.intp)
        self.moon_targets.setflags(write=False)
        self.gravitational_parameters_m3_s2 = gravitational_parameters_m3_s2
        self.positions_m = np.empty((len(names), 3), dtype=float)
        self.velocities_mps = np.empty_like(self.positions_m)
        self.relative_m = np.zeros(3)
        self.history: StateHistoryProvider | None = None
        self.step_payloads: Mapping[str, object] | None = None
        self.epoch_tdb: Epoch
        self.earth_fixed2inertial_matrix: np.ndarray
        self.newtonian_evaluation: PointMassGravityEvaluation | None = None
        self.moon_fixed2inertial_matrix = np.eye(3)
        self._figure_cache: dict = {}
        # id(field) -> coefficient array of this evaluation (degree 2 or J2(t) applied).
        self.coefficient_overrides: dict[int, np.ndarray] = {}
        self.lunar_inertia = None

    def load_relative_state(self, relative_state, emb_position, emb_velocity) -> None:
        """Place the Earth and the Moon about the barycentre: r_E = r_B - mu_M y, r_M = r_B + mu_E y."""
        state = np.asarray(relative_state, dtype=float)
        mu = self.gravitational_parameters_m3_s2
        earth, moon = mu[self.earth_index], mu[self.moon_index]
        f_earth, f_moon = earth / (earth + moon), moon / (earth + moon)
        self.relative_m = state[:3].copy()
        self.positions_m[self.earth_index] = emb_position - f_moon * state[:3]
        self.velocities_mps[self.earth_index] = emb_velocity - f_moon * state[3:6]
        self.positions_m[self.moon_index] = emb_position + f_earth * state[:3]
        self.velocities_mps[self.moon_index] = emb_velocity + f_earth * state[3:6]

    def load_evaluation(self, batch, index, epoch, relative_state, history, *, moon_matrix=None, step_payloads=None):
        self.epoch_tdb = epoch
        self.earth_fixed2inertial_matrix = batch.earth_fixed2inertial_matrix[index]
        self.positions_m[:] = batch.positions_m[index]
        self.velocities_mps[:] = batch.velocities_mps[index]
        self.gravitational_parameters_m3_s2 = batch.gravitational_parameters_m3_s2[index]
        self.load_relative_state(relative_state, batch.emb_positions_m[index], batch.emb_velocities_mps[index])
        self.history = history
        self.step_payloads = step_payloads
        self.newtonian_evaluation = None
        self.moon_fixed2inertial_matrix = (
            batch.moon_fixed2inertial_matrix[index] if moon_matrix is None else moon_matrix
        )
        # Corrector iterations at one epoch have different candidate states.
        self._figure_cache.clear()
        self.coefficient_overrides.clear()
        self.lunar_inertia = None
        return self

    def figure_accelerations(self, source, targets, field):
        """PA/fixed positions and direct non-spherical accelerations.

        Cache each target separately so a force's batch can serve individual
        torque models. Entries live for exactly one candidate evaluation.
        """
        cache = self._figure_cache.setdefault((source, id(field)), {})
        missing = [index for index in targets if index not in cache]
        if missing:
            R = self.earth_fixed2inertial_matrix if source == self.earth_index else self.moon_fixed2inertial_matrix
            relative = (self.positions_m[missing] - self.positions_m[source]) @ R
            accelerations = field.nonspherical_accelerations(relative, self.coefficient_overrides.get(id(field)))
            cache.update({index: (r, a) for index, r, a in zip(missing, relative, accelerations, strict=True)})
        return tuple(np.asarray([cache[index][component] for index in targets]) for component in (0, 1))


@dataclass(frozen=True, slots=True)
class RotationEvaluation:
    derivative: np.ndarray
    torques: dict[str, np.ndarray]
    viscous_torque: np.ndarray
    pressure_torque: np.ndarray
    geodetic_rate_pa_rad_s: np.ndarray
    inertia: LunarInertiaEvaluation | None


def quaternion_derivative(quaternion, angular_velocity) -> np.ndarray:
    """q_dot = 1/2 q ⊗ (0, omega_PA), without state projection."""
    q = np.asarray(quaternion)
    w = np.asarray(angular_velocity)
    return 0.5 * np.concatenate(([-q[1:] @ w], q[0] * w + cross3(q[1:], w)))


class LunarForceGroup(parameters.LunarForceGroup):
    def _contribution(self, model, inputs: DynamicsEvaluationContext, body: str = "moon") -> np.ndarray:
        if model is self._point_mass_model:
            inputs.newtonian_evaluation = NewtonianPointMassForce.evaluate(cast(NewtonianPointMassForce, model), inputs)
        return model.acceleration(inputs) if body == "moon" else model.earth_acceleration(inputs)

    def compute_accelerations(self, inputs: DynamicsEvaluationContext, body: str = "moon") -> np.ndarray:
        total = np.zeros(3)
        for model in self._active_models:
            total += self._contribution(model, inputs, body)
        return total

    def acceleration_terms(self, inputs: DynamicsEvaluationContext, body: str = "moon") -> dict[str, np.ndarray]:
        return {model.name: self._contribution(model, inputs, body) for model in self._active_models}


class LunarTorqueGroup(torque_parameters.LunarTorqueGroup):
    def evaluate(self, inputs):
        terms = {model.name: model.torque(inputs) for model in self.models}
        return sum(terms.values(), start=np.zeros(3)), terms


class LunarRotationDynamics(RotationParameters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._core_diagonal = np.diag(self.core_inertia)
        self._mantle_diagonal = None
        if np.allclose(self.mantle_inertia, np.diag(np.diag(self.mantle_inertia)), rtol=0.0, atol=1e-15):
            self._mantle_diagonal = np.diag(self.mantle_inertia)

    def evaluate(self, inputs, state) -> RotationEvaluation:
        q, wm = state[:4], state[4:7]
        mantle = self.mantle_inertia if inputs.lunar_inertia is None else inputs.lunar_inertia.total - self.core_inertia
        inertia_dot = np.zeros((3, 3)) if inputs.lunar_inertia is None else inputs.lunar_inertia.derivative
        total, terms = LunarTorqueGroup.evaluate(self.torque_group, inputs)
        nu = self.geodetic_rate(inputs) if self.geodetic_precession else np.zeros(3)
        viscous, pressure = np.zeros(3), np.zeros(3)
        if self.core is not None:
            wc = state[7:10]
            viscous = self.core.viscosity_per_s * (wc - wm)
            # Pressure torque uses the core spin for DE430 and DE440 alike.
            pressure = (self.core_inertia[2, 2] - self.core_inertia[0, 0]) * wc[2] * np.array([-wc[1], wc[0], 0.0])
        cmb = viscous + pressure
        mantle_rhs = total + cmb - cross3(wm - nu, mantle @ wm) - inertia_dot @ wm
        if inputs.lunar_inertia is None and self._mantle_diagonal is not None:
            mantle_dot = mantle_rhs / self._mantle_diagonal
        else:
            # Deformation makes the mantle tensor time-varying; retain a
            # stable general solve for that case.
            mantle_dot = np.linalg.solve(mantle, mantle_rhs)
        parts = [quaternion_derivative(q, wm), mantle_dot]
        if self.core is not None:
            core_rhs = -cmb - cross3(wm - nu, self.core_inertia @ wc)
            parts.append(core_rhs / self._core_diagonal)
        return RotationEvaluation(np.concatenate(parts), terms, viscous, pressure, nu, inputs.lunar_inertia)

    @staticmethod
    def geodetic_rate(inputs) -> np.ndarray:
        """De Sitter precession, gamma_PPN=1, SI/TDB, in candidate PA axes.

        nu = 3/(2c^2) sum_j GM_j (r_jM x v_jM) / r_jM^3 over Earth and Sun;
        the Moon's position and velocity are relative to each source. DE440 Eq. 44
        prints the opposite cross product and uses Earth-Moon velocity in the solar term."""
        earth, moon, sun = (inputs.body_indices[name] for name in ("EARTH", "MOON", "SUN"))
        p, v, mu = inputs.positions_m, inputs.velocities_mps, inputs.gravitational_parameters_m3_s2
        rate = np.zeros(3)
        for source in (earth, sun):
            r, w = p[moon] - p[source], v[moon] - v[source]
            rate += mu[source] * cross3(r, w) / np.linalg.norm(r) ** 3
        return inputs.moon_fixed2inertial_matrix.T @ (1.5 / C**2 * rate)


class ReferenceDynamics(LunarDynamics):
    force_group: LunarForceGroup
    rotation: LunarRotationDynamics | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inputs = DynamicsEvaluationContext(self.body_names, self.gravitational_parameters_m3_s2)

    def derivatives(
        self,
        epoch_tdb: Epoch,
        vector,
        batch: EpochBatch | None = None,
        *,
        history=None,
        index: int = 0,
        rigid: bool = False,
        step: StepData | None = None,
    ) -> np.ndarray:
        """Return the state derivative; ``rigid`` uses the constant-inertia backward seed model.

        ``step`` carries the candidate-independent data of :meth:`prepare_step`.
        """
        y = np.asarray(vector, dtype=float)
        batch = self.epoch_batch((epoch_tdb,)) if batch is None else batch
        inputs = self._load_evaluation(epoch_tdb, batch, y, history, index=index, rigid=rigid, step=step)
        moon = self.force_group.compute_accelerations(inputs, "moon")
        earth = self.force_group.compute_accelerations(inputs, "earth")
        parts = [y[3:6], moon - earth]
        if self.rotation is not None:
            parts.append(self.rotation.evaluate(inputs, y[6:]).derivative)
        return np.concatenate(parts)

    def _load_evaluation(self, epoch, batch, vector, history, *, index=0, rigid=False, step: StepData | None = None):
        step = self.prepare_step(epoch, batch, history, index=index, rigid=rigid) if step is None else step
        matrix = None if self.rotation is None else quaternion_matrix(vector[6:10])
        inputs = self._inputs.load_evaluation(
            batch,
            index,
            epoch,
            vector,
            history,
            moon_matrix=matrix,
            step_payloads=step.force_payloads,
        )
        deformation = None if self.rotation is None else self.rotation.deformation
        if deformation is not None:
            inertia = step.lunar_inertia
            inputs.lunar_inertia = inertia
            if inertia is not None and inertia.degree2 is not None:
                inputs.coefficient_overrides[id(deformation.field)] = deformation.field.coefficients_with_degree2(
                    inertia.degree2
                )
        return inputs


def reference_inputs(system, epoch, batch, vector, history=None, *, index=0):
    oracle = ReferenceDynamics.__new__(ReferenceDynamics)
    oracle._inputs = DynamicsEvaluationContext(system.body_names, system.gravitational_parameters_m3_s2)
    oracle.rotation = system.rotation
    oracle._fused = system._fused
    return oracle._load_evaluation(epoch, batch, vector, history, index=index)


def inertia_gravity(
    inertia, *, gravitational_parameter_m3_s2, radius_m, c21_static=0.0, s21_static=0.0, s22_static=0.0
):
    coefficients = np.zeros((2, 3, 3))
    coefficients[0, 0, 0] = 1.0
    coefficients[:, 2, :3] = degree2_coefficients(
        inertia,
        c21_static=c21_static,
        s21_static=s21_static,
        s22_static=s22_static,
    )
    return GravityCoefficients(coefficients, gravitational_parameter_m3_s2, radius_m)


def evaluate_newtonian_point_mass_system(positions, gm):
    delta = positions[None, :, :] - positions[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    inv = 1.0 / distance
    acceleration = np.sum((gm[None, :] * inv**3)[:, :, None] * delta, axis=1)
    potential = np.sum(gm[None, :] * inv, axis=1)
    return PointMassGravityEvaluation(acceleration, potential)


def evaluate_eih_correction(positions, velocities, gm, newtonian, targets):
    correction = np.zeros_like(positions)
    for i in targets:
        vi = velocities[i]
        for j in range(len(gm)):
            if i == j:
                continue
            d = positions[j] - positions[i]
            vj = velocities[j]
            aj = newtonian.accelerations_mps2[j]
            inv = 1.0 / np.linalg.norm(d)
            factor = (
                -4 * newtonian.potentials_m2_s2[i]
                - newtonian.potentials_m2_s2[j]
                + vi @ vi
                + 2 * (vj @ vj)
                - 4 * (vi @ vj)
                - 1.5 * (d @ vj * inv) * (d @ vj * inv)
                + 0.5 * (d @ aj)
            )
            correction[i] += (
                gm[j] * (inv * inv * inv) * (d * factor + (d @ (4 * vi - 3 * vj)) * (vj - vi))
                + (3.5 * inv * gm[j]) * aj
            )
    return correction * (1.0 / C**2)


def earth_tide_relative_acceleration(r, vectors, gm, mu_earth, mu_moon, radius, k20, k21, k22):
    x, y, z = r
    rho = np.array([x, y, 0.0])
    rho_sq = rho @ rho
    common = r / np.linalg.norm(r) ** 2
    result = np.zeros(3)
    for delayed, mu in zip(vectors, gm, strict=True):
        r0, r1, r2 = delayed
        rho0 = np.array([r0[0], r0[1], 0.0])
        rho1 = np.array([r1[0], r1[1], 0.0])
        rho2 = np.array([r2[0], r2[1], 0.0])
        dot1, dot2 = rho @ rho1, rho @ rho2
        t0 = (rho0 @ rho0) * rho + np.array([0.0, 0.0, 2 * z * r0[2] ** 2])
        t0 -= 5 * ((z * r0[2]) ** 2 + 0.5 * rho_sq * (rho0 @ rho0)) * common
        t0 += (r0 @ r0) * r
        t1 = 2 * z * r1[2] * rho1 + np.array([0.0, 0.0, 2 * dot1 * r1[2]])
        t1 -= 10 * z * r1[2] * dot1 * common
        t2 = 2 * dot2 * rho2 - (rho2 @ rho2) * rho
        t2 -= 5 * (dot2**2 - 0.5 * rho_sq * (rho2 @ rho2)) * common
        scale = 1.5 * (mu_earth + mu_moon) / mu_earth
        scale *= mu * radius**5 / np.linalg.norm(r) ** 5
        result += (
            scale * k20 * (t0 / np.linalg.norm(r0) ** 5)
            + scale * k21 * (t1 / np.linalg.norm(r1) ** 5)
            + scale * k22 * (t2 / np.linalg.norm(r2) ** 5)
        )
    return result
