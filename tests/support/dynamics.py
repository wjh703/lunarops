"""Analytic ephemerides, a Moon field and complete force/torque systems for the dynamics tests."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lunarops.classes.dynamics import (
    EarthFigureForce,
    EarthFigureTorque,
    EarthTideForce,
    EarthTideModel,
    EihPointMassForce,
    GravitatingBody,
    LenseThirringForce,
    LunarCoreParameters,
    LunarDeformation,
    LunarDeformationParameters,
    LunarDynamics,
    LunarFigureTorque,
    LunarForceGroup,
    LunarRotationDynamics,
    LunarTorqueGroup,
    MoonFigureForce,
    NewtonianPointMassForce,
    OrientationProvider,
    PointMassGravityTorque,
    SolarJ2Force,
    SolarRadiationPressureForce,
    SolarSourceParameters,
    de440_earth_fixed2inertial_matrix,
    inertial2fixed_matrix_from_pole,
    make_gravity_coefficients,
    make_j2_coefficients,
)
from lunarops.classes.dynamics.gravity import GravityField
from lunarops.classes.dynamics.solar_environment import SolarGmVariation
from lunarops.classes.ephemerides import BodyState, Ephemeris
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.context import RunContext
from lunarops.fileio.mass_catalog import BodyMass, MassCatalog, write_mass_catalog
from lunarops.programs.registry import get_program
from tests.support import reference_dynamics as oracle

EPOCH = Epoch(2451545.0, 0.0, TimeScale.TDB)
EARTH_MU = 3.98600435507e14
MOON_MU = 4.902800118e12
SUN_GM = 1.32712440041279419e20
JUPITER_GM = 1.26712764e17
EARTH_RADIUS = 6378136.6
MOON_RADIUS = 1738000.0
# Lunar inertia per M R^2, an attitude quaternion and an Earth-to-Moon relative state.
K = np.diag([0.393, 0.3932, 0.3935])
Q = np.array([0.7, 0.2, -0.4, 0.1]) / np.linalg.norm([0.7, 0.2, -0.4, 0.1])
RELATIVE = np.array([3.84e8, 2e7, -1e7, 0.0, 1000.0, 0.0])
PA_TO_INERTIAL = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
SOLAR_POLE = (
    float(np.cos(np.deg2rad(63.87)) * np.cos(np.deg2rad(286.13))),
    float(np.cos(np.deg2rad(63.87)) * np.sin(np.deg2rad(286.13))),
    float(np.sin(np.deg2rad(63.87))),
)


class FunctionEphemeris(Ephemeris):
    """Ephemeris defined by ``state(name, seconds_since_J2000) -> (position, velocity)``."""

    def __init__(self, state=None):
        self._function = state if state is not None else (lambda name, t: (np.zeros(3), np.zeros(3)))

    @property
    def source_path(self):
        return None

    def body_state_bcrs(self, name, epoch):
        position, velocity = self._function(name, EPOCH.seconds_until(epoch))
        return BodyState(np.asarray(position, dtype=float), np.asarray(velocity, dtype=float))

    def body_position_bcrs(self, name, epoch):
        return self.body_state_bcrs(name, epoch).position_m

    def body_state_matrix_bcrs(self, bodies, epoch):
        return np.asarray(
            [
                np.r_[state.position_m, state.velocity_mps]
                for state in (self.body_state_bcrs(body, epoch) for body in bodies)
            ]
        ).reshape(-1, 6)

    def body_state_matrices_bcrs(self, bodies, epochs):
        return np.asarray([self.body_state_matrix_bcrs(bodies, epoch) for epoch in epochs])

    def body_accelerations_bcrs(self, body, epochs):
        return np.zeros((len(epochs), 3))

    def close(self):
        return None


def _solar_system_state(name, t):
    if name in ("EARTH", "EARTH MOON BARYCENTER"):
        a = 2.0e-7 * t
        return 1.496e11 * np.array([np.cos(a), np.sin(a), 0.0]), 29780.0 * np.array([-np.sin(a), np.cos(a), 0.0])
    if name == "SUN":
        velocity = np.array([1.0, 2.0, 0.1])
        return np.array([1.0e9, -3.0e8, 2.0e7]) + t * velocity, velocity
    b = 1.7e-8 * t + 1.0
    return 7.78e11 * np.array([np.cos(b), np.sin(b), 0.02]), 13060.0 * np.array([-np.sin(b), np.cos(b), 0.0])


def solar_system():
    """Smooth analytic Earth-Moon barycentre (also returned for the Earth), Sun and Jupiter."""
    return FunctionEphemeris(_solar_system_state)


def _circular_moon_state(name, t):
    if name == "EARTH":
        return np.zeros(3), np.zeros(3)
    if name == "SUN":
        return np.array([1.49e11, 2e10, 0.0]), np.zeros(3)
    angle = t * 1000 / 3.844e8
    position = 3.844e8 * np.array([np.cos(angle), np.sin(angle), 0])
    velocity = 1000 * np.array([-np.sin(angle), np.cos(angle), 0])
    if name == "EARTH MOON BARYCENTER":
        fraction = MOON_MU / (EARTH_MU + MOON_MU)
        return fraction * position, fraction * velocity
    return position, velocity


def circular_moon_ephemeris():
    """Earth at rest, a fixed Sun, a Moon on a 3.844e8 m circle at 1 km/s and their GM-weighted barycentre."""
    return FunctionEphemeris(_circular_moon_state)


class History:
    """Body states for the Earth tide and a delayed Hermite sample for the deformation."""

    def __init__(self, state):
        self.state = np.array(state, dtype=float)
        self.ephemeris = solar_system()

    def states(self, names, epoch):
        t = EPOCH.seconds_until(epoch)
        rows = []
        for name in names:
            body = self.ephemeris.body_state_bcrs("EARTH" if name == "MOON" else name, epoch)
            if name == "MOON":
                rows.append(
                    np.r_[body.position_m + RELATIVE[:3] * (1.0 + 1.0e-9 * t), body.velocity_mps + RELATIVE[3:6]]
                )
            else:
                rows.append(np.r_[body.position_m, body.velocity_mps])
        return np.asarray(rows)

    def integration_sample(self, epoch):
        state = self.state.copy()
        state[:3] = RELATIVE[:3] * (1.0 + 1.0e-9 * EPOCH.seconds_until(epoch))
        derivative = np.zeros_like(state)
        derivative[10:13] = [1.0e-13, -2.0e-14, 3.0e-15]
        return state, derivative


def make_system(*, rotation, forces=(), ephemeris=None):
    return LunarDynamics(
        GravitatingBody("EARTH", EARTH_MU),
        GravitatingBody("MOON", MOON_MU),
        perturbing_bodies=(),
        ephemeris=FunctionEphemeris() if ephemeris is None else ephemeris,
        force_group=LunarForceGroup(forces),
        rotation=rotation,
    )


def oracle_inputs(system, rotation_state, *, epoch=EPOCH):
    """State vector and the oracle's evaluation context for ``RELATIVE`` plus a rotation state."""
    y = np.concatenate((RELATIVE, rotation_state))
    return y, oracle.reference_inputs(system, epoch, system.epoch_batch((epoch,)), y)


def moon_field():
    rng = np.random.default_rng(3)
    cosine = np.tril(rng.normal(size=(5, 5)) * 1.0e-5)
    sine = np.tril(rng.normal(size=(5, 5)) * 1.0e-5)
    cosine[0, 0] = 1.0
    cosine[1, :2] = 0.0
    sine[0, :] = 0.0
    sine[1, :2] = 0.0
    sine[:, 0] = 0.0
    return GravityField(make_gravity_coefficients(cosine, sine, gm_m3_s2=MOON_MU, radius_m=MOON_RADIUS))


def earth_field():
    cosine = np.zeros((6, 6))
    cosine[0, 0] = 1.0
    cosine[2, 0], cosine[3, 0], cosine[4, 0], cosine[5, 0] = -4.841694e-4, 9.57e-7, 5.4e-7, 6.8e-8
    return GravityField(make_gravity_coefficients(cosine, np.zeros((6, 6)), gm_m3_s2=EARTH_MU, radius_m=EARTH_RADIUS))


ALL_FORCES = (
    "point_mass",
    "eih_1pn",
    "solar_j2",
    "figure_earth",
    "figure_moon",
    "tide",
    "lense_thirring",
    "solar_radiation_pressure",
)
ALL_TORQUES = ("figure:EARTH", "figure:SUN", "quadrupole:JUPITER BARYCENTER", "figure:EARTH_J2")


def _as_oracle(model):
    return getattr(oracle, type(model).__name__)(**{f.name: getattr(model, f.name) for f in fields(model) if f.init})


def build(*, fused, rotation=True, core=True, deformation=True, forces=None, torques=None, geodetic=True):
    """Every supported force and torque, as the native system (``fused``) or the oracle; each call owns its fields."""
    variation = SolarGmVariation(-9.0e-14, 2451545.0)
    field = moon_field()
    solar = SolarSourceParameters(SUN_GM, 6.96e8, spin_axis=SOLAR_POLE)
    tide = EarthTideModel(
        EARTH_MU,
        MOON_MU,
        tide_raiser_gm={"MOON": MOON_MU, "SUN": SUN_GM},
        tide_raisers=("MOON", "SUN"),
        solar_gm_variation=variation,
    )
    force_models = [
        NewtonianPointMassForce(),
        EihPointMassForce(),
        SolarJ2Force(
            GravityField(make_j2_coefficients(gm_m3_s2=SUN_GM, radius_m=6.96e8, j2=2.2e-7)),
            inertial2fixed_matrix_from_pole(SOLAR_POLE),
        ),
        EarthFigureForce(
            earth_field(),
            partners=("MOON", "SUN", "JUPITER BARYCENTER"),
            j2_rate_per_year=-2.6e-11,
            j2_quadratic_per_year2=1.0e-13,
        ),
        MoonFigureForce(field, ("EARTH", "SUN", "JUPITER BARYCENTER")),
        EarthTideForce(tide),
        LenseThirringForce(solar),
        SolarRadiationPressureForce(solar, 1.0e-9),
    ]
    if not fused:
        force_models = [_as_oracle(model) for model in force_models]
    dynamics_rotation = None
    state = RELATIVE
    if rotation:
        deforming = (
            LunarDeformation(
                LunarDeformationParameters(0.0242, 0.14519388 * 86400.0, 2.6617e-6, c21_static=2.0e-9),
                K,
                MOON_RADIUS,
                EARTH_MU,
                MOON_MU,
                field,
            )
            if deformation
            else None
        )
        torque_models = [
            LunarFigureTorque("EARTH", field),
            LunarFigureTorque("SUN", field),
            PointMassGravityTorque("JUPITER BARYCENTER", K),
            EarthFigureTorque(K, EARTH_RADIUS, 1.0826e-3, -2.6e-11, 1.0e-13),
        ]
        torque_models = [model for model in torque_models if torques is None or model.name in torques]
        if not fused:
            torque_models = [_as_oracle(model) for model in torque_models]
        dynamics_rotation = (LunarRotationDynamics if fused else oracle.LunarRotationDynamics)(
            K,
            MOON_RADIUS,
            (LunarTorqueGroup if fused else oracle.LunarTorqueGroup)(torque_models),
            core=LunarCoreParameters(0.0007, 0.000246, 7.0e-14) if core else None,
            geodetic_precession=geodetic,
            deformation=deforming,
        )
        state = np.concatenate((RELATIVE, Q, [1.0e-7, -2.0e-8, 2.7e-6], [2.0e-8, 1.0e-8, 2.6e-6] if core else []))
    system = (LunarDynamics if fused else oracle.ReferenceDynamics)(
        GravitatingBody("EARTH", EARTH_MU),
        GravitatingBody("MOON", MOON_MU),
        perturbing_bodies=(GravitatingBody("SUN", SUN_GM), GravitatingBody("JUPITER BARYCENTER", JUPITER_GM)),
        ephemeris=solar_system(),
        force_group=(LunarForceGroup if fused else oracle.LunarForceGroup)(force_models, forces),
        earth_fixed2inertial_matrix_provider=OrientationProvider(de440_earth_fixed2inertial_matrix),
        moon_fixed2inertial_matrix_provider=None if rotation else OrientationProvider(fixed_lunar_axes),
        rotation=dynamics_rotation,
        solar_gm_variation=variation,
    )
    return system, state


def fixed_lunar_axes(epoch):
    """Module level so the epoch-input recipe stays picklable."""
    return PA_TO_INERTIAL


class AnalyticRunContext(RunContext):
    """Run context whose ephemeris declaration resolves to ``circular_moon_ephemeris``."""

    def create_class(self, *args, **kwargs):
        return circular_moon_ephemeris()


def orbit_program_config(root: Path, *, eih: bool, direction: int = 1):
    """A resolved 20-minute LunarOrbitPropagation configuration on the analytic ephemeris."""
    catalog = MassCatalog(
        (
            BodyMass("EARTH", EARTH_MU, ("major",), 399),
            BodyMass("MOON", MOON_MU, ("major",), 301),
            BodyMass("SUN", 1.32712440018e20, ("major",), 10),
        ),
        name="analytic test",
    )
    write_mass_catalog(catalog, root / "masses.txt")
    config = get_program("LunarOrbitPropagation")[0].schema.resolve(
        {
            "inputFileMassCatalog": "masses.txt",
            "outputFileOrbit": "orbit.dat",
            "outputFileAccelerationDiagnostics": "accel.dat",
            "outputFileMetadata": "meta.txt",
            "ephemerides": {"type": "calceph", "directory": ".", "lunarRelativisticScaleConvention": "alreadyScaled"},
            "initialTdbJd1": EPOCH.jd1,
            "durationSeconds": direction * 1200,
            "stepSeconds": 60,
            "outputStepSeconds": 60,
            "startupStepSeconds": 15,
            "integratorOrder": 3,
            "historyInterpolationOrder": 3,
            "trajectoryInterpolationOrder": 3,
            "externalBodyIds": ["SUN"],
            "includeEih": eih,
            "accelerationDiagnosticsStepSeconds": 60,
        }
    )
    return config, AnalyticRunContext(working_dir=root)
