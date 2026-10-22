"""The fused native right-hand side reproduces the Python force and torque reference path."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_lunar_dynamics import EARTH_MU, MOON_MU, Provider
from test_lunar_rotation import EPOCH, RADIUS, RELATIVE, K, Q

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
from lunarops.classes.ephemerides import BodyState

SUN_GM = 1.32712440041279419e20
JUPITER_GM = 1.26712764e17
EARTH_RADIUS = 6378136.6
STATE_BLOCKS = (slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), slice(13, 16))
PA_TO_INERTIAL = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
SOLAR_POLE = (
    float(np.cos(np.deg2rad(63.87)) * np.cos(np.deg2rad(286.13))),
    float(np.cos(np.deg2rad(63.87)) * np.sin(np.deg2rad(286.13))),
    float(np.sin(np.deg2rad(63.87))),
)


def fixed_lunar_axes(epoch):
    """Module level so the epoch-input recipe stays picklable."""
    return PA_TO_INERTIAL


class SolarSystem(Provider):
    """Smooth analytic Earth, Sun and Jupiter; the integrated Moon is not part of the ephemeris."""

    def __init__(self, earth_acceleration=(1.0e-7, -2.0e-7, 3.0e-8)):
        self.earth_acceleration = np.asarray(earth_acceleration, dtype=float)

    def body_state_bcrs(self, name, epoch):
        t = EPOCH.seconds_until(epoch)
        if name == "EARTH":
            a = 2.0e-7 * t
            return BodyState(
                1.496e11 * np.array([np.cos(a), np.sin(a), 0.0]), 29780.0 * np.array([-np.sin(a), np.cos(a), 0.0])
            )
        if name == "SUN":
            velocity = np.array([1.0, 2.0, 0.1])
            return BodyState(np.array([1.0e9, -3.0e8, 2.0e7]) + t * velocity, velocity)
        b = 1.7e-8 * t + 1.0
        return BodyState(
            7.78e11 * np.array([np.cos(b), np.sin(b), 0.02]), 13060.0 * np.array([-np.sin(b), np.cos(b), 0.0])
        )

    def body_acceleration_bcrs(self, name, epoch):
        return self.earth_acceleration.copy()


class History:
    """Body states for the Earth tide and the delayed Hermite sample for the deformation."""

    def __init__(self, state):
        self.state = np.array(state, dtype=float)
        self.ephemeris = SolarSystem()

    def __call__(self, names, epoch):
        t = EPOCH.seconds_until(epoch)
        rows = []
        for name in names:
            if name == "MOON":
                earth = self.ephemeris.body_state_bcrs("EARTH", epoch)
                moon = RELATIVE[:3] * (1.0 + 1.0e-9 * t)
                rows.append(np.r_[earth.position_m + moon, earth.velocity_mps + RELATIVE[3:6]])
            else:
                body = self.ephemeris.body_state_bcrs(name, epoch)
                rows.append(np.r_[body.position_m, body.velocity_mps])
        return np.asarray(rows)

    def integration_sample(self, epoch):
        t = EPOCH.seconds_until(epoch)
        state = self.state.copy()
        state[:3] = RELATIVE[:3] * (1.0 + 1.0e-9 * t)
        derivative = np.zeros_like(state)
        derivative[10:13] = [1.0e-13, -2.0e-14, 3.0e-15]
        return state, derivative


def _moon_field():
    rng = np.random.default_rng(3)
    cosine = np.tril(rng.normal(size=(5, 5)) * 1.0e-5)
    sine = np.tril(rng.normal(size=(5, 5)) * 1.0e-5)
    cosine[0, 0] = 1.0
    cosine[1, :2] = 0.0
    sine[0, :] = 0.0
    sine[1, :2] = 0.0
    sine[:, 0] = 0.0
    return GravityField(make_gravity_coefficients(cosine, sine, gm_m3_s2=MOON_MU, radius_m=RADIUS))


def _earth_field():
    cosine = np.zeros((6, 6))
    cosine[0, 0] = 1.0
    cosine[2, 0], cosine[3, 0], cosine[4, 0], cosine[5, 0] = -4.841694e-4, 9.57e-7, 5.4e-7, 6.8e-8
    return GravityField(make_gravity_coefficients(cosine, np.zeros((6, 6)), gm_m3_s2=EARTH_MU, radius_m=EARTH_RADIUS))


ALL_FORCES = (
    "point_mass", "eih_1pn", "solar_j2", "figure_earth", "figure_moon", "tide", "lense_thirring",
    "solar_radiation_pressure",
)
ALL_TORQUES = ("figure:EARTH", "figure:SUN", "quadrupole:JUPITER BARYCENTER", "figure:EARTH_J2")


def build(
    *, fused, rotation=True, core=True, deformation=True, forces=None, torques=None, geodetic=True,
    earth_acceleration=(1.0e-7, -2.0e-7, 3.0e-8),
):
    """A dynamics with every supported force and torque; each call owns its fields."""
    variation = SolarGmVariation(-9.0e-14, 2451545.0)
    moon_field = _moon_field()
    sun = make_j2_coefficients(gm_m3_s2=SUN_GM, radius_m=6.96e8, j2=2.2e-7)
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
        SolarJ2Force(GravityField(sun), inertial2fixed_matrix_from_pole(SOLAR_POLE)),
        EarthFigureForce(_earth_field(), j2_rate_per_year=-2.6e-11, j2_quadratic_per_year2=1.0e-13),
        MoonFigureForce(moon_field, ("EARTH", "SUN", "JUPITER BARYCENTER")),
        EarthTideForce(tide),
        LenseThirringForce(solar),
        SolarRadiationPressureForce(solar, 1.0e-9),
    ]
    dynamics_rotation = None
    state = RELATIVE
    if rotation:
        deforming = (
            LunarDeformation(
                LunarDeformationParameters(0.0242, 0.14519388 * 86400.0, 2.6617e-6, c21_static=2.0e-9),
                K,
                RADIUS,
                EARTH_MU,
                MOON_MU,
                moon_field,
            )
            if deformation
            else None
        )
        torque_models = [
            LunarFigureTorque("EARTH", moon_field),
            LunarFigureTorque("SUN", moon_field),
            PointMassGravityTorque("JUPITER BARYCENTER", K),
            EarthFigureTorque(K, EARTH_RADIUS, 1.0826e-3, -2.6e-11, 1.0e-13),
        ]
        torque_models = [model for model in torque_models if torques is None or model.name in torques]
        dynamics_rotation = LunarRotationDynamics(
            K,
            RADIUS,
            LunarTorqueGroup(torque_models),
            core=LunarCoreParameters(0.0007, 0.000246, 7.0e-14) if core else None,
            geodetic_precession=geodetic,
            deformation=deforming,
        )
        state = np.concatenate((RELATIVE, Q, [1.0e-7, -2.0e-8, 2.7e-6], [2.0e-8, 1.0e-8, 2.6e-6] if core else []))
    system = LunarDynamics(
        GravitatingBody("EARTH", EARTH_MU),
        GravitatingBody("MOON", MOON_MU),
        perturbing_bodies=(GravitatingBody("SUN", SUN_GM), GravitatingBody("JUPITER BARYCENTER", JUPITER_GM)),
        ephemeris=SolarSystem(earth_acceleration),
        force_group=LunarForceGroup(force_models, forces),
        earth_fixed2inertial_matrix_provider=OrientationProvider.from_scalar(de440_earth_fixed2inertial_matrix),
        moon_fixed2inertial_matrix_provider=None
        if rotation
        else OrientationProvider.from_scalar(fixed_lunar_axes),
        rotation=dynamics_rotation,
        solar_gm_variation=variation,
        fused_rhs=fused,
    )
    return system, state


def assert_blocks_close(actual, expected):
    """Per-block tolerance: positions, quaternion and angular rates differ by many orders of magnitude."""
    assert actual.shape == expected.shape
    for block in STATE_BLOCKS:
        if block.start >= len(expected):
            break
        reference = expected[block]
        np.testing.assert_allclose(actual[block], reference, rtol=1e-10, atol=1e-10 * np.max(np.abs(reference)))


VARIANTS = [
    pytest.param({"rotation": False}, id="translation-only"),
    pytest.param({"core": False, "deformation": False}, id="rigid-mantle"),
    pytest.param({}, id="deforming-core"),
]


@pytest.mark.parametrize("options", VARIANTS)
@pytest.mark.parametrize("seconds", [0.0, 3.1e6, -4.0e7])
def test_fused_rhs_matches_reference(options, seconds):
    fused, state = build(fused=True, **options)
    reference, _ = build(fused=False, **options)
    assert fused._fused is not None
    assert reference._fused is None
    history = History(state)
    epoch = EPOCH.shifted(seconds)
    candidate = state * (1.0 + 1.0e-6 * np.linspace(-1.0, 1.0, len(state)))
    expected = reference.derivatives(epoch, candidate, history=history)
    actual = fused.derivatives(epoch, candidate, history=history)
    assert_blocks_close(actual, expected)


def test_prepared_step_serves_every_corrector_candidate():
    fused, state = build(fused=True)
    reference, _ = build(fused=False)
    history = History(state)
    epoch = EPOCH.shifted(1.0e6)
    fused_data = fused.build_epoch_data(epoch)
    reference_data = reference.build_epoch_data(epoch)
    step = fused.prepare_step(fused_data, history)
    results = []
    for scale in (1.0, 1.0 + 3.0e-7, 1.0 - 8.0e-7):
        candidate = state * scale
        expected = reference.derivatives(epoch, candidate, reference_data, history=history)
        actual = fused.derivatives(epoch, candidate, fused_data, history=history, step=step)
        assert_blocks_close(actual, expected)
        results.append(actual)
    assert not np.array_equal(results[0], results[1])


def test_rigid_seed_matches_reference():
    fused, state = build(fused=True)
    reference, _ = build(fused=False)
    history = History(state)
    epoch = EPOCH.shifted(-2.0e6)
    assert_blocks_close(
        fused.derivatives(epoch, state, history=history, rigid=True),
        reference.derivatives(epoch, state, history=history, rigid=True),
    )


NO_EARTH_ACCELERATION = (0.0, 0.0, 0.0)


@pytest.mark.parametrize("rotation", [False, True], ids=["prescribed-moon", "integrated-moon"])
@pytest.mark.parametrize("force", [name for name in ALL_FORCES if name != "eih_1pn"])
def test_each_force_alone_matches_reference(force, rotation):
    """With the Earth's own acceleration removed, the acceleration block is exactly one force term."""
    options = {"forces": (force,), "earth_acceleration": NO_EARTH_ACCELERATION, "rotation": rotation, "core": False,
               "deformation": False}
    fused, state = build(fused=True, **options)
    reference, _ = build(fused=False, **options)
    history = History(state)
    epoch = EPOCH.shifted(2.2e6)
    expected = reference.derivatives(epoch, state, history=history)[3:6]
    actual = fused.derivatives(epoch, state, history=history)[3:6]
    assert np.all(expected != 0.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0.0)


def test_eih_term_matches_reference_as_a_difference_from_point_mass():
    def eih_only(fused):
        results = []
        for names in (("point_mass",), ("point_mass", "eih_1pn")):
            system, state = build(fused=fused, forces=names, earth_acceleration=NO_EARTH_ACCELERATION, rotation=False)
            results.append(system.derivatives(EPOCH.shifted(1.0e6), state)[3:6])
        return results[1] - results[0]

    expected = eih_only(False)
    assert np.all(expected != 0.0)
    np.testing.assert_allclose(eih_only(True), expected, rtol=1e-6)


def _torque_state(state, *, mantle_rate=(0.0, 0.0, 0.0), core_rate=None):
    state = state.copy()
    state[10:13] = mantle_rate
    if core_rate is not None:
        state[13:16] = core_rate
    return state


@pytest.mark.parametrize("torque", ALL_TORQUES)
def test_each_torque_alone_matches_reference(torque):
    """A non-rotating mantle has no gyroscopic term, so the angular acceleration is I^-1 times the torque."""
    options = {"forces": (), "torques": (torque,), "core": False, "deformation": False, "geodetic": False}
    fused, state = build(fused=True, **options)
    reference, _ = build(fused=False, **options)
    state = _torque_state(state)
    history = History(state)
    epoch = EPOCH.shifted(2.2e6)
    expected = reference.derivatives(epoch, state, history=history)[10:13]
    actual = fused.derivatives(epoch, state, history=history)[10:13]
    assert np.all(expected != 0.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0.0)


def test_core_mantle_coupling_matches_reference():
    options = {"forces": (), "torques": (), "deformation": False, "geodetic": False}
    fused, state = build(fused=True, **options)
    reference, _ = build(fused=False, **options)
    state = _torque_state(state, core_rate=(3.0e-6, -2.0e-6, 2.6e-6))
    history = History(state)
    epoch = EPOCH.shifted(2.2e6)
    expected = reference.derivatives(epoch, state, history=history)
    actual = fused.derivatives(epoch, state, history=history)
    for block in (slice(10, 13), slice(13, 16)):
        assert np.all(expected[block] != 0.0)
        np.testing.assert_allclose(actual[block], expected[block], rtol=1e-11, atol=0.0)


def test_geodetic_precession_matches_reference_as_a_difference():
    def effect(fused):
        results = []
        for geodetic in (False, True):
            system, state = build(
                fused=fused, forces=(), torques=(), core=False, deformation=False, geodetic=geodetic
            )
            state = _torque_state(state, mantle_rate=(1.0e-6, -4.0e-7, 2.7e-6))
            results.append(system.derivatives(EPOCH.shifted(1.0e6), state, history=History(state))[10:13])
        return results[1] - results[0]

    expected = effect(False)
    assert np.all(expected != 0.0)
    np.testing.assert_allclose(effect(True), expected, rtol=1e-6)


def test_quaternion_kinematics_use_the_raw_state():
    """The state is not renormalized by the integrator; the derivative must follow the raw quaternion."""
    fused, state = build(fused=True, forces=(), torques=(), core=False, deformation=False, geodetic=False)
    reference, _ = build(fused=False, forces=(), torques=(), core=False, deformation=False, geodetic=False)
    state = state.copy()
    state[6:10] *= 1.0003
    history = History(state)
    np.testing.assert_allclose(
        fused.derivatives(EPOCH, state, history=history)[6:10],
        reference.derivatives(EPOCH, state, history=history)[6:10],
        rtol=1e-12,
    )


def test_zero_quaternion_is_rejected():
    fused, state = build(fused=True)
    state = state.copy()
    state[6:10] = 0.0
    with pytest.raises(ValueError, match="attitude quaternion"):
        fused.derivatives(EPOCH, state, history=History(state))


def test_unknown_force_model_falls_back_to_the_reference_path():
    class Extra:
        name = "extra"

        def acceleration(self, inputs):
            return np.array([1.0e-9, 0.0, 0.0])

    system = LunarDynamics(
        GravitatingBody("EARTH", EARTH_MU),
        GravitatingBody("MOON", MOON_MU),
        perturbing_bodies=(),
        ephemeris=Provider(),
        force_group=LunarForceGroup([NewtonianPointMassForce(), Extra()]),
    )
    assert system._fused is None
    total = system.accelerations(EPOCH, RELATIVE).total_accelerations_mps2
    np.testing.assert_array_equal(system.derivatives(EPOCH, RELATIVE)[3:6], total[1])
    assert total[1, 0] != 0.0
