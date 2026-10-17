from dataclasses import replace

import numpy as np
import pytest

from lunarops.classes.dynamics import (
    EarthTideModel,
    EarthTideParameters,
    EihPointMassForce,
    GravitatingBody,
    LenseThirringForce,
    LunarDynamics,
    LunarForceGroup,
    NewtonianPointMassForce,
    SolarJ2Force,
    SolarRadiationPressureForce,
    SolarSourceParameters,
    evaluate_eih_correction,
    make_j2_coefficients,
)
from lunarops.classes.dynamics.gravity import GravityField
from lunarops.classes.dynamics.solar_environment import AU_M, KuiperBeltRing, SolarGmVariation
from lunarops.classes.dynamics.torques import PointMassGravityTorque
from lunarops.classes.ephemerides import BodyState, Ephemeris
from lunarops.classes.time import Epoch, TimeScale

EPOCH = Epoch(2451545, 0, TimeScale.TDB)
SUN_GM = 1.3271244004127942e20
POINT_GM = 5.5227699716988214e-13 * AU_M**3 / 86400**2
PARAMETERS = (("EARTH", 3.98600435507e14), ("MOON", 4.902800118e12), ("SUN", SUN_GM))


class Provider(Ephemeris):
    def body_state_matrix_bcrs(self, bodies, epoch):
        return np.asarray([np.concatenate((self.body_state_bcrs(body, epoch).position_m,
                                          self.body_state_bcrs(body, epoch).velocity_mps)) for body in bodies]).reshape(-1, 6)

    def body_state_matrices_bcrs(self, bodies, epochs):
        return np.asarray([self.body_state_matrix_bcrs(bodies, epoch) for epoch in epochs])

    def body_accelerations_bcrs(self, body, epochs):
        return np.asarray([self.body_acceleration_bcrs(body, epoch) for epoch in epochs])

    @property
    def source_path(self):
        return None

    def body_state_bcrs(self, name, epoch):
        return BodyState(np.array([AU_M, 0, 0]) if name == "EARTH" else np.zeros(3), np.zeros(3))

    def body_position_bcrs(self, name, epoch):
        return self.body_state_bcrs(name, epoch).position_m

    def body_acceleration_bcrs(self, name, epoch):
        return np.zeros(3)

    def close(self):
        pass


def ring():
    return KuiperBeltRing(POINT_GM, SUN_GM, 44 * AU_M, 36, np.deg2rad(84381.448 / 3600), EPOCH.jd1)


def dynamics(*, belt=None, variation=None):
    return LunarDynamics(
        GravitatingBody(*PARAMETERS[0]), GravitatingBody(*PARAMETERS[1]),
        (GravitatingBody(*PARAMETERS[2]),), Provider(),
        LunarForceGroup((NewtonianPointMassForce(), EihPointMassForce())),
        kuiper_belt_ring=belt, solar_gm_variation=variation,
    )


@pytest.mark.parametrize("days", [-3652.5, 0, 3652.5])
def test_solar_drift_uses_explicit_reference_epoch_in_both_directions(days):
    drift = SolarGmVariation(-9e-14, EPOCH.jd1)
    value = drift.factor(EPOCH.shifted(days * 86400))
    assert value == 1 - 9e-14 * days / 365.25
    other = SolarGmVariation(-9e-14, 2440400.5)
    if days == 0:
        assert other.factor(EPOCH) != value


def test_ring_geometry_and_analytic_axis_field():
    belt = ring()
    sun = np.array([1e8, 2e8, -3e8])
    velocity = np.array([10, 20, 30])
    r, v = belt.circular_states(EPOCH.shifted(987654), sun, velocity)
    normal = np.array([0, -np.sin(belt.obliquity_rad), np.cos(belt.obliquity_rad)])
    relative = r - sun
    np.testing.assert_allclose(np.linalg.norm(relative, axis=1), belt.radius_m, rtol=2e-16)
    np.testing.assert_allclose(relative @ normal, 0, atol=.001)
    np.testing.assert_allclose(relative.mean(axis=0), 0, atol=.001)
    np.testing.assert_allclose(np.einsum("ij,ij->i", relative, v - velocity), 0, atol=40)
    distance = .03 * belt.radius_m
    delta = r - (sun + distance * normal)
    actual = np.sum(POINT_GM * delta / np.linalg.norm(delta, axis=1)[:, None]**3, axis=0)
    expected = -36 * POINT_GM * distance * normal / (belt.radius_m**2 + distance**2)**1.5
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-28)


def test_ring_phase_and_dense_quadrature_are_negligible_for_moon_field():
    belt = ring()
    point = np.array([AU_M + 3.8e8, .1 * AU_M, -.05 * AU_M])

    def acceleration(b):
        r, _ = b.states(EPOCH, np.zeros(3), np.zeros(3))
        d = r - point
        return np.sum(b.point_gm_m3_s2 * d / np.linalg.norm(d, axis=1)[:, None]**3, axis=0)

    baseline = acceleration(belt)
    for alternate in (replace(belt, phase_rad=.074), replace(belt, point_count=4096, point_gm_m3_s2=36 * POINT_GM / 4096)):
        np.testing.assert_allclose(acceleration(alternate), baseline, rtol=0, atol=3e-27)
    assert np.dot(baseline, point) > 0  # Interior ring attracts radially outwards.


def test_ring_prepropagation_recovers_independent_kepler_circles():
    belt = ring()
    duration = 365.25 * 86400
    belt = belt.prepare(Provider(), ("SUN",), np.array([SUN_GM]), EPOCH, duration, None)
    for seconds in (0, duration / 2, duration):
        epoch = EPOCH.shifted(seconds)
        r, v = belt.states(epoch, np.zeros(3), np.zeros(3))
        expected_r, expected_v = belt.circular_states(epoch, np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(r, expected_r, rtol=0, atol=.015)
        np.testing.assert_allclose(v, expected_v, rtol=0, atol=2e-11)


@pytest.mark.parametrize("days", [-365.25, 365.25])
def test_epoch_gm_and_ring_enter_cached_newtonian_eih_and_history(days):
    dyn = dynamics(belt=ring(), variation=SolarGmVariation(-.01, EPOCH.jd1))
    epoch = EPOCH.shifted(days * 86400)
    dyn.kuiper_belt_ring = dyn.kuiper_belt_ring.prepare(dyn.ephemeris, ("EARTH", "SUN"),
                                np.array([PARAMETERS[0][1], SUN_GM]), EPOCH, days * 86400,
                                dyn.solar_gm_variation)
    data = dyn.build_epoch_data(epoch)
    state = np.array([3.8e8, 0, 0, 0, 1000, 0])
    context = dyn._load_evaluation(data, state, None, None)
    assert len(context.body_names) == 39
    assert context.gravitational_parameters_m3_s2[2] == SUN_GM * (1 - .01 * days / 365.25)
    from lunarops._dynamics_core import point_mass_symmetric
    from lunarops.classes.dynamics.forces import PointMassGravityEvaluation
    direct = PointMassGravityEvaluation(*point_mass_symmetric(context.positions_m, context.gravitational_parameters_m3_s2))
    breakdown = dyn.accelerations(epoch, state, data)
    np.testing.assert_allclose(breakdown.accelerations_by_force_mps2["point_mass"][1], direct.accelerations_mps2[1], rtol=2e-15, atol=1e-28)
    eih = evaluate_eih_correction(context.positions_m, context.velocities_mps, context.gravitational_parameters_m3_s2, direct, np.array([1]))
    np.testing.assert_allclose(breakdown.accelerations_by_force_mps2["eih_1pn"][1], eih[1], rtol=2e-15)
    np.testing.assert_array_equal(breakdown.total_accelerations_mps2[0], 0)
    assert dyn.history_state_matrix(data).shape == (2, 6)
    assert dyn.history_body_names == ("EARTH", "SUN")
    batch = dyn.build_epoch_data_batch((epoch, EPOCH))
    np.testing.assert_array_equal(batch[0].positions_m, data.positions_m)
    np.testing.assert_array_equal(batch[0].gravitational_parameters_m3_s2, data.gravitational_parameters_m3_s2)
    np.testing.assert_array_equal(
        batch[0].point_mass_gravity_cache.prescribed_body_accelerations_mps2,
        data.point_mass_gravity_cache.prescribed_body_accelerations_mps2,
    )
    np.testing.assert_array_equal(
        batch[0].point_mass_gravity_cache.prescribed_body_potentials_m2_s2,
        data.point_mass_gravity_cache.prescribed_body_potentials_m2_s2,
    )
    at_reference = dyn.accelerations(EPOCH, state, batch[1])
    assert dyn._inputs.gravitational_parameters_m3_s2[2] == SUN_GM
    assert np.linalg.norm(at_reference.total_accelerations_mps2[1] - breakdown.total_accelerations_mps2[1]) > 0


def test_ring_adds_moon_absolute_force_without_subtracting_earth_ring():
    baseline = dynamics()
    augmented = dynamics(belt=ring())
    state = np.array([3.8e8, 0, 0, 0, 1000, 0])
    inputs = augmented._load_evaluation(augmented.build_epoch_data(EPOCH), state, None, None)
    delta = inputs.positions_m[3:] - inputs.positions_m[1]
    expected = np.sum(POINT_GM * delta / np.linalg.norm(delta, axis=1)[:, None]**3, axis=0)
    before = baseline.accelerations(EPOCH, state).accelerations_by_force_mps2["point_mass"][1]
    after = augmented.accelerations(EPOCH, state).accelerations_by_force_mps2["point_mass"][1]
    np.testing.assert_allclose(after - before, expected, rtol=0, atol=5e-18)


def test_solar_gm_is_shared_by_j2_lt_srp_and_torque():
    dyn = dynamics(variation=SolarGmVariation(-.1, EPOCH.jd1))
    state = np.array([3.8e8, 1e7, 0, 0, 1000, 0])
    reference = dyn.build_epoch_data(EPOCH)
    future = dyn.build_epoch_data(EPOCH.shifted(365.25 * 86400))
    field = GravityField(make_j2_coefficients(gm_m3_s2=SUN_GM, radius_m=6.96e8, j2=2e-7))
    solar = SolarSourceParameters(SUN_GM)
    models = (SolarJ2Force(field, np.eye(3)), LenseThirringForce(solar),
              SolarRadiationPressureForce(solar, {"MOON": 3e-16}))
    for model in models:
        before = model.acceleration(dyn._load_evaluation(reference, state, None, None))
        after = model.acceleration(dyn._load_evaluation(future, state, None, None))
        np.testing.assert_allclose(after, .9 * before, rtol=3e-15)
    torque = PointMassGravityTorque("SUN", np.diag([.39, .3901, .3902]))
    before = torque.torque(dyn._load_evaluation(reference, state, None, None))
    after = torque.torque(dyn._load_evaluation(future, state, None, None))
    np.testing.assert_allclose(after, .9 * before, rtol=3e-15)


def test_solar_gm_drift_scales_sun_raised_earth_tide():
    provider = Provider()

    def history(names, epoch):
        return np.array([np.r_[provider.body_state_bcrs(n, epoch).position_m, np.zeros(3)] for n in names])

    tide = EarthTideModel(PARAMETERS[0][1], PARAMETERS[1][1],
                          tide_parameters=EarthTideParameters(tau_orb_days=(0, 0, 0)),
                          tide_raisers=("SUN",), tide_raiser_gm={"SUN": SUN_GM},
                          solar_gm_variation=SolarGmVariation(-.1, EPOCH.jd1))
    earth = np.array([AU_M, 0, 0]); moon = earth + np.array([3.8e8, 1e7, 0])
    before = tide.relative_acceleration(EPOCH, earth, moon, np.eye(3), history)
    after = tide.relative_acceleration(EPOCH.shifted(365.25 * 86400), earth, moon, np.eye(3), history)
    np.testing.assert_allclose(after, .9 * before, rtol=2e-15)
