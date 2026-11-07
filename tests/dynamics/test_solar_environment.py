"""Secular solar GM and the Kuiper-belt ring."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lunarops.classes.dynamics import (
    EarthTideParameters,
    EihPointMassForce,
    GravitatingBody,
    LunarDynamics,
    LunarForceGroup,
    NewtonianPointMassForce,
    make_j2_coefficients,
)
from lunarops.classes.dynamics.gravity import GravityField
from lunarops.classes.dynamics.solar_environment import AU_M, KuiperBeltRing, SolarGmVariation
from tests.support import reference_dynamics as oracle
from tests.support.dynamics import EARTH_MU, EPOCH, MOON_MU, FunctionEphemeris
from tests.tolerances import DYNAMICS_REL, REL, Tolerance, assert_close, assert_vector_close

SUN_GM = 1.3271244004127942e20
POINT_GM = 5.5227699716988214e-13 * AU_M**3 / 86400**2
YEAR_S = 365.25 * 86400
STATE = np.array([3.8e8, 0, 0, 0, 1000, 0])


def _ephemeris():
    near_earth = ("EARTH", "EARTH MOON BARYCENTER")
    return FunctionEphemeris(
        lambda name, t: (np.array([AU_M, 0, 0]) if name in near_earth else np.zeros(3), np.zeros(3))
    )


def ring():
    return KuiperBeltRing(POINT_GM, SUN_GM, 44 * AU_M, 36, np.deg2rad(84381.448 / 3600), EPOCH.jd1)


def dynamics(*, belt=None, variation=None):
    return LunarDynamics(
        GravitatingBody("EARTH", EARTH_MU),
        GravitatingBody("MOON", MOON_MU),
        (GravitatingBody("SUN", SUN_GM),),
        _ephemeris(),
        LunarForceGroup((NewtonianPointMassForce(), EihPointMassForce())),
        kuiper_belt_ring=belt,
        solar_gm_variation=variation,
    )


@pytest.mark.parametrize("days", [-3652.5, 0, 3652.5])
def test_solar_gm_drifts_linearly_from_its_reference_epoch(days):
    drift = SolarGmVariation(-9e-14, EPOCH.jd1)
    assert_close(drift.factor(EPOCH.shifted(days * 86400)), 1 - 9e-14 * days / 365.25, REL)
    assert SolarGmVariation(-9e-14, 2440400.5).factor(EPOCH) != pytest.approx(1.0, abs=1e-15)


def test_ring_is_a_circle_about_the_sun_with_the_analytic_axial_field():
    belt = ring()
    sun, velocity = np.array([1e8, 2e8, -3e8]), np.array([10, 20, 30])
    r, v = belt.circular_states(EPOCH.shifted(987654), sun, velocity)
    normal = np.array([0, -np.sin(belt.obliquity_rad), np.cos(belt.obliquity_rad)])
    relative = r - sun
    metre = Tolerance("RING_GEOMETRY_M", atol=1e-3, unit="m")

    assert_close(np.linalg.norm(relative, axis=1), belt.radius_m, REL)
    assert_close(relative @ normal, 0, metre)
    assert_close(relative.mean(axis=0), 0, metre)
    assert_close(np.einsum("ij,ij->i", relative, v - velocity), 0, Tolerance("RING_TANGENTIAL", atol=40))
    # On the axis, the ring of N points attracts like N GM d / (R^2 + d^2)^1.5 towards its plane.
    distance = 0.03 * belt.radius_m
    delta = r - (sun + distance * normal)
    actual = np.sum(POINT_GM * delta / np.linalg.norm(delta, axis=1)[:, None] ** 3, axis=0)
    assert_vector_close(actual, -36 * POINT_GM * distance * normal / (belt.radius_m**2 + distance**2) ** 1.5, REL)


def test_ring_phase_and_point_count_are_negligible_for_the_moon():
    belt = ring()
    point = np.array([AU_M + 3.8e8, 0.1 * AU_M, -0.05 * AU_M])

    def acceleration(b):
        r, _ = b.states(EPOCH, np.zeros(3), np.zeros(3))
        d = r - point
        return np.sum(b.point_gm_m3_s2 * d / np.linalg.norm(d, axis=1)[:, None] ** 3, axis=0)

    baseline = acceleration(belt)
    for alternate in (
        replace(belt, phase_rad=0.074),
        replace(belt, point_count=4096, point_gm_m3_s2=36 * POINT_GM / 4096),
    ):
        assert_vector_close(acceleration(alternate), baseline, Tolerance("RING_QUADRATURE", atol=3e-27, unit="m/s^2"))
    assert np.dot(baseline, point) > 0  # an interior ring attracts radially outwards


@pytest.mark.slow
def test_prepropagated_ring_follows_independent_kepler_circles():
    duration = YEAR_S
    belt = ring().prepare(_ephemeris(), ("SUN",), np.array([SUN_GM]), EPOCH, duration, None)
    for seconds in (0, duration / 2, duration):
        epoch = EPOCH.shifted(seconds)
        r, v = belt.states(epoch, np.zeros(3), np.zeros(3))
        expected_r, expected_v = belt.circular_states(epoch, np.zeros(3), np.zeros(3))
        # Numerical propagation of 44 AU circles over one year.
        assert_close(r, expected_r, Tolerance("RING_PROPAGATION_M", atol=0.015, unit="m"))
        assert_close(v, expected_v, Tolerance("RING_PROPAGATION_MPS", atol=2e-11, unit="m/s"))


@pytest.mark.parametrize("days", [-365.25, 365.25])
def test_epoch_gm_and_ring_points_enter_the_newtonian_and_eih_terms(days):
    belt = ring()
    system = dynamics(belt=belt, variation=SolarGmVariation(-0.01, EPOCH.jd1))
    epoch = EPOCH.shifted(days * 86400)
    system.kuiper_belt_ring = system.kuiper_belt_ring.prepare(
        system.ephemeris, ("EARTH", "SUN"), np.array([EARTH_MU, SUN_GM]), EPOCH, days * 86400, system.solar_gm_variation
    )
    data = system.epoch_batch((epoch,))
    context = oracle.reference_inputs(system, epoch, data, STATE)
    newtonian = oracle.evaluate_newtonian_point_mass_system(context.positions_m, context.gravitational_parameters_m3_s2)
    eih = oracle.evaluate_eih_correction(
        context.positions_m, context.velocities_mps, context.gravitational_parameters_m3_s2, newtonian, np.array([1])
    )
    breakdown = system.terms(epoch, STATE, data)

    assert len(context.body_names) == 3 + belt.point_count
    assert system.prescribed_body_names == ("EARTH MOON BARYCENTER", "SUN")
    assert_close(context.gravitational_parameters_m3_s2[2], SUN_GM * (1 - 0.01 * days / 365.25), REL)
    assert_vector_close(breakdown["forces"]["point_mass"], newtonian.accelerations_mps2[1], DYNAMICS_REL)
    assert_vector_close(breakdown["forces"]["eih_1pn"], eih[1], DYNAMICS_REL)
    # A two-epoch batch reproduces the single-epoch inputs and uses the reference GM at the reference epoch.
    batch = system.epoch_batch((epoch, EPOCH))
    for name in (
        "positions_m",
        "gravitational_parameters_m3_s2",
        "source_accelerations_mps2",
        "source_potentials_m2_s2",
    ):
        assert_close(getattr(batch, name)[0], getattr(data, name)[0], REL, err_msg=name)
    assert_close(batch.gravitational_parameters_m3_s2[1, 2], SUN_GM, REL)


def test_ring_attracts_the_moon_and_the_earth_as_point_masses():
    baseline, augmented = dynamics(), dynamics(belt=ring())
    inputs = oracle.reference_inputs(augmented, EPOCH, augmented.epoch_batch((EPOCH,)), STATE)
    for key, body in (("forces", 1), ("earth_forces", 0)):
        delta = inputs.positions_m[3:] - inputs.positions_m[body]
        expected = np.sum(POINT_GM * delta / np.linalg.norm(delta, axis=1)[:, None] ** 3, axis=0)
        before = baseline.terms(EPOCH, STATE)[key]["point_mass"]
        after = augmented.terms(EPOCH, STATE)[key]["point_mass"]
        # The ring force (1e-14 m/s^2) is taken from accelerations 1e8 times larger.
        assert_vector_close(after - before, expected, Tolerance("RING_FORCE", atol=5e-18, unit="m/s^2"), err_msg=key)


@pytest.mark.parametrize(
    "model",
    [
        lambda: oracle.SolarJ2Force(
            GravityField(make_j2_coefficients(gm_m3_s2=SUN_GM, radius_m=6.96e8, j2=2e-7)), np.eye(3)
        ),
        lambda: oracle.LenseThirringForce(oracle.SolarSourceParameters(SUN_GM)),
        lambda: oracle.SolarRadiationPressureForce(oracle.SolarSourceParameters(SUN_GM), 3e-16),
    ],
    ids=["solar-j2", "lense-thirring", "radiation-pressure"],
)
def test_solar_gm_drift_scales_every_solar_force(model):
    system = dynamics(variation=SolarGmVariation(-0.1, EPOCH.jd1))
    state = np.array([3.8e8, 1e7, 0, 0, 1000, 0])
    future = EPOCH.shifted(YEAR_S)
    force = model()
    before = force.acceleration(oracle.reference_inputs(system, EPOCH, system.epoch_batch((EPOCH,)), state))
    after = force.acceleration(oracle.reference_inputs(system, future, system.epoch_batch((future,)), state))
    assert_vector_close(after, 0.9 * before, REL)


def test_solar_gm_drift_scales_the_solar_torque_and_the_sun_raised_tide():
    system = dynamics(variation=SolarGmVariation(-0.1, EPOCH.jd1))
    state = np.array([3.8e8, 1e7, 0, 0, 1000, 0])
    future = EPOCH.shifted(YEAR_S)
    torque = oracle.PointMassGravityTorque("SUN", np.diag([0.39, 0.3901, 0.3902]))
    before = torque.torque(oracle.reference_inputs(system, EPOCH, system.epoch_batch((EPOCH,)), state))
    after = torque.torque(oracle.reference_inputs(system, future, system.epoch_batch((future,)), state))
    assert_vector_close(after, 0.9 * before, REL)

    ephemeris = _ephemeris()
    history = SimpleNamespace(
        states=lambda names, epoch: np.array(
            [np.r_[ephemeris.body_state_bcrs(n, epoch).position_m, np.zeros(3)] for n in names]
        )
    )
    tide = oracle.EarthTideModel(
        EARTH_MU,
        MOON_MU,
        tide_parameters=EarthTideParameters(tau_orb_days=(0, 0, 0)),
        tide_raisers=("SUN",),
        tide_raiser_gm={"SUN": SUN_GM},
        solar_gm_variation=SolarGmVariation(-0.1, EPOCH.jd1),
    )
    earth = np.array([AU_M, 0, 0])
    moon = earth + np.array([3.8e8, 1e7, 0])
    assert_vector_close(
        tide.relative_acceleration(future, earth, moon, np.eye(3), history),
        0.9 * tide.relative_acceleration(EPOCH, earth, moon, np.eye(3), history),
        REL,
    )
