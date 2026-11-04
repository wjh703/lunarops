"""Translational RHS, the Earth tide on a prescribed Earth, and the ABM integrator on Kepler orbits."""

from types import SimpleNamespace

import numpy as np
import pytest

from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    EarthFigureForce,
    EarthTideForce,
    EarthTideParameters,
    GravitatingBody,
    IntegratorSettings,
    LunarDynamics,
    LunarForceGroup,
    MoonFigureForce,
    NewtonianPointMassForce,
    OrientationProvider,
    make_gravity_coefficients,
)
from lunarops.classes.dynamics.gravity import GravityField
from tests.support import reference_dynamics as oracle
from tests.support.dynamics import EARTH_MU, EPOCH, MOON_MU, FunctionEphemeris, make_system
from tests.tolerances import DYNAMICS_REL, REL, Tolerance, assert_close, assert_vector_close

RADIUS = 3.844e8
MEAN_MOTION = np.sqrt(EARTH_MU / RADIUS**3)
CIRCULAR = np.array([RADIUS, 0.0, 0.0, 0.0, RADIUS * MEAN_MOTION, 0.0])
TEN_DAYS = 10 * 86400.0


def _two_body():
    return make_system(rotation=None, forces=[NewtonianPointMassForce()])


def _circular_position(times):
    return np.vstack([RADIUS * np.cos(MEAN_MOTION * times), RADIUS * np.sin(MEAN_MOTION * times), 0 * times])


def test_point_mass_rhs_is_the_keplerian_acceleration_about_a_prescribed_earth():
    state = np.array([3.5e8, 0.7e8, -0.4e8, 10.0, 1000.0, -5.0])
    r = state[:3]
    derivative = _two_body().derivatives(EPOCH, state)

    assert_close(derivative[:3], state[3:], REL)
    assert_vector_close(derivative[3:], -EARTH_MU * r / np.linalg.norm(r) ** 3, REL)


def test_abm_converges_at_fifth_order_for_order_four():
    """PECE with an order-4 predictor and its corrector is globally fifth order."""
    errors = []
    for step in (7200.0, 3600.0, 1800.0):
        trajectory = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=step, order=4)).integrate(
            _two_body(), EPOCH, CIRCULAR, TEN_DAYS)
        errors.append(np.max(np.linalg.norm(trajectory.states[:3] - _circular_position(trajectory.times_s), axis=0)))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((2**4.5 < ratios) & (ratios < 2**5.5)), ratios


def test_high_order_abm_follows_the_analytic_circle_and_conserves_kepler_invariants():
    trajectory = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=1800.0, order=8)).integrate(
        _two_body(), EPOCH, CIRCULAR, TEN_DAYS)
    positions, velocities = trajectory.states[:3], trajectory.states[3:6]
    energy = 0.5 * np.sum(velocities**2, axis=0) - EARTH_MU / np.linalg.norm(positions, axis=0)
    momentum = np.cross(positions.T, velocities.T)

    # The analytic circle is reproduced to the round-off floor of 3.8e8 m positions (about 2e-6 m).
    assert_vector_close(positions.T, _circular_position(trajectory.times_s).T, Tolerance("KEPLER_M", atol=1e-5, unit="m"))
    assert_close(energy, energy[0], Tolerance("KEPLER_ENERGY", rtol=1e-12))
    assert_vector_close(momentum, np.broadcast_to(momentum[0], momentum.shape), Tolerance("KEPLER_MOMENTUM", rtol=1e-12))


def test_backward_integration_returns_to_the_initial_state():
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=1800.0, order=8))
    eccentric = np.array([RADIUS, 0.0, 0.0, 0.0, 1.1 * RADIUS * MEAN_MOTION, 30.0])
    forward = integrator.integrate(_two_body(), EPOCH, eccentric, TEN_DAYS)
    final = forward.states[:, -1]
    backward = integrator.integrate(_two_body(), EPOCH.shifted(TEN_DAYS), final, -TEN_DAYS)

    assert_vector_close(backward.states[:3, -1], eccentric[:3], Tolerance("REVERSIBILITY_M", atol=1e-5, unit="m"))
    assert_vector_close(backward.states[3:, -1], eccentric[3:], Tolerance("REVERSIBILITY_MPS", atol=1e-11, unit="m/s"))


def test_short_propagation_is_rejected_before_seeding():
    # Seeding would need epoch data; this system offers none.
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=900.0, order=4))
    with pytest.raises(ValueError, match="needs at least order-1 = 3 steps; got 2"):
        integrator.integrate(SimpleNamespace(rotation=None), EPOCH, np.zeros(6), 1800.0)


@pytest.mark.parametrize("raiser", ["MOON", "SUN"])
def test_earth_tide_on_a_prescribed_earth_recovers_the_relative_acceleration_and_keeps_the_emb(raiser):
    positions = {"EARTH": np.zeros(3), "MOON": np.array([3.5e8, 0.7e8, -0.4e8]),
                 "SUN": np.array([1.2e11, -0.8e11, 0.1e11])}
    history = SimpleNamespace(states=lambda names, epoch: np.array([np.r_[positions[name], np.zeros(3)]
                                                                    for name in names]))
    tide = oracle.EarthTideModel(EARTH_MU, MOON_MU, tide_parameters=EarthTideParameters(tau_orb_days=(0, 0, 0),
                                                                                         tau_rot_days=(0, 0, 0)),
                                 tide_raisers=(raiser,), tide_raiser_gm={"MOON": MOON_MU, "SUN": 1.3271244e20})
    relative_tide = tide.relative_acceleration(EPOCH, positions["EARTH"], positions["MOON"], np.eye(3), history)
    # The tide leaves the Earth-Moon barycentre unaffected; this is the Earth's prescribed share.
    earth_tide = -MOON_MU / (EARTH_MU + MOON_MU) * relative_tide
    dynamics = LunarDynamics(GravitatingBody("EARTH", EARTH_MU), GravitatingBody("MOON", MOON_MU),
                             perturbing_bodies=(), ephemeris=FunctionEphemeris(earth_acceleration=earth_tide),
                             force_group=LunarForceGroup((EarthTideForce(tide),)))
    state = np.concatenate((positions["MOON"], np.zeros(3)))

    assert_vector_close(dynamics.derivatives(EPOCH, state, history=history)[3:], relative_tide, DYNAMICS_REL)
    moon_tide = dynamics.terms(EPOCH, state, history=history)["forces"]["tide"]
    barycentre = (EARTH_MU * earth_tide + MOON_MU * moon_tide) / (EARTH_MU + MOON_MU)
    assert np.linalg.norm(barycentre) <= 1e-14 * np.linalg.norm(moon_tide)


@pytest.mark.parametrize("source", ["EARTH", "MOON"])
def test_figure_acceleration_rotates_with_the_body_fixed_frames(source):
    # A tilted, nonsymmetric rotation and a non-axisymmetric field expose reversed transforms.
    rotation = np.array([[0.36, -0.8, 0.48], [0.48, 0.6, 0.64], [-0.8, 0.0, 0.6]])
    cosine, sine = np.zeros((3, 3)), np.zeros((3, 3))
    cosine[0, 0], cosine[2, 0], cosine[2, 2], sine[2, 1] = 1.0, -1e-4, 2e-5, 3e-5
    field = GravityField(make_gravity_coefficients(cosine, sine, gm_m3_s2=EARTH_MU if source == "EARTH" else MOON_MU,
                                                   radius_m=1e6, name="rotation test"))
    force = EarthFigureForce(field) if source == "EARTH" else MoonFigureForce(field, ("EARTH",))

    def system(matrix):
        provider = OrientationProvider(lambda epoch: matrix)
        return LunarDynamics(GravitatingBody("EARTH", EARTH_MU), GravitatingBody("MOON", MOON_MU), perturbing_bodies=(),
                             ephemeris=FunctionEphemeris(), force_group=LunarForceGroup((force,)),
                             earth_fixed2inertial_matrix_provider=provider, moon_fixed2inertial_matrix_provider=provider)

    position = np.array([3.5e8, 0.7e8, -0.4e8])
    aligned = system(np.eye(3)).terms(EPOCH, np.concatenate((position, np.zeros(3))))["forces"]
    rotated = system(rotation).terms(EPOCH, np.concatenate((rotation @ position, np.zeros(3))))["forces"]

    assert_vector_close(next(iter(rotated.values())), rotation @ next(iter(aligned.values())), REL)
