"""The native fused right-hand side against the NumPy oracle, term by term."""

import pickle
from dataclasses import fields

import numpy as np
import pytest

from lunarops._dynamics_core import point_mass_symmetric, point_mass_symmetric_batch
from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    GravitatingBody,
    IntegratorSettings,
    LenseThirringForce,
    LunarDynamics,
    LunarForceGroup,
    LunarRotationDynamics,
    LunarTorqueGroup,
    NewtonianPointMassForce,
    SolarRadiationPressureForce,
    SolarSourceParameters,
)
from lunarops.classes.dynamics.epoch_data import SerialEpochSource, batch_epochs
from lunarops.classes.dynamics.history import EphemerisHistory
from tests.support import reference_dynamics as oracle
from tests.support.dynamics import (
    ALL_FORCES,
    ALL_TORQUES,
    EARTH_MU,
    EPOCH,
    MOON_MU,
    MOON_RADIUS,
    SUN_GM,
    FunctionEphemeris,
    History,
    K,
    build,
    solar_system,
)
from tests.tolerances import DYNAMICS_REL, REL, Tolerance, assert_close, assert_vector_close

# Positions, quaternion and angular rates differ by many orders of magnitude; compare each block.
STATE_BLOCKS = (slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), slice(13, 16))
VARIANTS = [
    pytest.param({"rotation": False}, id="translation-only"),
    pytest.param({"core": False, "deformation": False}, id="rigid-mantle"),
    pytest.param({}, id="deforming-core"),
]


def assert_blocks_close(actual, expected):
    assert actual.shape == expected.shape
    for block in STATE_BLOCKS[: (len(expected) + 2) // 3]:
        assert_vector_close(actual[block], expected[block], DYNAMICS_REL, err_msg=str(block))


def _pair(**options):
    fused, state = build(fused=True, **options)
    reference, _ = build(fused=False, **options)
    return fused, reference, state


@pytest.mark.parametrize("options", VARIANTS)
@pytest.mark.parametrize("seconds", [0.0, 3.1e6, -4.0e7])
def test_full_derivative_matches_the_oracle(options, seconds):
    fused, reference, state = _pair(**options)
    history = History(state)
    epoch = EPOCH.shifted(seconds)
    candidate = state * (1.0 + 1.0e-6 * np.linspace(-1.0, 1.0, len(state)))

    assert_blocks_close(fused.derivatives(epoch, candidate, history=history),
                        reference.derivatives(epoch, candidate, history=history))


def test_rigid_seed_derivative_matches_the_oracle():
    fused, reference, state = _pair()
    history = History(state)
    epoch = EPOCH.shifted(-2.0e6)
    assert_blocks_close(fused.derivatives(epoch, state, history=history, rigid=True),
                        reference.derivatives(epoch, state, history=history, rigid=True))


def test_one_prepared_step_serves_every_corrector_candidate():
    fused, reference, state = _pair()
    history = History(state)
    epoch = EPOCH.shifted(1.0e6)
    fused_data, reference_data = fused.epoch_batch((epoch,)), reference.epoch_batch((epoch,))
    step = fused.prepare_step(epoch, fused_data, history)
    results = []
    for scale in (1.0, 1.0 + 3.0e-7, 1.0 - 8.0e-7):
        actual = fused.derivatives(epoch, state * scale, fused_data, history=history, step=step)
        assert_blocks_close(actual, reference.derivatives(epoch, state * scale, reference_data, history=history))
        results.append(actual)
    assert not np.array_equal(results[0], results[1])


@pytest.mark.parametrize("rotation", [False, True], ids=["prescribed-moon", "integrated-moon"])
@pytest.mark.parametrize("force", [name for name in ALL_FORCES if name != "eih_1pn"])
def test_each_force_alone_matches_the_oracle(force, rotation):
    """The acceleration block is one force term acting on the Moon minus the same term on the Earth."""
    fused, reference, state = _pair(forces=(force,), rotation=rotation, core=False, deformation=False)
    history = History(state)
    epoch = EPOCH.shifted(2.2e6)
    expected = reference.derivatives(epoch, state, history=history)[3:6]

    assert np.linalg.norm(expected) > 0.0
    assert_vector_close(fused.derivatives(epoch, state, history=history)[3:6], expected, DYNAMICS_REL)


def test_eih_correction_matches_the_oracle_as_a_difference_from_point_mass():
    def eih_only(fused):
        results = []
        for names in (("point_mass",), ("point_mass", "eih_1pn")):
            system, state = build(fused=fused, forces=names, rotation=False)
            results.append(system.derivatives(EPOCH.shifted(1.0e6), state)[3:6])
        return results[1] - results[0]

    expected = eih_only(False)
    assert np.linalg.norm(expected) > 0.0
    # The difference of two accelerations 1e8 times larger keeps about seven digits.
    assert_vector_close(eih_only(True), expected, Tolerance("EIH_DIFFERENCE", rtol=1e-6))


def _rotation_state(state, *, mantle_rate=(0.0, 0.0, 0.0), core_rate=None):
    state = state.copy()
    state[10:13] = mantle_rate
    if core_rate is not None:
        state[13:16] = core_rate
    return state


@pytest.mark.parametrize("torque", ALL_TORQUES)
def test_each_torque_alone_matches_the_oracle(torque):
    """A non-rotating mantle has no gyroscopic term, so the angular acceleration is I^-1 times the torque."""
    fused, reference, state = _pair(forces=(), torques=(torque,), core=False, deformation=False, geodetic=False)
    state = _rotation_state(state)
    history = History(state)
    epoch = EPOCH.shifted(2.2e6)
    expected = reference.derivatives(epoch, state, history=history)[10:13]

    assert np.linalg.norm(expected) > 0.0
    assert_vector_close(fused.derivatives(epoch, state, history=history)[10:13], expected, DYNAMICS_REL)


def test_core_mantle_coupling_matches_the_oracle():
    fused, reference, state = _pair(forces=(), torques=(), deformation=False, geodetic=False)
    state = _rotation_state(state, core_rate=(3.0e-6, -2.0e-6, 2.6e-6))
    history = History(state)
    epoch = EPOCH.shifted(2.2e6)
    expected = reference.derivatives(epoch, state, history=history)
    actual = fused.derivatives(epoch, state, history=history)
    for block in (slice(10, 13), slice(13, 16)):
        assert np.linalg.norm(expected[block]) > 0.0
        assert_vector_close(actual[block], expected[block], DYNAMICS_REL)


def test_geodetic_precession_matches_the_oracle_as_a_difference():
    def effect(fused):
        results = []
        for geodetic in (False, True):
            system, state = build(fused=fused, forces=(), torques=(), core=False, deformation=False, geodetic=geodetic)
            state = _rotation_state(state, mantle_rate=(1.0e-6, -4.0e-7, 2.7e-6))
            results.append(system.derivatives(EPOCH.shifted(1.0e6), state, history=History(state))[10:13])
        return results[1] - results[0]

    expected = effect(False)
    assert np.linalg.norm(expected) > 0.0
    assert_vector_close(effect(True), expected, Tolerance("DE_SITTER_DIFFERENCE", rtol=1e-6))


def test_quaternion_kinematics_follow_the_raw_unnormalized_state():
    fused, reference, state = _pair(forces=(), torques=(), core=False, deformation=False, geodetic=False)
    state = state.copy()
    state[6:10] *= 1.0003
    history = History(state)
    assert_vector_close(fused.derivatives(EPOCH, state, history=history)[6:10],
                        reference.derivatives(EPOCH, state, history=history)[6:10], DYNAMICS_REL)


@pytest.mark.parametrize("options", VARIANTS)
def test_native_term_diagnostics_match_the_oracle_and_sum_to_the_derivative(options):
    fused, reference, state = _pair(**options)
    history = History(state)
    epoch = EPOCH.shifted(1.0e6)
    derivative = fused.derivatives(epoch, state, history=history)
    inputs = reference._load_evaluation(epoch, reference.epoch_batch((epoch,)), state, history)
    actual = fused.terms(epoch, state, history=history)

    assert_blocks_close(actual["derivative"], derivative)
    for key, body in (("forces", "moon"), ("earth_forces", "earth")):
        for name, expected in reference.force_group.acceleration_terms(inputs, body).items():
            if np.linalg.norm(expected) == 0.0:
                assert np.linalg.norm(actual[key][name]) == 0.0, (key, name)
            else:
                assert_vector_close(actual[key][name], expected, DYNAMICS_REL, err_msg=f"{key} {name}")
    relative = sum(actual["forces"].values()) - sum(actual["earth_forces"].values())
    assert_vector_close(relative, derivative[3:6], DYNAMICS_REL)
    if reference.rotation is not None:
        expected = reference.rotation.evaluate(inputs, state[6:])
        for vector, oracle_vector in zip(actual["torques"], expected.torques.values(), strict=True):
            assert_vector_close(vector, oracle_vector, DYNAMICS_REL)
        oracle_rotation = (expected.viscous_torque, expected.pressure_torque, expected.geodetic_rate_pa_rad_s)
        for vector, oracle_vector in zip(actual["rotation"], oracle_rotation, strict=True):
            assert_vector_close(vector, oracle_vector, DYNAMICS_REL)
        if expected.inertia is not None:
            core = reference.rotation.core_inertia
            for vector, oracle_vector in zip(actual["inertia"], (expected.inertia.total, expected.inertia.total - core,
                                                                 expected.inertia.derivative), strict=True):
                assert_close(vector, oracle_vector, REL)


class _Extra:
    name = "extra"


@pytest.mark.parametrize(("make", "message"), [
    (lambda: {"force_group": LunarForceGroup([NewtonianPointMassForce(), _Extra()])}, "unsupported: force model"),
    (lambda: {"rotation": LunarRotationDynamics(K, MOON_RADIUS, LunarTorqueGroup([_Extra()]))},
     "unsupported: torque model"),
    (lambda: {"force_group": LunarForceGroup([LenseThirringForce(SolarSourceParameters(SUN_GM))])},
     "unsupported: .* needs SUN"),
    (lambda: {"force_group": LunarForceGroup([SolarRadiationPressureForce(SolarSourceParameters(SUN_GM), 1e-13)])},
     "unsupported: .* needs SUN"),
], ids=["unknown-force", "unknown-torque", "lense-thirring-without-sun", "radiation-pressure-without-sun"])
def test_unsupported_models_are_rejected_at_construction(make, message):
    with pytest.raises(ValueError, match=message):
        LunarDynamics(GravitatingBody("EARTH", EARTH_MU), GravitatingBody("MOON", MOON_MU), perturbing_bodies=(),
                      ephemeris=FunctionEphemeris(), **make())


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_zero_quaternion_is_rejected():
    fused, state = build(fused=True)
    state = state.copy()
    state[6:10] = 0.0
    with pytest.raises(ValueError, match="attitude quaternion"):
        fused.derivatives(EPOCH, state, history=History(state))


def test_epoch_inputs_rebuilt_from_a_pickled_recipe_match_the_local_arrays():
    system, _ = build(fused=True, rotation=False)
    recipe = pickle.loads(pickle.dumps(system.epoch_inputs))
    epochs = batch_epochs(EPOCH, [0.0, 2700.0, -5400.0, 1.0e7])
    local = system.epoch_inputs.build(system.ephemeris, epochs)
    remote = recipe.build(solar_system(), epochs)
    for field in fields(local):
        np.testing.assert_array_equal(getattr(remote, field.name), getattr(local, field.name), err_msg=field.name)


def test_integrator_accepts_an_external_epoch_source():
    system, state = build(fused=True, rotation=False)
    integrator = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=600.0, order=4, startup_step_s=75.0))
    history = History(state)
    default = integrator.integrate(system, EPOCH, state, 86400.0, external_history_provider=history)
    source = SerialEpochSource(pickle.loads(pickle.dumps(system.epoch_inputs)), solar_system())
    streamed = integrator.integrate(system, EPOCH, state, 86400.0, external_history_provider=history,
                                    epoch_source=source)
    # Same epoch data in the same order, so the same arithmetic.
    np.testing.assert_array_equal(streamed.states, default.states)


def test_startup_history_places_earth_and_moon_about_the_barycentre():
    rows = {"EARTH": np.array([1.0, 2.0, -3.0, 4.0, 5.0, 6.0]), "MOON": np.array([0.1, 0.2, -0.3, 0.4, 0.5, 0.6]),
            "SUN": np.arange(6, dtype=float)}
    rows["EARTH MOON BARYCENTER"] = (EARTH_MU * rows["EARTH"] + MOON_MU * rows["MOON"]) / (EARTH_MU + MOON_MU)
    ephemeris = FunctionEphemeris(lambda name, t: (rows[name][:3], rows[name][3:]))

    actual = EphemerisHistory(ephemeris, EARTH_MU, MOON_MU).states(("MOON", "SUN", "EARTH"), EPOCH)

    # With the ephemeris relative state the barycentric placement returns the ephemeris Earth and Moon.
    assert_close(actual, [rows["MOON"], rows["SUN"], rows["EARTH"]], Tolerance("RELATIVE_STATE_ROUND_OFF", atol=1e-14))


@pytest.mark.parametrize("epochs,bodies", [(0, 4), (1, 0), (3, 1), (64, 128)])
def test_batched_source_gravity_matches_the_per_epoch_kernel(epochs, bodies):
    rng = np.random.default_rng(617)
    positions = rng.normal(size=(epochs, bodies, 3)) * 1e11
    gm = np.exp(rng.uniform(10, 45, size=(epochs, bodies)))
    acceleration, potential = point_mass_symmetric_batch(positions, gm)

    assert acceleration.shape == positions.shape and potential.shape == gm.shape
    for index in range(epochs):
        expected_acceleration, expected_potential = point_mass_symmetric(positions[index], gm[index])
        assert_vector_close(acceleration[index], expected_acceleration, REL)
        assert_close(potential[index], expected_potential, REL)


def test_batched_source_gravity_matches_the_newtonian_oracle():
    rng = np.random.default_rng(5)
    positions = rng.normal(size=(2, 6, 3)) * 1e11
    gm = np.exp(rng.uniform(10, 45, size=(2, 6)))
    acceleration, potential = point_mass_symmetric_batch(positions, gm)
    for index in range(2):
        expected = oracle.evaluate_newtonian_point_mass_system(positions[index], gm[index])
        assert_vector_close(acceleration[index], expected.accelerations_mps2, DYNAMICS_REL)
        assert_close(potential[index], expected.potentials_m2_s2, REL)


@pytest.mark.parametrize("bad_epoch", [0, 3, 7])
def test_coincident_bodies_are_reported_from_any_parallel_epoch(bad_epoch):
    positions = np.random.default_rng(118).normal(size=(8, 64, 3))
    positions[bad_epoch, 1] = positions[bad_epoch, 0]
    with pytest.raises(ValueError, match="coincident"):
        point_mass_symmetric_batch(positions, np.ones((8, 64)))
