import numpy as np
import pytest

from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    BodyRole,
    DynamicalBody,
    EarthMoonBarycentricState,
    EarthMoonDynamics,
    EarthMoonIntegrationState,
    EarthTideForce,
    EarthTideModel,
    EarthTideParameters,
    FigureForce,
    ForceGroup,
    IntegratorSettings,
    LunarInertiaModel,
    LunarInertiaParameters,
    PointMassForce,
    de440_earth_dynamics_frame,
    de440_earth_pole_inertial,
)
from lunarops.classes.dynamics.context import DynamicsContext, PreparedEpoch
from lunarops.classes.dynamics.force_models import LunarInertiaFigureForce
from lunarops.classes.dynamics.forces import (
    eih_acceleration,
    evaluate_point_mass_system,
    prepare_point_mass_system,
)
from lunarops.classes.dynamics.gravity import GravityField, make_gravity_coefficients
from lunarops.classes.ephemerides import BodyState
from lunarops.classes.time import Epoch, TimeScale

EPOCH = Epoch(2451545.0, 0.0, TimeScale.TDB)
EARTH_MU = 3.98600435507e14
MOON_MU = 4.902800118e12


def _rotate_z(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def make_dynamics():
    bodies = (("EARTH", EARTH_MU), ("MOON", MOON_MU))
    return EarthMoonDynamics(
        DynamicalBody("EARTH", EARTH_MU, BodyRole.INTEGRATED),
        DynamicalBody("MOON", MOON_MU, BodyRole.INTEGRATED),
        force_group=ForceGroup((PointMassForce(bodies),)),
    )


def test_physical_and_integration_states_round_trip():
    earth = BodyState([-4.7e6, 0.0, 0.0], [0.0, -0.012, 0.0])
    moon = BodyState([3.796e8, 0.0, 0.0], [0.0, 1022.0, 0.0])
    physical = EarthMoonBarycentricState(EPOCH, earth, moon)
    integration = physical.to_integration(EARTH_MU, MOON_MU)
    assert isinstance(integration, EarthMoonIntegrationState)
    recovered = integration.to_barycentric(EARTH_MU, MOON_MU)
    np.testing.assert_allclose(recovered.earth.position_m, earth.position_m)
    np.testing.assert_allclose(recovered.moon.velocity_mps, moon.velocity_mps)


def test_newtonian_force_and_emb_relative_projection():
    physical = EarthMoonBarycentricState(
        EPOCH,
        BodyState([-4.7e6, 0.0, 0.0], [0.0, 0.0, 0.0]),
        BodyState([3.796e8, 0.0, 0.0], [0.0, 1022.0, 0.0]),
    )
    dynamics = make_dynamics()
    breakdown = dynamics.accelerations(EPOCH, physical.to_integration(EARTH_MU, MOON_MU).to_vector())
    assert set(breakdown.terms_mps2) == {"point_mass"}
    earth = breakdown.accelerations_mps2[breakdown.body_names.index("EARTH")]
    moon = breakdown.accelerations_mps2[breakdown.body_names.index("MOON")]
    np.testing.assert_allclose(breakdown.relative_mps2, moon - earth)


def test_integrator_is_independent_of_force_composition():
    dynamics = make_dynamics()
    physical = EarthMoonBarycentricState(
        EPOCH,
        BodyState([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        BodyState([3.844e8, 0.0, 0.0], [0.0, 1022.0, 0.0]),
    )
    initial = physical.to_integration(EARTH_MU, MOON_MU)
    trajectory = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=3600.0, order=2)).integrate(
        dynamics, EPOCH, initial.to_vector(), 3600.0
    )
    assert trajectory.vector(EPOCH.shifted(3600.0)).shape == (12,)
    assert trajectory.vector(EPOCH.shifted(1800.0)).shape == (12,)


def test_accepted_step_observer_receives_regular_nodes_and_prepared_epoch():
    dynamics = make_dynamics()
    physical = EarthMoonBarycentricState(
        EPOCH,
        BodyState([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        BodyState([3.844e8, 0.0, 0.0], [0.0, 1022.0, 0.0]),
    )
    accepted = []

    def observer(epoch, state, prepared, history):
        accepted.append((EPOCH.seconds_until(epoch), state.copy(), prepared.epoch_tdb, history))

    AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=900.0, order=2)).integrate(
        dynamics,
        EPOCH,
        physical.to_integration(EARTH_MU, MOON_MU).to_vector(),
        2700.0,
        accepted_step_observer=observer,
    )
    assert [item[0] for item in accepted] == [0.0, 900.0, 1800.0, 2700.0]
    assert all(item[2] == EPOCH.shifted(item[0]) for item in accepted)


def test_integrator_sampled_query_supports_reverse_arc():
    dynamics = make_dynamics()
    physical = EarthMoonBarycentricState(
        EPOCH,
        BodyState([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        BodyState([3.844e8, 0.0, 0.0], [0.0, 1022.0, 0.0]),
    )
    initial = physical.to_integration(EARTH_MU, MOON_MU)
    trajectory = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=900.0, order=2)).integrate(
        dynamics, EPOCH, initial.to_vector(), -3600.0
    )
    assert trajectory.vector(EPOCH.shifted(-1800.0)).shape == (12,)


def test_paper_abmd_settings_and_regular_grid():
    settings = IntegratorSettings()
    assert settings.step_s == 5400.0
    assert settings.order == 13
    assert settings.corrector_iterations == 2
    assert settings.interpolation_order == 13
    assert settings.startup_step_s == 675.0

    dynamics = make_dynamics()
    physical = EarthMoonBarycentricState(
        EPOCH,
        BodyState([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        BodyState([3.844e8, 0.0, 0.0], [0.0, 1022.0, 0.0]),
    )
    trajectory = AdamsBashforthMoultonIntegrator(settings).integrate(
        dynamics, EPOCH, physical.to_integration(EARTH_MU, MOON_MU).to_vector(), 13 * settings.step_s
    )
    np.testing.assert_array_equal(trajectory.times_s, np.arange(14) * settings.step_s)
    assert trajectory.history_buffer_capacity == 14
    assert trajectory.history_buffer_max_nodes == 14


def test_fixed_grid_rejects_partial_main_step():
    dynamics = make_dynamics()
    initial = np.zeros(12)
    with pytest.raises(ValueError, match="integer multiple"):
        AdamsBashforthMoultonIntegrator().integrate(dynamics, EPOCH, initial, 5401.0)


def test_startup_step_must_divide_main_step():
    with pytest.raises(ValueError, match="integer multiple"):
        IntegratorSettings(step_s=5400.0, startup_step_s=700.0)


def test_order_13_pecec_matches_exponential_solution():
    class ExponentialDynamics:
        @staticmethod
        def prepare_epoch(epoch):
            return None

        @staticmethod
        def derivatives(epoch, state, prepared=None, *, history=None):
            return state

    settings = IntegratorSettings(step_s=0.1, startup_step_s=0.0125)
    trajectory = AdamsBashforthMoultonIntegrator(settings).integrate(ExponentialDynamics(), EPOCH, np.array([1.0]), 2.0)
    np.testing.assert_allclose(trajectory.states[0, -1], np.exp(2.0), rtol=0.0, atol=2e-13)


def test_multichannel_acceleration_shares_one_prepared_epoch():
    dynamics = make_dynamics()
    physical = EarthMoonBarycentricState(
        EPOCH,
        BodyState([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        BodyState([3.844e8, 0.0, 0.0], [0.0, 1022.0, 0.0]),
    )
    vector = physical.to_integration(EARTH_MU, MOON_MU).to_vector()
    prepared = dynamics.prepare_epoch(EPOCH)
    batch = dynamics.accelerations_batch(EPOCH, np.vstack((vector, vector)), prepared)
    assert batch.accelerations_mps2.shape == (2, 2, 3)
    np.testing.assert_array_equal(batch.accelerations_mps2[0], batch.accelerations_mps2[1])


def test_derivatives_fast_path_does_not_build_diagnostic_terms():
    dynamics = make_dynamics()
    vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.844e8, 0.0, 0.0, 0.0, 1022.0, 0.0])
    derivative = dynamics.derivatives(EPOCH, vector, dynamics.prepare_epoch(EPOCH))
    assert derivative.shape == (12,)


def test_prepared_external_newtonian_system_matches_full_direct_sum():
    positions = np.array([[1.0, 2.0, 3.0], [5.0, -2.0, 1.0], [20.0, 4.0, -8.0], [-7.0, 11.0, 13.0]])
    gm = np.array([2.0, 3.0, 5.0, 7.0])
    direct = evaluate_point_mass_system(positions, gm)
    prepared = prepare_point_mass_system(positions, gm, integrated_count=2)
    cached = evaluate_point_mass_system(positions, gm, prepared)
    np.testing.assert_allclose(cached.accelerations_mps2, direct.accelerations_mps2, rtol=2e-15)
    np.testing.assert_allclose(cached.potentials_m2_s2, direct.potentials_m2_s2, rtol=2e-15)


def test_eih_target_rows_use_complete_newtonian_auxiliary_system():
    positions = np.array(
        [[1.0e9, 2.0e9, 3.0e9], [5.0e9, -2.0e9, 1.0e9], [2.0e10, 4.0e9, -8.0e9], [-7.0e9, 1.1e10, 1.3e10]]
    )
    velocities = np.array([[10.0, 20.0, 30.0], [50.0, -20.0, 10.0], [3.0, 4.0, -8.0], [-7.0, 11.0, 13.0]])
    gm = np.array([2.0e14, 3.0e12, 5.0e16, 7.0e10])
    newtonian = evaluate_point_mass_system(positions, gm)
    targets_only = eih_acceleration(positions, velocities, gm, newtonian, [0, 1])
    all_targets = eih_acceleration(positions, velocities, gm, newtonian, np.arange(4))
    np.testing.assert_allclose(targets_only[:2], all_targets[:2], rtol=1e-15)
    np.testing.assert_array_equal(targets_only[2:], 0.0)


def test_earth_tide_delayed_geometry_uses_current_fixed_frame_basis():
    earth = np.array([1.2e9, -2.5e9, 0.7e9])
    moon_relative = np.array([3.70e8, 8.0e7, 4.0e7])
    sun_relative = np.array([1.2e11, -7.0e10, 2.5e10])

    def fixed_inertial_history(names, epoch):
        positions = {
            "EARTH": earth,
            "MOON": earth + moon_relative,
            "SUN": earth + sun_relative,
        }
        return np.array([np.r_[positions[name], np.zeros(3)] for name in names])

    current_angle = 0.63
    current_frame = _rotate_z(current_angle)

    def changing_frame(epoch):
        elapsed = EPOCH.seconds_until(epoch)
        return _rotate_z(current_angle + 7.29211514670698e-5 * elapsed)

    parameters = EarthTideParameters(tau_rot_days=(0.0, 0.0, 0.0))
    raiser_gm = {"MOON": MOON_MU, "SUN": 1.3271244004193938e20}
    changing = EarthTideModel(
        EARTH_MU,
        MOON_MU,
        fixed_inertial_history,
        changing_frame,
        parameters=parameters,
        tide_raiser_gm=raiser_gm,
    )
    frozen = EarthTideModel(
        EARTH_MU,
        MOON_MU,
        fixed_inertial_history,
        lambda epoch: current_frame,
        parameters=parameters,
        tide_raiser_gm=raiser_gm,
    )

    changing_result = changing.relative_acceleration(EPOCH, earth, earth + moon_relative)
    frozen_result = frozen.relative_acceleration(EPOCH, earth, earth + moon_relative)
    np.testing.assert_allclose(changing_result, frozen_result, rtol=2e-15, atol=1e-30)


def test_gravity_field_batch_matches_scalar_evaluation():
    pytest.importorskip("pyshtools")
    cosine = np.zeros((4, 4))
    sine = np.zeros_like(cosine)
    cosine[0, 0] = 1.0
    cosine[2, 0] = -2.0e-4
    sine[2, 1] = 3.0e-4
    coefficients = make_gravity_coefficients(cosine, sine, gm_m3_s2=3.986e14, radius_m=6.378e6, name="test")
    field = GravityField(coefficients)
    positions = np.array(
        [
            [7.0e6, 2.0e6, 1.0e6],
            [-4.0e7, 3.0e7, 2.0e7],
            [2.0e8, -5.0e7, 4.0e7],
        ]
    )
    batch = field.figure_accelerations(positions)
    scalar = np.vstack([field.figure_acceleration(position) for position in positions])
    np.testing.assert_allclose(batch, scalar, rtol=1e-13, atol=1e-18)


def test_de440_earth_pole_and_frame_at_j2000():
    pole = de440_earth_pole_inertial(EPOCH)
    frame = de440_earth_dynamics_frame(EPOCH)
    np.testing.assert_allclose(
        pole,
        [-2.7071057021861346e-5, -2.5593428847018424e-5, 0.9999999993060672],
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(frame[2], pole, rtol=0.0, atol=2e-15)


def test_figure_partners_are_independent_of_acceleration_target_mask():
    pytest.importorskip("pyshtools")
    cosine = np.zeros((3, 3))
    sine = np.zeros_like(cosine)
    cosine[0, 0] = 1.0
    cosine[2, 0] = -2.0e-4
    field = GravityField(make_gravity_coefficients(cosine, sine, gm_m3_s2=EARTH_MU, radius_m=6.378e6, name="Earth"))
    force = FigureForce({"EARTH": field}, {"EARTH": ("MOON", "SUN")}, name="figure_earth")
    positions = np.array(((0.0, 0.0, 0.0), (3.844e8, 0.0, 0.0), (1.5e11, 0.0, 0.0)))
    gm = (EARTH_MU, MOON_MU, 1.3271244004193938e20)
    context = DynamicsContext(("EARTH", "MOON", "SUN"), gm)
    context.bind_prepared(
        PreparedEpoch(EPOCH, context.body_names, positions, np.zeros_like(positions), {"EARTH": np.eye(3)})
    )
    acceleration = force.acceleration(context, target_mask=np.array((True, True, False)))
    moon_direct, sun_direct = field.figure_accelerations(positions[1:])
    expected_earth = -(MOON_MU * moon_direct + gm[2] * sun_direct) / EARTH_MU
    np.testing.assert_allclose(acceleration[0], expected_earth, rtol=2e-15)
    np.testing.assert_allclose(acceleration[1], moon_direct, rtol=2e-15)
    np.testing.assert_array_equal(acceleration[2], 0.0)


def test_earth_tide_rejects_one_sided_force_configuration():
    group = ForceGroup(
        (EarthTideForce(None),),
        {"EARTH": ("tide",), "MOON": ()},
    )
    with pytest.raises(ValueError, match="enabled for EARTH and MOON together"):
        group.bind(("EARTH", "MOON"))


def test_lunar_inertia_coefficients_are_adapted_to_gravity_field():
    pytest.importorskip("pyshtools")
    cosine = np.zeros((3, 3))
    sine = np.zeros_like(cosine)
    cosine[0, 0] = 1.0
    cosine[2, 0] = -2.0e-4
    coefficients = make_gravity_coefficients(
        cosine,
        sine,
        gm_m3_s2=MOON_MU,
        radius_m=1.738e6,
        name="dynamic Moon",
    )

    class FixedInertia:
        @staticmethod
        def field(epoch, static_coefficients, *, history=None):
            return static_coefficients.copy(), {}

    force = LunarInertiaFigureForce(
        {"MOON": GravityField(coefficients)},
        {"MOON": ("EARTH",)},
        inertia=FixedInertia(),
    )
    context = DynamicsContext(("EARTH", "MOON"), (EARTH_MU, MOON_MU))
    context.bind_prepared(
        PreparedEpoch(
            EPOCH,
            ("EARTH", "MOON"),
            np.array(((0.0, 0.0, 0.0), (3.844e8, 0.0, 0.0))),
            np.zeros((2, 3)),
            {"MOON": np.eye(3)},
        )
    )

    acceleration = force.acceleration(context, target_mask=np.ones(2, dtype=bool))
    assert acceleration.shape == (2, 3)
    assert np.all(np.isfinite(acceleration))


def test_lunar_inertia_deformation_switches_isolate_tide_and_spin():
    pytest.importorskip("pyshtools")
    earth = np.array([0.0, 0.0, 0.0])
    moon = np.array([3.844e8, 0.0, 0.0])

    def history(names, epoch):
        states = {"EARTH": np.r_[earth, np.zeros(3)], "MOON": np.r_[moon, np.zeros(3)]}
        return np.array([states[name] for name in names])

    def attitude(epoch):
        angle = 2.7e-6 * EPOCH.seconds_until(epoch)
        return _rotate_z(angle)

    common = dict(delay_days=0.0, attitude_difference_step_s=10.0)
    full = LunarInertiaModel(
        EARTH_MU,
        MOON_MU,
        history,
        attitude,
        parameters=LunarInertiaParameters(**common),
    ).evaluate(EPOCH)
    undistorted = LunarInertiaModel(
        EARTH_MU,
        MOON_MU,
        history,
        attitude,
        parameters=LunarInertiaParameters(
            **common,
            include_tidal_deformation=False,
            include_rotational_deformation=False,
        ),
    ).evaluate(EPOCH)
    tide_only = LunarInertiaModel(
        EARTH_MU,
        MOON_MU,
        history,
        attitude,
        parameters=LunarInertiaParameters(**common, include_rotational_deformation=False),
    ).evaluate(EPOCH)
    spin_only = LunarInertiaModel(
        EARTH_MU,
        MOON_MU,
        history,
        attitude,
        parameters=LunarInertiaParameters(**common, include_tidal_deformation=False),
    ).evaluate(EPOCH)

    np.testing.assert_array_equal(undistorted["tidal_inertia"], 0.0)
    np.testing.assert_array_equal(undistorted["rotational_inertia"], 0.0)
    np.testing.assert_array_equal(tide_only["rotational_inertia"], 0.0)
    np.testing.assert_array_equal(spin_only["tidal_inertia"], 0.0)
    np.testing.assert_allclose(
        full["normalized_inertia"],
        tide_only["normalized_inertia"]
        + spin_only["normalized_inertia"]
        - undistorted["normalized_inertia"],
        rtol=0.0,
        atol=1e-16,
    )
