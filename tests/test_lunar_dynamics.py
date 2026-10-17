import numpy as np
import pytest

from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    BarycentricEarthMoonState,
    EarthTideForce,
    EarthTideModel,
    EarthTideParameters,
    FigureForce,
    GravitatingBody,
    IntegratorSettings,
    LunarDynamics,
    LunarForceGroup,
    MoonRelativeState,
    NewtonianPointMassForce,
    OrientationProvider,
    make_gravity_coefficients,
)
from lunarops.classes.dynamics.gravity import GravityField
from lunarops.classes.dynamics.history import LunarNodeBuffer
from lunarops.classes.ephemerides import BodyState, Ephemeris
from lunarops.classes.time import Epoch, TimeScale

EPOCH = Epoch(2451545.0, 0.0, TimeScale.TDB)
EARTH_MU = 3.98600435507e14
MOON_MU = 4.902800118e12


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
        return BodyState(np.zeros(3), np.zeros(3))

    def body_position_bcrs(self, name, epoch):
        return self.body_state_bcrs(name, epoch).position_m.copy()

    def body_acceleration_bcrs(self, name, epoch):
        return np.zeros(3)

    def close(self):
        return None


def make_dynamics():
    return LunarDynamics(
        perturbing_bodies=(),
        earth_body=GravitatingBody("EARTH", EARTH_MU),
        moon_body=GravitatingBody("MOON", MOON_MU),
        ephemeris=Provider(),
        force_group=LunarForceGroup((NewtonianPointMassForce(),), ("point_mass",)),
    )


def test_relative_state_round_trip():
    physical = BarycentricEarthMoonState(
        EPOCH,
        BodyState([-4.7e6, 0.0, 0.0], [0.0, -0.012, 0.0]),
        BodyState([3.796e8, 0.0, 0.0], [0.0, 1022.0, 0.0]),
    )
    relative = MoonRelativeState.from_barycentric_state(physical)
    recovered = relative.to_barycentric_state(physical.earth)
    np.testing.assert_allclose(recovered.relative_state.position_m, physical.relative_state.position_m)


def test_lunar_rhs_and_integrator_use_six_vector():
    dynamics = make_dynamics()
    vector = np.array([3.844e8, 0.0, 0.0, 0.0, 1022.0, 0.0])
    derivative = dynamics.derivatives(EPOCH, vector)
    assert derivative.shape == (6,)
    trajectory = AdamsBashforthMoultonIntegrator(IntegratorSettings(step_s=900.0, order=2)).integrate(
        dynamics, EPOCH, vector, 2700.0
    )
    assert trajectory.vector(EPOCH.shifted(1800.0)).shape == (6,)


def test_moon_target_mask_excludes_prescribed_earth():
    dynamics = make_dynamics()
    epoch_data = dynamics.build_epoch_data(EPOCH)
    breakdown = dynamics.accelerations(EPOCH, np.array([3.844e8, 0.0, 0.0, 0.0, 1022.0, 0.0]), epoch_data)
    np.testing.assert_array_equal(breakdown.total_accelerations_mps2[0], 0.0)
    assert set(breakdown.accelerations_by_force_mps2) == {"point_mass"}


@pytest.mark.parametrize("step", [0.1, 0.5])
@pytest.mark.parametrize("corrector_iterations", [1, 2])
def test_abm_stores_rhs_at_accepted_orbit_state(step, corrector_iterations):
    class Oscillator:
        rotation = None

        def derivatives(self, epoch, state, epoch_data=None, **kwargs):
            return np.concatenate((state[3:], -state[:3]))

    system = Oscillator()
    nodes = LunarNodeBuffer(2, 6, 1)
    for time in (-step, 0.0):
        state = np.array([np.cos(time), 0, 0, -np.sin(time), 0, 0])
        nodes.append(time, state, system.derivatives(EPOCH, state), np.zeros((1, 6)))
    integrator = AdamsBashforthMoultonIntegrator(
        IntegratorSettings(step_s=step, order=2, corrector_iterations=corrector_iterations)
    )
    accepted, derivative = integrator._compute_abm_step(system, nodes, None, EPOCH.shifted(step), None, step)
    np.testing.assert_array_equal(derivative, system.derivatives(EPOCH.shifted(step), accepted))


@pytest.mark.parametrize("raiser", ["MOON", "SUN"])
def test_earth_tide_with_prescribed_earth_recovers_relative_acceleration(raiser):
    positions = {
        "EARTH": np.zeros(3),
        "MOON": np.array([3.5e8, 0.7e8, -0.4e8]),
        "SUN": np.array([1.2e11, -0.8e11, 0.1e11]),
    }

    def states(names, epoch):
        return np.array([np.concatenate((positions[name], np.zeros(3))) for name in names])

    tide = EarthTideModel(
        EARTH_MU,
        MOON_MU,
        tide_parameters=EarthTideParameters(tau_orb_days=(0, 0, 0), tau_rot_days=(0, 0, 0)),
        tide_raisers=(raiser,),
        tide_raiser_gm={"MOON": MOON_MU, "SUN": 1.3271244e20},
    )
    relative_tide = tide.relative_acceleration(EPOCH, positions["EARTH"], positions["MOON"], np.eye(3), states)
    # Both papers leave the EMB unaffected, including their approximation
    # for Sun-raised tides. This is the Earth's prescribed barycentric share.
    earth_tide = -MOON_MU / (EARTH_MU + MOON_MU) * relative_tide

    class TideProvider(Provider):
        def body_acceleration_bcrs(self, name, epoch):
            return earth_tide

    dynamics = LunarDynamics(
        perturbing_bodies=(),
        earth_body=GravitatingBody("EARTH", EARTH_MU),
        moon_body=GravitatingBody("MOON", MOON_MU),
        ephemeris=TideProvider(),
        force_group=LunarForceGroup((EarthTideForce(tide),)),
    )
    state = np.concatenate((positions["MOON"], np.zeros(3)))
    derivative = dynamics.derivatives(EPOCH, state, history=states)
    np.testing.assert_allclose(derivative[3:], relative_tide, rtol=2e-15, atol=0)
    acceleration = dynamics.accelerations(EPOCH, state, history=states).total_accelerations_mps2
    np.testing.assert_array_equal(acceleration[0], np.zeros(3))
    np.testing.assert_allclose(
        (EARTH_MU * earth_tide + MOON_MU * acceleration[1]) / (EARTH_MU + MOON_MU),
        0,
        atol=2e-30,
    )


@pytest.mark.parametrize("source", ["EARTH", "MOON"])
def test_figure_acceleration_rotates_with_earth_and_lunar_matrices(source):
    # A tilted, nonsymmetric rotation and a non-axisymmetric field expose
    # reversed transforms that identity matrices and pure J2 would hide.
    rotation = np.array([[0.36, -0.8, 0.48], [0.48, 0.6, 0.64], [-0.8, 0.0, 0.6]])
    cosine = np.zeros((3, 3))
    sine = np.zeros_like(cosine)
    cosine[0, 0] = 1.0
    cosine[2, 0] = -1e-4
    cosine[2, 2] = 2e-5
    sine[2, 1] = 3e-5
    field = make_gravity_coefficients(
        cosine,
        sine,
        gm_m3_s2=EARTH_MU if source == "EARTH" else MOON_MU,
        radius_m=1e6,
        name="rotation test",
    )
    force = FigureForce({source: GravityField(field)}, {source: ("MOON" if source == "EARTH" else "EARTH",)})

    def system(**rotation_providers):
        return LunarDynamics(
            perturbing_bodies=(),
        earth_body=GravitatingBody("EARTH", EARTH_MU),
            moon_body=GravitatingBody("MOON", MOON_MU),
            ephemeris=Provider(),
            force_group=LunarForceGroup((force,)),
            **rotation_providers,
        )

    position = np.array([3.5e8, 0.7e8, -0.4e8])
    with pytest.raises(ValueError, match="orientation"):
        system()
    aligned = system(
        earth_fixed2inertial_matrix_provider=OrientationProvider.from_scalar(lambda epoch: np.eye(3)),
        moon_fixed2inertial_matrix_provider=OrientationProvider.from_scalar(lambda epoch: np.eye(3)),
    ).accelerations(EPOCH, np.concatenate((position, np.zeros(3))))
    rotated = system(
        earth_fixed2inertial_matrix_provider=OrientationProvider.from_scalar(lambda epoch: rotation),
        moon_fixed2inertial_matrix_provider=OrientationProvider.from_scalar(lambda epoch: rotation),
    )
    epoch_data = rotated.build_epoch_data(EPOCH)
    np.testing.assert_array_equal(epoch_data.earth_fixed2inertial_matrix, rotation)
    np.testing.assert_array_equal(epoch_data.moon_fixed2inertial_matrix, rotation)
    assert not epoch_data.earth_fixed2inertial_matrix.flags.writeable
    assert not epoch_data.moon_fixed2inertial_matrix.flags.writeable
    actual = rotated.accelerations(EPOCH, np.concatenate((rotation @ position, np.zeros(3))), epoch_data)
    np.testing.assert_allclose(
        actual.total_accelerations_mps2,
        aligned.total_accelerations_mps2 @ rotation.T,
        rtol=1e-12,
        atol=1e-25,
    )
