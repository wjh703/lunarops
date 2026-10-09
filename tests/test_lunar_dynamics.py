import numpy as np
import pytest

from lunarops.classes.dynamics import (
    AdamsBashforthMoultonIntegrator,
    BarycentricEarthMoonState,
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
from lunarops.classes.ephemerides import BodyState, Ephemeris
from lunarops.classes.time import Epoch, TimeScale

EPOCH = Epoch(2451545.0, 0.0, TimeScale.TDB)
EARTH_MU = 3.98600435507e14
MOON_MU = 4.902800118e12


class Provider(Ephemeris):
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
    bodies = (("EARTH", EARTH_MU), ("MOON", MOON_MU))
    return LunarDynamics(
        earth_body=GravitatingBody("EARTH", EARTH_MU),
        moon_body=GravitatingBody("MOON", MOON_MU),
        ephemeris=Provider(),
        force_group=LunarForceGroup((NewtonianPointMassForce(bodies),), ("point_mass",)),
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
    trajectory = AdamsBashforthMoultonIntegrator(
        IntegratorSettings(step_s=900.0, order=2)
    ).integrate(dynamics, EPOCH, vector, 2700.0)
    assert trajectory.vector(EPOCH.shifted(1800.0)).shape == (6,)


def test_moon_target_mask_excludes_prescribed_earth():
    dynamics = make_dynamics()
    epoch_data = dynamics.build_epoch_data(EPOCH)
    breakdown = dynamics.accelerations(EPOCH, np.array([3.844e8, 0.0, 0.0, 0.0, 1022.0, 0.0]), epoch_data)
    np.testing.assert_array_equal(breakdown.total_accelerations_mps2[0], 0.0)
    assert set(breakdown.accelerations_by_force_mps2) == {"point_mass"}


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
        cosine, sine, gm_m3_s2=EARTH_MU if source == "EARTH" else MOON_MU,
        radius_m=1e6, name="rotation test",
    )
    force = FigureForce({source: field}, {source: ("MOON" if source == "EARTH" else "EARTH",)})

    def system(**rotation_providers):
        return LunarDynamics(
            earth_body=GravitatingBody("EARTH", EARTH_MU),
            moon_body=GravitatingBody("MOON", MOON_MU),
            ephemeris=Provider(), force_group=LunarForceGroup((force,)), **rotation_providers,
        )

    position = np.array([3.5e8, 0.7e8, -0.4e8])
    aligned = system().accelerations(EPOCH, np.concatenate((position, np.zeros(3))))
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
        actual.total_accelerations_mps2, aligned.total_accelerations_mps2 @ rotation.T, rtol=1e-12, atol=1e-25,
    )
