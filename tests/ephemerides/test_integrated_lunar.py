"""IntegratedLunarEphemeris: propagated Moon-Earth state and orientation about the wrapped barycentre."""

import numpy as np
import pytest

from lunarops.classes.dynamics.rotation import euler_matrix
from lunarops.classes.ephemerides import IntegratedLunarEphemeris
from lunarops.classes.frames.coordinate_scaling import LunarRelativisticScale
from lunarops.classes.time import Epoch, TimeScale
from lunarops.fileio.numeric_table import write_numeric_table
from tests.support.synthetic import LinearEphemeris

EPOCH = Epoch(2451545.0, 0.25, TimeScale.TDB)
EARTH_GM, MOON_GM = 3.98600435507e14, 4.902800118e12
STEP = 3600.0
RATE = 2.66e-6


def _state(offset):
    """A smooth Moon-like orbit and rotation: (r, v, euler angles, mantle angular velocity)."""
    angle = RATE * offset
    radius = 3.84e8 * (1 + 0.05 * np.cos(0.9 * angle))
    radius_dot = -3.84e8 * 0.05 * 0.9 * RATE * np.sin(0.9 * angle)
    unit = np.array([np.cos(angle), np.sin(angle) * np.cos(0.4), np.sin(angle) * np.sin(0.4)])
    unit_dot = RATE * np.array([-np.sin(angle), np.cos(angle) * np.cos(0.4), np.cos(angle) * np.sin(0.4)])
    phi, theta, psi = 0.1 + 1e-9 * offset, 0.4 + 1e-3 * np.sin(angle), 1.2 + angle
    phi_dot, theta_dot, psi_dot = 1e-9, 1e-3 * RATE * np.cos(angle), RATE
    omega = [
        phi_dot * np.sin(theta) * np.sin(psi) + theta_dot * np.cos(psi),
        phi_dot * np.sin(theta) * np.cos(psi) - theta_dot * np.sin(psi),
        phi_dot * np.cos(theta) + psi_dot,
    ]
    return np.r_[radius * unit, radius_dot * unit + radius * unit_dot, phi, theta, psi, omega]


def _inner():
    ephemeris = LinearEphemeris(
        {
            "EARTH MOON BARYCENTER": ([1.4e11, 2.0e10, -3.0e9], [-4.0e3, 2.9e4, 1.0e2]),
            "SUN": ([1.0e8, -2.0e8, 3.0e7], [1.0, 2.0, -1.0]),
            # Deliberately not about the barycentre: the wrapper must not read them.
            "EARTH": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "MOON": ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        }
    )
    ephemeris.relativistic_scale = LunarRelativisticScale("alreadyScaled", 0.0)
    return ephemeris


def _tables(tmp_path, *, count=200, rotation_epoch=EPOCH, rotation_step=STEP):
    metadata = {"initialTdbJd1": EPOCH.jd1, "initialTdbJd2": EPOCH.jd2}
    offsets = STEP * np.arange(-20, count - 20, dtype=float)
    states = np.asarray([_state(offset) for offset in offsets])
    orbit = ("offset_tdb_s", *(f"relative_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")))
    write_numeric_table(
        tmp_path / "orbit.dat", "lunarOrbit", orbit, np.column_stack((offsets, states[:, :6])), metadata=metadata
    )
    rotation = (
        "offset_tdb_s",
        *(f"euler_{angle}_rad" for angle in ("phi", "theta", "psi")),
        *(f"mantle_omega_pa_{axis}_rad_s" for axis in "xyz"),
        "extra",
    )
    rotation_offsets = rotation_step * np.arange(-20, count - 20, dtype=float)
    rotation_states = np.asarray([_state(offset) for offset in rotation_offsets])
    write_numeric_table(
        tmp_path / "rotation.dat",
        "lunarRotation",
        rotation,
        np.column_stack((rotation_offsets, rotation_states[:, 6:], np.zeros(count))),
        metadata={"initialTdbJd1": rotation_epoch.jd1, "initialTdbJd2": rotation_epoch.jd2},
    )
    return tmp_path / "orbit.dat", tmp_path / "rotation.dat"


def test_the_moon_and_the_earth_follow_the_propagation_about_the_wrapped_barycentre(tmp_path):
    inner = _inner()
    ephemeris = IntegratedLunarEphemeris(inner, *_tables(tmp_path), EARTH_GM, MOON_GM)
    assert ephemeris.initial_epoch == EPOCH
    assert ephemeris.span_s == (-20 * STEP, 179 * STEP)
    assert ephemeris.source_path == inner.source_path and ephemeris.relativistic_scale is inner.relativistic_scale
    for offset in (-19.5 * STEP, 1234.5, 40 * STEP, 150.25 * STEP):
        epoch = EPOCH.shifted(offset)
        truth = _state(offset)
        relative = ephemeris.relative_state_matrix("MOON", "EARTH", epoch)
        np.testing.assert_allclose(relative[:3], truth[:3], rtol=0, atol=1e-6)
        np.testing.assert_allclose(relative[3:], truth[3:6], rtol=0, atol=1e-11)
        np.testing.assert_array_equal(ephemeris.relative_state_matrix("EARTH", "MOON", epoch), -relative)
        states = ephemeris.body_state_matrix_bcrs(("MOON", "SUN", "EARTH"), epoch)
        barycentre, sun = inner.body_state_matrix_bcrs(("EARTH MOON BARYCENTER", "SUN"), epoch)
        np.testing.assert_array_equal(states[1], sun)
        np.testing.assert_allclose(
            (EARTH_GM * states[2] + MOON_GM * states[0]) / (EARTH_GM + MOON_GM), barycentre, rtol=1e-15
        )
        np.testing.assert_allclose(states[0] - states[2], relative, rtol=0, atol=1e-4)
        np.testing.assert_array_equal(ephemeris.body_state_bcrs("MOON", epoch).position_m, states[0, :3])
        np.testing.assert_array_equal(ephemeris.body_position_bcrs("EARTH", epoch), states[2, :3])
        many = ephemeris.body_state_matrices_bcrs(("EARTH", "MOON"), (epoch, epoch))
        np.testing.assert_array_equal(many[1], states[[2, 0]])
    # A table node is returned unchanged.
    np.testing.assert_array_equal(
        ephemeris.relative_state_matrix("MOON", "EARTH", EPOCH.shifted(7 * STEP)), _state(7 * STEP)[:6]
    )
    np.testing.assert_array_equal(
        ephemeris.body_state_matrix_bcrs(("SUN",), EPOCH), inner.body_state_matrix_bcrs(("SUN",), EPOCH)
    )


def test_the_orientation_is_the_propagated_euler_rotation(tmp_path):
    ephemeris = IntegratedLunarEphemeris(_inner(), *_tables(tmp_path), EARTH_GM, MOON_GM)
    orientation = ephemeris.lunar_orientation
    for offset in (100.0, 33.3 * STEP):
        epoch = EPOCH.shifted(offset)
        truth = _state(offset)
        matrix = orientation.pa_to_lcrs_matrix(epoch)
        np.testing.assert_allclose(matrix, euler_matrix(truth[6:9]), rtol=0, atol=1e-13)
        np.testing.assert_allclose(orientation.angular_velocity_rad_s(epoch), truth[9:12], rtol=0, atol=1e-19)
        np.testing.assert_array_equal(orientation.pa_to_lcrs_matrices((epoch,))[0], matrix)
        assert orientation.longitude_libration_correction_rad(epoch) == 0.0


def test_epochs_outside_the_tables_and_inconsistent_tables_are_rejected(tmp_path):
    ephemeris = IntegratedLunarEphemeris(_inner(), *_tables(tmp_path), EARTH_GM, MOON_GM)
    with pytest.raises(ValueError, match="outside the lunarOrbit table"):
        ephemeris.relative_state_matrix("MOON", "EARTH", EPOCH.shifted(-21 * STEP))
    with pytest.raises(ValueError, match="outside the lunarRotation table"):
        ephemeris.lunar_orientation.pa_to_lcrs_matrix(EPOCH.shifted(180 * STEP))
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="different initial epochs"):
        IntegratedLunarEphemeris(_inner(), *_tables(other, rotation_epoch=EPOCH.shifted(1.0)), EARTH_GM, MOON_GM)
    with pytest.raises(ValueError, match="lunarOrbit"):
        IntegratedLunarEphemeris(_inner(), tmp_path / "rotation.dat", tmp_path / "rotation.dat", EARTH_GM, MOON_GM)
