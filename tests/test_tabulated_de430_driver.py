from __future__ import annotations

import numpy as np

from lunarops.classes.ephemerides import (
    BodyState,
    FixedLunarOrientation,
    LunarRelativisticScale,
    TabulatedDe430Driver,
)
from lunarops.classes.time import Epoch, TimeScale

J2000 = Epoch(2_451_545.0, 0.0, TimeScale.TDB)


class _Base:
    source_path = None
    relativistic_scale = LunarRelativisticScale.from_convention("alreadyScaled")
    lunar_orientation = FixedLunarOrientation(lambda _: np.eye(3))

    def body_state_bcrs(self, body, epoch_tdb):
        return BodyState(np.full(3, 100.0), np.full(3, 10.0))

    def body_position_bcrs(self, body, epoch_tdb):
        return self.body_state_bcrs(body, epoch_tdb).position_m.copy()

    def body_acceleration_bcrs(self, body, epoch_tdb):
        return np.zeros(3)

    def close(self):
        return None


def _write_driver(tmp_path):
    prefix = tmp_path / "driver"
    (tmp_path / "driver_metadata.txt").write_text(
        "version=1\ninitial_tdb_jd=2451545.0\nstart_seconds=0\nend_seconds=10\n"
        "state_step_seconds=10\nstate_count=2\nbody_count=1\n"
        "frame_step_seconds=10\nframe_count=2\n",
        encoding="ascii",
    )
    (tmp_path / "driver_body_ids.txt").write_text("10\n", encoding="ascii")
    states = np.zeros((2, 1, 6), dtype="<f8")
    states[0, 0] = (0, 0, 0, 1, 0, 0)
    states[1, 0] = (10, 0, 0, 1, 0, 0)
    states.tofile(tmp_path / "driver_states.bin")
    frames = np.empty((2, 2, 3, 3), dtype="<f8")
    frames[0] = np.eye(3)
    frames[1, 1] = np.eye(3)
    angle = np.pi / 2
    frames[1, 0] = ((np.cos(angle), np.sin(angle), 0), (-np.sin(angle), np.cos(angle), 0), (0, 0, 1))
    frames.tofile(tmp_path / "driver_frames.bin")
    return prefix


def test_driver_interpolates_state_and_orientation(tmp_path):
    driver = TabulatedDe430Driver(
        _Base(), _write_driver(tmp_path), bodies=("SUN",), replace_lunar_orientation=True
    )
    state = driver.body_state_bcrs("SUN", J2000.shifted(5.0))
    assert np.allclose(state.position_m, (5, 0, 0), rtol=0, atol=1e-14)
    assert np.allclose(state.velocity_mps, (1, 0, 0), rtol=0, atol=1e-14)
    frame = driver.lunar_orientation.pa_to_lcrs_matrix(J2000.shifted(5.0))
    assert np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-14)
    assert np.isclose(np.linalg.det(frame), 1.0, rtol=0, atol=1e-14)


def test_driver_delegates_unselected_body(tmp_path):
    driver = TabulatedDe430Driver(_Base(), _write_driver(tmp_path))
    state = driver.body_state_bcrs("SUN", J2000.shifted(5.0))
    assert np.array_equal(state.position_m, np.full(3, 100.0))
