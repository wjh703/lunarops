"""Tabulated DE430 replacements for controlled hybrid-driver experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from lunarops.base.array_validation import readonly_matrix3x3
from lunarops.classes.ephemerides.body_ids import BODY_BY_NAIF_ID, body_name
from lunarops.classes.ephemerides.ephemeris_provider import BcrsAccelerationProvider, BodyState, Ephemeris
from lunarops.classes.ephemerides.lunar_orientation import (
    FixedLunarOrientation,
    LunarOrientationProvider,
)
from lunarops.classes.relativistic.lunar_scale import LunarRelativisticScale
from lunarops.classes.time import Epoch, TimeScale, require_tdb_epoch


@runtime_checkable
class De430DriverSource(Ephemeris, BcrsAccelerationProvider, Protocol):
    @property
    def lunar_orientation(self) -> LunarOrientationProvider: ...

    @property
    def relativistic_scale(self) -> LunarRelativisticScale: ...


class TabulatedDe430Driver:
    """Replace selected CALCEPH states or frame histories with saved DE430 samples."""

    def __init__(
        self,
        base: De430DriverSource,
        prefix: str | Path,
        *,
        bodies=(),
        replace_external_bodies: bool = False,
        replace_lunar_orientation: bool = False,
    ) -> None:
        if not isinstance(base, De430DriverSource):
            raise TypeError("base must provide BCRS states, accelerations, lunar orientation, and relativistic scale.")
        self.base = base
        self.prefix = Path(prefix).expanduser().resolve()
        metadata = self._read_metadata(Path(f"{self.prefix}_metadata.txt"))
        if metadata.get("version") != "1":
            raise ValueError("Unsupported tabulated DE430 driver version.")
        self._initial_epoch = Epoch(float(metadata["initial_tdb_jd"]), 0.0, TimeScale.TDB)
        self._start_s = float(metadata["start_seconds"])
        self._end_s = float(metadata["end_seconds"])
        self._state_step_s = float(metadata["state_step_seconds"])
        self._state_count = int(metadata["state_count"])
        self._frame_step_s = float(metadata["frame_step_seconds"])
        self._frame_count = int(metadata["frame_count"])
        body_ids = Path(f"{self.prefix}_body_ids.txt").read_text(encoding="ascii").splitlines()
        if len(body_ids) != int(metadata["body_count"]):
            raise ValueError("Tabulated DE430 body ID count does not match metadata.")
        names = tuple(str(BODY_BY_NAIF_ID.get(int(value), f"NAIF:{int(value)}")) for value in body_ids)
        self._body_index = {name: index for index, name in enumerate(names)}
        selected = frozenset(body_name(value) for value in bodies)
        self._bodies = frozenset(set(names) - {"EARTH", "MOON"}) if replace_external_bodies else selected
        missing = self._bodies - self._body_index.keys()
        if missing:
            raise KeyError(f"Tabulated DE430 driver has no bodies: {sorted(missing)}")
        self._states = np.memmap(
            f"{self.prefix}_states.bin", dtype="<f8", mode="r", shape=(self._state_count, len(names), 6)
        )
        self._frames = np.memmap(
            f"{self.prefix}_frames.bin", dtype="<f8", mode="r", shape=(self._frame_count, 2, 3, 3)
        )
        base_orientation = getattr(base, "lunar_orientation", None)
        if not isinstance(base_orientation, LunarOrientationProvider):
            raise TypeError("Base provider must expose a LunarOrientationProvider.")
        self._lunar_orientation = (
            FixedLunarOrientation(
                lambda epoch: readonly_matrix3x3(self._frame(0, epoch).T, name="pa_to_lcrs_matrix"),
                correction=base_orientation.correction,
                correction_at_epoch=base_orientation.longitude_libration_correction_rad,
            )
            if replace_lunar_orientation
            else base_orientation
        )

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, str]:
        metadata = {}
        for line in path.read_text(encoding="ascii").splitlines():
            key, value = line.split("=", 1)
            metadata[key] = value
        return metadata

    @property
    def source_path(self) -> Path:
        return self.prefix.parent

    @property
    def relativistic_scale(self) -> LunarRelativisticScale:
        return self.base.relativistic_scale

    @property
    def lunar_orientation(self) -> LunarOrientationProvider:
        return self._lunar_orientation

    def _offset_seconds(self, epoch_tdb: Epoch) -> float:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        offset = float(self._initial_epoch.seconds_until(epoch))
        if offset < self._start_s - 1e-6 or offset > self._end_s + 1e-6:
            raise ValueError(f"Epoch offset {offset} s is outside tabulated DE430 coverage.")
        return min(self._end_s, max(self._start_s, offset))

    @staticmethod
    def _interval(offset: float, start: float, step: float, count: int) -> tuple[int, float]:
        coordinate = (offset - start) / step
        left = min(max(int(np.floor(coordinate)), 0), count - 2)
        return left, coordinate - left

    def _state_vector(self, name: str, offset: float) -> np.ndarray:
        left, fraction = self._interval(offset, self._start_s, self._state_step_s, self._state_count)
        if abs(fraction) < 1e-13:
            return np.array(self._states[left, self._body_index[name]], copy=True)
        if abs(fraction - 1.0) < 1e-13:
            return np.array(self._states[left + 1, self._body_index[name]], copy=True)
        first = self._states[left, self._body_index[name]]
        second = self._states[left + 1, self._body_index[name]]
        step = self._state_step_s
        u2, u3 = fraction**2, fraction**3
        position = (
            (2 * u3 - 3 * u2 + 1) * first[:3]
            + (u3 - 2 * u2 + fraction) * step * first[3:]
            + (-2 * u3 + 3 * u2) * second[:3]
            + (u3 - u2) * step * second[3:]
        )
        velocity = (
            (6 * u2 - 6 * fraction) * first[:3] / step
            + (3 * u2 - 4 * fraction + 1) * first[3:]
            + (-6 * u2 + 6 * fraction) * second[:3] / step
            + (3 * u2 - 2 * fraction) * second[3:]
        )
        return np.concatenate((position, velocity))

    def body_state_bcrs(self, body: str, epoch_tdb: Epoch) -> BodyState:
        name = body_name(body)
        if name not in self._bodies:
            return self.base.body_state_bcrs(name, epoch_tdb)
        state = self._state_vector(name, self._offset_seconds(epoch_tdb))
        return BodyState(state[:3], state[3:])

    def body_position_bcrs(self, body: str, epoch_tdb: Epoch) -> np.ndarray:
        return np.array(self.body_state_bcrs(body, epoch_tdb).position_m, copy=True)

    def body_acceleration_bcrs(self, body: str, epoch_tdb: Epoch) -> np.ndarray:
        name = body_name(body)
        if name not in self._bodies:
            return self.base.body_acceleration_bcrs(name, epoch_tdb)
        offset = self._offset_seconds(epoch_tdb)
        left, fraction = self._interval(offset, self._start_s, self._state_step_s, self._state_count)
        first = self._states[left, self._body_index[name]]
        second = self._states[left + 1, self._body_index[name]]
        step = self._state_step_s
        acceleration = (
            (12 * fraction - 6) * first[:3] / step**2
            + (6 * fraction - 4) * first[3:] / step
            + (-12 * fraction + 6) * second[:3] / step**2
            + (6 * fraction - 2) * second[3:] / step
        )
        return np.asarray(acceleration, dtype=float)

    def _frame(self, frame_index: int, epoch_tdb: Epoch) -> np.ndarray:
        offset = self._offset_seconds(epoch_tdb)
        left, fraction = self._interval(offset, self._start_s, self._frame_step_s, self._frame_count)
        if abs(fraction) < 1e-13:
            return np.array(self._frames[left, frame_index], copy=True)
        rotations = Rotation.from_matrix(np.asarray(self._frames[left : left + 2, frame_index]))
        return Slerp((0.0, 1.0), rotations)((fraction,)).as_matrix()[0]

    def earth_fixed_to_inertial_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        return readonly_matrix3x3(self._frame(1, epoch_tdb).T, name="earth_fixed_to_inertial_matrix")

    def close(self) -> None:
        self.base.close()


__all__ = ["De430DriverSource", "TabulatedDe430Driver"]
