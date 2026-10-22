"""CALCEPH-backed BCRS body states and lunar orientation."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Self

import numpy as np

from lunarops.classes.ephemerides.body_ids import body_name, naif_id
from lunarops.classes.ephemerides.ephemeris_provider import BodyState
from lunarops.classes.ephemerides.lunar_orientation import (
    CalcephLunarOrientation,
    LongitudeLibrationCorrection,
    normalize_longitude_libration_correction,
)
from lunarops.classes.frames.coordinate_scaling import LunarRelativisticScale, LunarRelativisticScaleConvention
from lunarops.classes.time import Epoch, TimeScale, TimeScaleConverter, require_tdb_epoch

_J2000_TT = Epoch(2_451_545.0, 0.0, TimeScale.TT)


class CalcephEphemeris:
    """Read SPICE position and lunar attitude kernels through CALCEPH."""

    def __init__(
        self,
        kernel_directory: str | Path,
        *,
        lunar_relativistic_scale_convention: LunarRelativisticScaleConvention | str,
        longitude_libration_correction: LongitudeLibrationCorrection | str | None = None,
    ) -> None:
        directory = Path(kernel_directory).expanduser()
        if not directory.is_dir():
            raise ValueError(f"CALCEPH SPICE kernel directory not found: {directory}")
        from lunarops.config.fingerprints import calceph_kernel_paths

        kernels = calceph_kernel_paths(directory)
        position_kernels = tuple(path for path in kernels if path.suffix == ".bsp")
        orientation_kernels = tuple(path for path in kernels if path.suffix == ".bpc")
        if not position_kernels or not orientation_kernels:
            raise ValueError(f"SPICE kernel directory must contain .bsp and .bpc kernels: {directory}")
        try:
            from calcephpy import CalcephBin, Constants
        except (ImportError, OSError) as exc:  # pragma: no cover
            raise ImportError(f"CALCEPH requires a working calcephpy installation: {exc}") from exc

        self._source_path = directory
        self._scale = LunarRelativisticScale.from_convention(lunar_relativistic_scale_convention)
        self._handle = CalcephBin.open([str(path) for path in (*position_kernels, *orientation_kernels)])
        self._lock = RLock()
        self._state_units = Constants.UNIT_KM + Constants.UNIT_SEC + Constants.USE_NAIFID
        self._angle_units = Constants.UNIT_RAD + Constants.UNIT_SEC + Constants.USE_NAIFID
        self._position_targets: frozenset[int] | None = None
        self._target_ids: dict[str, int] = {}
        self._resolved_bodies: dict[tuple, tuple[tuple[str, int], ...]] = {}
        self._state_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        try:
            orientation_target = self._find_orientation_target(Constants)
            correction = normalize_longitude_libration_correction(longitude_libration_correction)
            j2000_tdb = (
                TimeScaleConverter().tt2tdb(_J2000_TT) if correction is LongitudeLibrationCorrection.INPOP21A else None
            )
            self.lunar_orientation = CalcephLunarOrientation(
                self._lunar_angles_rad,
                self._lunar_angles_many_rad,
                correction,
                j2000_tdb,
            )
            self._orientation_target = orientation_target
        except Exception:
            self.close()
            raise

    @property
    def source_path(self) -> Path:
        return self._source_path

    @property
    def relativistic_scale(self) -> LunarRelativisticScale:
        return self._scale

    def _require_handle(self):
        if self._handle is None:
            raise RuntimeError("CALCEPH provider is closed.")
        return self._handle

    def _find_orientation_target(self, constants) -> int:
        handle = self._require_handle()
        if handle.gettimescale() != constants.TDB:
            raise ValueError("CALCEPH SPICE kernels must use TDB.")
        records = [handle.getorientrecordindex2(index) for index in range(1, int(handle.getorientrecordcount()) + 1)]
        targets = {int(record[0]) for record in records}
        if len(targets) != 1:
            raise ValueError(f"Expected one lunar orientation target, found {sorted(targets)}.")
        frames = {int(record[3]) for record in records}
        if frames != {1}:
            raise ValueError(f"Lunar orientation must be relative to ICRF frame 1, found {sorted(frames)}.")
        return targets.pop()

    def _position_record_targets(self) -> frozenset[int]:
        if self._position_targets is None:
            handle = self._require_handle()
            count = int(handle.getpositionrecordcount())
            self._position_targets = frozenset(
                int(handle.getpositionrecordindex2(index)[0]) for index in range(1, count + 1)
            )
        return self._position_targets

    def _resolve_target(self, body: str) -> tuple[str, int]:
        name = body_name(body)
        cached_target = self._target_ids.get(name)
        if cached_target is not None:
            return name, cached_target
        target_id = naif_id(name)
        if target_id is None:
            target_id = self._require_handle().getidbyname(name, self._state_units)
            if target_id is None:
                raise KeyError(f"Unknown CALCEPH body name: {body!r}")
            target_id = int(target_id)
        if target_id != 0:
            available = self._position_record_targets()
            if target_id not in available:
                barycenter = naif_id(f"{name} BARYCENTER")
                hint = f"; try {name} BARYCENTER" if barycenter in available else ""
                raise KeyError(f"No CALCEPH position record for {name!r} (NAIF {target_id}){hint}.")
        self._target_ids[name] = target_id
        return name, target_id

    def body_state_bcrs(self, body: str, epoch_tdb: Epoch) -> BodyState:
        values = self.body_state_matrix_bcrs((body,), epoch_tdb)[0]
        return BodyState(values[:3], values[3:])

    def body_state_matrix_bcrs(self, bodies, epoch_tdb: Epoch) -> np.ndarray:
        names = tuple(body_name(body) for body in bodies)
        with self._lock:
            missing = tuple(name for name in names if (name, epoch_tdb.jd1, epoch_tdb.jd2) not in self._state_cache)
            if missing:
                values = self.body_state_matrices_bcrs(missing, (epoch_tdb,))[0]
                for name, value in zip(missing, values, strict=True):
                    self._state_cache[(name, epoch_tdb.jd1, epoch_tdb.jd2)] = value
            result = np.asarray([self._state_cache[(name, epoch_tdb.jd1, epoch_tdb.jd2)] for name in names]).reshape(len(names), 6)
            for name in names:
                self._state_cache.move_to_end((name, epoch_tdb.jd1, epoch_tdb.jd2))
            while len(self._state_cache) > 512:
                self._state_cache.popitem(last=False)
        return result

    def body_state_matrices_bcrs(self, bodies, epochs) -> np.ndarray:
        """Read all requested bodies at an epoch array in one CALCEPH batch."""
        epoch_array = tuple(require_tdb_epoch(epoch, name="epoch_tdb") for epoch in epochs)
        bodies = tuple(bodies)
        result = np.empty((len(epoch_array), len(bodies), 6), dtype=float)
        if not epoch_array or not bodies:
            return result
        jd1 = np.asarray([epoch.jd1 for epoch in epoch_array], dtype=float)
        jd2 = np.asarray([epoch.jd2 for epoch in epoch_array], dtype=float)
        with self._lock:
            handle = self._require_handle()
            resolved = self._resolved_bodies.get(bodies)
            if resolved is None:
                resolved = tuple(self._resolve_target(body) for body in bodies)
                self._resolved_bodies[bodies] = resolved
            names = tuple(name for name, _ in resolved)
            targets = tuple(target for _, target in resolved)
            for body_index, (name, target_id) in enumerate(zip(names, targets, strict=True)):
                values = np.asarray(
                    handle.compute_unit(jd1, jd2, target_id, 0, self._state_units),
                    dtype=float,
                )
                if values.shape != (6, len(epoch_array)):
                    raise RuntimeError(
                        f"CALCEPH returned {values.shape} states for {name}; expected (6, {len(epoch_array)})."
                    )
                result[:, body_index, :] = values[:6].T * 1000.0
        return result

    def body_position_bcrs(self, body: str, epoch_tdb: Epoch) -> np.ndarray:
        return np.array(self.body_state_bcrs(body, epoch_tdb).position_m, copy=True)

    def body_accelerations_bcrs(self, body: str, epochs) -> np.ndarray:
        """Read one body's BCRS acceleration at an epoch array."""
        epoch_array = tuple(require_tdb_epoch(epoch, name="epoch_tdb") for epoch in epochs)
        result = np.empty((len(epoch_array), 3), dtype=float)
        if not epoch_array:
            return result
        jd1 = np.asarray([epoch.jd1 for epoch in epoch_array], dtype=float)
        jd2 = np.asarray([epoch.jd2 for epoch in epoch_array], dtype=float)
        with self._lock:
            _, target_id = self._resolve_target(body)
            values = np.asarray(
                self._require_handle().compute_order(jd1, jd2, target_id, 0, self._state_units, 2),
                dtype=float,
            )
        if values.shape != (9, len(epoch_array)):
            raise RuntimeError(f"CALCEPH returned {values.shape} accelerations; expected (9, {len(epoch_array)}).")
        result[:, :] = values[6:9].T * 1000.0
        return result

    def _lunar_angles_rad(self, epoch_tdb: Epoch) -> np.ndarray:
        return self._lunar_angles_many_rad((epoch_tdb,))[0]

    def _lunar_angles_many_rad(self, epochs) -> np.ndarray:
        epoch_array = tuple(require_tdb_epoch(epoch, name="epoch_tdb") for epoch in epochs)
        result = np.empty((len(epoch_array), 6), dtype=float)
        if not epoch_array:
            return result
        jd1 = np.asarray([epoch.jd1 for epoch in epoch_array], dtype=float)
        jd2 = np.asarray([epoch.jd2 for epoch in epoch_array], dtype=float)
        with self._lock:
            angles = np.asarray(
                self._require_handle().orient_unit(jd1, jd2, self._orientation_target, self._angle_units),
                dtype=float,
            )
        if angles.shape != (6, len(epoch_array)):
            raise RuntimeError(f"CALCEPH returned {angles.shape} lunar orientations; expected (6, {len(epoch_array)}).")
        result[:, :] = angles.T
        return result

    def close(self) -> None:
        with self._lock:
            self._state_cache.clear()
            handle = self._handle
            self._handle = None
            if handle is not None:
                handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["CalcephEphemeris"]
