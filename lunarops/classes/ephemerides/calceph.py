"""CALCEPH implementation of the LLR ephemeris interface."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import RLock

import numpy as np

from lunarops.base.array_validation import finite_array, readonly_matrix3x3
from lunarops.classes.relativistic import (
    LunarRelativisticScaleConvention,
    l_b_minus_l_l_for_convention,
    normalize_lunar_relativistic_scale_convention,
)
from lunarops.classes.time import Epoch, TimeScale, TimeScaleConverter

from .base import (
    BodyState,
    Ephemeris,
    LongitudeLibrationCorrectionType,
    require_tdb_epoch,
)
from .body_ids import body_name as canonical_body_name
from .body_ids import naif_id
from .longitude_libration import (
    LongitudeLibrationCorrectionModel,
    make_longitude_libration_correction_model,
    normalize_longitude_libration_correction_type,
)

_J2000_TT_JD1 = 2451545.0
_J2000_TT_JD2 = 0.0
_SPICE_KERNEL_SUFFIX_ORDER = (".bsp", ".bpc")


def _passive_rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _passive_rotation_x(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, sine], [0.0, -sine, cosine]],
        dtype=float,
    )


class CalcephEphemeris(Ephemeris):
    """SPICE position and lunar-orientation kernels read through CALCEPH."""

    def __init__(
        self,
        kernel_directory: str | Path,
        *,
        lunar_relativistic_scale_convention: LunarRelativisticScaleConvention | str,
        longitude_libration_correction_type: (LongitudeLibrationCorrectionType | str | None) = None,
    ) -> None:
        directory = Path(kernel_directory).expanduser()
        if not directory.exists():
            raise FileNotFoundError(f"CALCEPH SPICE kernel directory not found: {directory}")
        if not directory.is_dir():
            raise ValueError(f"CALCEPH SPICE kernel source must be a directory: {directory}")
        try:
            from calcephpy import CalcephBin, Constants
        except (ImportError, OSError) as exc:  # pragma: no cover
            raise ImportError(
                f"The CALCEPH ephemeris requires a working calcephpy installation. Original import error: {exc}"
            ) from exc

        self._source_file = directory
        self._lunar_relativistic_scale_convention = normalize_lunar_relativistic_scale_convention(
            lunar_relativistic_scale_convention
        )
        self._l_b_minus_l_l = l_b_minus_l_l_for_convention(self._lunar_relativistic_scale_convention)
        self._longitude_libration_correction_type = normalize_longitude_libration_correction_type(
            longitude_libration_correction_type
        )
        self._longitude_libration_correction_model: LongitudeLibrationCorrectionModel = (
            make_longitude_libration_correction_model(self._longitude_libration_correction_type)
        )
        self._j2000_tdb: Epoch | None = None
        kernel_files = self._discover_spice_kernels(directory)
        self._handle = CalcephBin.open([str(kernel) for kernel in kernel_files])
        self._query_lock = RLock()
        self._prefetched = False
        self._target_id_cache: dict[str, int] = {}
        self._position_target_ids: frozenset[int] | None = None
        self._validated_position_target_ids: set[int] = set()
        self._state_units = Constants.UNIT_KM + Constants.UNIT_SEC + Constants.USE_NAIFID
        self._angle_units = Constants.UNIT_RAD + Constants.UNIT_SEC + Constants.USE_NAIFID
        try:
            self._orientation_target = self._spice_orientation_target(Constants)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _discover_spice_kernels(directory: Path) -> tuple[Path, ...]:
        by_suffix = {suffix: tuple(sorted(directory.glob(f"*{suffix}"))) for suffix in _SPICE_KERNEL_SUFFIX_ORDER}
        if not by_suffix[".bsp"]:
            raise ValueError(f"CALCEPH SPICE bundle contains no .bsp position kernel: {directory}")
        if not by_suffix[".bpc"]:
            raise ValueError(f"CALCEPH SPICE bundle contains no .bpc orientation kernel: {directory}")
        return tuple(kernel for suffix in _SPICE_KERNEL_SUFFIX_ORDER for kernel in by_suffix[suffix])

    def _spice_orientation_target(self, constants) -> int:
        handle = self._require_open_handle()
        timescale = handle.gettimescale()
        if timescale != constants.TDB:
            raise ValueError(f"CALCEPH SPICE bundle must use TDB, but gettimescale() returned {timescale}.")
        record_count = int(handle.getorientrecordcount())
        if record_count < 1:
            raise ValueError("CALCEPH SPICE bundle contains no orientation records.")
        records = [handle.getorientrecordindex2(index) for index in range(1, record_count + 1)]
        targets = {int(record[0]) for record in records}
        if len(targets) != 1:
            raise ValueError(
                f"CALCEPH SPICE bundle must contain exactly one orientation target; found {sorted(targets)}."
            )
        reference_frames = {int(record[3]) for record in records}
        if reference_frames != {1}:
            raise ValueError(
                "CALCEPH lunar orientation records must be relative to ICRF "
                "(SPICE frame code 1, historically labeled J2000); "
                f"found {sorted(reference_frames)}."
            )

        return targets.pop()

    @property
    def source_file_path(self) -> Path:
        return self._source_file

    def position_record_targets(self) -> tuple[int, ...]:
        """Return all target IDs with position records in the opened kernels."""
        return tuple(sorted(self._load_position_target_ids()))

    def _load_position_target_ids(self) -> frozenset[int]:
        if self._position_target_ids is None:
            handle = self._require_open_handle()
            count = int(handle.getpositionrecordcount())
            self._position_target_ids = frozenset(
                int(handle.getpositionrecordindex2(index)[0]) for index in range(1, count + 1)
            )
        return self._position_target_ids

    def _require_position_target(self, name: str, target_id: int) -> None:
        if target_id == 0 or target_id in self._validated_position_target_ids:
            return
        handle = self._require_open_handle()
        if not hasattr(handle, "getpositionrecordcount"):
            return
        available = self._load_position_target_ids()
        if target_id in available:
            self._validated_position_target_ids.add(target_id)
            return
        barycenter_name = f"{name} BARYCENTER"
        barycenter_id = naif_id(barycenter_name)
        hint = ""
        if barycenter_id is not None and barycenter_id in available:
            hint = f"; use {barycenter_name!r} (NAIF {barycenter_id}) for this kernel"
        raise KeyError(f"Body {name!r} (NAIF {target_id}) has no position record in the opened CALCEPH kernels{hint}")

    def resolve_body_id(self, body_name: str) -> int:
        """Resolve a CALCEPH name or ``NAIF:<id>`` target dynamically."""
        normalized = canonical_body_name(body_name)
        cached = self._target_id_cache.get(normalized)
        if cached is not None:
            return cached
        target = naif_id(normalized)
        if target is not None:
            self._target_id_cache[normalized] = target
            return target
        handle = self._require_open_handle()
        target = handle.getidbyname(normalized, self._state_units)
        if target is None:
            raise KeyError(f"Unknown CALCEPH body name: {body_name!r}")
        result = int(target)
        self._target_id_cache[normalized] = result
        return result

    @property
    def l_b_minus_l_l(self) -> float:
        return self._l_b_minus_l_l

    @property
    def lunar_relativistic_scale_convention(
        self,
    ) -> LunarRelativisticScaleConvention:
        return self._lunar_relativistic_scale_convention

    @property
    def longitude_libration_correction_type(
        self,
    ) -> LongitudeLibrationCorrectionType:
        return self._longitude_libration_correction_type

    def _require_open_handle(self):
        if self._handle is None:
            raise RuntimeError("CALCEPH ephemeris is closed.")
        return self._handle

    def close(self) -> None:
        with self._query_lock:
            handle = self._handle
            self._handle = None
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:  # pragma: no cover - backend cleanup detail
                    raise RuntimeError("CALCEPH ephemeris close() failed.") from exc

    def body_state_bcrs(self, body_name: str, epoch_tdb: Epoch) -> BodyState:
        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        normalized_body_name = canonical_body_name(body_name)
        target_id = self.resolve_body_id(body_name)
        self._require_position_target(normalized_body_name, target_id)
        with self._query_lock:
            values = self._require_open_handle().compute_unit(
                epoch_tdb.jd1,
                epoch_tdb.jd2,
                target_id,
                0,
                self._state_units,
            )
        state = np.asarray(values, dtype=float)
        if state.size < 6:
            raise RuntimeError(
                f"CALCEPH returned {state.size} state values for {normalized_body_name}; expected at least six."
            )
        return BodyState(
            position_m=state[:3] * 1000.0,
            velocity_mps=state[3:6] * 1000.0,
        )

    def body_states_bcrs(self, body_names: Sequence[str], epoch_tdb: Epoch) -> dict[str, BodyState]:
        """Return typed states built from one contiguous batch query."""
        names = tuple(canonical_body_name(name) for name in body_names)
        vectors = self.body_state_vectors_bcrs(names, epoch_tdb)
        return {name: BodyState(vector[:3], vector[3:]) for name, vector in zip(names, vectors, strict=True)}

    def body_state_vectors_bcrs(self, body_names: Sequence[str], epoch_tdb: Epoch) -> np.ndarray:
        """Return all requested targets in a contiguous SI array after prefetch."""
        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        names = tuple(canonical_body_name(name) for name in body_names)
        with self._query_lock:
            handle = self._require_open_handle()
            if not self._prefetched:
                handle.prefetch()
                self._prefetched = True
            target_ids = tuple(self.resolve_body_id(name) for name in names)
            for name, target_id in zip(names, target_ids, strict=True):
                self._require_position_target(name, target_id)
            result = np.empty((len(names), 6), dtype=float)
            for index, target_id in enumerate(target_ids):
                values = np.asarray(
                    handle.compute_unit(epoch_tdb.jd1, epoch_tdb.jd2, target_id, 0, self._state_units),
                    dtype=float,
                )
                if values.size < 6:
                    raise RuntimeError(
                        f"CALCEPH returned {values.size} values for {names[index]}; expected at least six"
                    )
                result[index] = values[:6] * 1000.0
        return result

    def _lunar_orientation_angles_rad(self, epoch_tdb: Epoch) -> np.ndarray:
        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        with self._query_lock:
            values = self._require_open_handle().orient_unit(
                epoch_tdb.jd1,
                epoch_tdb.jd2,
                self._orientation_target,
                self._angle_units,
            )
        angles = np.asarray(values, dtype=float)
        if angles.size < 3:
            raise RuntimeError("CALCEPH libration target returned fewer than three angles.")
        result = np.array(angles[:3], dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def body_acceleration_bcrs(self, body_name: str, epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        name = canonical_body_name(body_name)
        target_id = self.resolve_body_id(name)
        self._require_position_target(name, target_id)
        with self._query_lock:
            values = np.asarray(
                self._require_open_handle().compute_order(
                    epoch.jd1,
                    epoch.jd2,
                    target_id,
                    0,
                    self._state_units,
                    2,
                ),
                dtype=float,
            )
        if values.shape != (9,):
            raise RuntimeError("CALCEPH order-2 query must return nine components.")
        return finite_array(values[6:9] * 1000.0, size=3, name="acceleration_mps2", copy=True, readonly=True)

    def _j2000_tdb_epoch(self) -> Epoch:
        if self._j2000_tdb is None:
            converter = TimeScaleConverter()
            self._j2000_tdb = converter.tt2tdb(Epoch(_J2000_TT_JD1, _J2000_TT_JD2, TimeScale.TT))
        return self._j2000_tdb

    def longitude_libration_correction_rad(self, epoch_tdb: Epoch) -> float:
        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        if self.longitude_libration_correction_type is LongitudeLibrationCorrectionType.NONE:
            return 0.0
        return self._longitude_libration_correction_model.correction_rad(
            epoch_tdb,
            j2000_epoch_tdb=self._j2000_tdb_epoch(),
        )

    def pa2lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        phi, theta, psi = self._lunar_orientation_angles_rad(epoch_tdb)
        psi += self.longitude_libration_correction_rad(epoch_tdb)
        lcrs_to_pa_matrix = _passive_rotation_z(psi) @ _passive_rotation_x(theta) @ _passive_rotation_z(phi)
        return readonly_matrix3x3(
            lcrs_to_pa_matrix.T,
            name="pa2lcrs_matrix",
        )


def load_calceph_ephemeris(
    kernel_directory: str | Path,
    *,
    lunar_relativistic_scale_convention: LunarRelativisticScaleConvention | str,
    longitude_libration_correction_type: (LongitudeLibrationCorrectionType | str | None) = None,
) -> CalcephEphemeris:
    return CalcephEphemeris(
        kernel_directory,
        lunar_relativistic_scale_convention=lunar_relativistic_scale_convention,
        longitude_libration_correction_type=longitude_libration_correction_type,
    )


__all__ = ["CalcephEphemeris", "load_calceph_ephemeris"]
